"""
endloss_secondpass.py — STANDALONE lightweight test of the end-loss second-pass
idea on a real model, with layerwise CLC as the comparison baseline.

Memory-frugal rewrite (v2). The earlier version stored each Linear's lattice as
five dense float32 tensors with per-ELEMENT scale/zp (~20x the bf16 weight) and
kept five full model copies resident on the GPU — a 1B model could OOM a 24GB
card. This version:
  * compact lattice: int16 codes, per-GROUP scale/zp, int8 frozen flip-dir,
    residual kept only when CLC needs it;
  * one working model on the GPU at a time; arms run sequentially and reload FP
    weights from a CPU snapshot between them; teacher built only for --loss-kind kl;
  * end-loss backward sets requires_grad only on the candidate layers.

Scientific content is unchanged — one clean contrast at equal flip budget:

    BASELINE (CLC, layerwise):   flip codes to cancel each output row's
             ACTIVATION-WEIGHTED reconstruction error, per Linear, independently.
             Local objective; no end loss; no cross-layer signal.
    OURS     (end-loss 2nd pass): flip codes scored by the gradient of the MODEL
             END LOSS, accepted on the real end-loss forward. c_i = dL_end/d(delta_i)
             is a total derivative through all downstream layers => cross-layer at
             first order WITHOUT any explicit global state.

Same RTN grid, same +-1 lattice flips, same budget, same PPL eval => the only
difference is the objective.

NOTE: run this yourself (GPU-first; lattice math float32, per-group). Nothing runs
on import. Start small (Qwen2-0.5B / Llama-3.2-1B). An end-loss backward needs the
full-model graph, so "lightweight" means a small model, few --layers, tiny
--grad-batch. On 24GB, a 1B model with --loss-kind ce and --layers 0-3 fits; KL
over a 128k-vocab model is far heavier (full logits with grad) — prefer ce first.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

LATTICE_DTYPE = torch.float32


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; using cpu")
        return torch.device("cpu")
    return torch.device(pref)


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 1. Compact fixed-grid RTN lattice. W_eff = (codes - zp) * scale, scale/zp per
#    group. codes int16 (mutable), fdir int8 (frozen complementary flip dir).
# ============================================================================
@dataclass
class Lattice:
    codes: torch.Tensor        # (out, padded) int16   -- mutable integer codes
    fdir: torch.Tensor         # (out, padded) int8    -- frozen complementary flip dir
    scale_g: torch.Tensor      # (out, n_groups) float -- per-group scale
    zp_g: torch.Tensor         # (out, n_groups) float -- per-group zero point
    gs: int
    n_groups: int
    in_features: int
    min_int: int
    max_int: int
    res: Optional[torch.Tensor] = None   # (out, padded) float, = pre - codes0 (CLC only)

    @property
    def padded(self) -> int:
        return self.n_groups * self.gs

    def _expand(self, tg: torch.Tensor) -> torch.Tensor:
        out = self.codes.shape[0]
        return tg.unsqueeze(-1).expand(out, self.n_groups, self.gs).reshape(out, self.padded)

    def scale_full(self) -> torch.Tensor:
        return self._expand(self.scale_g.to(LATTICE_DTYPE))

    def weight(self) -> torch.Tensor:
        s = self.scale_full()
        z = self._expand(self.zp_g.to(LATTICE_DTYPE))
        w = (self.codes.to(LATTICE_DTYPE) - z) * s
        return w[:, : self.in_features]

    def step(self) -> torch.Tensor:
        """Weight-space change of a single +-1 complementary flip (out, padded)."""
        return self.fdir.to(LATTICE_DTYPE) * self.scale_full()

    def valid(self) -> torch.Tensor:
        tgt = self.codes.to(torch.int32) + self.fdir.to(torch.int32)
        return (tgt >= self.min_int) & (tgt <= self.max_int)


@torch.no_grad()
def rtn_quantize(w: torch.Tensor, bits: int, group_size: int, keep_residual: bool) -> Lattice:
    dev = w.device
    w = w.to(LATTICE_DTYPE)
    out, in_features = w.shape
    gs = in_features if group_size is None or group_size <= 0 else group_size
    n_groups = (in_features + gs - 1) // gs
    padded = n_groups * gs
    if padded > in_features:
        wp = torch.zeros(out, padded, device=dev, dtype=w.dtype)
        wp[:, :in_features] = w
    else:
        wp = w
    wg = wp.reshape(out, n_groups, gs)
    wmin = wg.min(dim=2)[0]
    wmax = wg.max(dim=2)[0]
    max_int = 2 ** bits - 1
    scale_g = ((wmax - wmin) / max_int).clamp(min=1e-8)      # (out, n_groups)
    zp_g = torch.round(-wmin / scale_g).clamp(0, max_int)    # (out, n_groups)
    scale_full = scale_g.unsqueeze(-1).expand(out, n_groups, gs).reshape(out, padded)
    zp_full = zp_g.unsqueeze(-1).expand(out, n_groups, gs).reshape(out, padded)
    pre = wp / scale_full + zp_full
    codes = torch.round(pre).clamp(0, max_int)
    fdir = torch.sign(pre - codes)
    fdir[fdir == 0] = 1.0
    lat = Lattice(
        codes=codes.to(torch.int16), fdir=fdir.to(torch.int8),
        scale_g=scale_g.to(torch.float16), zp_g=zp_g.to(torch.float16),
        gs=gs, n_groups=n_groups, in_features=in_features, min_int=0, max_int=max_int,
        res=(pre - codes).to(torch.float16) if keep_residual else None,
    )
    del pre, scale_full, zp_full, wp, wg
    return lat


def target_linears(model) -> dict:
    out = {}
    for name, m in model.model.layers.named_modules():
        if isinstance(m, nn.Linear):
            out[f"model.layers.{name}"] = m
    return out


@torch.no_grad()
def quantize_model_rtn(model, bits: int, group_size: int, keep_residual: bool) -> dict:
    lats = {}
    for name, m in target_linears(model).items():
        lat = rtn_quantize(m.weight.data, bits, group_size, keep_residual)
        m.weight.data.copy_(lat.weight().to(m.weight.dtype))
        lats[name] = lat
        _free()
    return lats


@torch.no_grad()
def write_back(model, lats: dict):
    lin = target_linears(model)
    for name, lat in lats.items():
        lin[name].weight.data.copy_(lat.weight().to(lin[name].weight.dtype))


@torch.no_grad()
def _apply_flip(lat: Lattice, rows: torch.Tensor, cols: torch.Tensor):
    lat.codes[rows, cols] += lat.fdir[rows, cols].to(lat.codes.dtype)
    lat.codes.clamp_(lat.min_int, lat.max_int)


# ============================================================================
# 2. BASELINE: layerwise CLC (standalone faithful reimplementation).
#    err_o = sum_i (pre - codes)*scale * a_i  (== (w_fp - w_q) . a);  flip to cancel.
# ============================================================================
def _james_stein(mean: torch.Tensor) -> torch.Tensor:
    p = mean.numel()
    if p < 3:
        return mean
    gm = mean.mean(); dev = mean - gm; ss = (dev ** 2).sum()
    if ss < 1e-10:
        return mean
    var = ((mean - gm).abs().mean()) ** 2
    shr = (((p - 2) * var.clamp(min=1e-8)) / ss).clamp(0, 1)
    return gm + (1 - shr) * dev


def _knee_threshold(a_abs: torch.Tensor) -> torch.Tensor:
    s = torch.sort(a_abs, descending=True)[0]
    half = s[: a_abs.numel() // 2]
    n = half.numel()
    if n < 3:
        return s[min(int(0.05 * s.numel()), s.numel() - 1)]
    y = half.float().cpu()
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    xn = torch.linspace(0, 1, n)
    line = yn[0] + (yn[-1] - yn[0]) * xn
    knee = int(torch.argmax((yn - line).abs()).item())
    return s[knee]


@torch.no_grad()
def clc_correct_linear(lat: Lattice, act_mean: torch.Tensor, max_flip_percent: float,
                       use_js: bool = True):
    """Apply CLC in place. Returns (n_flips, flipped_rows, flipped_cols)."""
    assert lat.res is not None, "CLC needs the lattice built with keep_residual=True"
    dev = lat.codes.device
    d, pin = lat.in_features, lat.padded
    a = torch.zeros(pin, device=dev, dtype=LATTICE_DTYPE)
    a[:d] = (_james_stein(act_mean) if use_js else act_mean).to(dev, LATTICE_DTYPE)

    scale = lat.scale_full()
    err = (lat.res.to(LATTICE_DTYPE) * scale * a[None, :]).sum(dim=1)   # per-row current error
    fdir = lat.fdir.to(LATTICE_DTYPE)
    impact = a[None, :] * fdir * scale
    tgt_sign = torch.sign(err).unsqueeze(1)
    valid = (torch.sign(impact) == tgt_sign) & lat.valid()

    thr = _knee_threshold(a.abs())
    valid &= (a.abs() <= thr).unsqueeze(0)

    rc = lat.res.abs().to(LATTICE_DTYPE).clone()
    rc[~valid] = -1.0
    order = torch.argsort(rc, dim=1, descending=True)
    imp_s = torch.gather(impact, 1, order) * torch.gather(valid.long(), 1, order)
    cum = torch.cumsum(imp_s, dim=1)
    resid = (err.unsqueeze(1) - cum).abs()
    allr = torch.cat([err.abs().unsqueeze(1), resid], dim=1)
    best_k = torch.argmin(allr, dim=1)
    idxr = torch.arange(pin, device=dev).unsqueeze(0)
    take = (idxr < best_k.unsqueeze(1)) & torch.gather(valid, 1, order).bool()
    fd_s = torch.gather(fdir, 1, order)
    fd_s[~take] = 0.0
    budget = int(max_flip_percent * d)
    cnt = (fd_s != 0).long().cumsum(1)        # true count (v1 used signed cumsum = a bug)
    fd_s[cnt > budget] = 0.0

    flipped = torch.zeros_like(lat.codes, dtype=torch.bool)
    flipped.scatter_(1, order, fd_s != 0)
    lat.codes += (torch.zeros_like(lat.codes).scatter_(1, order, fd_s.to(lat.codes.dtype)))
    lat.codes.clamp_(lat.min_int, lat.max_int)
    idx = flipped.nonzero(as_tuple=False)
    return int(idx.shape[0]), idx[:, 0].contiguous(), idx[:, 1].contiguous()


@torch.no_grad()
def run_clc_baseline(model, lats: dict, act_means: dict, max_flip_percent: float):
    total = 0
    flips = {}
    for name, lat in lats.items():
        if name in act_means:
            n, r, c = clc_correct_linear(lat, act_means[name], max_flip_percent)
            total += n
            if n > 0:
                flips[name] = (r, c)
    write_back(model, lats)
    return {"method": "clc_layerwise", "flips_total": total}, flips


# ============================================================================
# 3. Activation means (for CLC) + end loss.
# ============================================================================
@torch.no_grad()
def collect_act_means(model, calib_ids, device, batch: int = 2) -> dict:
    lin = target_linears(model)
    acc = {n: None for n in lin}
    cnt = {n: 0 for n in lin}
    handles = []
    for n, m in lin.items():
        def hook(_m, inp, _o, key=n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).to(LATTICE_DTYPE)
            s = x.sum(0)
            acc[key] = s if acc[key] is None else acc[key] + s
            cnt[key] += x.shape[0]
        handles.append(m.register_forward_hook(hook))
    for i in range(0, len(calib_ids), batch):
        ids = torch.stack(calib_ids[i:i + batch]).to(device)
        model(ids)
        _free()
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(cnt[n], 1)) for n in lin if acc[n] is not None}


def end_loss(student, teacher, ids, device, kind: str) -> torch.Tensor:
    logits_s = student(ids.to(device)).logits
    if kind == "ce":
        sl = logits_s[:, :-1].reshape(-1, logits_s.shape[-1]).float()
        lab = ids[:, 1:].reshape(-1).to(device)
        return nn.functional.cross_entropy(sl, lab)
    with torch.no_grad():
        logp_t = torch.log_softmax(teacher(ids.to(device)).logits.float(), dim=-1)
    logp_s = torch.log_softmax(logits_s.float(), dim=-1)
    return (logp_t.exp() * (logp_t - logp_s)).sum(-1).mean()


@torch.no_grad()
def eval_end_loss(student, teacher, ids_list, device, kind, batch: int = 2) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(ids_list), batch):
        ids = torch.stack(ids_list[i:i + batch]).to(device)
        tot += float(end_loss(student, teacher, ids, device, kind)) * ids.shape[0]
        n += ids.shape[0]
        _free()
    return tot / max(n, 1)


# ============================================================================
# 4. OURS: end-loss-guided second pass. Predict with the end-loss gradient (one
#    backward => c for all candidates), accept on the real end-loss forward.
# ============================================================================
@dataclass
class EndLossConfig:
    max_flip_percent: float = 0.02
    layers: Optional[set] = None
    batch_init: int = 256
    batch_max: int = 4096
    refresh_every: int = 4
    max_rounds: int = 60
    grad_batch: int = 2
    loss_kind: str = "kl"
    seed: int = 0


@dataclass
class CostLedger:
    backwards: int = 0
    forwards: int = 0


def _layer_of(name: str) -> int:
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except Exception:
        return -1


def _cand_names(lats, layers) -> list:
    return [n for n in lats if layers is None or _layer_of(n) in layers]


def endloss_gradient(student, teacher, lats, opt_ids, device, cfg) -> dict:
    """c_i = (dL_end/dW)_i * step_i. requires_grad only on candidate layers."""
    lin = target_linears(student)
    names = _cand_names(lats, cfg.layers)
    for n, m in lin.items():
        want = n in names
        m.weight.requires_grad_(want)
        if m.weight.grad is not None:
            m.weight.grad = None
    for i in range(0, len(opt_ids), cfg.grad_batch):
        ids = torch.stack(opt_ids[i:i + cfg.grad_batch]).to(device)
        end_loss(student, teacher, ids, device, cfg.loss_kind).backward()
        _free()
    c = {}
    with torch.no_grad():
        for n in names:
            g = lin[n].weight.grad
            if g is None:
                continue
            step = lats[n].step()[:, : lats[n].in_features]
            c[n] = (g.to(LATTICE_DTYPE) * step)
    for m in lin.values():
        m.weight.requires_grad_(False)
        m.weight.grad = None
    _free()
    return c


def run_endloss_secondpass(student, teacher, lats, opt_ids, acc_ids, device,
                           cfg: EndLossConfig) -> dict:
    torch.manual_seed(cfg.seed)
    ledger = CostLedger()
    t0 = time.time()
    names = _cand_names(lats, cfg.layers)

    used = {n: torch.zeros_like(lats[n].codes, dtype=torch.bool) for n in names}
    blocked = {n: torch.zeros_like(lats[n].codes, dtype=torch.bool) for n in names}
    budget = {n: int(cfg.max_flip_percent * lats[n].in_features) for n in names}
    row_cnt = {n: torch.zeros(lats[n].codes.shape[0], dtype=torch.long, device=device) for n in names}

    base_L = eval_end_loss(student, teacher, acc_ids, device, cfg.loss_kind); ledger.forwards += 1
    L0 = base_L
    bsize = cfg.batch_init
    since_refresh = 0
    c = endloss_gradient(student, teacher, lats, opt_ids, device, cfg); ledger.backwards += 1
    flips_total = 0
    rounds = 0

    for rnd in range(cfg.max_rounds):
        rounds = rnd + 1
        cand = []
        for n in names:
            lat = lats[n]; d = lat.in_features; cval = c.get(n)
            if cval is None:
                continue
            elig = (lat.valid()[:, :d] & ~used[n][:, :d] & ~blocked[n][:, :d]
                    & (cval < 0) & (row_cnt[n] < budget[n])[:, None])
            idx = elig.nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            vals = cval[idx[:, 0], idx[:, 1]]
            cand.append((vals, torch.full_like(vals, names.index(n)), idx[:, 0], idx[:, 1]))
        if not cand:
            break
        allv = torch.cat([x[0] for x in cand]); alln = torch.cat([x[1] for x in cand]).long()
        allr = torch.cat([x[2] for x in cand]).long(); allc = torch.cat([x[3] for x in cand]).long()
        k = min(int(bsize), allv.numel())
        sel = torch.topk(allv, k, largest=False).indices

        proposal = {n: [] for n in names}
        tmp = {n: row_cnt[n].clone() for n in names}
        for s in sel.tolist():
            n = names[int(alln[s])]; r, col = int(allr[s]), int(allc[s])
            if tmp[n][r] < budget[n]:
                proposal[n].append((r, col)); tmp[n][r] += 1
        if not any(proposal.values()):
            break

        snapshot = {n: lats[n].codes.clone() for n in proposal if proposal[n]}
        for n, rc in proposal.items():
            if rc:
                rows = torch.tensor([r for r, _ in rc], device=device)
                cols = torch.tensor([col for _, col in rc], device=device)
                _apply_flip(lats[n], rows, cols)
        write_back(student, {n: lats[n] for n in proposal if proposal[n]})
        newL = eval_end_loss(student, teacher, acc_ids, device, cfg.loss_kind); ledger.forwards += 1
        n_prop = sum(len(v) for v in proposal.values())

        if newL < base_L - 1e-9:
            base_L = newL
            for n, rc in proposal.items():
                for r, col in rc:
                    used[n][r, col] = True; row_cnt[n][r] += 1
            flips_total += n_prop
            bsize = min(int(bsize * 2), cfg.batch_max)
            since_refresh += 1
            if since_refresh >= cfg.refresh_every:
                c = endloss_gradient(student, teacher, lats, opt_ids, device, cfg)
                ledger.backwards += 1; since_refresh = 0
        else:
            for n in proposal:
                if proposal[n]:
                    lats[n].codes.copy_(snapshot[n])
            write_back(student, {n: lats[n] for n in proposal if proposal[n]})
            if n_prop <= 1:
                for n, rc in proposal.items():
                    for r, col in rc:
                        blocked[n][r, col] = True
            bsize = max(1, bsize // 2)
        del snapshot; _free()

    write_back(student, lats)
    return {"method": f"endloss_2ndpass[{cfg.loss_kind}]", "flips_total": flips_total,
            "rounds": rounds, "end_loss_before": L0, "end_loss_after": base_L,
            "cost": {"backwards": ledger.backwards, "forwards": ledger.forwards},
            "time_s": time.time() - t0}


# ============================================================================
# 5. Audits — the real-model headline + the premise check.
# ============================================================================
@torch.no_grad()
def audit_grad_vs_truth(student, teacher, lats, c, opt_ids, device, cfg, n_samples: int = 48) -> dict:
    names = [n for n in c]
    if not names:
        return {}
    L0 = eval_end_loss(student, teacher, opt_ids, device, cfg.loss_kind)
    pred, true = [], []
    g = torch.Generator().manual_seed(cfg.seed)
    for _ in range(n_samples):
        n = names[int(torch.randint(len(names), (1,), generator=g))]
        lat = lats[n]; d = lat.in_features
        elig = lat.valid()[:, :d].nonzero(as_tuple=False)
        if elig.numel() == 0:
            continue
        r, col = elig[int(torch.randint(elig.shape[0], (1,), generator=g))].tolist()
        pred.append(float(c[n][r, col]))
        snap = lat.codes[r, col].item()
        _apply_flip(lat, torch.tensor([r], device=device), torch.tensor([col], device=device))
        write_back(student, {n: lat})
        true.append(eval_end_loss(student, teacher, opt_ids, device, cfg.loss_kind) - L0)
        lat.codes[r, col] = snap; write_back(student, {n: lat})
    if len(pred) < 3:
        return {"n": len(pred)}
    p = torch.tensor(pred); t = torch.tensor(true)
    return {"n": len(pred),
            "pearson_pred_vs_true": float(torch.corrcoef(torch.stack([p, t]))[0, 1]),
            "sign_disagree": float(((p.sign() != t.sign()) & (t.abs() > 1e-12)).float().mean()),
            "pred_abs_mean": float(p.abs().mean()), "true_abs_mean": float(t.abs().mean())}


@torch.no_grad()
def audit_clc_vs_endloss(clc_flips: dict, c: dict) -> dict:
    """Of the flips CLC made, fraction the end-loss gradient calls HARMFUL (c>0)."""
    harmful, total = 0, 0
    for name, (rows, cols) in clc_flips.items():
        if name not in c:
            continue
        cv = c[name][rows, cols]
        harmful += int((cv > 0).sum().item()); total += cv.numel()
    return {"clc_flips_scored": total,
            "frac_harmful_by_endloss": (harmful / total) if total else float("nan")}


# ============================================================================
# 6. PPL eval + data.
# ============================================================================
@torch.no_grad()
def eval_ppl(model, ids_1d: torch.Tensor, device, seqlen: int = 2048,
             max_chunks: int = 0) -> float:
    model.eval()
    n = ids_1d.numel()
    nll, cnt, done = 0.0, 0, 0
    for i in range(0, n - 1, seqlen):
        j = min(i + seqlen, n - 1)
        logits = model(ids_1d[i:j].unsqueeze(0).to(device)).logits
        tgt = ids_1d[i + 1:j + 1].to(device)
        nll += float(nn.functional.cross_entropy(logits[0].float(), tgt, reduction="sum"))
        cnt += tgt.numel(); done += 1
        _free()
        if max_chunks and done >= max_chunks:
            break
    return math.exp(nll / max(cnt, 1))


def load_data(tokenizer, n_calib: int, seqlen: int, seed: int):
    from datasets import load_dataset
    tr = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    te = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    g = torch.Generator().manual_seed(seed)
    enc = tokenizer("\n\n".join(tr["text"]), return_tensors="pt").input_ids[0]
    calib = []
    for _ in range(n_calib):
        s = int(torch.randint(0, enc.numel() - seqlen - 1, (1,), generator=g))
        calib.append(enc[s:s + seqlen])
    test_ids = tokenizer("\n\n".join(te["text"]), return_tensors="pt").input_ids[0]
    return calib, test_ids


# ============================================================================
# 7. Head-to-head harness, memory-frugal: one GPU model reused across arms.
# ============================================================================
def load_model(path, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    return m, tok


def G_recovery(ppl_ptq, ppl_ref, ppl_fp):
    if ppl_ptq <= ppl_fp:
        return float("nan")
    return (ppl_ptq - ppl_ref) / (ppl_ptq - ppl_fp)


def main():
    ap = argparse.ArgumentParser(description="Standalone end-loss second-pass test; CLC baseline")
    ap.add_argument("--model-path", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--opt-seqs", type=int, default=16)
    ap.add_argument("--acc-seqs", type=int, default=8)
    ap.add_argument("--grad-batch", type=int, default=2)
    ap.add_argument("--flip-percent", type=float, default=0.02)
    ap.add_argument("--loss-kind", choices=["kl", "ce"], default="ce")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--ppl-max-chunks", type=int, default=0, help="0 = full wikitext2 test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    def parse_layers(spec, n):
        if spec.strip().lower() in ("", "all"):
            return None
        out = set()
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
            elif part.strip():
                out.add(int(part))
        return {i for i in out if 0 <= i < n}

    print(f"[load] {args.model_path} on {device} ({args.dtype})")
    model, tok = load_model(args.model_path, device, dtype)
    n_layers = len(model.model.layers)
    layers = parse_layers(args.layers, n_layers)
    calib, test_ids = load_data(tok, args.n_calib, args.seqlen, args.seed)
    opt_ids = calib[: args.opt_seqs]
    acc_ids = calib[args.opt_seqs: args.opt_seqs + args.acc_seqs]

    # FP weights kept on CPU; one GPU model reused across arms.
    fp_cpu = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    def reset_fp():
        model.load_state_dict(fp_cpu); _free()

    results = {}
    ppl_fp = eval_ppl(model, test_ids, device, args.seqlen, args.ppl_max_chunks)
    results["fp"] = {"ppl": ppl_fp}; print(f"[fp]  ppl={ppl_fp:.4f}")

    # ---- arm 1: RTN ----
    reset_fp()
    lats = quantize_model_rtn(model, args.bits, args.group_size, keep_residual=False)
    ppl_rtn = eval_ppl(model, test_ids, device, args.seqlen, args.ppl_max_chunks)
    results["rtn"] = {"ppl": ppl_rtn}; print(f"[rtn] ppl={ppl_rtn:.4f}")
    act_means = collect_act_means(model, calib, device)
    del lats; _free()

    # ---- arm 2: RTN + CLC (baseline) ----
    reset_fp()
    lats_clc = quantize_model_rtn(model, args.bits, args.group_size, keep_residual=True)
    clc_log, clc_flips = run_clc_baseline(model, lats_clc, act_means, args.flip_percent)
    ppl_clc = eval_ppl(model, test_ids, device, args.seqlen, args.ppl_max_chunks)
    results["rtn_clc"] = {"ppl": ppl_clc, **clc_log, "G_recovery": G_recovery(ppl_rtn, ppl_clc, ppl_fp)}
    print(f"[clc] ppl={ppl_clc:.4f} flips={clc_log['flips_total']} G={results['rtn_clc']['G_recovery']:.3f}")
    # drop residuals we no longer need; keep clc_flips (sparse) for the audit
    for lat in lats_clc.values():
        lat.res = None
    del lats_clc; _free()

    # ---- teacher (only for KL) ----
    teacher = None
    if args.loss_kind == "kl":
        import copy
        reset_fp()
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        _free()

    # ---- arm 3: RTN + end-loss second pass (ours) ----
    reset_fp()
    lats_el = quantize_model_rtn(model, args.bits, args.group_size, keep_residual=False)
    cfg = EndLossConfig(max_flip_percent=args.flip_percent, layers=layers,
                        loss_kind=args.loss_kind, grad_batch=args.grad_batch, seed=args.seed)
    el_log = run_endloss_secondpass(model, teacher, lats_el, opt_ids, acc_ids, device, cfg)
    ppl_el = eval_ppl(model, test_ids, device, args.seqlen, args.ppl_max_chunks)
    results["rtn_endloss"] = {"ppl": ppl_el, **el_log, "G_recovery": G_recovery(ppl_rtn, ppl_el, ppl_fp)}
    print(f"[ours] ppl={ppl_el:.4f} flips={el_log['flips_total']} cost={el_log['cost']} "
          f"G={results['rtn_endloss']['G_recovery']:.3f}")

    if args.audit:
        c = endloss_gradient(model, teacher, lats_el, opt_ids, device, cfg)
        a1 = audit_grad_vs_truth(model, teacher, lats_el, c, opt_ids, device, cfg)
        a2 = audit_clc_vs_endloss(clc_flips, c)
        results["audit"] = {"grad_vs_truth": a1, "clc_vs_endloss": a2}
        print(f"[audit] grad_vs_truth={a1}")
        print(f"[audit] clc_vs_endloss={a2}")

    print("\n==== SUMMARY (lower ppl better; G_recovery higher better) ====")
    for k, v in results.items():
        print(f"  {k:<14}: {v}")
    print("\nFair-comparison invariants: same RTN grid, same +-1 lattice flips, "
          "same flip budget, same PPL eval. Only the objective differs.")


if __name__ == "__main__":
    main()