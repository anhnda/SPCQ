"""
endloss_secondpass.py — a STANDALONE lightweight test of the research idea on a
real model, with layerwise CLC as the comparison baseline.

This is NOT the QK-ALC framework and shares no code with it. QK-ALC is a
different method (attention Q-K interaction correction, per-layer softmax-KL
proxy). Here we test the actual hypothesis: does an END-LOSS-guided second pass
over a fixed PTQ grid beat a purely-local layerwise correction, at equal flip
budget? The scientific content is one clean contrast:

    BASELINE  (CLC, layerwise):  flip integer codes to cancel each output row's
              ACTIVATION-WEIGHTED RECONSTRUCTION error, per Linear, independently.
              Local objective. No end loss. No cross-layer signal.

    OURS      (end-loss 2nd pass): flip codes scored by the gradient of the MODEL
              END LOSS (teacher-FP vs student-quant KL on calibration), accepted
              on the real end-loss forward. The end-loss gradient c_i = dL/d(delta_i)
              is a TOTAL derivative through all downstream layers, so it carries
              cross-layer information at first order WITHOUT any explicit global
              state. That is exactly the cheap test of "is there cross-layer,
              end-loss-relevant value beyond per-layer local correction?".

Both run on the SAME RTN grid, the SAME lattice flip convention (+-1 toward the
other side of pre_round, in-range, per-row budget), and the SAME PPL eval, so the
only thing that differs is the objective. Equal budget => fair head-to-head.

Also reports, as a real-model analogue of the toy "headline":
  * audit: predicted c_i vs TRUE single-flip end-loss change (finite difference);
  * disagreement: fraction of CLC-chosen flips the end-loss gradient says are
    HARMFUL (sign c_i > 0). If that fraction is ~0, local and global agree and the
    whole premise is moot; if it is large, end loss sees something CLC cannot.
  * two-axis cost ledger (H1d): #end-loss backwards (c refreshes) and #end-loss
    forwards (acceptance checks) — never hide state/scoring cost behind forwards.

NOTE: run this yourself (GPU-first; lattice math in float32). Nothing runs on
import; __main__ is a manual harness. Start on a SMALL model (Qwen2-0.5B): an
end-loss backward needs the full-model graph, so "lightweight" holds at <=1B, not
at 8B. Models with QK-norm/partial-RoPE are irrelevant here (weight-space method,
works on every nn.Linear), unlike QK-ALC.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

# GPU-first per workflow preference. Lattice/stat math is float32 so pre_round is
# fine enough to decide rounding directions; the model itself can stay bf16.
LATTICE_DTYPE = torch.float32


def pick_device(pref: str) -> torch.device:
    if pref == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(pref)


# ============================================================================
# 1. Fixed-grid RTN quantization, keeping the integer lattice so a second pass
#    can flip codes. (Group-wise asymmetric min-max; same math as a standard RTN.)
# ============================================================================
@dataclass
class Lattice:
    """Mutable integer lattice of one Linear. W_eff = (codes - zp) * scale."""
    codes: torch.Tensor        # (out, padded_in) float (integer-valued)
    pre: torch.Tensor          # (out, padded_in) pre-round real codes
    scale: torch.Tensor        # (out, padded_in)
    zp: torch.Tensor           # (out, padded_in)
    min_int: int
    max_int: int
    in_features: int
    codes0: torch.Tensor = field(default=None)   # codes before any 2nd pass (for budget/freeze)

    def weight(self) -> torch.Tensor:
        w = (self.codes - self.zp) * self.scale
        return w[:, : self.in_features]

    def flip_dir(self) -> torch.Tensor:
        """Complementary direction: toward the other adjacent level of pre_round."""
        d = torch.sign(self.pre - self.codes)
        d[d == 0] = 1.0
        return d

    def step(self) -> torch.Tensor:
        """Weight-space change of a single +-1 complementary flip, per coord."""
        return self.flip_dir() * self.scale

    def valid(self, min_residual: float = 0.0) -> torch.Tensor:
        tgt = self.codes + self.flip_dir()
        v = (tgt >= self.min_int) & (tgt <= self.max_int)
        if min_residual > 0:
            v &= (self.pre - self.codes).abs() >= min_residual
        return v


@torch.no_grad()
def rtn_quantize(w: torch.Tensor, bits: int, group_size: int) -> Lattice:
    w = w.to(LATTICE_DTYPE)
    out, in_features = w.shape
    gs = in_features if group_size is None or group_size <= 0 else group_size
    n_groups = (in_features + gs - 1) // gs
    padded = n_groups * gs
    if padded > in_features:
        wp = torch.zeros(out, padded, device=w.device, dtype=w.dtype)
        wp[:, :in_features] = w
    else:
        wp = w
    wg = wp.reshape(out, n_groups, gs)
    wmin = wg.min(dim=2, keepdim=True)[0]
    wmax = wg.max(dim=2, keepdim=True)[0]
    max_int = 2 ** bits - 1
    scale = ((wmax - wmin) / max_int).clamp(min=1e-8)
    zp = torch.round(-wmin / scale).clamp(0, max_int)
    scale_f = scale.expand(out, n_groups, gs).reshape(out, padded).contiguous()
    zp_f = zp.expand(out, n_groups, gs).reshape(out, padded).contiguous()
    pre = wp / scale_f + zp_f
    codes = torch.round(pre).clamp(0, max_int)
    return Lattice(codes=codes, pre=pre, scale=scale_f, zp=zp_f, min_int=0,
                   max_int=max_int, in_features=in_features, codes0=codes.clone())


def target_linears(model) -> dict:
    """Every nn.Linear inside the decoder stack (q,k,v,o,gate,up,down,...)."""
    out = {}
    for name, m in model.model.layers.named_modules():
        if isinstance(m, nn.Linear):
            out[f"model.layers.{name}"] = m
    return out


@torch.no_grad()
def quantize_model_rtn(model, bits: int, group_size: int) -> dict:
    """Quantize every decoder Linear in place (fake-quant). Returns {name: Lattice}."""
    lats = {}
    for name, m in target_linears(model).items():
        lat = rtn_quantize(m.weight.data, bits, group_size)
        lat_dev = m.weight.device
        for t in ("codes", "pre", "scale", "zp", "codes0"):
            setattr(lat, t, getattr(lat, t).to(lat_dev))
        m.weight.data.copy_(lat.weight().to(m.weight.dtype))
        lats[name] = lat
    return lats


@torch.no_grad()
def write_back(model, lats: dict):
    lin = target_linears(model)
    for name, lat in lats.items():
        lin[name].weight.data.copy_(lat.weight().to(lin[name].weight.dtype))


# ============================================================================
# 2. BASELINE: layerwise CLC (faithful standalone reimplementation).
#    Per output row, flip codes (complementary to pre_round) to cancel the
#    activation-weighted reconstruction error  e_o = sum_i (w_fp - w_q)_oi * a_i,
#    with James-Stein shrinkage of activation means, knee-based outlier exclusion,
#    and a per-row flip budget. Local objective, applied per Linear independently.
# ============================================================================
def _james_stein(mean: torch.Tensor) -> torch.Tensor:
    p = mean.numel()
    if p < 3:
        return mean
    gm = mean.mean()
    dev = mean - gm
    ss = (dev ** 2).sum()
    if ss < 1e-10:
        return mean
    var = ((mean - gm).abs().mean()) ** 2
    var = var.clamp(min=1e-8)
    shr = (((p - 2) * var) / ss).clamp(0, 1)
    return gm + (1 - shr) * dev


def _knee(values: torch.Tensor) -> int:
    y = values.detach().float().cpu()
    n = y.numel()
    if n < 3:
        return n // 2
    ymin, ymax = y.min(), y.max()
    if (ymax - ymin) < 1e-10:
        return n // 2
    yn = (y - ymin) / (ymax - ymin)
    xn = torch.linspace(0, 1, n)
    line = yn[0] + (yn[-1] - yn[0]) * xn
    return int(torch.argmax((yn - line).abs()).item())


@torch.no_grad()
def clc_correct_linear(lat: Lattice, act_mean: torch.Tensor,
                       max_flip_percent: float, use_js: bool = True) -> int:
    """Apply CLC to one Linear's lattice in place. Returns #flips. act_mean: (in,)."""
    dev, dt = lat.codes.device, lat.codes.dtype
    d, pin = lat.in_features, lat.codes.shape[1]
    a = torch.zeros(pin, device=dev, dtype=dt)
    a[:d] = (_james_stein(act_mean) if use_js else act_mean).to(dev, dt)

    w_fp = lat.pre * lat.scale - lat.zp * lat.scale   # == (pre - zp)*scale, the FP-ish target on grid
    w_q = (lat.codes - lat.zp) * lat.scale
    err = ((w_fp - w_q) * a[None, :]).sum(dim=1)      # per-row current error

    flip_dir = lat.flip_dir()
    impact = a[None, :] * flip_dir * lat.scale        # weight-err change of each flip
    tgt_sign = torch.sign(err).unsqueeze(1)
    valid = (torch.sign(impact) == tgt_sign) & lat.valid()

    thr_idx = _knee(torch.sort(a.abs(), descending=True)[0][: pin // 2])
    thr = torch.sort(a.abs(), descending=True)[0][thr_idx]
    valid &= (a.abs() <= thr).unsqueeze(0)            # exclude activation outliers

    rc = (lat.pre - lat.codes).abs().clone()          # prefer high-rounding-cost coords
    rc[~valid] = -1.0
    order = torch.argsort(rc, dim=1, descending=True)
    imp_s = torch.gather(impact, 1, order) * torch.gather(valid.long(), 1, order)
    cum = torch.cumsum(imp_s, dim=1)
    resid = (err.unsqueeze(1) - cum).abs()
    allr = torch.cat([err.abs().unsqueeze(1), resid], dim=1)
    best_k = torch.argmin(allr, dim=1)
    idxr = torch.arange(pin, device=dev).unsqueeze(0)
    take = (idxr < best_k.unsqueeze(1)) & torch.gather(valid, 1, order).bool()
    fd_s = torch.gather(flip_dir, 1, order)
    fd_s[~take] = 0.0
    budget = int(max_flip_percent * d)
    within = fd_s.cumsum(1) <= budget if budget > 0 else torch.zeros_like(fd_s, dtype=torch.bool)
    # note: cumsum of signed dirs is not a count; match repo behaviour (count via !=0)
    cnt = (fd_s != 0).long().cumsum(1)
    within = cnt <= budget
    fd_s[~within] = 0.0
    lat.codes.scatter_add_(1, order, fd_s)
    lat.codes.clamp_(lat.min_int, lat.max_int)
    return int((fd_s != 0).sum().item())


@torch.no_grad()
def run_clc_baseline(model, lats: dict, act_means: dict, max_flip_percent: float) -> dict:
    total = 0
    for name, lat in lats.items():
        if name in act_means:
            total += clc_correct_linear(lat, act_means[name], max_flip_percent)
    write_back(model, lats)
    return {"method": "clc_layerwise", "flips_total": total}


# ============================================================================
# 3. End loss + activation-mean collection.
# ============================================================================
@torch.no_grad()
def collect_act_means(model, calib_ids, device, batch: int = 4) -> dict:
    """Mean input activation per decoder Linear (for CLC), via forward hooks."""
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
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(cnt[n], 1)) for n in lin if acc[n] is not None}


def end_loss(student, teacher, ids, device, kind: str = "kl") -> torch.Tensor:
    """Teacher-FP vs student-quant KL (default) or student LM cross-entropy.
    Teacher logits computed on the fly and detached (no large logit storage)."""
    logits_s = student(ids.to(device)).logits
    if kind == "ce":
        shift_logits = logits_s[:, :-1].reshape(-1, logits_s.shape[-1])
        shift_labels = ids[:, 1:].reshape(-1).to(device)
        return nn.functional.cross_entropy(shift_logits.float(), shift_labels)
    with torch.no_grad():
        logits_t = teacher(ids.to(device)).logits
        logp_t = torch.log_softmax(logits_t.float(), dim=-1)
    logp_s = torch.log_softmax(logits_s.float(), dim=-1)
    p_t = logp_t.exp()
    return (p_t * (logp_t - logp_s)).sum(-1).mean()


@torch.no_grad()
def eval_end_loss(student, teacher, ids_list, device, kind, batch: int = 4) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(ids_list), batch):
        ids = torch.stack(ids_list[i:i + batch]).to(device)
        tot += float(end_loss(student, teacher, ids, device, kind)) * ids.shape[0]
        n += ids.shape[0]
    return tot / max(n, 1)


# ============================================================================
# 4. OURS: end-loss-guided second pass over the fixed grid.
#    Predict cheaply with the end-loss GRADIENT (one backward => c for ALL
#    candidates at once), accept EXACTLY on the real end-loss forward. Batch
#    halving = discrete trust region. c re-backward on a schedule (it drifts).
# ============================================================================
@dataclass
class EndLossConfig:
    max_flip_percent: float = 0.02   # per output row, fraction of in_features
    layers: Optional[set] = None     # None = all decoder layers
    batch_init: int = 256            # flips proposed per accept-check (global)
    batch_max: int = 4096
    refresh_every: int = 4           # accepted batches between gradient refreshes
    max_rounds: int = 60
    grad_batch: int = 8              # calib seqs per backward (opt split)
    loss_kind: str = "kl"            # "kl" (teacher-student) | "ce" (LM)
    seed: int = 0


@dataclass
class CostLedger:
    backwards: int = 0   # end-loss gradients (state/scoring cost -- Feasibility B axis)
    forwards: int = 0    # end-loss evaluations (acceptance axis)


def _layer_of(name: str) -> int:
    # "model.layers.<i>.<...>"
    try:
        return int(name.split("model.layers.")[1].split(".")[0])
    except Exception:
        return -1


def endloss_gradient(student, teacher, lats, opt_ids, device, cfg) -> dict:
    """One backward of the end loss -> per-candidate c_i = (dL/dW)_i * step_i.
    Returns {name: c_tensor (out, in)}. Most negative c => best predicted flip."""
    lin = target_linears(student)
    for m in lin.values():
        m.weight.requires_grad_(True)
        if m.weight.grad is not None:
            m.weight.grad = None
    # accumulate grad over the opt split
    for i in range(0, len(opt_ids), cfg.grad_batch):
        ids = torch.stack(opt_ids[i:i + cfg.grad_batch]).to(device)
        loss = end_loss(student, teacher, ids, device, cfg.loss_kind)
        loss.backward()
    c = {}
    with torch.no_grad():
        for name, lat in lats.items():
            g = lin[name].weight.grad
            if g is None:
                continue
            step = lat.step()[:, : lat.in_features]           # (out, in) weight change of a flip
            c[name] = (g.to(LATTICE_DTYPE) * step)            # dL for the +flip, first order
    for m in lin.values():
        m.weight.requires_grad_(False)
        m.weight.grad = None
    return c


@torch.no_grad()
def _apply_flip(lat: Lattice, rows, cols):
    fd = lat.flip_dir()
    lat.codes[rows, cols] += fd[rows, cols]
    lat.codes.clamp_(lat.min_int, lat.max_int)


def run_endloss_secondpass(student, teacher, lats, opt_ids, acc_ids, device,
                           cfg: EndLossConfig) -> dict:
    """Greedy-by-gradient proposal + exact real-forward acceptance with batch
    halving. Equal-budget, same lattice convention as CLC => fair comparison."""
    torch.manual_seed(cfg.seed)
    ledger = CostLedger()
    t0 = time.time()
    names = [n for n in lats if cfg.layers is None or _layer_of(n) in cfg.layers]

    # per-row budget + usage bookkeeping (frozen coords also count against budget)
    used = {n: torch.zeros_like(lats[n].codes, dtype=torch.bool) for n in names}
    blocked = {n: torch.zeros_like(lats[n].codes, dtype=torch.bool) for n in names}
    budget = {n: int(cfg.max_flip_percent * lats[n].in_features) for n in names}
    row_cnt = {n: torch.zeros(lats[n].codes.shape[0], dtype=torch.long, device=device)
               for n in names}

    base_L = eval_end_loss(student, teacher, acc_ids, device, cfg.loss_kind)
    ledger.forwards += 1
    L0 = base_L
    bsize = cfg.batch_init
    since_refresh = 0
    c = endloss_gradient(student, teacher, lats, opt_ids, device, cfg)
    ledger.backwards += 1
    flips_total = 0
    rounds = 0
    disagree_tracked = []  # (for the audit; filled lazily below)

    for rnd in range(cfg.max_rounds):
        rounds = rnd + 1
        # ---- rank all eligible candidates by c (most negative first), global ----
        cand = []  # (c_val, name, row, col)
        for n in names:
            lat = lats[n]
            d = lat.in_features
            cval = c.get(n)
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
        allv = torch.cat([c0[0] for c0 in cand])
        alln = torch.cat([c0[1] for c0 in cand]).long()
        allr = torch.cat([c0[2] for c0 in cand]).long()
        allc = torch.cat([c0[3] for c0 in cand]).long()
        k = min(int(bsize), allv.numel())
        sel = torch.topk(allv, k, largest=False).indices   # most-negative c

        # ---- enforce per-row budget within the proposed batch ----
        proposal = {n: [] for n in names}
        tmp_row = {n: row_cnt[n].clone() for n in names}
        for s in sel.tolist():
            n = names[int(alln[s])]
            r, col = int(allr[s]), int(allc[s])
            if tmp_row[n][r] < budget[n]:
                proposal[n].append((r, col))
                tmp_row[n][r] += 1
        if not any(proposal.values()):
            break

        # ---- apply, evaluate exactly on acceptance split ----
        snapshot = {n: lats[n].codes.clone() for n in proposal if proposal[n]}
        for n, rc in proposal.items():
            if not rc:
                continue
            rows = torch.tensor([r for r, _ in rc], device=device)
            cols = torch.tensor([col for _, col in rc], device=device)
            _apply_flip(lats[n], rows, cols)
        write_back(student, {n: lats[n] for n in proposal if proposal[n]})
        newL = eval_end_loss(student, teacher, acc_ids, device, cfg.loss_kind)
        ledger.forwards += 1
        n_prop = sum(len(v) for v in proposal.values())

        if newL < base_L - 1e-9:
            # accept: commit, mark used, grow batch
            base_L = newL
            for n, rc in proposal.items():
                for r, col in rc:
                    used[n][r, col] = True
                    row_cnt[n][r] += 1
            flips_total += n_prop
            bsize = min(int(bsize * 2), cfg.batch_max)
            since_refresh += 1
            if since_refresh >= cfg.refresh_every:
                c = endloss_gradient(student, teacher, lats, opt_ids, device, cfg)
                ledger.backwards += 1
                since_refresh = 0
        else:
            # reject: revert and halve; if the batch was a single flip, block it
            for n in proposal:
                if proposal[n]:
                    lats[n].codes.copy_(snapshot[n])
            write_back(student, {n: lats[n] for n in proposal if proposal[n]})
            if n_prop <= 1:
                for n, rc in proposal.items():
                    for r, col in rc:
                        blocked[n][r, col] = True
            bsize = max(1, bsize // 2)

    write_back(student, lats)
    return {
        "method": f"endloss_2ndpass[{cfg.loss_kind}]",
        "flips_total": flips_total,
        "rounds": rounds,
        "end_loss_before": L0,
        "end_loss_after": base_L,
        "cost": {"backwards": ledger.backwards, "forwards": ledger.forwards},
        "time_s": time.time() - t0,
    }


# ============================================================================
# 5. Audits — the real-model "headline" (does end-loss gradient predict true
#    single-flip change?) and the premise check (does CLC disagree with it?).
# ============================================================================
@torch.no_grad()
def audit_grad_vs_truth(student, teacher, lats, c, opt_ids, device, cfg,
                        n_samples: int = 64) -> dict:
    """For a sample of candidate flips: predicted c_i vs TRUE single-flip end-loss
    change (finite difference on the opt split). Analogue of the toy headline."""
    names = [n for n in lats if (cfg.layers is None or _layer_of(n) in cfg.layers) and n in c]
    if not names:
        return {}
    L0 = eval_end_loss(student, teacher, opt_ids, device, cfg.loss_kind)
    pred, true = [], []
    g = torch.Generator(device="cpu"); g.manual_seed(cfg.seed)
    for _ in range(n_samples):
        n = names[int(torch.randint(len(names), (1,), generator=g))]
        lat = lats[n]
        d = lat.in_features
        elig = lat.valid()[:, :d].nonzero(as_tuple=False)
        if elig.numel() == 0:
            continue
        r, col = elig[int(torch.randint(elig.shape[0], (1,), generator=g))].tolist()
        pred.append(float(c[n][r, col]))
        snap = lat.codes[r, col].item()
        _apply_flip(lat, torch.tensor([r], device=device), torch.tensor([col], device=device))
        write_back(student, {n: lat})
        true.append(eval_end_loss(student, teacher, opt_ids, device, cfg.loss_kind) - L0)
        lat.codes[r, col] = snap
        write_back(student, {n: lat})
    if len(pred) < 3:
        return {"n": len(pred)}
    p = torch.tensor(pred); t = torch.tensor(true)
    pear = float(torch.corrcoef(torch.stack([p, t]))[0, 1])
    sign_dis = float(((p.sign() != t.sign()) & (t.abs() > 1e-12)).float().mean())
    return {"n": len(pred), "pearson_pred_vs_true": pear,
            "sign_disagree": sign_dis,
            "pred_abs_mean": float(p.abs().mean()), "true_abs_mean": float(t.abs().mean())}


@torch.no_grad()
def audit_clc_vs_endloss(lats_clc_before, lats_after_clc, c) -> dict:
    """Premise check: of the flips CLC actually made, what fraction does the
    end-loss gradient say are HARMFUL (c_i > 0)? ~0 => local and global agree
    (premise moot). Large => end loss sees cross-layer effects CLC cannot."""
    harmful, total = 0, 0
    for name, lat in lats_after_clc.items():
        if name not in c:
            continue
        before = lats_clc_before[name]
        d = lat.in_features
        flipped = (lat.codes[:, :d] != before.codes[:, :d])
        idx = flipped.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        cv = c[name][idx[:, 0], idx[:, 1]]
        harmful += int((cv > 0).sum().item())
        total += cv.numel()
    return {"clc_flips_scored": total,
            "frac_harmful_by_endloss": (harmful / total) if total else float("nan")}


# ============================================================================
# 6. PPL eval (sliding-window, standard) + data.
# ============================================================================
@torch.no_grad()
def eval_ppl(model, ids_1d: torch.Tensor, device, seqlen: int = 2048, stride: int = 2048) -> float:
    model.eval()
    n = ids_1d.numel()
    nll, cnt = 0.0, 0
    for i in range(0, n - 1, stride):
        j = min(i + seqlen, n - 1)
        inp = ids_1d[i:j].unsqueeze(0).to(device)
        tgt = ids_1d[i + 1:j + 1].unsqueeze(0).to(device)
        logits = model(inp).logits
        ll = nn.functional.cross_entropy(
            logits[0].float(), tgt[0], reduction="sum")
        nll += float(ll)
        cnt += tgt.numel()
    return math.exp(nll / max(cnt, 1))


def load_data(tokenizer, n_calib: int, seqlen: int, seed: int):
    """WikiText-2: calibration chunks (train) + one long test stream for PPL."""
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
# 7. Head-to-head harness: FP | RTN | RTN+CLC (baseline) | RTN+EndLoss (ours).
# ============================================================================
def load_model(path, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
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
    ap.add_argument("--opt-seqs", type=int, default=32)      # backward split
    ap.add_argument("--acc-seqs", type=int, default=16)      # acceptance split
    ap.add_argument("--flip-percent", type=float, default=0.02)
    ap.add_argument("--loss-kind", choices=["kl", "ce"], default="kl")
    ap.add_argument("--layers", default="all", help='e.g. "all" or "0-5,11"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true", help="run grad-vs-truth + CLC-disagreement audits")
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
    fp_model, tok = load_model(args.model_path, device, dtype)
    n_layers = len(fp_model.model.layers)
    layers = parse_layers(args.layers, n_layers)
    calib, test_ids = load_data(tok, args.n_calib, args.seqlen, args.seed)
    opt_ids = calib[: args.opt_seqs]
    acc_ids = calib[args.opt_seqs: args.opt_seqs + args.acc_seqs]

    results = {}
    ppl_fp = eval_ppl(fp_model, test_ids, device, args.seqlen)
    results["fp"] = {"ppl": ppl_fp}
    print(f"[fp]  ppl={ppl_fp:.4f}")

    # teacher = untouched FP copy (end-loss target); keep on same device
    teacher = copy.deepcopy(fp_model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---------- arm 1: RTN ----------
    student = copy.deepcopy(fp_model).eval()
    lats_rtn = quantize_model_rtn(student, args.bits, args.group_size)
    ppl_rtn = eval_ppl(student, test_ids, device, args.seqlen)
    results["rtn"] = {"ppl": ppl_rtn}
    print(f"[rtn] ppl={ppl_rtn:.4f}")

    act_means = collect_act_means(student, calib, device)

    # ---------- arm 2: RTN + CLC (BASELINE) ----------
    student_clc = copy.deepcopy(fp_model).eval()
    lats_clc = quantize_model_rtn(student_clc, args.bits, args.group_size)
    lats_clc_before = {n: copy.deepcopy(l) for n, l in lats_clc.items()}
    clc_log = run_clc_baseline(student_clc, lats_clc, act_means, args.flip_percent)
    ppl_clc = eval_ppl(student_clc, test_ids, device, args.seqlen)
    results["rtn_clc"] = {"ppl": ppl_clc, **clc_log,
                          "G_recovery": G_recovery(ppl_rtn, ppl_clc, ppl_fp)}
    print(f"[clc] ppl={ppl_clc:.4f} flips={clc_log['flips_total']} "
          f"G={results['rtn_clc']['G_recovery']:.3f}")

    # ---------- arm 3: RTN + end-loss second pass (OURS) ----------
    student_el = copy.deepcopy(fp_model).eval()
    lats_el = quantize_model_rtn(student_el, args.bits, args.group_size)
    cfg = EndLossConfig(max_flip_percent=args.flip_percent, layers=layers,
                        loss_kind=args.loss_kind, seed=args.seed)
    el_log = run_endloss_secondpass(student_el, teacher, lats_el, opt_ids, acc_ids, device, cfg)
    ppl_el = eval_ppl(student_el, test_ids, device, args.seqlen)
    results["rtn_endloss"] = {"ppl": ppl_el, **el_log,
                              "G_recovery": G_recovery(ppl_rtn, ppl_el, ppl_fp)}
    print(f"[ours] ppl={ppl_el:.4f} flips={el_log['flips_total']} "
          f"cost={el_log['cost']} G={results['rtn_endloss']['G_recovery']:.3f}")

    # ---------- audits (optional) ----------
    if args.audit:
        c = endloss_gradient(student_el, teacher, lats_el, opt_ids, device, cfg)
        a1 = audit_grad_vs_truth(student_el, teacher, lats_el, c, opt_ids, device, cfg)
        a2 = audit_clc_vs_endloss(lats_clc_before, lats_clc, c)
        results["audit"] = {"grad_vs_truth": a1, "clc_vs_endloss": a2}
        print(f"[audit] grad_vs_truth={a1}")
        print(f"[audit] clc_vs_endloss={a2}")

    print("\n==== SUMMARY (lower ppl better; G_recovery higher better) ====")
    for k, v in results.items():
        print(f"  {k:<14}: {v}")
    print("\nFair-comparison invariants: same RTN grid, same +-1 lattice flips, "
          "same flip budget, same PPL eval. Only the objective differs "
          "(CLC local recon vs end-loss gradient).")


if __name__ == "__main__":
    main()
