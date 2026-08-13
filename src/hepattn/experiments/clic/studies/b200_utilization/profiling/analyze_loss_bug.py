"""Loss-broadcast bug analysis: buggy vs intended mask losses on real CLIC data.

Investigation only - nothing here touches production code.

Definitions used throughout:
  V1 "buggy"    = the code as it exists on HEAD (and the clic-paper tag): boolean
                  indexing collapses [B,N,C] -> [Nv,C], then the [B,1,C] pad mask
                  broadcasts against the flattened object axis -> [B,Nv,C].
  V2 "intended" = what the docstrings describe: each object masked by its OWN
                  event's pad mask, normalised by its own event's valid count.
  V0 "pre-bug"  = the pre-b3985a3 code: no pad mask at all; mean over all C
                  (padded logits are finfo.min so BCE contributes ~0 but dilutes
                  the denominator). The paper's original plotted run trained on this.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

REPO = "/blue/avery/m.mazza/projects/fastml/hepattn"
sys.path.insert(0, REPO + "/src")

from hepattn.models.loss import (  # noqa: E402  (uncompiled module-level fns)
    mask_bce_loss,
    mask_dice_loss,
    mask_focal_loss,
    mask_kl_div_loss,
)

OUT = {}
FMIN = torch.finfo(torch.float32).min


# ----------------------------------------------------------------------------
# Reference (intended, V2) implementations
# ----------------------------------------------------------------------------
def ref_bce_loop(pred, tgt, ovm, ipm, sw=None):
    """Per-event python loop; the unarguable reference for the docstring semantics."""
    parts = []
    for b in range(pred.shape[0]):
        p, t = pred[b][ovm[b]], tgt[b][ovm[b]]
        w = sw[b][ovm[b]] if sw is not None else None
        el = F.binary_cross_entropy_with_logits(p, t, weight=w, reduction="none")
        el = el * ipm[b].unsqueeze(0)
        parts.append(el.sum(-1) / ipm[b].sum().clamp_min(1.0))
    return torch.cat(parts).mean()


def ref_bce_vec(pred, tgt, ovm, ipm, sw=None):
    """Vectorised intended BCE (the 'mulmask' rewrite)."""
    el = F.binary_cross_entropy_with_logits(pred, tgt, weight=sw, reduction="none")
    el = el * ipm.unsqueeze(1).to(el.dtype)
    per_obj = el.sum(-1) / ipm.sum(-1, keepdim=True).clamp_min(1.0)
    w = ovm.to(per_obj.dtype)
    return (per_obj * w).sum() / w.sum().clamp_min(1.0)


def ref_dice_loop(pred, tgt, ovm, ipm):
    parts = []
    for b in range(pred.shape[0]):
        p = pred[b][ovm[b]].sigmoid() * ipm[b].unsqueeze(0)
        t = tgt[b][ovm[b]]
        num = 2 * (p * t).sum(-1)
        den = p.sum(-1) + t.sum(-1)
        parts.append(1 - (num + 1) / (den + 1))
    return torch.cat(parts).mean()


def ref_dice_vec(pred, tgt, ovm, ipm):
    p = pred.sigmoid() * ipm.unsqueeze(1).to(pred.dtype)
    num = 2 * (p * tgt).sum(-1)
    den = p.sum(-1) + tgt.sum(-1)
    per_obj = 1 - (num + 1) / (den + 1)
    w = ovm.to(per_obj.dtype)
    return (per_obj * w).sum() / w.sum().clamp_min(1.0)


def v0_bce(pred, tgt, ovm, sw=None):
    """Pre-b3985a3 semantics: mean over valid objects x ALL constituents."""
    el = F.binary_cross_entropy_with_logits(pred[ovm], tgt[ovm], weight=None if sw is None else sw[ovm], reduction="none")
    return el.mean()


# ----------------------------------------------------------------------------
# Closed-form equivalents of the BUGGY computation (no [B,Nv,C] materialisation)
# ----------------------------------------------------------------------------
def buggy_bce_closed(pred, tgt, ovm, ipm, sw=None):
    """mean_n sum_c el[n,c] * w[c],  w[c] = mean_b pad[b,c]/V[b]."""
    p, t = pred[ovm], tgt[ovm]
    w = sw[ovm] if sw is not None else None
    el = F.binary_cross_entropy_with_logits(p, t, weight=w, reduction="none")  # [Nv,C]
    V = ipm.sum(-1, keepdim=True).clamp_min(1.0)  # [B,1]
    wc = (ipm.to(el.dtype) / V).mean(0)  # [C]
    return (el * wc).sum(-1).mean()


def buggy_dice_closed(pred, tgt, ovm, ipm):
    """mean_{b,n} of dice(object n truncated to event b's pad support).

    Requires contiguous pad masks (valid-first padding) - asserted by caller.
    """
    p = pred[ovm].sigmoid()  # [Nv,C]
    t = tgt[ovm]
    Pc = p.cumsum(-1)  # [Nv,C]
    Ic = (p * t).cumsum(-1)
    Tn = t.sum(-1)  # [Nv]
    V = ipm.sum(-1).long()  # [B]
    assert (V > 0).all(), "V=0 event would need special-casing"
    # group events by their V value: loss over [B,Nv] == count-weighted loss over unique V
    uV, counts = torch.unique(V, return_counts=True)
    P_vn = Pc.index_select(1, uV - 1).T  # [U,Nv] : Pc[:, v-1]
    I_vn = Ic.index_select(1, uV - 1).T
    loss_vn = 1 - (2 * I_vn + 1) / (P_vn + Tn.unsqueeze(0) + 1)  # [U,Nv]
    w = counts.to(loss_vn.dtype) / V.numel()
    return (loss_vn.mean(1) * w).sum()


# ----------------------------------------------------------------------------
# Load real CLIC data
# ----------------------------------------------------------------------------
def load_real(n_events):
    from hepattn.experiments.clic.pflow_data import CLICDataset

    t0 = time.time()
    ds = CLICDataset(
        filepath=REPO + "/data/clic/train_clic_fix.root",
        inputs={"node": []},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=REPO + "/src/hepattn/experiments/clic/configs/clic_var_transform.yaml",
        num_events=n_events,
        num_objects=150,
        max_nodes=160,
        incidence_cutval=0.01,
    )
    n = len(ds)
    node_valid = torch.zeros(n, 160, dtype=torch.bool)
    part_valid = torch.zeros(n, 150, dtype=torch.bool)
    pn_valid = torch.zeros(n, 150, 160, dtype=torch.bool)
    for i in range(n):
        _, lab = ds[i]
        node_valid[i] = lab["node_valid"]
        part_valid[i] = lab["particle_valid"]
        pn_valid[i] = lab["particle_node_valid"]
        if i % 2000 == 0:
            print(f"  event {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"loaded {n} events in {time.time() - t0:.0f}s", flush=True)
    return node_valid, part_valid, pn_valid, torch.tensor(ds.n_nodes), torch.tensor(ds.n_tracks)


def make_logits(tgt, node_valid, regime, gen):
    """Surrogate predictions on REAL targets. Padded logits get finfo.min as
    ObjectHitMaskTask.forward does (task.py:562)."""
    s, ns = {"early": (0.0, 2.0), "mid": (2.0, 1.5), "late": (6.0, 1.0)}[regime]
    logits = s * (2.0 * tgt - 1.0) + ns * torch.randn(tgt.shape, generator=gen)
    logits = logits.masked_fill(~node_valid.unsqueeze(1), FMIN)
    return logits


def grad_metrics(ga, gb):
    fa, fb = ga.flatten(), gb.flatten()
    cos = F.cosine_similarity(fa, fb, dim=0).item()
    return {"cosine": cos, "norm_ratio": (fa.norm() / fb.norm()).item()}


def batch_metrics(pred_f, tgt_f, ovm, ipm, sw, use_direct=False):
    """Compute buggy vs intended losses + grads for one batch. pred_f is a leaf."""
    res = {}

    # ---- BCE ----
    pred = pred_f.clone().requires_grad_(True)
    lb = buggy_bce_closed(pred, tgt_f, ovm, ipm, sw)
    gb = torch.autograd.grad(lb, pred)[0]
    pred2 = pred_f.clone().requires_grad_(True)
    lc = ref_bce_vec(pred2, tgt_f, ovm, ipm, sw)
    gc = torch.autograd.grad(lc, pred2)[0]
    res["bce"] = {
        "buggy": lb.item(),
        "intended": lc.item(),
        "rel_diff": (lb / lc - 1).item(),
        "grad": grad_metrics(gb, gc),
    }
    if use_direct:
        ld = mask_bce_loss(pred_f, tgt_f, object_valid_mask=ovm, input_pad_mask=ipm, sample_weight=sw)
        res["bce"]["direct_buggy"] = ld.item()
        res["bce"]["closed_form_matches_direct"] = bool(torch.isclose(ld, lb.detach(), rtol=1e-5))
        # V0 for context
        res["bce"]["v0_prebug"] = v0_bce(pred_f, tgt_f, ovm, sw).item()

    # ---- dice ----
    pred = pred_f.clone().requires_grad_(True)
    lb = buggy_dice_closed(pred, tgt_f, ovm, ipm)
    gb = torch.autograd.grad(lb, pred)[0]
    pred2 = pred_f.clone().requires_grad_(True)
    lc = ref_dice_vec(pred2, tgt_f, ovm, ipm)
    gc = torch.autograd.grad(lc, pred2)[0]
    res["dice"] = {
        "buggy": lb.item(),
        "intended": lc.item(),
        "rel_diff": (lb / lc - 1).item(),
        "grad": grad_metrics(gb, gc),
    }
    if use_direct:
        ld = mask_dice_loss(pred_f, tgt_f, object_valid_mask=ovm, input_pad_mask=ipm)
        res["dice"]["direct_buggy"] = ld.item()
        res["dice"]["closed_form_matches_direct"] = bool(torch.isclose(ld, lb.detach(), rtol=1e-5))

    # ---- production combination 5*bce + 1*dice ----
    pred = pred_f.clone().requires_grad_(True)
    ltot_b = 5.0 * buggy_bce_closed(pred, tgt_f, ovm, ipm, sw) + 1.0 * buggy_dice_closed(pred, tgt_f, ovm, ipm)
    gb = torch.autograd.grad(ltot_b, pred)[0]
    pred2 = pred_f.clone().requires_grad_(True)
    ltot_c = 5.0 * ref_bce_vec(pred2, tgt_f, ovm, ipm, sw) + 1.0 * ref_dice_vec(pred2, tgt_f, ovm, ipm)
    gc = torch.autograd.grad(ltot_c, pred2)[0]
    res["combined_5bce_1dice"] = {
        "buggy": ltot_b.item(),
        "intended": ltot_c.item(),
        "rel_diff": (ltot_b / ltot_c - 1).item(),
        "grad": grad_metrics(gb, gc),
    }
    return res


def main():
    # ------------------------------------------------------------------
    # Part 0: toy-scale proof that closed forms == actual buggy code, and
    # that the actual code != per-event reference
    # ------------------------------------------------------------------
    print("=== Part 0: toy verification ===", flush=True)
    g = torch.Generator().manual_seed(123)
    B, N, C = 6, 8, 11
    ipm = torch.zeros(B, C, dtype=torch.bool)
    for b, v in enumerate([3, 5, 7, 9, 11, 6]):
        ipm[b, :v] = True
    ovm = torch.rand(B, N, generator=g) > 0.35
    tgt = ((torch.rand(B, N, C, generator=g) > 0.7) & ipm.unsqueeze(1)).float()
    pred = 2 * torch.randn(B, N, C, generator=g)
    pred = pred.masked_fill(~ipm.unsqueeze(1), FMIN)
    sw = tgt + 1.0 * (1 - tgt)

    toy = {}
    lb_direct = mask_bce_loss(pred, tgt, object_valid_mask=ovm, input_pad_mask=ipm, sample_weight=sw)
    toy["bce"] = {
        "direct_buggy": lb_direct.item(),
        "closed_form": buggy_bce_closed(pred, tgt, ovm, ipm, sw).item(),
        "ref_loop": ref_bce_loop(pred, tgt, ovm, ipm, sw).item(),
        "ref_vec": ref_bce_vec(pred, tgt, ovm, ipm, sw).item(),
    }
    ld_direct = mask_dice_loss(pred, tgt, object_valid_mask=ovm, input_pad_mask=ipm)
    toy["dice"] = {
        "direct_buggy": ld_direct.item(),
        "closed_form": buggy_dice_closed(pred, tgt, ovm, ipm).item(),
        "ref_loop": ref_dice_loop(pred, tgt, ovm, ipm).item(),
        "ref_vec": ref_dice_vec(pred, tgt, ovm, ipm).item(),
    }
    # focal: confirm same broadcast fires (shape check via internal compute)
    lf = mask_focal_loss(pred, tgt, object_valid_mask=ovm, input_pad_mask=ipm, sample_weight=sw)
    # intended focal reference (loop)
    parts = []
    for b in range(B):
        p, t = pred[b][ovm[b]], tgt[b][ovm[b]]
        w = sw[b][ovm[b]]
        pr = p.sigmoid()
        ce = F.binary_cross_entropy_with_logits(p, t, weight=w, reduction="none")
        ce = ce * ipm[b].unsqueeze(0)
        prm = pr * ipm[b].unsqueeze(0)
        p_t = prm * t + (1 - prm) * (1 - t)
        fl = ce * ((1 - p_t) ** 2.0)
        parts.append(fl.sum(-1) / ipm[b].sum().clamp_min(1.0))
    toy["focal"] = {"direct_buggy": lf.item(), "ref_loop": torch.cat(parts).mean().item()}
    # kl_div: does masked_fill with [B,1,C] mask on [Nv,C] tensor even run?
    try:
        lk = mask_kl_div_loss(pred.softmax(-1), tgt, object_valid_mask=ovm, input_pad_mask=ipm)
        toy["kl_div"] = {"runs": True, "value": lk.item()}
    except RuntimeError as e:
        toy["kl_div"] = {"runs": False, "error": str(e)[:200]}
    OUT["toy"] = toy
    print(json.dumps(toy, indent=1), flush=True)

    # grad equivalence of closed form vs direct code at toy scale
    p1 = pred.clone().requires_grad_(True)
    g1 = torch.autograd.grad(mask_bce_loss(p1, tgt, object_valid_mask=ovm, input_pad_mask=ipm, sample_weight=sw), p1)[0]
    p2 = pred.clone().requires_grad_(True)
    g2 = torch.autograd.grad(buggy_bce_closed(p2, tgt, ovm, ipm, sw), p2)[0]
    OUT["toy_grad_closed_eq_direct_bce"] = bool(torch.allclose(g1, g2, atol=1e-6))
    p1 = pred.clone().requires_grad_(True)
    g1 = torch.autograd.grad(mask_dice_loss(p1, tgt, object_valid_mask=ovm, input_pad_mask=ipm), p1)[0]
    p2 = pred.clone().requires_grad_(True)
    g2 = torch.autograd.grad(buggy_dice_closed(p2, tgt, ovm, ipm), p2)[0]
    OUT["toy_grad_closed_eq_direct_dice"] = bool(torch.allclose(g1, g2, atol=1e-6))
    print("closed-form grad == direct grad:", OUT["toy_grad_closed_eq_direct_bce"], OUT["toy_grad_closed_eq_direct_dice"], flush=True)

    import os

    if os.environ.get("SMOKE"):
        print("SMOKE mode: stopping after toy verification", flush=True)
        return

    # ------------------------------------------------------------------
    # Part 1: real data
    # ------------------------------------------------------------------
    print("=== Part 1: loading real CLIC data ===", flush=True)
    N_EVENTS = 12288
    node_valid, part_valid, pn_valid, n_nodes, n_tracks = load_real(N_EVENTS)
    n = node_valid.shape[0]

    # pad contiguity check (needed by dice closed form)
    ar = torch.arange(160).unsqueeze(0)
    assert (node_valid == (ar < n_nodes.unsqueeze(1))).all(), "pad masks are not contiguous!"
    OUT["pad_contiguous"] = True

    V = n_nodes.float()
    nv_obj = part_valid.sum(-1).float()
    OUT["V_distribution"] = {
        "n_events": int(n),
        "min": V.min().item(),
        "max": V.max().item(),
        "mean": V.mean().item(),
        "std": V.std().item(),
        "quantiles_1_5_25_50_75_95_99": [torch.quantile(V, q).item() for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]],
        "max_nodes": 160,
        "n_tracks_mean": n_tracks.float().mean().item(),
        "valid_objects_per_event_mean": nv_obj.mean().item(),
        "valid_objects_per_event_std": nv_obj.std().item(),
    }
    print(json.dumps(OUT["V_distribution"], indent=1), flush=True)

    # data-only weight distortion: buggy per-object total weight sum_{c<Ve} w[c]
    # (intended total weight is exactly 1 for every object)
    ipm_all = node_valid.float()
    wc_pop = (ipm_all / ipm_all.sum(-1, keepdim=True)).mean(0)  # population w[c] (B=12288)
    cum_wc = torch.cat([torch.zeros(1), wc_pop.cumsum(0)])
    tot_w = cum_wc[n_nodes.long()]  # per event: sum_{c<Ve} w[c]
    OUT["object_total_weight_buggy_vs_1"] = {
        "description": "sum_c<V_e w[c]: total loss weight the buggy BCE gives an object of an event with V_e valid nodes (intended = 1.0 for all)",
        "min": tot_w.min().item(),
        "max": tot_w.max().item(),
        "mean": tot_w.mean().item(),
        "at_V=[20,40,60,80,100,120,140,159]": [cum_wc[v].item() for v in [20, 40, 60, 80, 100, 120, 140, 159]],
        "wc_first_last": [wc_pop[0].item(), wc_pop[-1].item()],
    }
    print(json.dumps(OUT["object_total_weight_buggy_vs_1"], indent=1), flush=True)

    # ------------------------------------------------------------------
    # Part 2: direct-code validation on a small real batch
    # ------------------------------------------------------------------
    print("=== Part 2: direct validation on real batch (B=64) ===", flush=True)
    gen = torch.Generator().manual_seed(7)
    idx = torch.randperm(n, generator=gen)[:64]
    tgt = pn_valid[idx].float()
    ipm = node_valid[idx]
    ovm = part_valid[idx]
    sw = tgt + 1.0 * (1 - tgt)  # null_weight = 1.0 (base.yaml)
    real_direct = {}
    for regime in ["early", "mid", "late"]:
        pred = make_logits(tgt, ipm, regime, gen)
        real_direct[regime] = batch_metrics(pred, tgt, ovm, ipm, sw, use_direct=True)
        # also loop-reference cross-check
        real_direct[regime]["bce"]["ref_loop"] = ref_bce_loop(pred, tgt, ovm, ipm, sw).item()
        real_direct[regime]["dice"]["ref_loop"] = ref_dice_loop(pred, tgt, ovm, ipm).item()
    OUT["real_B64_direct"] = real_direct
    print(json.dumps(real_direct, indent=1), flush=True)

    # ------------------------------------------------------------------
    # Part 3: batch-size dependence (closed forms, validated above)
    # ------------------------------------------------------------------
    print("=== Part 3: batch-size sweep ===", flush=True)
    sweep = {}
    for Bsz, nrep in [(32, 16), (256, 8), (2048, 6)]:
        for regime in ["early", "late"]:
            key = f"B{Bsz}_{regime}"
            reps = []
            for r in range(nrep):
                gen2 = torch.Generator().manual_seed(1000 + 17 * r + Bsz)
                idx = torch.randperm(n, generator=gen2)[:Bsz]
                tgt = pn_valid[idx].float()
                ipm = node_valid[idx]
                ovm = part_valid[idx]
                sw = tgt + 1.0 * (1 - tgt)
                pred = make_logits(tgt, ipm, regime, gen2)
                reps.append(batch_metrics(pred, tgt, ovm, ipm, sw))
            agg = {}
            for lossname in ["bce", "dice", "combined_5bce_1dice"]:
                rd = np.array([x[lossname]["rel_diff"] for x in reps])
                cos = np.array([x[lossname]["grad"]["cosine"] for x in reps])
                nr = np.array([x[lossname]["grad"]["norm_ratio"] for x in reps])
                lb = np.array([x[lossname]["buggy"] for x in reps])
                lc = np.array([x[lossname]["intended"] for x in reps])
                agg[lossname] = {
                    "buggy_mean": lb.mean(),
                    "intended_mean": lc.mean(),
                    "rel_diff_mean": rd.mean(),
                    "rel_diff_std": rd.std(),
                    "grad_cos_mean": cos.mean(),
                    "grad_cos_std": cos.std(),
                    "grad_norm_ratio_mean": nr.mean(),
                }
            sweep[key] = agg
            print(key, json.dumps(agg, indent=1, default=float), flush=True)
    OUT["batch_size_sweep"] = sweep

    # w[c] batch-composition dependence: how much does the effective weight
    # vector move between batches of the same size / different sizes?
    wc_stats = {}
    for Bsz in [32, 256, 2048]:
        wcs = []
        for r in range(24):
            gen2 = torch.Generator().manual_seed(5000 + r)
            idx = torch.randperm(n, generator=gen2)[:Bsz]
            ipm = node_valid[idx].float()
            wcs.append((ipm / ipm.sum(-1, keepdim=True).clamp_min(1.0)).mean(0))
        wcs = torch.stack(wcs)
        # relative RMS fluctuation of w[c] across batches, averaged over c (weighted by wc)
        m = wcs.mean(0)
        rel_rms = ((wcs.std(0) / m.clamp_min(1e-12)) * (m / m.sum())).sum().item()
        dev_pop = ((m - wc_pop).abs() / wc_pop.clamp_min(1e-12) * (wc_pop / wc_pop.sum())).sum().item()
        wc_stats[f"B{Bsz}"] = {"weighted_rel_rms_across_batches": rel_rms, "weighted_rel_dev_from_population": dev_pop}
    OUT["wc_composition_dependence"] = wc_stats
    print(json.dumps(wc_stats, indent=1), flush=True)

    with open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w") as f:
        json.dump(OUT, f, indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
