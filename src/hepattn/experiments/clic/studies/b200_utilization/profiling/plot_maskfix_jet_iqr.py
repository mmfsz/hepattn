"""Jet-E response median + IQR: mask-loss fix (v2) vs the v6 baseline it was forked from.

This is the decisive physics test for the mask-loss broadcast bug (LOSS_BUG_ANALYSIS.md).
The bug provably changed the training signal (gradient cosine ~0.90); this asks whether
that changes reconstruction quality.

Three arms, and they are NOT of equal evidential weight:

1. `v6 baseline (buggy loss)` — 3xL4, per-GPU batch 256, global 768.
2. `v6 maskfix 3xL4 (v2 loss)` — identical to (1) in config, hardware, per-GPU batch, data
   and epoch count. Differs in exactly the loss functions (`mask_bce`/`mask_dice` ->
   `mask_bce_v2`/`mask_dice_v2`) and the matcher solver (scipy -> lap1015_late, verified
   assignment-equivalent). **This pair is the clean A/B; it carries the physics argument.**
3. `v6 maskfix 1xB200 (v2 loss)` — same fixed losses, but global batch 2048 (vs 768) with
   the LR *not* batch-scaled (job 38598204, run as a timing measurement). A CONFOUNDED
   arm: fix and batch/LR change together. Drawn dashed. It is the weaker kind of evidence
   — "a third, differently-trained mask-fixed model lands in the same place" — not a
   controlled comparison, and any difference it shows cannot be pinned on the loss fix.

Evaluation is identical across all three (`configs/eval.yaml`, fp32, batch 256, same test
file), so the eval path contributes nothing to the differences.

Both `mpflow` (regression-refined) and `mpflow_proxy` (incidence-weighted, what the GLOW
paper's Fig. 3/4 curves use) branches are plotted -- the choice shifts the median ~0.03
and the IQR 0.01-0.02, so the A/B is only meaningful within a branch.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/profiling/plot_maskfix_jet_iqr.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

L = Path("/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs")
OUT = Path(__file__).resolve().parent

# (display name, run folder, ckpt stem). RUNS[0] is the baseline every other arm is compared to.
BASE = "v6 baseline (buggy loss)"
L4FIX = "v6 maskfix 3xL4 (v2 loss)"
B200FIX = "v6 maskfix 1xB200 (v2 loss)"
RUNS = [
    (BASE, "clic_v6_20260605-T113014", "epoch=195-val_loss=3.78800"),
    (L4FIX, "clic_v6_maskfix_20260731-T152547", "epoch=199-val_loss=3.36249"),
    (B200FIX, "clic_v6_maskfix_20260803-T132601", "epoch=198-val_loss=3.45147"),
]
BRANCHES = ["mpflow", "mpflow_proxy"]
COLORS = {BASE: "#ffa600", L4FIX: "#003f5c", B200FIX: "#bc5090"}
# The B200 arm is dashed to keep its confound (global batch 2048, unscaled LR) visible.
LINESTYLES = {BASE: "-", L4FIX: "-", B200FIX: "--"}

networks = []
for disp, folder, stem in RUNS:
    p = L / folder / "ckpts" / f"{stem}__test.root"
    if not p.exists():
        print(f"MISSING: {disp} -> {p}")
        continue
    for br in BRANCHES:
        networks.append({"name": f"{disp} [{br}]", "path": str(p), "network_type": br, "ind_threshold": 0.65})

# Jet clustering is ~7 min; cache the per-jet residuals so the bootstrap can be re-run free.
# The cache is keyed to the arm set: `reorder_and_find_intersection` restricts every arm to the
# events ALL arms have, so adding an arm shifts the others too and the old file cannot be reused.
CACHE = OUT / f"maskfix_jet_residuals_{len(RUNS)}arm.npz"
_want = {n["name"] for n in networks}
if CACHE.exists() and _want <= set(np.load(CACHE, allow_pickle=True).files):
    print(f"Loading cached residuals from {CACHE}")
    _z = np.load(CACHE, allow_pickle=True)
    RES = {k: _z[k].item() for k in _z.files}
else:
    config = PerformanceConfig.from_dict({
        "truth_path": "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root",
        "networks": networks,
    })
    perf = Performance(config)
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=8)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    RES = {
        name: {"ref_e": np.asarray(perf.data[name]["jet_residuals"]["ref_e"]),
               "e_rel": np.asarray(perf.data[name]["jet_residuals"]["e_rel"])}
        for name in perf.data
        if "jet_residuals" in perf.data[name]
    }
    np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in RES.items()})
    print(f"Cached residuals to {CACHE}")

e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2


def curve(name, idx=None):
    """Median and IQR of the jet-E response per truth-E bin. `idx` selects a bootstrap resample."""
    res = RES[name]
    ref_e, e_rel = res["ref_e"], res["e_rel"]
    if idx is not None:
        ref_e, e_rel = ref_e[idx], e_rel[idx]
    med = np.full(len(mids), np.nan)
    iqr = np.full(len(mids), np.nan)
    for i, (a, b) in enumerate(zip(e_bins[:-1], e_bins[1:], strict=False)):
        m = (ref_e > a) & (ref_e < b)
        if m.sum() == 0:
            continue
        v = e_rel[m]
        med[i] = np.percentile(v, 50)
        iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
    return med, iqr


def bootstrap_iqr(name, n=400, seed=0):
    """Per-bin IQR standard error, by resampling matched jets with replacement."""
    rng = np.random.default_rng(seed)
    n_jets = len(RES[name]["ref_e"])
    draws = np.stack([curve(name, rng.integers(0, n_jets, n_jets))[1] for _ in range(n)])
    return draws.std(axis=0), draws


fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
print(f"\n{'run':>34} " + " ".join(f"E{int(m):>4}" for m in mids))
print("-" * 34 + "-" * 60)

results, boots = {}, {}
for row, br in enumerate(BRANCHES):
    a1, a2 = axes[row]
    for disp, _f, _s in RUNS:
        name = f"{disp} [{br}]"
        if name not in RES:
            continue
        med, iqr = curve(name)
        err, draws = bootstrap_iqr(name)
        results[name] = (med, iqr)
        boots[name] = draws
        print(f"{name:>34} IQR " + " ".join(f"{x:.3f}" for x in iqr))
        ls = LINESTYLES[disp]
        a1.plot(mids, med, "o", ls=ls, label=disp, color=COLORS[disp], markersize=3)
        a2.errorbar(mids, iqr, yerr=err, fmt="o", ls=ls, label=disp, color=COLORS[disp], markersize=3, capsize=2)
    if "Pandora" in RES:
        medp, iqrp = curve("Pandora")
        if row == 0:
            print(f"{'Pandora':>34} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
        a1.plot(mids, medp, "s--", label="Pandora", color="#666666", markersize=3)
        a2.plot(mids, iqrp, "s--", label="Pandora", color="#666666", markersize=3)

    a1.axhline(0, ls="--", color="k", alpha=0.4)
    a1.set(xlabel="Jet truth E [GeV]", ylabel="Median jet E response", title=f"Median response — {br}")
    a2.set(xlabel="Jet truth E [GeV]", ylabel="IQR of jet E response", title=f"IQR (rises = the glow_jet_iqr discrepancy) — {br}")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.grid(alpha=0.3)

fig.suptitle(
    "Mask-loss fix (v2) vs v6 baseline — 200 epochs, identical eval\n"
    "solid = clean A/B (3xL4, global batch 768)   |   dashed = B200 arm, global batch 2048 "
    "with unscaled LR (confounded)",
    fontsize=11,
)
outpath = OUT / "maskfix_jet_iqr.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")

# --- headline numbers: low-E vs high-E IQR, and the rise across the range
LO, HI = 1, 8  # bins 20-40 GeV and 160-180 GeV
print("\nIQR at low E (20-40) -> high E (160-180), and the rise (+- bootstrap SE):")
for name, (_med, iqr) in results.items():
    d = boots[name]
    rise = d[:, HI] - d[:, LO]
    print(f"  {name:<42} {iqr[LO]:.4f} -> {iqr[HI]:.4f}   rise {iqr[HI] - iqr[LO]:+.4f} +- {rise.std():.4f}")

# Is each maskfix arm distinguishable from the baseline? The models are evaluated on the same
# events, so independent resampling makes this test *conservative* (it overstates the error).
print("\nmaskfix - baseline, per branch (bootstrap; |z| < 2 => not distinguishable):")
for disp, _f, _s in RUNS[1:]:
    for br in BRANCHES:
        nb, nm = f"{RUNS[0][0]} [{br}]", f"{disp} [{br}]"
        if nb not in boots or nm not in boots:
            continue
        tag = f"{disp.split()[2]} {br}"
        for lbl, i in [("IQR @ 160-180", HI), ("IQR @ 20-40", LO)]:
            d = boots[nm][:, i] - boots[nb][:, i]
            print(f"  {tag:<21} {lbl:<14} {d.mean():+.4f} +- {d.std():.4f}   z = {d.mean() / d.std():+.2f}")
        d = (boots[nm][:, HI] - boots[nm][:, LO]) - (boots[nb][:, HI] - boots[nb][:, LO])
        print(f"  {tag:<21} {'rise':<14} {d.mean():+.4f} +- {d.std():.4f}   z = {d.mean() / d.std():+.2f}")
