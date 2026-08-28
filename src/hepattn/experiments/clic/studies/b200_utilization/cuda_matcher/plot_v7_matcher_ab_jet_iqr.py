"""v7 GPU-matcher vs CPU-matcher: jet-E response median + IQR (Tier 1 of STUDY.md §4.3).

The physics question the training curves cannot answer. Type-A (loss) matching flatters a model
against the truth particle the matcher itself chose; this script uses the type-C path -- cluster
the reconstructed particles into jets, match jets to truth jets on kinematics, and look at the
energy response -- which is how Pandora is scored and the only honest reading (§4.1).

ARMS. The host arm (job 40400036) was still training when this was first run, so the controlled
comparison is taken at the last epoch BOTH arms had, epoch 169, evaluated from the two
checkpoints with everything else held fixed. The device arm's converged epoch-197 point is
plotted too, as the answer to "where does the device arm actually land", but it is NOT the A/B:
it has 28 more epochs than the epoch-169 pair. Add the host arm's converged point to ARMS when
job 40400036 finishes and the pair at 197-ish becomes the headline.

Both arms are evaluated with the HOST solver (studies/model_size/submit_eval_v7.sh forces it),
so the eval path is identical and contributes nothing to any difference seen here.

Thresholds for "is a difference real", from STUDY.md §4.3 -- 2 sigma_tot, dominated by
training-to-training scatter, not by the test sample:
    global jet-E IQR   0.0073
    high-E bin         0.0164
The bootstrap below measures only sigma_stat, which is the smaller half; a difference inside
the bootstrap error is certainly not real, and one outside it still has to clear the threshold.

Binning is the 20 GeV grid the earlier jet-IQR studies used, kept so the high-E numbers stay
comparable to what is already on record (STUDY.md §4.3's binning note).

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/cuda_matcher/plot_v7_matcher_ab_jet_iqr.py
"""

import sys
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

L = Path("/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs")
OUT = Path(__file__).resolve().parent

DEV169 = "device jv @169"
HOST169 = "host lap1015 @169"
DEV197 = "device jv @197"
# (display name, run folder, ckpt stem). The first two are the paired A/B.
ARMS = [
    (DEV169, "clic_v7_cudamatch_b200_b2048_20260827-T130314", "epoch=169-val_loss=4.12771"),
    (HOST169, "clic_v7_maskfix_b200_b2048_20260827-T120002", "epoch=169-val_loss=4.11265"),
    (DEV197, "clic_v7_cudamatch_b200_b2048_20260827-T130314", "epoch=197-val_loss=4.05102"),
]
PAIR = (DEV169, HOST169)
BRANCHES = ["mpflow", "mpflow_proxy"]
COLORS = {DEV169: "#bc5090", HOST169: "#003f5c", DEV197: "#ffa600"}
LINESTYLES = {DEV169: "-", HOST169: "-", DEV197: "--"}

networks = []
for disp, folder, stem in ARMS:
    p = L / folder / "ckpts" / f"{stem}__test.root"
    if not p.exists():
        print(f"MISSING: {disp} -> {p}")
        continue
    networks.extend({"name": f"{disp} [{br}]", "path": str(p), "network_type": br, "ind_threshold": 0.65} for br in BRANCHES)

# Jet clustering is the expensive step; cache the per-jet residuals so the bootstrap is free on
# re-runs. Keyed to the arm set: reorder_and_find_intersection restricts every arm to the events
# ALL arms have, so adding an arm shifts the others and an old cache cannot be reused.
CACHE = OUT / f"v7_matcher_ab_residuals_{len(networks)}net.npz"
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
        name: {"ref_e": np.asarray(perf.data[name]["jet_residuals"]["ref_e"]), "e_rel": np.asarray(perf.data[name]["jet_residuals"]["e_rel"])}
        for name in perf.data
        if "jet_residuals" in perf.data[name]
    }
    np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in RES.items()})
    print(f"Cached residuals to {CACHE}")

e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2
LO, HI = 1, 8  # 20-40 GeV and 160-180 GeV


def curve(name, idx=None):
    """Per-bin median and IQR of the jet-E response, plus the global IQR. `idx` = bootstrap draw."""
    res = RES[name]
    ref_e, e_rel = res["ref_e"], res["e_rel"]
    if idx is not None:
        ref_e, e_rel = ref_e[idx], e_rel[idx]
    med = np.full(len(mids), np.nan)
    iqr = np.full(len(mids), np.nan)
    for i, (a, b) in enumerate(pairwise(e_bins)):
        m = (ref_e > a) & (ref_e < b)
        if m.sum() == 0:
            continue
        v = e_rel[m]
        med[i] = np.percentile(v, 50)
        iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
    glob = np.percentile(e_rel, 75) - np.percentile(e_rel, 25)
    return med, iqr, glob, np.percentile(e_rel, 50)


def bootstrap(name, n=400, seed=0):
    """sigma_stat on the per-bin IQR and the global IQR, by resampling matched jets."""
    rng = np.random.default_rng(seed)
    n_jets = len(RES[name]["ref_e"])
    draws = [curve(name, rng.integers(0, n_jets, n_jets)) for _ in range(n)]
    return (np.stack([d[1] for d in draws]), np.array([d[2] for d in draws]), np.array([d[3] for d in draws]))


fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
results, boots, globals_, medians = {}, {}, {}, {}

# Jet counts per bin: STUDY.md P0.1 asks for these explicitly, and the low-E threshold is open
# until they are on record.
ref0 = RES[f"{ARMS[0][0]} [{BRANCHES[0]}]"]["ref_e"]
print("\njets per truth-E bin (identical across arms — the same matched jets):")
print("  " + "  ".join(f"E{int(m)}:{int(((ref0 > a) & (ref0 < b)).sum())}" for m, (a, b) in zip(mids, pairwise(e_bins), strict=False)))
print(f"  total matched jets: {len(ref0)}")

for row, br in enumerate(BRANCHES):
    a1, a2 = axes[row]
    for disp, _f, _s in ARMS:
        name = f"{disp} [{br}]"
        if name not in RES:
            continue
        med, iqr, glob, gmed = curve(name)
        biqr, bglob, bgmed = bootstrap(name)
        results[name] = (med, iqr)
        boots[name] = biqr
        globals_[name] = (glob, bglob.std())
        medians[name] = (gmed, bgmed.std())
        ls = LINESTYLES[disp]
        a1.plot(mids, med, "o", ls=ls, label=disp, color=COLORS[disp], markersize=3)
        a2.errorbar(mids, iqr, yerr=biqr.std(axis=0), fmt="o", ls=ls, label=disp, color=COLORS[disp], markersize=3, capsize=2)
    a1.axhline(0, ls="--", color="k", alpha=0.4)
    a1.set(xlabel="Jet truth E [GeV]", ylabel="Median jet E response", title=f"Median response — {br}")
    a2.set(xlabel="Jet truth E [GeV]", ylabel="IQR of jet E response", title=f"IQR — {br}")
    for a in (a1, a2):
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

fig.suptitle(
    "v7 batch-2048 B200 — GPU matcher (jv) vs CPU matcher (lap1015), identical configs but for the solver\n"
    "solid = the controlled pair at epoch 169   |   dashed = device arm converged at epoch 197 (28 epochs further, not the A/B)",
    fontsize=11,
)
outpath = OUT / "v7_matcher_ab_jet_iqr.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")

print("\nglobal jet-E median response (+- bootstrap sigma_stat) — energy SCALE, a different failure mode from resolution:")
for name, (g, e) in medians.items():
    print(f"  {name:<34} {g:+.4f} +- {e:.4f}")

print("\nglobal jet-E IQR (+- bootstrap sigma_stat):")
for name, (g, e) in globals_.items():
    print(f"  {name:<34} {g:.4f} +- {e:.4f}")

print("\nper-bin IQR:")
print(f"{'arm':>34} " + " ".join(f"E{int(m):>5}" for m in mids))
for name, (_med, iqr) in results.items():
    print(f"{name:>34} " + " ".join(f"{x:.3f}" for x in iqr))

# The A/B itself. Same events for both arms, so resampling them independently OVERSTATES the
# error -- the test is conservative.
print("\nA/B, device - host at epoch 169 (bootstrap sigma_stat; thresholds from STUDY.md 4.3):")
for br in BRANCHES:
    nd, nh = f"{PAIR[0]} [{br}]", f"{PAIR[1]} [{br}]"
    if nd not in boots or nh not in boots:
        print(f"  {br}: arm missing, skipped")
        continue
    for lbl, i, thr in [("IQR @ 0-20 (low E)", 0, 0.0170), ("IQR @ 20-40", LO, None), ("IQR @ 160-180 (high E)", HI, 0.0164)]:
        d = boots[nd][:, i] - boots[nh][:, i]
        delta = results[nd][1][i] - results[nh][1][i]
        t = f"   threshold {thr:.4f} -> {'REAL' if abs(delta) > thr else 'not detectable'}" if thr else ""
        print(f"  {br:<12} {lbl:<24} {delta:+.4f} +- {d.std():.4f}  z = {d.mean() / d.std():+.2f}{t}")
    gd, gh = globals_[nd][0], globals_[nh][0]
    se = np.hypot(globals_[nd][1], globals_[nh][1])
    print(f"  {br:<12} {'IQR global':<24} {gd - gh:+.4f} +- {se:.4f}  threshold 0.0073 -> {'REAL' if abs(gd - gh) > 0.0073 else 'not detectable'}")
    # No published threshold exists for the median: STUDY.md 4.2 measured training-to-training
    # scatter for the IQR only. So this row gets sigma_stat and no verdict -- deliberately.
    md, mh = medians[nd][0], medians[nh][0]
    mse = np.hypot(medians[nd][1], medians[nh][1])
    print(f"  {br:<12} {'median global (scale)':<24} {md - mh:+.4f} +- {mse:.4f}  sigma_stat only — no sigma_repro on record for the median")

# The sign check of STUDY.md §4.3: a real difference moves both conventions the same way.
print("\nsign check (both conventions must agree before a difference is believed):")
for lbl, kind in [("IQR global", "iqr"), ("IQR high-E", "hi"), ("median global", "med")]:
    ds = []
    for br in BRANCHES:
        nd, nh = f"{PAIR[0]} [{br}]", f"{PAIR[1]} [{br}]"
        if kind == "iqr":
            ds.append(globals_[nd][0] - globals_[nh][0])
        elif kind == "med":
            ds.append(medians[nd][0] - medians[nh][0])
        else:
            ds.append(results[nd][1][HI] - results[nh][1][HI])
    agree = "same sign" if ds[0] * ds[1] > 0 else "OPPOSITE signs -> reads as noise"
    print(f"  {lbl:<16} mpflow {ds[0]:+.4f}   mpflow_proxy {ds[1]:+.4f}   {agree}")
