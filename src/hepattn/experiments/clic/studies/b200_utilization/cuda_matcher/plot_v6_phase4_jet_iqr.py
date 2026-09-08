"""Phase 4 — physics equivalence of the GPU matcher, on v6, both arms converged.

§6's third ship condition. Phases 0-3 established that the device solver is exact on real cost
matrices and 2.3-3.1x faster inside training; none of that says the model it trains is the same
model. This does.

  device : clic_v6_maskfix_20260827-T101733  job 40393482  device_solver=jv          11 h 22 m
  host   : clic_v6_maskfix_20260803-T132601  job 38598204  lap1015_late, n_jobs 16   (the
           baseline submit_phase4_device_training_b200.sh was written against, deliberately
           reused rather than retrained)

Their resolved configs differ in exactly the matcher block -- `default_solver`,
`parallel_solver`, and the four `device_solver_*` keys that did not exist yet in August -- and
in nothing else: same 10,126,115 parameters, same batch 2048, same 1x B200, same 200 epochs,
same seed, same data. Twenty-four days apart, so the torch build is not identical; that is the
one uncontrolled variable and it is noted rather than hidden.

Unlike the v7 A/B (plot_v7_matcher_ab_jet_iqr.py, next to this file), which had to be taken at
epoch 169 because its host arm was still training, both arms here ran the full schedule. Each
is evaluated at its own lowest-val_loss checkpoint, which is what the v6 baseline's original
evaluation did.

⚠️ Both arms are scored with the HOST solver at evaluation time, so the eval path contributes
nothing to any difference below. What is being compared is the two TRAINED MODELS.

Thresholds, from the model-size study's STUDY.md §4.3 -- 2 sigma_tot, dominated by
training-to-training scatter (sigma_repro ~ 0.0036 global), not by the test sample:
    global jet-E IQR  0.0073     high-E bin (160-180)  0.0164
The bootstrap here measures sigma_stat only, the smaller half. **A difference inside the
bootstrap error is certainly noise; one outside it still has to clear the threshold.** Note
what this means for a PASS: the test can only say "no degradation larger than 0.0073", which
is a bound, not a proof of identity.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/cuda_matcher/plot_v6_phase4_jet_iqr.py
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

DEV = "device (jv)"
HOST = "host (lap1015)"
ARMS = [
    (DEV, "clic_v6_maskfix_20260827-T101733", "epoch=196-val_loss=3.38756"),
    (HOST, "clic_v6_maskfix_20260803-T132601", "epoch=198-val_loss=3.45147"),
]
BRANCHES = ["mpflow", "mpflow_proxy"]
COLORS = {DEV: "#bc5090", HOST: "#003f5c"}

networks = []
for disp, folder, stem in ARMS:
    p = L / folder / "ckpts" / f"{stem}__test.root"
    if not p.exists():
        print(f"MISSING: {disp} -> {p}")
        continue
    networks.extend({"name": f"{disp} [{br}]", "path": str(p), "network_type": br, "ind_threshold": 0.65} for br in BRANCHES)

# Cache the per-jet residuals; the clustering is the expensive step and the bootstrap is then
# free to re-run. Keyed to the arm set -- reorder_and_find_intersection restricts every arm to
# the events all arms share, so an old cache from a different arm set is not reusable.
CACHE = OUT / f"v6_phase4_residuals_{len(networks)}net.npz"
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

# The 20 GeV grid the earlier jet-IQR work used, kept so the high-E number stays comparable.
e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2
LO, HI = 1, 8


def curve(name, idx=None):
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
    rng = np.random.default_rng(seed)
    n_jets = len(RES[name]["ref_e"])
    draws = [curve(name, rng.integers(0, n_jets, n_jets)) for _ in range(n)]
    return (np.stack([d[1] for d in draws]), np.array([d[2] for d in draws]), np.array([d[3] for d in draws]))


fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
results, boots, globals_, medians = {}, {}, {}, {}

# NOT the same jets for every arm, despite sharing every event. `hung_match_jets` runs per
# network and the dr < 0.1 / pt_min cuts are then applied per network, so an arm that
# reconstructs a jet slightly differently keeps or loses it. Counts spread ~0.6% across arms.
# The header used to claim the jets were identical, which is false and invites reading a change
# of quoted arm as a change in the sample.
ref0 = RES[f"{ARMS[0][0]} [{BRANCHES[0]}]"]["ref_e"]
_counts = {k: len(v["ref_e"]) for k, v in RES.items()}
print(f"\njets per truth-E bin, for {ARMS[0][0]} [{BRANCHES[0]}] (arms share events, not jets):")
print("  " + "  ".join(f"E{int(m)}:{int(((ref0 > a) & (ref0 < b)).sum())}" for m, (a, b) in zip(mids, pairwise(e_bins), strict=False)))
print(f"  total matched jets: {len(ref0)}   (across all arms: {min(_counts.values())}-{max(_counts.values())})")

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
        a1.plot(mids, med, "o-", label=disp, color=COLORS[disp], markersize=3)
        a2.errorbar(mids, iqr, yerr=biqr.std(axis=0), fmt="o-", label=disp, color=COLORS[disp], markersize=3, capsize=2)
    a1.axhline(0, ls="--", color="k", alpha=0.4)
    a1.set(xlabel="Jet truth E [GeV]", ylabel="Median jet E response", title=f"Median response — {br}")
    a2.set(xlabel="Jet truth E [GeV]", ylabel="IQR of jet E response", title=f"IQR — {br}")
    for a in (a1, a2):
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

fig.suptitle(
    "Phase 4 — v6, 200 epochs, batch 2048, 1x B200: GPU matcher (jv) vs CPU matcher (lap1015)\n"
    "identical configs but for the solver; both arms scored with the host solver at evaluation",
    fontsize=11,
)
outpath = OUT / "v6_phase4_jet_iqr.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")

print("\nglobal jet-E median response (+- bootstrap sigma_stat) — energy SCALE, a different failure mode from resolution:")
for name, (g, e) in medians.items():
    print(f"  {name:<34} {g:+.4f} +- {e:.4f}")

print("\nglobal jet-E IQR (+- bootstrap sigma_stat):")
for name, (g, e) in globals_.items():
    print(f"  {name:<28} {g:.4f} +- {e:.4f}")

print("\nper-bin IQR:")
print(f"{'arm':>28} " + " ".join(f"E{int(m):>5}" for m in mids))
for name, (_med, iqr) in results.items():
    print(f"{name:>28} " + " ".join(f"{x:.3f}" for x in iqr))

# Both arms see the same events, so independent resampling overstates the error: conservative.
print("\nPhase 4 verdict, device - host (thresholds from STUDY.md 4.3):")
for br in BRANCHES:
    nd, nh = f"{DEV} [{br}]", f"{HOST} [{br}]"
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

# The sign check of STUDY.md 4.3: a genuine capacity loss moves both conventions the same way.
print("\nsign check (both conventions must agree before a difference is believed):")
for lbl, i in [("IQR global", None), ("IQR high-E", HI), ("median global", "med")]:
    ds = []
    for br in BRANCHES:
        nd, nh = f"{DEV} [{br}]", f"{HOST} [{br}]"
        if i == "med":
            ds.append(medians[nd][0] - medians[nh][0])
        elif i is None:
            ds.append(globals_[nd][0] - globals_[nh][0])
        else:
            ds.append(results[nd][1][i] - results[nh][1][i])
    agree = "same sign" if ds[0] * ds[1] > 0 else "OPPOSITE signs -> reads as noise"
    print(f"  {lbl:<16} mpflow {ds[0]:+.4f}   mpflow_proxy {ds[1]:+.4f}   {agree}")
