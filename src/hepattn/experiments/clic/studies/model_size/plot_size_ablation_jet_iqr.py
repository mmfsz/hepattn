"""Size ablation: jet-E response median + IQR for every arm against the paper-tag reference.

The Tier-1 reading, and the first physics this branch's model-size study produces. The training
curves cannot answer the question being asked: type-A (loss) matching flatters a model against the
truth particle the matcher itself chose, so a smaller model that has learned to choose differently
can look unchanged in val_loss and still be worse. This script uses the type-C path -- cluster the
reconstructed particles into jets, match jets to truth jets on kinematics, look at the energy
response -- which is how Pandora is scored and the only honest reading.

Ported from the head-based study on `main`; the binning, the bootstrap and the two-convention sign
check are that script's. What is NOT ported is its threshold test, and the reason is in the next
paragraph.

⚠️ NO VERDICT COLUMN. Head's script printed REAL / not-detectable against thresholds of 0.0073
(global IQR), 0.0164 (high-E) and 0.0170 (low-E). Those are properties of head's code, measured
from four seed trainings of head's reference, and they do not transfer: this branch differs in the
feed-forward activation, the q/k/v norms and the incidence head, and the training-to-training
scatter is its own unmeasured quantity. Every delta below therefore carries its bootstrap
sigma_stat and nothing else. sigma_stat is the SMALLER half of the error -- it measures only the
finiteness of the test sample, not the scatter between two trainings of the same config -- so:

    |delta| < sigma_stat     certainly not a real difference
    |delta| > sigma_stat     not excluded, and not yet established either

Read the output as an ORDERING of the arms, not as a set of verdicts. It becomes a verdict when a
seed set is trained on this code and sigma_repro is measured; the bar on an arm-vs-reference
difference is then 2*sqrt(2)*sigma_repro, the sqrt(2) because both sides are independently trained.

ARMS live in `arm_sets.py`. In the default `all` set the arms are the three pairs C5/C4/C3 and the
triple C1 of the 2^3 factorial over {A2, A3, A4}; the three singles were never trained here. Unlike
head's table C1 is ON the canvas -- it trained on this code, and whether that survives the physics
is the question this directory exists to answer. Every arm is at its own lowest-val_loss
checkpoint, which for C4 and C3 is not the last epoch.

Every arm was trained with the GPU matcher (`device_solver: jv`) and every one is EVALUATED with
the host solver -- `configs/eval.yaml` forces it -- so the eval path is identical across arms and
contributes nothing to any difference seen here.

Binning is the 20 GeV grid head's jet-IQR studies used, kept so the shape of these curves can be
laid beside those even though the numbers cannot be compared.

Run from the clic experiment dir (via submit_size_plots.sh -- jet clustering wants cores):
    pixi run -e clic python studies/model_size/plot_size_ablation_jet_iqr.py
"""

import hashlib
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn-paper/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CACHE_DIR = OUT / "cache"

# The truth file is the shared CLIC test sample; it lives in the head clone's data directory and
# is read, never written, so both branches use the one copy. The arms' own eval configs point at
# `test_clic_common_infer.root` in that same directory.
TRUTH = "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root"

# The arm table -- which runs, which checkpoints, which colours -- lives in `arm_sets.py`, shared
# with the other plot scripts. ARM_SET picks the set; see that module for what each one asks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_sets import select  # noqa: E402

ARM_SET, SET = select()
PREFIX = SET.prefix
# (display name, .root path, trainable params), the shape the drawing code below wants.
# The .root path comes from `Arm.root`, never rebuilt here: an arm may point at a non-default
# evaluation of its checkpoint (`test_suff`), and a second copy of the filename convention
# silently reads the wrong file when it does.
ARMS = [(a.label, a.root, a.params) for a in SET.arms]
REF = SET.baseline.label  # the baseline of THIS set; every delta below is taken against it
PARAMS = {a.label: a.params for a in SET.arms}
COLORS = {a.label: a.color for a in SET.arms}
# Heavy lines are the ones a reader is meant to READ rather than scan past: the set's baseline,
# and any arm the set's question is actually about.
LINEWIDTHS = {a.label: (2.4 if a.heavy else 1.5) for a in SET.arms}
MARKERS = {a.label: a.marker for a in SET.arms}

BRANCHES = ["mpflow", "mpflow_proxy"]

networks = []
for disp, root, _p in ARMS:
    if not root.exists():
        print(f"MISSING: {disp} -> {root}")
        continue
    networks.extend({"name": f"{disp} [{br}]", "path": str(root), "network_type": br, "ind_threshold": 0.65} for br in BRANCHES)

# Jet clustering is the expensive step; cache the per-jet residuals so the bootstrap is free on
# re-runs. Keyed to the arm set: reorder_and_find_intersection restricts every arm to the events
# ALL arms have, so adding an arm shifts the others and an old cache cannot be reused.
_key = hashlib.sha256("|".join(sorted(n["path"] for n in networks)).encode()).hexdigest()[:12]
CACHE = CACHE_DIR / f"{PREFIX}_residuals_{len(networks)}net_{_key}.npz"
_want = {n["name"] for n in networks}
if CACHE.exists() and _want <= set(np.load(CACHE, allow_pickle=True).files):
    print(f"Loading cached residuals from {CACHE}")
    _z = np.load(CACHE, allow_pickle=True)
    RES = {k: _z[k].item() for k in _z.files}
else:
    config = PerformanceConfig.from_dict({"truth_path": TRUTH, "networks": networks})
    perf = Performance(config)
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=16)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    RES = {
        name: {"ref_e": np.asarray(perf.data[name]["jet_residuals"]["ref_e"]), "e_rel": np.asarray(perf.data[name]["jet_residuals"]["e_rel"])}
        for name in perf.data
        if "jet_residuals" in perf.data[name]
    }
    CACHE_DIR.mkdir(exist_ok=True)
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

# Jet counts per bin, quoted for ONE named arm because the counts are arm-dependent and not the
# same matched jets: every arm is jet-matched to truth separately and then cut on dr < 0.1, so an
# arm that reconstructs a jet slightly differently keeps or loses it. The spread across arms is
# printed alongside so a change in it cannot be misread as a change in the event intersection --
# that is identical for every arm.
ref_name = f"{REF} [{BRANCHES[0]}]"
ref0 = RES[ref_name]["ref_e"]
print(f"\njets per truth-E bin, for {ref_name}:")
print("  " + "  ".join(f"E{int(m)}:{int(((ref0 > a) & (ref0 < b)).sum())}" for m, (a, b) in zip(mids, pairwise(e_bins), strict=False)))
print(f"  low-E 0-50 GeV: {int(((ref0 > 0) & (ref0 < 50)).sum())}")
_counts = {k: len(v["ref_e"]) for k, v in RES.items()}
print(f"  total matched jets: {len(ref0)}   (across all arms: {min(_counts.values())}-{max(_counts.values())})")

for row, br in enumerate(BRANCHES):
    a1, a2 = axes[row]
    for disp, _root, _p in ARMS:
        name = f"{disp} [{br}]"
        if name not in RES:
            continue
        med, iqr, glob, gmed = curve(name)
        biqr, bglob, bgmed = bootstrap(name)
        results[name] = (med, iqr)
        boots[name] = biqr
        globals_[name] = (glob, bglob.std())
        medians[name] = (gmed, bgmed.std())
        style = {"color": COLORS[disp], "lw": LINEWIDTHS[disp], "markersize": 4}
        a1.plot(mids, med, MARKERS[disp] + "-", label=disp, **style)
        a2.errorbar(mids, iqr, yerr=biqr.std(axis=0), fmt=MARKERS[disp] + "-", label=disp, capsize=2, **style)
    a1.axhline(0, ls="--", color="k", alpha=0.4)
    a1.set(xlabel="Jet truth E [GeV]", ylabel="Median jet E response", title=f"Median response — {br}")
    a2.set(xlabel="Jet truth E [GeV]", ylabel="IQR of jet E response", title=f"IQR — {br}")
    for a in (a1, a2):
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

fig.suptitle(SET.title, fontsize=11)
FIG.mkdir(exist_ok=True)
outpath = FIG / f"{PREFIX}_jet_iqr.png"
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

# The ablation itself. Same events for every arm, so resampling them independently OVERSTATES the
# error -- the comparison is conservative in that one respect, and silent about sigma_repro, which
# is the larger one and is not measured on this code. No verdicts: see the module docstring.
print("\narm - reference, per convention. sigma_stat ONLY — no sigma_repro on this code, so no verdicts:")
for br in BRANCHES:
    nr = f"{REF} [{br}]"
    print(f"\n  --- {br} ---")
    for disp, _root, _p in ARMS[1:]:
        na = f"{disp} [{br}]"
        if na not in boots or nr not in boots:
            print(f"    {disp}: arm missing, skipped")
            continue
        for lbl, i in [("IQR @ 0-20 (low E)", 0), ("IQR @ 20-40", LO), ("IQR @ 160-180 (high E)", HI)]:
            d = boots[na][:, i] - boots[nr][:, i]
            delta = results[na][1][i] - results[nr][1][i]
            print(f"    {disp:<16} {lbl:<24} {delta:+.4f} +- {d.std():.4f}  z = {d.mean() / d.std():+.2f}")
        ga, gr = globals_[na][0], globals_[nr][0]
        se = np.hypot(globals_[na][1], globals_[nr][1])
        print(f"    {disp:<16} {'IQR global':<24} {ga - gr:+.4f} +- {se:.4f}")
        ma, mr = medians[na][0], medians[nr][0]
        mse = np.hypot(medians[na][1], medians[nr][1])
        print(f"    {disp:<16} {'median global (scale)':<24} {ma - mr:+.4f} +- {mse:.4f}")

# The sign check: a real difference moves both conventions the same way. This is the one test here
# that does NOT need sigma_repro -- it asks for consistency, not for significance -- so it is the
# strongest statement this script can make until a seed set exists.
print("\nsign check (both conventions must agree before a difference is believed):")
for lbl, kind in [("IQR global", "iqr"), ("IQR high-E", "hi"), ("median global", "med")]:
    print(f"  {lbl}:")
    for disp, _root, _p in ARMS[1:]:
        ds = []
        for br in BRANCHES:
            na, nr = f"{disp} [{br}]", f"{REF} [{br}]"
            if kind == "iqr":
                ds.append(globals_[na][0] - globals_[nr][0])
            elif kind == "med":
                ds.append(medians[na][0] - medians[nr][0])
            else:
                ds.append(results[na][1][HI] - results[nr][1][HI])
        agree = "same sign" if ds[0] * ds[1] > 0 else "OPPOSITE signs -> reads as noise"
        print(f"    {disp:<16} mpflow {ds[0]:+.4f}   mpflow_proxy {ds[1]:+.4f}   {agree}")

# The headline table: what each arm costs in physics per parameter removed. Deliberately last, and
# deliberately without a "winner" -- the verdict belongs in the study notes, not in a script, and
# on this code it needs an error bar that has not been measured yet.
print("\nsummary — parameters against global jet-E IQR (mpflow):")
gr = globals_[f"{REF} [mpflow]"][0]
print(f"  {'arm':<18} {'params':>9} {'ratio':>7} {'IQR':>8} {'dIQR':>8} {'sigma_stat':>11}")
for disp, _root, p in ARMS:
    g, e = globals_[f"{disp} [mpflow]"]
    print(f"  {disp:<18} {p:>9,} {p / PARAMS[REF]:>6.3f}x {g:>8.4f} {g - gr:>+8.4f} {e:>11.4f}")
print("\n  sigma_stat is the test-sample error only. The bar an arm must clear is")
print("  2*sqrt(2)*sigma_repro, and sigma_repro has not been measured on this code.")
