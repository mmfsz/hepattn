"""Capacity ablation ON THE PAPER TAG: does the falling jet-E IQR survive shrinking the model?

Task 5 showed the paper-tag code reproduces the paper's FALLING IQR, while our drifted HEAD
rises. Task 1 showed that shrinking HEAD 14x (v7, ~0.8M params) leaves HEAD's rising IQR intact,
i.e. capacity is not what broke it there. This is the mirror experiment: the paper tag's own
code narrowed to the same v7 width (dim 256->64, 12.1M -> 0.82M params, everything else the
pristine paper recipe) -- run `clic_paper_small_20260811-T212339`, 200/200 epochs.

Curves are plotted in BOTH conventions: `mpflow_proxy` (incidence-weighted proxy kinematics, the
convention the paper's own notebooks/performance.ipynb uses for Fig. 4) and `mpflow` (the
regression-refined output). Compare like with like -- the paper's 0.072->0.042 is a PROXY curve.

Run:  pixi run -e clic python studies/glow_jet_iqr/07_paper_tag_small/plot_paper_small_iqr.py
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig

L = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs"
LPAPER = "/blue/avery/m.mazza/projects/fastml/hepattn-clic-paper/src/hepattn/experiments/clic/logs"
OUT = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/glow_jet_iqr/07_paper_tag_small"

PAPER_ROOT = f"{LPAPER}/clic_paper_20260715-T173417/ckpts/epoch=194-val_loss=4.01237__test.root"
PAPER_SMALL_ROOT = f"{LPAPER}/clic_paper_small_20260811-T212339/ckpts/epoch=196-val_loss=4.37156__test.root"
HEAD_SMALL_ROOT = f"{L}/clic_v7_20260706-T181417/ckpts/epoch=192-val_loss=4.33644__test.root"   # job 36450143, 2-node
HEAD_SMALL2_ROOT = f"{L}/clic_v7_20260706-T181418/ckpts/epoch=197-val_loss=4.25845__test.root"  # job 36472892, 1-node
BASE_ROOT = f"{L}/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800__test.root"

# (display name, root path, network_type, color, linestyle)
RUNS = [
    ("paper-tag 12.1M PROXY", PAPER_ROOT, "mpflow_proxy", "#c1121f", "-"),
    ("paper-tag 12.1M regr.", PAPER_ROOT, "mpflow", "#c1121f", ":"),
    ("paper-tag 0.82M PROXY", PAPER_SMALL_ROOT, "mpflow_proxy", "#7b2cbf", "-"),
    ("paper-tag 0.82M regr.", PAPER_SMALL_ROOT, "mpflow", "#7b2cbf", ":"),
    ("HEAD v7 0.70M 2-node PROXY", HEAD_SMALL_ROOT, "mpflow_proxy", "#2a9d8f", "-"),
    ("HEAD v7 0.70M 2-node regr.", HEAD_SMALL_ROOT, "mpflow", "#2a9d8f", ":"),
    ("HEAD v7 0.70M 1-node PROXY", HEAD_SMALL2_ROOT, "mpflow_proxy", "#8ecae6", "-"),
    ("HEAD v7 0.70M 1-node regr.", HEAD_SMALL2_ROOT, "mpflow", "#8ecae6", ":"),
    ("HEAD v6 10.1M PROXY", BASE_ROOT, "mpflow_proxy", "#ffa600", "-"),
    ("HEAD v6 10.1M regr.", BASE_ROOT, "mpflow", "#ffa600", ":"),
]

e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2
lo, hi = e_bins[:-1], e_bins[1:]

# The jet clustering + Hungarian jet matching below costs ~13 min for 10 curves, and every
# cosmetic tweak to the figure would otherwise pay it again. Cache the binned curves (a few
# hundred floats) and replot from them. Delete the .npz (or pass --recompute) after re-evaluating
# a checkpoint or changing the binning / jet cuts, or you will silently plot stale numbers.
CACHE = f"{OUT}/jet_iqr_paper_small_curves.npz"
recompute = "--recompute" in sys.argv or not os.path.exists(CACHE)

if recompute:
    networks = []
    for disp, p, ntype, _c, _ls in RUNS:
        if os.path.exists(p):
            networks.append({"name": disp, "path": p, "network_type": ntype, "ind_threshold": 0.65})
        else:
            print(f"SKIP (missing): {disp} -> {p}")

    config = PerformanceConfig.from_dict({
        "truth_path": "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root",
        "networks": networks,
    })
    perf = Performance(config)
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=20)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)

    def _bin(name):
        res = perf.data[name]["jet_residuals"]
        med = np.full(len(mids), np.nan)
        iqr = np.full(len(mids), np.nan)
        for i, (a, b) in enumerate(zip(lo, hi, strict=False)):
            m = (res["ref_e"] > a) & (res["ref_e"] < b)
            if m.sum() == 0:
                continue
            v = res["e_rel"][m]
            med[i] = np.percentile(v, 50)
            iqr[i] = np.percentile(v, 75) - np.percentile(v, 25)
        return med, iqr

    names = list(perf.network_names)
    cached = {}
    for name in names:
        med, iqr = _bin(name)
        cached[f"med::{name}"] = med
        cached[f"iqr::{name}"] = iqr
    np.savez(CACHE, names=np.array(names, dtype=object), **cached)
    print(f"Cached curves -> {CACHE}")
else:
    print(f"Replotting from cache (pass --recompute to redo the jet clustering): {CACHE}")

_z = np.load(CACHE, allow_pickle=True)
network_names = list(_z["names"])


def curve(name):
    return _z[f"med::{name}"], _z[f"iqr::{name}"]


styles = {disp: (c, ls) for disp, _p, _t, c, ls in RUNS}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
print(f"\n{'run':>24} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    # trend sign over the paper's quoted range: first vs last populated bin
    ok = ~np.isnan(iqr)
    trend = iqr[ok][-1] - iqr[ok][0]
    print(f"{name:>24} med " + " ".join(f"{x:+.3f}" for x in med))
    print(f"{name:>24} IQR " + " ".join(f"{x:.3f}" for x in iqr) + f"   [trend {trend:+.3f}]")
    c, ls = styles.get(name, (None, "-"))
    lw = 2.2 if "PROXY" in name else 1.3
    a1.plot(mids, med, ls, marker="o", label=name, color=c, markersize=3, lw=lw)
    a2.plot(mids, iqr, ls, marker="o", label=name, color=c, markersize=3, lw=lw)
if "Pandora" in network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>24} med " + " ".join(f"{x:+.3f}" for x in medp))
    print(f"{'Pandora':>24} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]")
a1.set_ylabel("Median jet E response")
a1.set_title("Median (paper Fig. 4 left: GLOW ~0, Pandora ~+0.02)")
a2.set_xlabel("Jet truth E [GeV]")
a2.set_ylabel("IQR of jet E response")
a2.set_title("IQR (paper Fig. 4 right: GLOW 0.072->0.042, FALLING)")

# 11 curves: a per-axes legend covers the data (it sat on top of the regression curves in the
# first version). One shared legend under both panels instead. Solid = proxy, dotted = regression.
handles, labels = a2.get_legend_handles_labels()
# "outside lower center" makes constrained_layout RESERVE the strip for the legend; a plain
# "lower center" + bbox_to_anchor fights constrained_layout and lands on the x-axis labels.
fig.legend(handles, labels, fontsize=7.5, ncol=4, loc="outside lower center", frameon=False)
fig.suptitle("Capacity ablation on the paper tag: does the falling IQR survive 12.1M -> 0.82M?\n"
             "(solid = PROXY, the paper's Fig. 4 convention;  dotted = regression output)",
             fontsize=11)
outpath = f"{OUT}/jet_iqr_paper_small.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")
