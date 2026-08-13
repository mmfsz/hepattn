"""Proxy-vs-regression overlay: does the paper plot the PROXY output, not the regression?

Discovery (2026-07-20): the paper tag's own notebooks/performance.ipynb evaluates GLOW with
network_type "mpflow_proxy" (incidence-weighted proxy kinematics), NOT "mpflow" (the
regression-refined output) that this whole study has plotted so far. Every eval root stores
both branch sets, so we can re-plot both conventions with no re-evaluation.

Run:  pixi run -e clic python studies/glow_jet_iqr/05_reproduce_paper_tag/plot_paper_iqr_proxy.py
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
OUT = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/glow_jet_iqr/05_reproduce_paper_tag"

PAPER_ROOT = f"{LPAPER}/clic_paper_20260715-T173417/ckpts/epoch=194-val_loss=4.01237__test.root"
BASE_ROOT = f"{L}/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800__test.root"

# (display name, root path, network_type, color, linestyle)
RUNS = [
    ("paper-tag PROXY (paper convention)", PAPER_ROOT, "mpflow_proxy", "#c1121f", "-"),
    ("paper-tag regression",               PAPER_ROOT, "mpflow",       "#c1121f", ":"),
    ("v6 baseline PROXY",                  BASE_ROOT,  "mpflow_proxy", "#ffa600", "-"),
    ("v6 baseline regression",             BASE_ROOT,  "mpflow",       "#ffa600", ":"),
]

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

e_bins = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
mids = (e_bins[:-1] + e_bins[1:]) / 2
lo, hi = e_bins[:-1], e_bins[1:]


def curve(name):
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


styles = {disp: (c, ls) for disp, _p, _t, c, ls in RUNS}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
print(f"\n{'run':>36} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in perf.network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    print(f"{name:>36} med " + " ".join(f"{x:+.3f}" for x in med))
    print(f"{name:>36} IQR " + " ".join(f"{x:.3f}" for x in iqr))
    c, ls = styles.get(name, (None, "-"))
    lw = 2.2 if "PROXY" in name else 1.3
    a1.plot(mids, med, ls, marker="o", label=name, color=c, markersize=3, lw=lw)
    a2.plot(mids, iqr, ls, marker="o", label=name, color=c, markersize=3, lw=lw)
if "Pandora" in perf.network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>36} med " + " ".join(f"{x:+.3f}" for x in medp))
    print(f"{'Pandora':>36} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]")
a1.set_ylabel("Median jet E response")
a1.set_title("Median (paper Fig. 4 left: GLOW ~0, Pandora ~+0.02)")
a1.legend(fontsize=7)
a2.set_xlabel("Jet truth E [GeV]")
a2.set_ylabel("IQR of jet E response")
a2.set_title("IQR (paper Fig. 4 right: GLOW 0.072->0.042)")
a2.legend(fontsize=7)
fig.suptitle("CLIC Glow: proxy (paper notebook convention) vs regression output")
outpath = f"{OUT}/jet_iqr_proxy_vs_regression.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")
