"""Jet E-response median + IQR overlay for the newly-evaluated runs in the investigation.

Answers two tasks at once, now that every training has an eval ROOT:
  - Task 3 (precision): does fp32 (32-true) change the rising-IQR trend vs the bf16 baseline?
  - Task 1 (capacity):  do BOTH 700k v7 runs (1-node + 2-node) show the same trend? (triple check)

Baseline = the plotted 3xL4 bf16 run; Pandora is the reference. Reuses the Performance pipeline.

Run:  pixi run -e clic python studies/glow_jet_iqr/plot_new_evals_iqr.py
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
OUT = "/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/glow_jet_iqr"

# (display name, run folder, ckpt stem, color)
RUNS = [
    ("v6 3xL4 bf16 (baseline)", "clic_v6_20260605-T113014", "epoch=195-val_loss=3.78800", "#ffa600"),
    ("v6 fp32 6xL4",            "clic_v6_fp32_20260707-T150453", "epoch=197-val_loss=3.74189", "#7a5195"),
    ("v7 700k 2-node",          "clic_v7_20260706-T181417", "epoch=192-val_loss=4.33644", "#bc5090"),
    ("v7 700k 1-node",          "clic_v7_20260706-T181418", "epoch=197-val_loss=4.25845", "#ef5675"),
]

networks = []
for disp, folder, stem, _c in RUNS:
    p = f"{L}/{folder}/ckpts/{stem}__test.root"
    if os.path.exists(p):
        networks.append({"name": disp, "path": p, "network_type": "mpflow", "ind_threshold": 0.65})
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


colors = {disp: c for disp, _f, _s, c in RUNS}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
print(f"\n{'run':>26} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in perf.network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    print(f"{name:>26} IQR " + " ".join(f"{x:.3f}" for x in iqr))
    c = colors.get(name)
    a1.plot(mids, med, "o-", label=name, color=c, markersize=3)
    a2.plot(mids, iqr, "o-", label=name, color=c, markersize=3)
if "Pandora" in perf.network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>26} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]")
a1.set_ylabel("Median jet E response")
a1.set_title("Jet E response median")
a1.legend(fontsize=8)
a2.set_xlabel("Jet truth E [GeV]")
a2.set_ylabel("IQR of jet E response")
a2.set_title("Jet E response IQR (rises = the discrepancy)")
a2.legend(fontsize=8)
fig.suptitle("CLIC Glow: fp32 + both 700k runs vs bf16 baseline & Pandora")
outpath = f"{OUT}/jet_iqr_new_evals.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")
