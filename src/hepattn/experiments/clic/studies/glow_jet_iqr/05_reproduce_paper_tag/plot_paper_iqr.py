"""Task 5 DECISIVE overlay: paper-tag (clic-paper, 12.1M) jet-E IQR vs HEAD runs + Pandora.

The paper-tag run trains the paper's actual code (tag fb90390, separate clone
hepattn-clic-paper). If its IQR FALLS with jet E (like paper Fig. 4) while every
HEAD run RISES, the post-paper model refactor is confirmed as the cause.

Run:  pixi run -e clic python studies/glow_jet_iqr/05_reproduce_paper_tag/plot_paper_iqr.py
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

# (display name, full eval-root path, color)
RUNS = [
    ("paper-tag 12.1M (clic-paper)",
     f"{LPAPER}/clic_paper_20260715-T173417/ckpts/epoch=194-val_loss=4.01237__test.root", "#c1121f"),
    ("v6 3xL4 bf16 (baseline HEAD)",
     f"{L}/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800__test.root", "#ffa600"),
    ("v6 fp32 6xL4",
     f"{L}/clic_v6_fp32_20260707-T150453/ckpts/epoch=197-val_loss=3.74189__test.root", "#7a5195"),
    ("v7 700k 2-node",
     f"{L}/clic_v7_20260706-T181417/ckpts/epoch=192-val_loss=4.33644__test.root", "#bc5090"),
    ("v7 700k 1-node",
     f"{L}/clic_v7_20260706-T181418/ckpts/epoch=197-val_loss=4.25845__test.root", "#ef5675"),
]

networks = []
for disp, p, _c in RUNS:
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


colors = {disp: c for disp, _p, c in RUNS}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
print(f"\n{'run':>30} " + " ".join(f"E{int(m):>3}" for m in mids))
for name in perf.network_names:
    if name == "Pandora":
        continue
    med, iqr = curve(name)
    print(f"{name:>30} IQR " + " ".join(f"{x:.3f}" for x in iqr))
    c = colors.get(name)
    lw = 2.2 if name.startswith("paper-tag") else 1.2
    a1.plot(mids, med, "o-", label=name, color=c, markersize=3, lw=lw)
    a2.plot(mids, iqr, "o-", label=name, color=c, markersize=3, lw=lw)
if "Pandora" in perf.network_names:
    medp, iqrp = curve("Pandora")
    print(f"{'Pandora':>30} IQR " + " ".join(f"{x:.3f}" for x in iqrp))
    a1.plot(mids, medp, "s--", label="Pandora", color="#003f5c", markersize=3)
    a2.plot(mids, iqrp, "s--", label="Pandora", color="#003f5c", markersize=3)

a1.axhline(0, ls="--", color="k", alpha=0.4)
a1.set_xlabel("Jet truth E [GeV]")
a1.set_ylabel("Median jet E response")
a1.set_title("Jet E response median")
a1.legend(fontsize=8)
a2.set_xlabel("Jet truth E [GeV]")
a2.set_ylabel("IQR of jet E response")
a2.set_title("Jet E response IQR (paper Fig. 4: falls with E)")
a2.legend(fontsize=8)
fig.suptitle("CLIC Glow task 5: paper-tag (fb90390) vs HEAD runs & Pandora")
outpath = f"{OUT}/jet_iqr_paper_vs_head.png"
fig.savefig(outpath, dpi=140)
print(f"\nSaved: {outpath}")
