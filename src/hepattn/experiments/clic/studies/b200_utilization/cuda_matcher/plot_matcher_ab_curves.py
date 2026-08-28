"""GPU-matcher vs CPU-matcher: paired training/validation curves (Tier 0 of the model-size
study's STUDY.md §4.3), for both A/Bs this study has run.

These are the cleanest A/Bs in the project. Within each pair the resolved configs differ in the
matcher block -- `default_solver`, `parallel_solver`, `device_solver` -- plus the run name and
output directory, and in nothing else: same seed, data, batch, schedule and 1x B200 geometry.
So every curve below is a *paired* comparison, and the loss values ARE directly comparable,
unlike the mask-fix A/B this script is modelled on (there the objective itself changed; here
only the algorithm that solves the assignment does).

  v6  device : clic_v6_maskfix_20260827-T101733             job 40393482  device_solver=jv
      host   : clic_v6_maskfix_20260803-T132601             job 38598204  lap1015_late, n_jobs 16
      -- both arms ran the full 200 epochs. They are 24 days apart, so the torch build is not
         identical; that is the one uncontrolled variable and it is noted, not hidden.

  v7  device : clic_v7_cudamatch_b200_b2048_20260827-T130314  job 40405423  device_solver=jv
      host   : clic_v7_maskfix_b200_b2048_20260827-T120002    job 40400036  lap1015_late
      -- the host arm was still training when this was written, so the comparison truncates at
         the last epoch both arms have. Nothing is hardcoded to an epoch count: re-run it when
         that arm finishes and the curves and tables extend themselves.

Both solvers are exact, so the assignment they return is the same up to ties, and the
expectation is that the curves lie on top of each other. Any visible separation is either
tie-breaking compounding through training, or a bug.

⚠️ These metrics use the *generous* type-A matching of §4.1 -- they score the model against the
truth particle the matcher itself chose, which is exactly the quantity a change of matcher could
flatter. They are the cheap check, not the verdict. The jet-level counterparts, where the
physics claim actually has to be made, are `plot_v6_phase4_jet_iqr.py` and
`plot_v7_matcher_ab_jet_iqr.py`.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/cuda_matcher/plot_matcher_ab_curves.py [--pair v6|v7|both]
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CLIC = Path(__file__).resolve().parents[3]
LOGS = CLIC / "logs"
OUT = Path(__file__).resolve().parent

PAIRS = {
    "v6": {
        "device": "clic_v6_maskfix_20260827-T101733",
        "host": "clic_v6_maskfix_20260803-T132601",
        "title": "v6 batch-2048 B200 (10.1 M params)",
    },
    "v7": {
        "device": "clic_v7_cudamatch_b200_b2048_20260827-T130314",
        "host": "clic_v7_maskfix_b200_b2048_20260827-T120002",
        "title": "v7 batch-2048 B200 (702 k params)",
    },
}
C_DEV, C_HOST = "#bc5090", "#003f5c"
SETTLE = 25  # epochs before this are the common startup transient, not a comparison

PANELS = [
    ("val/final_eff", "efficiency", "up"),
    ("val/final_pur", "purity", "up"),
    ("val/final_mask_purity", "mask purity", "up"),
    ("val/final_mask_recall", "mask recall", "up"),
    ("val/final_mask_exact_match", "mask exact match", "up"),
    ("val/final_obj_class_accuracy_macro", "class acc (macro)", "up"),
    ("val/final_regression_e_abs_res", "|E residual|", "down"),
    ("val/final_regression_e_proxy_abs_res", "|E residual| (proxy)", "down"),
    ("val/final_regression_pt_abs_res", "|pT residual|", "down"),
    ("val/final_regression_eta_abs_res", "|eta residual|", "down"),
    ("val/final_incidence_kl_div", "incidence KL", "down"),
    ("val/final_classification_object_ce", "object CE", "down"),
]


def series(df, col):
    """Per-epoch series for `col`; the CSVLogger writes sparse rows, so drop the NaNs."""
    if col not in df.columns:
        return None
    s = df.loc[df[col].notna(), ["epoch", col]].dropna()
    return None if s.empty else s.groupby("epoch", as_index=False)[col].mean()


def run(tag):
    cfg = PAIRS[tag]
    dev = pd.read_csv(LOGS / cfg["device"] / "csv_metrics" / "metrics.csv")
    host = pd.read_csv(LOGS / cfg["host"] / "csv_metrics" / "metrics.csv")
    last = int(min(dev["epoch"].max(), host["epoch"].max()))
    print(f"\n=== {tag}: device {int(dev['epoch'].max())} epochs, host {int(host['epoch'].max())} epochs -> comparing over the {last + 1} they share")

    # ------------------------------------------------------------ figure 1: total loss
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for lbl, df, c in [("device (jv)", dev, C_DEV), ("host (lap1015)", host, C_HOST)]:
        tr, va = series(df, "train/loss"), series(df, "val/loss")
        ax.plot(tr["epoch"], tr["train/loss"], color=c, alpha=0.35, lw=1, label=f"{lbl} train")
        ax.plot(va["epoch"], va["val/loss"], color=c, lw=1.6, label=f"{lbl} val")
    ax.set(xlabel="epoch", ylabel="total loss", yscale="log", title="Total loss — identical objective, so values ARE comparable")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # The difference is the whole point, so it gets its own axis rather than asking the eye to
    # separate two overlapping curves.
    ax = axes[1]
    m = series(dev, "val/loss").merge(series(host, "val/loss"), on="epoch", suffixes=("_dev", "_host"))
    m = m[m["epoch"] <= last]
    d = m["val/loss_dev"] - m["val/loss_host"]
    ax.plot(m["epoch"], d, color="k", lw=1.2)
    ax.axhline(0, ls="--", color="r", alpha=0.6)
    # Band and limits come from the settled region; the startup transient would otherwise set a
    # y-range that hides the part that matters.
    sd = d[m["epoch"] >= SETTLE].std()
    ax.axhspan(-sd, sd, color="0.7", alpha=0.35, label=f"±1 sd of epochs ≥{SETTLE} ({sd:.3f})")
    lim = 1.15 * max(abs(d[m["epoch"] >= SETTLE]).max(), sd)
    ax.set_ylim(-lim, lim)
    n_off = int((d[m["epoch"] < SETTLE].abs() > lim).sum())
    ax.set(
        xlabel="epoch",
        ylabel="val loss: device - host",
        title=f"Paired difference (0 = the two solvers train the same model)\n{n_off} early-transient epochs off-scale",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{cfg['title']} — GPU matcher (jv) vs CPU matcher (lap1015)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"{tag}_matcher_ab_loss.png", dpi=140)
    print("wrote", OUT / f"{tag}_matcher_ab_loss.png")

    # ----------------------------------------------- figure 2: physics-facing metrics
    def zoom(col, pad=0.12):
        """y-limits covering both arms over epochs >= SETTLE, with padding."""
        vals = [s.loc[s["epoch"] >= SETTLE, col] for s in (series(dev, col), series(host, col)) if s is not None]
        if not vals:
            return None, None
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        pad_ = pad * (hi - lo) or 1e-6
        return lo - pad_, hi + pad_

    fig, axes = plt.subplots(3, 4, figsize=(19, 11))
    summary = []
    for ax, (col, title, direction) in zip(axes.ravel(), PANELS, strict=True):
        vals = {}
        for lbl, df, c in [("device (jv)", dev, C_DEV), ("host (lap1015)", host, C_HOST)]:
            s = series(df, col)
            if s is None:
                continue
            ax.plot(s["epoch"], s[col], color=c, lw=1.3, label=lbl)
            # Mean over the last 10 epochs BOTH arms have, so the two numbers are paired.
            vals[lbl] = s.loc[(s["epoch"] >= last - 9) & (s["epoch"] <= last), col].mean()
        ax.set_title(f"{title}  ({'higher' if direction == 'up' else 'lower'} better)", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        # Zoom past the startup transient, else both arms are one flat line at this scale and
        # the panel cannot show whether they separate.
        lo, hi = zoom(col)
        if lo is not None:
            ax.set_ylim(lo, hi)
        if len(vals) == 2:
            delta = vals["device (jv)"] - vals["host (lap1015)"]
            summary.append((title, vals["device (jv)"], vals["host (lap1015)"], delta, 100 * delta / abs(vals["host (lap1015)"])))

    trunc = "" if last == int(dev["epoch"].max()) == int(host["epoch"].max()) else " (host arm still training)"
    fig.suptitle(
        f"{cfg['title']}: GPU matcher (jv) vs CPU matcher (lap1015) — configs identical but for the solver\n"
        f"curves compared over epochs 0-{last}{trunc}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{tag}_matcher_ab_metrics.png", dpi=130)
    print("wrote", OUT / f"{tag}_matcher_ab_metrics.png")

    print(f"\nmean over epochs {last - 9}-{last} (paired):")
    print(f"{'metric':<26} {'device':>10} {'host':>10} {'delta':>10} {'%':>8}")
    print("-" * 68)
    for title, a, b, delta, pct in summary:
        print(f"{title:<26} {a:>10.4f} {b:>10.4f} {delta:>+10.4f} {pct:>+7.2f}%")

    vl = m.loc[m["epoch"] >= last - 9]
    print(f"\nval total loss, epochs {last - 9}-{last}:")
    print(
        f"  device {vl['val/loss_dev'].mean():.5f}   host {vl['val/loss_host'].mean():.5f}   "
        f"delta {vl['val/loss_dev'].mean() - vl['val/loss_host'].mean():+.5f}"
    )
    print(f"  per-epoch difference over all {len(m)} common epochs: mean {d.mean():+.5f}, sd {d.std():.5f}")
    print(f"  ... over the settled epochs (>= {SETTLE}): mean {d[m['epoch'] >= SETTLE].mean():+.5f}, sd {sd:.5f}")


ap = argparse.ArgumentParser()
ap.add_argument("--pair", choices=[*PAIRS, "both"], default="both")
args = ap.parse_args()
for t in PAIRS if args.pair == "both" else [args.pair]:
    run(t)
