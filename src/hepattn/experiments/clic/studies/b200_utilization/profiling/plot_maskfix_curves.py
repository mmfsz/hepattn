"""Loss / metric curves for the mask-loss-fix A/B (job 38469247) vs the v6 baselines.

The corrected losses (`mask_bce_v2` / `mask_dice_v2`) change the objective by construction,
so **total loss values are not comparable across the fix** -- only curve *shape* is. What
*is* directly comparable is every metric that does not depend on the loss definition:
efficiency, purity, mask purity/recall, classification accuracy and the regression
residuals. Those are the panels that actually answer "is the fixed model better?".

Runs:
  - maskfix   : clic_v6_maskfix_20260731-T152547  (3xL4, bs 256/GPU, mask_*_v2)   [full CSV]
  - baseline  : clic_v6_20260605-T113014          (3xL4, bs 256/GPU, buggy loss)  [ckpt names only,
                predates the CSVLogger added 2026-07-07 -- val/loss total only]
  - v6 fp32   : clic_v6_fp32_20260707-T150453     (6xL4, fp32, buggy loss)        [full CSV]
                a *secondary* reference: same buggy loss with full per-metric history,
                but different precision and global batch, so treat gaps as indicative.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/profiling/plot_maskfix_curves.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CLIC = Path(__file__).resolve().parents[3]
LOGS = CLIC / "logs"
OUT = Path(__file__).resolve().parent

MASKFIX = "clic_v6_maskfix_20260731-T152547"
BASELINE = "clic_v6_20260605-T113014"
FP32 = "clic_v6_fp32_20260707-T150453"

C_FIX, C_BASE, C_FP32 = "#003f5c", "#ffa600", "#bc5090"
CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=([0-9.]+)\.ckpt$")


def from_csv(folder):
    return pd.read_csv(LOGS / folder / "csv_metrics" / "metrics.csv")


def val_from_ckpts(folder):
    """Fallback history for runs with no CSVLogger: parse the checkpoint filenames."""
    rows = [(int(m.group(1)), float(m.group(2).rstrip(".")))
            for p in (LOGS / folder / "ckpts").glob("*.ckpt")
            if (m := CKPT_RE.search(p.name))]
    return pd.DataFrame(sorted(rows), columns=["epoch", "loss"])


def series(df, col):
    """Per-epoch series for `col`, NaN rows dropped (CSVLogger writes sparse rows)."""
    if col not in df.columns:
        return None
    s = df.loc[df[col].notna(), ["epoch", col]].dropna()
    return None if s.empty else s.groupby("epoch", as_index=False)[col].mean()


fix = from_csv(MASKFIX)
fp32 = from_csv(FP32)
base_val = val_from_ckpts(BASELINE)

# ---------------------------------------------------------------- figure 1: total loss
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
tr = series(fix, "train/loss")
va = series(fix, "val/loss")
ax.plot(tr["epoch"], tr["train/loss"], color=C_FIX, alpha=0.45, lw=1, label="maskfix train")
ax.plot(va["epoch"], va["val/loss"], color=C_FIX, lw=1.6, label="maskfix val")
ax.plot(base_val["epoch"], base_val["loss"], color=C_BASE, lw=1.6, label="v6 baseline val (buggy loss)")
ax.set(xlabel="epoch", ylabel="total loss", title="Total loss — values NOT comparable\n(different objective), compare shape only")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# normalised to each run's own epoch-10 value: pure convergence-shape comparison
ax = axes[1]
for lbl, s, col, c in [
    ("maskfix", va, "val/loss", C_FIX),
    ("v6 baseline", base_val, "loss", C_BASE),
    ("v6 fp32", series(fp32, "val/loss"), "val/loss", C_FP32),
]:
    if s is None:
        continue
    ref = s.loc[s["epoch"] >= 10, col].iloc[0]
    ax.plot(s["epoch"], s[col] / ref, color=c, lw=1.6, label=lbl)
ax.set(xlabel="epoch", ylabel="val loss / own value at epoch 10",
       title="Convergence shape, self-normalised")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "maskfix_loss_curves.png", dpi=140)
print("wrote", OUT / "maskfix_loss_curves.png")

# ------------------------------------------- figure 2: loss-independent quality metrics
PANELS = [
    ("val/final_eff", "efficiency", "up"),
    ("val/final_pur", "purity", "up"),
    ("val/final_mask_purity", "mask purity", "up"),
    ("val/final_mask_recall", "mask recall", "up"),
    ("val/final_mask_exact_match", "mask exact match", "up"),
    ("val/final_obj_class_accuracy_macro", "class acc (macro)", "up"),
    ("val/final_regression_e_abs_res", "|E residual|", "down"),
    ("val/final_regression_pt_abs_res", "|pT residual|", "down"),
    ("val/final_regression_eta_abs_res", "|eta residual|", "down"),
    ("val/final_regression_e_proxy_abs_res", "|E residual| (proxy)", "down"),
    ("val/final_incidence_kl_div", "incidence KL", "down"),
    ("val/final_classification_object_ce", "object CE", "down"),
]

fig, axes = plt.subplots(3, 4, figsize=(19, 11))
summary = []
for ax, (col, title, direction) in zip(axes.ravel(), PANELS, strict=True):
    for lbl, df, c in [("maskfix (v2 loss)", fix, C_FIX), ("v6 fp32 (buggy loss)", fp32, C_FP32)]:
        s = series(df, col)
        if s is None:
            continue
        ax.plot(s["epoch"], s[col], color=c, lw=1.4, label=lbl)
    ax.set_title(f"{title}  ({'higher' if direction == 'up' else 'lower'} better)", fontsize=10)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # final-10-epoch mean for the table
    vals = {}
    for lbl, df in [("maskfix", fix), ("fp32", fp32)]:
        s = series(df, col)
        if s is not None:
            vals[lbl] = s.loc[s["epoch"] >= s["epoch"].max() - 9, col].mean()
    if len(vals) == 2:
        d = vals["maskfix"] - vals["fp32"]
        better = (d > 0) if direction == "up" else (d < 0)
        summary.append((title, vals["maskfix"], vals["fp32"], d, 100 * d / abs(vals["fp32"]), better))

fig.tight_layout()
fig.savefig(OUT / "maskfix_quality_metrics.png", dpi=130)
print("wrote", OUT / "maskfix_quality_metrics.png")

print(f"\n{'metric':<26} {'maskfix':>10} {'v6 fp32':>10} {'delta':>10} {'%':>8}  better?")
print("-" * 76)
for title, a, b, d, pct, better in summary:
    print(f"{title:<26} {a:>10.4f} {b:>10.4f} {d:>+10.4f} {pct:>+7.2f}%  {'YES' if better else 'no'}")

print("\nfinal-10-epoch mean of val total loss:")
print(f"  maskfix     {va.loc[va['epoch'] >= 190, 'val/loss'].mean():.5f}   (v2 objective)")
print(f"  v6 baseline {base_val.loc[base_val['epoch'] >= 190, 'loss'].mean():.5f}   (buggy objective — NOT comparable)")
