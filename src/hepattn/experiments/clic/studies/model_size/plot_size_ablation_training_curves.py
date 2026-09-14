"""Tier 0 for the size ablation: the per-epoch training curves, every arm on one canvas.

Ported from the head-based study on `main`. Two figures come out:

    <prefix>_training_loss.png     total loss, and each arm's paired difference from the baseline
    <prefix>_training_metrics.png  twelve validation metrics, one panel each

WHY THE LOSS VALUES ARE COMPARABLE HERE. Comparing raw loss between two runs is usually
meaningless, because the number depends on the objective. In this study the objective is IDENTICAL
across arms -- same task heads, same weights, same schedule, same data, same batch -- and only the
ARCHITECTURE the loss is computed through changes. That is what lets the right-hand panel plot a
difference at all. It holds within this directory only: a val_loss here is not comparable to one
from head's study, whose loss is computed through different feed-forwards and a different
incidence head.

⚠️ AND YET THE LOSS IS STILL NOT THE VERDICT, for a reason specific to this model and the most
transferable thing head's round 1 found. These are type-A (loss) numbers: the matcher picks which
truth particle each query is scored against, so a model that has learned to choose DIFFERENTLY can
look unchanged in val_loss and still be worse -- or look far worse and be fine. On head, A3 was
the worked example: +0.069 in val_loss, by a wide margin the worst arm, and +0.0006 in global
jet-E IQR, i.e. nothing at all. Rank arms with `plot_size_ablation_jet_iqr.py`. These curves answer
a different and narrower question: DID EACH ARM TRAIN?

THAT NARROW QUESTION IS WHY THIS SCRIPT MATTERS HERE. On head, C1 (A2+A3+A4) failed to train twice
over, and it was found in exactly these panels -- its `object CE` panel sat at 2.0 while every
other arm converged to 0.67 -- long before anyone clustered a jet. On the paper tag C1 completed
200 epochs at val_loss 4.96997, the bottom of a monotone ordering with no outlier. Whether the
per-head panels agree that it trained healthily is the first thing to read off this figure;
`plot_decoder_layers.py` then asks the same question layer by layer.

The twelve panels are the twelve head used, kept deliberately: they are the metrics the CSVLogger
writes every epoch and they cover all four task heads.

Arms and colours come from `arm_sets.py`; ARM_SET picks the set, exactly as for the other scripts.
Unlike them this script reads only `csv_metrics/metrics.csv`, so it needs no evaluation and no jet
clustering -- it runs in seconds on a login node:

    pixi run -e clic python studies/model_size/plot_size_ablation_training_curves.py
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_sets import select

FIG = Path(__file__).resolve().parent / "figures"

# Epochs before this are the common startup transient, shared by every arm and identical in shape.
# They are drawn but excluded from every y-range and every summary number: at full scale they set
# a range in which all the arms are one flat line, which is the failure mode this constant exists
# to avoid.
SETTLE = 25

# All four task heads, and both the "did it find the object" and "did it get the kinematics right"
# halves of each. Same twelve as head's study, so curves from the two can be laid side by side --
# as shapes, never as numbers.
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

ARM_SET, SET = select()
FRAMES = {}
for arm in SET.arms:
    if not arm.metrics.exists():
        print(f"MISSING {arm.label}: {arm.metrics}", flush=True)
        continue
    FRAMES[arm.key] = pd.read_csv(arm.metrics)
ARMS = [a for a in SET.arms if a.key in FRAMES]
if len(ARMS) < 2:
    raise SystemExit("need at least the baseline and one arm")
BASE = ARMS[0]
LAST = int(min(FRAMES[a.key]["epoch"].max() for a in ARMS))
print(
    f"comparing over the {LAST + 1} epochs every arm has (per-arm max: "
    + ", ".join(f"{a.key}:{int(FRAMES[a.key]['epoch'].max())}" for a in ARMS)
    + ")"
)


def series(key, col):
    """Per-epoch series for `col`; the CSVLogger writes sparse rows, so drop the NaNs.

    Values are averaged within an epoch rather than taking the last: validation logs one row per
    epoch here, but averaging is correct either way and does not assume that.
    """
    df = FRAMES[key]
    if col not in df.columns:
        return None
    s = df.loc[df[col].notna(), ["epoch", col]].dropna()
    return None if s.empty else s.groupby("epoch", as_index=False)[col].mean()


def lw(arm, bump=0.0):
    return (2.0 if arm.heavy else 1.2) + bump


FIG.mkdir(exist_ok=True)

# =============================================================================== figure 1: loss
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for arm in ARMS:
    tr, va = series(arm.key, "train/loss"), series(arm.key, "val/loss")
    if tr is not None:
        ax.plot(tr["epoch"], tr["train/loss"], color=arm.color, alpha=0.3, lw=0.9, ls=arm.dash)
    ax.plot(va["epoch"], va["val/loss"], color=arm.color, lw=lw(arm), ls=arm.dash, label=arm.label)
ax.set(xlabel="epoch", ylabel="total loss", yscale="log", title="Total loss — identical objective, so values ARE comparable")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# The difference is the whole point, so it gets its own axis rather than asking the eye to
# separate a bundle of overlapping curves on a log scale where they differ by ~1%.
ax = axes[1]
base_val = series(BASE.key, "val/loss")
diffs, tails = {}, {}
for arm in ARMS[1:]:
    m = series(arm.key, "val/loss").merge(base_val, on="epoch", suffixes=("_arm", "_base"))
    m = m[m["epoch"] <= LAST]
    d = m["val/loss_arm"] - m["val/loss_base"]
    ax.plot(m["epoch"], d, color=arm.color, lw=lw(arm), ls=arm.dash, label=arm.label)
    diffs[arm.key] = (m["epoch"], d)
    settled = d[m["epoch"] >= SETTLE]
    tails[arm.key] = (settled, d[m["epoch"] >= LAST - 9].mean())
ax.axhline(0, ls="--", color="r", alpha=0.6)
# Range from the CONVERGED HALF, not from `SETTLE`. The matcher A/B could use epoch 25 because
# both its arms had the same architecture and settled together; here they do not. A4 sits on a
# plateau until epoch ~53 and then drops two units in one epoch, and letting that set the range
# compresses every real difference -- which are 0.01 to 0.4 -- into the middle two percent of the
# axis. Epochs outside the frame are counted in the title rather than silently clipped.
CONVERGED = LAST // 2
lim = 1.15 * max(d[e >= CONVERGED].abs().max() for e, d in diffs.values())
ax.set_ylim(-lim, lim)
off = sum(int((d.abs() > lim).sum()) for _e, d in diffs.values())
ax.set(
    xlabel="epoch",
    ylabel=f"val loss: arm − {BASE.label}",
    title=f"Paired difference (0 = trains like the baseline)\ny-range from epochs ≥{CONVERGED}; {off} arm-epochs off-scale",
)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.suptitle(SET.title.split("\n")[0], fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = FIG / f"{SET.prefix}_training_loss.png"
fig.savefig(out, dpi=140)
print("wrote", out)

# ========================================================================= figure 2: metrics grid
fig, axes = plt.subplots(3, 4, figsize=(19, 11))
summary = {}
for ax, (col, title, direction) in zip(axes.ravel(), PANELS, strict=True):
    lo, hi = None, None
    for arm in ARMS:
        s = series(arm.key, col)
        if s is None:
            continue
        ax.plot(s["epoch"], s[col], color=arm.color, lw=lw(arm, 0.1), ls=arm.dash, label=arm.label)
        v = s.loc[s["epoch"] >= SETTLE, col]
        lo = v.min() if lo is None else min(lo, v.min())
        hi = v.max() if hi is None else max(hi, v.max())
        # Mean over the last 10 epochs EVERY arm has, so the numbers across arms are paired.
        summary.setdefault(title, {})[arm.key] = s.loc[(s["epoch"] >= LAST - 9) & (s["epoch"] <= LAST), col].mean()
    ax.set_title(f"{title}  ({'higher' if direction == 'up' else 'lower'} better)", fontsize=10)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    # Zoom past the startup transient, else every arm is one flat line at this scale and the panel
    # cannot show whether they separate at all.
    if lo is not None and hi > lo:
        pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

fig.suptitle(
    f"{SET.title.split(chr(10))[0]}\nvalidation metrics per epoch, compared over epochs 0-{LAST} — type-A (loss) matching, so these RANK arms wrongly; they answer 'did it train?'",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = FIG / f"{SET.prefix}_training_metrics.png"
fig.savefig(out, dpi=130)
print("wrote", out)

# ----------------------------------------------------------------------------------- the tables
w = max(len(a.label) for a in ARMS) + 2
print(f"\nmean over epochs {LAST - 9}-{LAST}, every arm minus the baseline ({BASE.label}):")
print(f"{'metric':<26} " + " ".join(f"{a.label[:14]:>15}" for a in ARMS))
print("-" * (26 + 16 * len(ARMS)))
for title, vals in summary.items():
    base = vals.get(BASE.key)
    row = [f"{vals[BASE.key]:>15.4f}"] + [f"{vals[a.key] - base:>+15.4f}" if a.key in vals else f"{'—':>15}" for a in ARMS[1:]]
    print(f"{title:<26} " + " ".join(row))
print(f"{'':<26} {'(absolute)':>15} " + " ".join(f"{'(Δ vs base)':>15}" for _ in ARMS[1:]))

print(f"\nval total loss, epochs {LAST - 9}-{LAST}:")
print(f"  {BASE.label:<{w}} {series(BASE.key, 'val/loss').pipe(lambda s: s.loc[s['epoch'] >= LAST - 9, 'val/loss'].mean()):.5f}")
for arm in ARMS[1:]:
    settled, tail = tails[arm.key]
    print(f"  {arm.label:<{w}} Δ {tail:+.5f}   over settled epochs (≥{SETTLE}): mean {settled.mean():+.5f}, sd {settled.std():.5f}")
