"""Where each arm's loss lives: the per-layer profile across the shared decoder heads.

THE OBSERVATION THIS DRAWS. The four task heads are SHARED nn.Modules run once per decoder layer
(deep supervision), so every layer can be scored with the same yardstick, and the CSVLogger has
been writing all of them every epoch since round 1. Read that way, an arm's damage can be located:
a model that is uniformly a little worse degrades at every layer, while a model with one bad block
tracks the reference through the intermediate layers and separates only at `final`.

THE QUESTION THIS DIRECTORY OPENED WITH. On head, C1 (A2+A3+A4) failed to train twice over, and
this figure is what localised the failure: at every one of the four intermediate layers C1 was
indistinguishable from the reference and from C2, and the entire failure was confined to the final
decoder layer, which took a healthy query set and turned it into an unusable one. That reading is
what the loop-starvation hypothesis was built on, and the round-3 factorial falsified the
hypothesis without touching the observation.

On the paper tag C1 trained: 200 epochs, val_loss 4.96997, the bottom of an ordering that is
monotone in parameter count with no outlier. So this script is not here to confirm a known failure
-- it is here to ask whether the last-layer degradation head saw is present at all on this code,
and if so whether it scales with the arms. Read the two right-hand panels first: if every arm's
curve is flat across L0..L3 and fans out only at `final`, the effect is the same one head saw; if
the arms are already separated at L1, this code degrades differently and head's diagnosis does not
carry over.

THE METRIC NAME DIFFERS FROM HEAD'S. Head logs `mask_mask_dice_v2`, this code logs
`mask_mask_dice` -- the `_v2` suffix belongs to the post-paper mask-loss normalisation. They are
not the same quantity and their values must not be compared across branches; the shape across
layers can be.

WHY THE PER-LAYER COMPARISON IS FAIR. Matching is done per layer, not once on the final layer and
reused (`MaskFormer._match_and_permute_outputs` stacks the layers as extra batch entries and
solves each), so every layer is scored under its OWN optimal assignment. A broken final layer
therefore cannot be handing the intermediate layers a bad permutation, and an intermediate layer
is not being flattered by one chosen elsewhere.

WHAT `layer_i` MEANS. In `MaskFormerDecoder.forward` the tasks for layer i run BEFORE layer i does
-- their mask output is thresholded into the `attn_mask` that gates that same layer's
cross-attention. So `layer_3` is the prediction made from queries that have been through 3 layers,
and `final` is the post-loop pass, the only prediction that has been through all 4.

Needs no evaluation and no jet clustering -- reads `csv_metrics/metrics.csv` only, so it runs in
seconds on a login node:

    pixi run -e clic python studies/model_size/plot_decoder_layers.py
"""

import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_sets import select

FIG = Path(__file__).resolve().parent / "figures"

# The five places the shared heads are evaluated, in the order the forward pass runs them.
LAYERS = ["layer_0", "layer_1", "layer_2", "layer_3", "final"]
# Paper-tag name: head's is `mask_mask_dice_v2`. See the module docstring.
CLIFF = "val/final_mask_mask_dice"

ARM_SET, SET = select()

FRAMES = {}
for arm in SET.arms:
    if not arm.metrics.exists():
        raise SystemExit(f"MISSING {arm.label}: {arm.metrics}")
    FRAMES[arm.key] = pd.read_csv(arm.metrics)


def epoch_of(arm):
    """The epoch of the checkpoint this arm was EVALUATED at, parsed from its stem.

    Reading the profile at the evaluated epoch rather than the last one keeps this figure and the
    physics figures describing the same weights -- C4's best checkpoint is epoch 189, not 199.

    Raises:
        SystemExit: if the arm's checkpoint stem carries no `epoch=` field to read.
    """
    m = re.search(r"epoch=(\d+)", arm.stem)
    if not m:
        raise SystemExit(f"cannot read an epoch out of {arm.stem!r}")
    return int(m.group(1))


def at_epoch(key, col, epoch):
    df = FRAMES[key]
    if col not in df.columns:
        return None
    s = df.loc[(df["epoch"] == epoch) & df[col].notna(), col]
    return float(s.mean()) if len(s) else None


def series(key, col):
    df = FRAMES[key]
    if col not in df.columns:
        return None
    s = df.loc[df[col].notna(), ["epoch", col]]
    return None if s.empty else s.groupby("epoch", as_index=False)[col].mean()


FIG.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# ------------------------------------------------------------------ 1. the final head, per epoch
# Head's version of this panel carried a shaded "initial basin" band at dice 0.50-0.66, the range
# its reference fell through at epoch ~11 and its C1 never left. No band is drawn here: that range
# is a property of head's `_v2` normalisation and drawing it on this metric would assert a
# threshold nobody has measured on this code. The panel shows the curves and lets them speak.
ax = axes[0]
for arm in SET.arms:
    s = series(arm.key, CLIFF)
    if s is None:
        continue
    ax.plot(s["epoch"], s[CLIFF], color=arm.color, ls=arm.dash, lw=2.2 if arm.heavy else 1.4, label=arm.label)
ax.set_xlabel("epoch")
ax.set_ylabel("val final mask dice loss  (lower is better)")
ax.set_title("Final-layer mask dice, per epoch", fontsize=11)
ax.set_xlim(0, 200)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="center right")

# ------------------------------------------------- 2. and 3. the profile across the five heads
for ax, col, name in (
    (axes[1], "mask_mask_dice", "mask dice loss"),
    (axes[2], "classification_object_ce", "object CE"),
):
    for arm in SET.arms:
        ep = epoch_of(arm)
        ys = [at_epoch(arm.key, f"val/{lay}_{col}", ep) for lay in LAYERS]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if not ys:
            continue
        ax.plot(xs, ys, color=arm.color, ls=arm.dash, lw=2.2 if arm.heavy else 1.4, marker=arm.marker, ms=7, label=f"{arm.label}  (ep {ep})")
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(["L0", "L1", "L2", "L3", "final"])
    ax.set_xlabel("where the shared head was evaluated")
    ax.set_ylabel(f"val {name}  (lower is better)")
    ax.set_title(f"{name} at each of the five evaluations", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    # The last decoder layer is where head's arms separated; highlighted so this canvas answers
    # the same question even when the answer is "they do not separate there".
    ax.axvspan(3.5, 4.35, color="#c00000", alpha=0.06, lw=0)
    ax.set_xlim(-0.35, 4.35)

fig.suptitle(SET.title, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = FIG / f"{SET.prefix}_decoder_layers.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")

# The numbers, printed so they are never transcribed by eye from the figure.
print("\nper-layer profile at each run's evaluated checkpoint")
for col in ("mask_mask_dice", "classification_object_ce"):
    print(f"\n  val {col}")
    print("    " + f"{'run':<26}" + "".join(f"{lay:>10}" for lay in LAYERS))
    for arm in SET.arms:
        ep = epoch_of(arm)
        row = "".join(f"{v:>10.4f}" if (v := at_epoch(arm.key, f"val/{lay}_{col}", ep)) is not None else f"{'-':>10}" for lay in LAYERS)
        print(f"    {arm.label + f' (ep {ep})':<26}{row}")
