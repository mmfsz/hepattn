"""How different are the GPU and CPU matchers? The three shadow layers, drawn.

Reads `matcher_shadow.jsonl` -- one line per shadow step, written by
`hepattn.callbacks.MatcherShadow` during job 40640611 (`clic_v6_cudamatch_shadow`, 200 epochs,
971 shadow steps, 10,240 matching problems solved BOTH ways at each) -- and turns it into the
answer to the question that opened the run: *technical metrics like the matching quality of the
algorithm, or intersection-over-union of GPU matcher vs CPU matcher, so we can start quantifying
how "different" they are.*

Three figures, in the order the argument has to be made. Each answers one question, and none of
them is redundant with the others -- that is the whole reason the callback records three layers
rather than one number.

  fig 1  `shadow_three_layers.png`   Is the GPU assignment optimal? Do the two solvers pick the
                                    same pairing? Does the difference matter physically?
                                    Its three rows are also written separately into
                                    `../slides/figures/`, one per slide of the deck.
  fig 2  `shadow_mechanism.png`      *Why* do they disagree, and what did the substitution buy?
  fig 3  `shadow_pair_delta.png`     What a disagreement costs the individual pair, versus what
                                    it costs the population.

⚠️ The one reading trap, and figure 2a exists to defuse it. The per-pair cost difference between
the two solvers' choices (`tie_gap`) GROWS through training, to a median of ~0.33 -- these are
not near-ties. Yet the TOTAL cost difference per problem stays at fp32 epsilon. Both facts are
true simultaneously because the disagreements are *cycles*: a set of targets swap queries among
themselves, one taking a cheaper entry and another a dearer one, summing to zero. That is the
signature of **multiple global optima**, not of one solver doing worse. The pre-registered
expectation in NOTES.md (2026-08-30) said `tie_gap` would be ~0; it is not, and the honest
version of the claim is stronger, not weaker.

Job 40705257 then dumped the per-pair arrays and identified the structure exactly: every
disagreement is a **transposition of two truth particles that own the same constituents**, so
each query scores identically against either, in cost and in mask IoU alike. See
`analyse_shadow_raw.py`. That is why row 3's two curves coincide bitwise rather than merely
closely.

Do not read row 3 as a formality. Its numbers alone -- distributions identical at every stored
quantile, while 66% of individual swapped pairs move by more than 0.01 -- already force the
conclusion that each query scores identically against both swapped targets; the dumps confirmed
that and supplied the physics, but did not discover it. What row 3 cannot do is *show* the
exactness: two overlapping lines look the same whether they agree to 1e-9 or to 1e-2, and only
the former means anything. Annotate the number, do not demote the panel.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/cuda_matcher/plot_shadow_matcher.py
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

CLIC = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DEFAULT_RUN = CLIC / "logs" / "clic_v6_cudamatch_shadow_20260830-T121009"

# The study's palette, kept identical to plot_v6_phase4_jet_iqr.py so the figures read as a set.
DEV = "#bc5090"  # the GPU (jv) solver
HOST = "#003f5c"  # the CPU (scipy) reference
CTL = "#7f8c8d"  # the agreeing-pair control: context, never a claim
FP32_EPS = np.finfo(np.float32).eps  # 1.19e-07


def load(path: Path) -> dict:
    """Flatten the .jsonl into arrays. Quantile keys become `<field>_<q>`.

    Raises:
        SystemExit: if the file holds no usable shadow records.
    """
    rows = [json.loads(line) for line in path.open() if line.strip()]
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    if bad:
        print(f"skipped {len(bad)} shadow steps that recorded an error")
    if not rows:
        raise SystemExit(f"no usable shadow records in {path}")

    d = {"step": np.array([r["step"] for r in rows], dtype=float)}
    # Fractional epoch: far more intuitive than a step index running to 97,200, and the
    # steps-per-epoch is inferred rather than hardcoded so this survives a different schedule.
    epoch = np.array([r["epoch"] for r in rows], dtype=float)
    spe = (d["step"][-1] - d["step"][0]) / max(epoch[-1] - epoch[0], 1)
    d["epoch"] = d["step"] / spe

    for key in ("problems_identical", "pairs_disagreeing", "problems_worse", "host_solve_s", "device_solve_s", "num_problems"):
        d[key] = np.array([r[key] for r in rows], dtype=float)
    for field in ("cost_gap", "tie_gap", "agreement", "assignment_iou"):
        for q in ("q0.01", "q0.5", "q0.99", "q1", "mean", "n"):
            d[f"{field}_{q}"] = np.array([r[field][q] for r in rows], dtype=float)
    for which in ("device", "host", "control", "delta"):
        for q in ("q0.01", "q0.5", "q0.99", "mean", "n"):
            d[f"mask_{which}_{q}"] = np.array([r["mask_iou"][which][q] for r in rows], dtype=float)
    d["interchangeable"] = np.array([r["mask_iou"].get("interchangeable", np.nan) for r in rows], dtype=float)
    d["rows"] = rows
    return d


def takeaway(ax, text: str, loc: str = "lower right") -> None:
    """One plain-language sentence per panel. The figures are for someone who was not here."""
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.text(
        0.985 if "right" in loc else 0.015,
        0.06,
        text,
        transform=ax.transAxes,
        ha="right" if "right" in loc else "left",
        va="bottom",
        fontsize=8.5,
        style="italic",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fffbe6", "ec": "#d9c97a", "lw": 0.8},
    )


# --------------------------------------------------------------------------- figure 1
def caption(ax, text: str, width: int = 128) -> None:
    """Put the panel's plain-language explanation ABOVE the axes, never over the data.

    These figures are read by people who were not here, so each panel has to say in words what
    it is showing. Inside the axes that box covers the curves at some zoom or other, and which
    curves it covers changes every time the data does -- so it goes in the margin, under the
    title, where it cannot collide with anything. ``width`` is in characters: the box is as wide
    as the axes allow at 8.5 pt, so it stays short.
    """
    wrapped = textwrap.fill(" ".join(text.split()), width=width)
    ax.set_title(ax.get_title(loc="left"), loc="left", fontweight="bold", pad=11 + 12.6 * (wrapped.count("\n") + 1))
    ax.text(
        0.0,
        1.012,
        wrapped,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#fffbe6", "ec": "#d9c97a", "lw": 0.8},
    )


# Each row is drawn by its own function so the three-panel figure and the three standalone
# slide figures cannot drift apart. `standalone` drops the title and the caption box, which the
# slide carries instead -- repeating them under a slide heading is noise.
def panel_optimality(ax, d: dict, standalone: bool = False) -> None:
    """Row 1: is the pairing the GPU picked as cheap as the CPU's?"""
    x = d["epoch"]
    # A line here would draw a forest of spikes to the axis: the gap is EXACTLY zero at most
    # steps, and only the nonzero ones carry information. Points, and the zeros counted in words.
    nonzero = d["cost_gap_q1"] > 0
    mean_nz = d["cost_gap_mean"] > 0
    ax.plot(x[nonzero], d["cost_gap_q1"][nonzero], "o", color=DEV, ms=2.5, alpha=0.55, label="worst of the 10,240 problems")
    ax.plot(x[mean_nz], d["cost_gap_mean"][mean_nz], "o", color=HOST, ms=2.5, alpha=0.55, label="mean over the 10,240 problems")
    ax.axhline(FP32_EPS, color="k", ls=":", lw=1.2)
    ax.text(x[-1], FP32_EPS * 1.35, "fp32 epsilon  ", ha="right", va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-11, 1e-4)
    ax.set_ylabel("cost of GPU pairing $-$ cost of CPU optimum")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if standalone:
        return
    ax.set_title("1.  Is the pairing the GPU picked as cheap as the CPU's?", loc="left", fontweight="bold")
    caption(
        ax,
        "Both solvers get the same cost matrix; this is how much more the GPU's answer costs. Zero means the GPU "
        f"found an optimum too. It is exactly zero at {(~nonzero).sum()} of the {len(x)} sampled steps and never exceeds "
        f"{d['cost_gap_q1'].max():.1e} anywhere — fp32 rounding, not a worse matching.",
    )


def panel_agreement(ax, d: dict, standalone: bool = False) -> None:
    """Row 2: do the two solvers pick the same pairing?"""
    x = d["epoch"]
    # Per-10,000 on the axis itself: "5.9 pairs in 10,000" is a quantity a reader can hold,
    # where 5.9e-04 is not. Both lines are per 10,000 OF THEIR OWN UNIT, said in the legend.
    ax.semilogy(x, d["pairs_disagreeing"] * 1e4, color=HOST, lw=1.2, label="(query, particle) pairs assigned differently — per 10,000 pairs")
    ax.semilogy(
        x, (1.0 - d["problems_identical"]) * 1e4, color=DEV, lw=1.0, label="matching problems with at least one such pair — per 10,000 problems"
    )
    drop = d["pairs_disagreeing"][0] / d["pairs_disagreeing"][-1]
    final_rate = d["pairs_disagreeing"][-1] * 1e4
    ax.set_ylabel("disagreements per 10,000")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if standalone:
        return
    ax.set_title("2.  Do they pick the same pairing?", loc="left", fontweight="bold")
    caption(
        ax,
        "The matcher's job is to decide which predicted object is scored against which true particle. Here: how many "
        f"of those decisions the two solvers make differently. Both curves fall {drop:.0f}x as training sharpens "
        f"the costs; at the end {final_rate:.1f} pairs in 10,000 differ.",
    )


def panel_overlap(ax, d: dict, standalone: bool = False) -> None:
    """Row 3: where they disagree, is one solver's choice worse?"""
    x = d["epoch"]
    ax.plot(x, d["mask_control_mean"], color=CTL, lw=2.0, label="pairs both solvers agree on  (reference)")
    ax.plot(x, d["mask_device_mean"], color=DEV, lw=2.0, label="pairs they disagree on — the GPU's choice")
    ax.plot(x, d["mask_host_mean"], color=HOST, lw=1.1, ls="--", label="pairs they disagree on — the CPU's choice")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("constituent overlap with the true particle\n(shared tracks + clusters / all in either)")
    ax.set_xlabel("epoch")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    if standalone:
        return
    ax.set_title("3.  Where they disagree, is one solver's choice worse?", loc="left", fontweight="bold")
    caption(
        ax,
        "A particle's mask is the set of input constituents it owns — reconstructed tracks and topological calorimeter "
        "clusters (the code calls them 'hits'). This is its overlap with the object the solver picked: 1 = identical, "
        "0 = disjoint. The two curves coincide exactly, and the raw per-pair dump (jobs 40705257 / 40779831, "
        "analyse_shadow_raw.py) says why: every disagreement swaps two truth particles that own the same constituents, "
        "so each query scores identically against either. Grey = the pairs both solvers agree on.",
    )


PANELS = (
    ("panel1_optimality", panel_optimality),
    ("panel2_agreement", panel_agreement),
    ("panel3_overlap", panel_overlap),
)


def fig_three_layers(d: dict, out: Path) -> None:
    """The argument in three rows: optimal, near-identical, and physically interchangeable."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12.2), sharex=True)
    fig.subplots_adjust(hspace=0.30)
    for ax, (_, draw) in zip(axes, PANELS, strict=True):
        draw(ax, d)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, d["epoch"].max())
    fig.suptitle(
        "GPU vs CPU Hungarian matching, measured on identical cost matrices\n"
        f"clic_v6 batch-2048, 200 epochs — {len(d['epoch'])} sampled steps x {int(d['num_problems'][0]):,} problems\n"
        "(2,048 events x 5 deeply-supervised decoder outputs, matched separately)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.0)
    fig.savefig(out, dpi=150)
    print("wrote", out)


def fig_split(d: dict, prefix: str, out_dir: Path) -> None:
    """The same three rows as standalone figures, one per slide.

    Wide and short: a slide gives ~1400 pt of width and only ~550 pt of height once the heading
    and a callout are placed, so the aspect ratio is set by the slide, not by the data. The
    title and caption box are dropped -- under a slide heading, with the slide's own text
    beside them, they are noise.

    These land in the DECK's directory, not this one, because typst sandboxes to the project
    root it is given: an `image("../cuda_matcher/...")` is refused with "would escape the
    project root". The deck already keeps its `logos/` the same way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in PANELS:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        draw(ax, d, standalone=True)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, d["epoch"].max())
        ax.set_xlabel("epoch")
        fig.tight_layout()
        out = out_dir / f"{prefix}_{name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out)


# --------------------------------------------------------------------------- figure 2
def fig_mechanism(d: dict, out: Path) -> None:
    """Why they disagree (equal-cost cycles, not ties), and what the substitution bought."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.8))
    x = d["epoch"]

    # (a) The trap: per-pair cost differences grow, the total stays at fp32 noise.
    a.fill_between(x, d["tie_gap_q0.01"], d["tie_gap_q0.99"], color=DEV, alpha=0.20, lw=0, label="1%-99% of differing pairs")
    a.plot(x, d["tie_gap_q0.5"], color=DEV, lw=1.6, label="median differing pair")
    nonzero = d["cost_gap_q1"] > 0
    a.plot(x[nonzero], d["cost_gap_q1"][nonzero], "o", color=HOST, ms=2.5, alpha=0.55, label="worst total gap, per problem")
    a.set_yscale("log")
    a.set_ylim(1e-9, 3)
    a.set_xlabel("epoch")
    a.set_ylabel("cost difference")
    a.set_title("Not ties — alternative optima", loc="left", fontweight="bold")
    takeaway(
        a,
        "Individual swapped pairs differ by O(0.3) in cost, yet\n"
        "the per-problem total is unchanged. The raw dump shows\n"
        "why: the swaps are transpositions of truth particles\n"
        "with identical constituent masks, so the two costs are\n"
        "the same two numbers exchanged.",
        loc="lower left",
    )

    # (b) The mechanism behind figure 1b: degenerate costs early, decisive costs late.
    s = b.scatter(d["tie_gap_q0.5"], d["pairs_disagreeing"], c=x, cmap="viridis", s=14, lw=0)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("median cost difference of a swapped pair")
    b.set_ylabel("fraction of pairs that disagree")
    b.set_title("Sharper costs, fewer swaps", loc="left", fontweight="bold")
    fig.colorbar(s, ax=b, label="epoch")
    b.grid(alpha=0.25)
    b.text(
        0.03,
        0.05,
        "Early the costs are degenerate, so two exact solvers\n"
        "split ties differently. As the model sharpens the\n"
        "optimum becomes unique and they converge.",
        transform=b.transAxes,
        fontsize=8.5,
        style="italic",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fffbe6", "ec": "#d9c97a", "lw": 0.8},
    )

    # (c) What it bought. Same problems, same step, both solvers timed.
    def _roll(v, w=25):
        pad = np.pad(v, (w // 2, w // 2), mode="edge")
        return np.array([np.median(pad[i : i + w]) for i in range(len(v))])

    c.plot(x, d["host_solve_s"], color=HOST, lw=0.5, alpha=0.3)
    c.plot(x, d["device_solve_s"], color=DEV, lw=0.5, alpha=0.3)
    c.plot(x, _roll(d["host_solve_s"]), color=HOST, lw=2.0, label="CPU: scipy x 16 threads")
    c.plot(x, _roll(d["device_solve_s"]), color=DEV, lw=2.0, label="GPU: batched jv")
    c.set_yscale("log")
    c.set_xlabel("epoch")
    c.set_ylabel("seconds to solve one step's 10,240 problems")
    c.set_title("What the substitution bought", loc="left", fontweight="bold")
    c.grid(alpha=0.25)
    ratio = np.median(d["host_solve_s"] / d["device_solve_s"])
    takeaway(c, f"Median {ratio:.1f}x faster on the identical problems.\n(Solve time only — see Phase 3 for end-to-end.)", loc="lower left")

    a.grid(alpha=0.25)
    fig.suptitle("Why the two matchers disagree, and what replacing one with the other bought", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print("wrote", out)


# --------------------------------------------------------------------------- figure 3
# --------------------------------------------------------------------------- figure 3
# Same idiom as PANELS above: one function per panel so the combined figure and the three
# standalone slide figures cannot drift apart. `standalone` drops the title and the yellow
# annotation box, which the slide's own text carries instead.
def panel_delta_spread(ax, d: dict, standalone: bool = False) -> None:
    """Delta 1: what a swap does to ONE pair."""
    x = d["epoch"]
    ax.plot(x, d["mask_delta_q0.99"], color=DEV, lw=1.2, alpha=0.85, label="99th percentile disagreeing pair")
    ax.plot(x, d["mask_delta_q0.5"], color="k", lw=1.6, label="median disagreeing pair")
    ax.plot(x, d["mask_delta_q0.01"], color=HOST, lw=1.2, alpha=0.85, label="1st percentile disagreeing pair")
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlim(0, x.max())
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\Delta$ mask IoU   (GPU's choice $-$ CPU's choice)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if standalone:
        return
    ax.set_title("Per pair: a real difference...", loc="left", fontweight="bold")
    takeaway(
        ax,
        "More than 1% of disagreeing pairs flip outright — IoU 1 under one\n"
        "solver, 0 under the other — and equally often in each direction.\n"
        "The median changes nothing. This is a real per-pair difference.",
        loc="lower left",
    )


def panel_delta_unchanged(ax, d: dict, standalone: bool = False) -> None:
    """Delta 2: how many swaps change nothing at all."""
    x = d["epoch"]
    ax.plot(x, d["interchangeable"], color=DEV, lw=1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, x.max())
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"fraction of disagreeing pairs with $|\Delta$ IoU$| < 0.01$")
    if standalone:
        return
    # NB the fraction plotted is the UNCHANGED one, so the reading is "one third of swaps leave
    # the pair alone", i.e. two thirds of them do move it.
    ax.set_title("...that two thirds of them do have", loc="left", fontweight="bold")
    late = d["epoch"] >= 150
    ax.text(
        0.03,
        0.06,
        f"At convergence {d['interchangeable'][late].mean():.0%} of disagreeing pairs leave the\nmask"
        " IoU untouched. The early value near 1 is not agreement --\nthe masks were empty, so every IoU was 0.",
        transform=ax.transAxes,
        fontsize=8.5,
        style="italic",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fffbe6", "ec": "#d9c97a", "lw": 0.8},
    )


def panel_delta_population(ax, d: dict, standalone: bool = False) -> None:
    """Delta 3: the population-level bias, over the converged half of training."""
    late = d["epoch"] >= 150
    md = d["mask_delta_mean"][late]
    lim = 6.0  # nano-IoU; set from the 1-99 percentile of md, which is +-4e-9
    inside = np.abs(md) <= lim * 1e-9
    ax.hist(md[inside] * 1e9, bins=np.linspace(-lim, lim, 41), color=DEV, alpha=0.75)
    ax.axvline(0, color="k", lw=1.4, ls="--")
    ax.set_xlabel(r"per-step mean $\Delta$ mask IoU   ($\times 10^{-9}$)")
    ax.set_ylabel(f"shadow steps (epochs 150-{d['epoch'].max():.0f})")
    if standalone:
        return
    ax.set_title("Population: no bias either way", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.97,
        f"Over {late.sum()} converged shadow steps the mean agrees to ~1e-9\n"
        f"({inside.sum()} of {late.sum()} shown; the {(~inside).sum()} outside reach {np.abs(md).max():.0e}). Verified\n"
        "mechanism: a disagreement moves values between\n"
        "particles that own the same constituents, so the multiset — and\nevery quantile of it — is unchanged. "
        "See analyse_shadow_raw.py.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        style="italic",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fffbe6", "ec": "#d9c97a", "lw": 0.8},
    )


DELTA_PANELS = (
    ("delta1_perpair", panel_delta_spread),
    ("delta2_unchanged", panel_delta_unchanged),
    ("delta3_population", panel_delta_population),
)


def fig_pair_delta(d: dict, out: Path) -> None:
    """The paired view: what a swap does to ONE pair, versus what it does to the population.

    The first draft of this figure drew the device / host / control mask-IoU distributions as
    1%-99% boxes. They came out full-width and said nothing: mask IoU is bounded on [0, 1] and
    the distribution is broad, so its 1st and 99th percentiles are 0 and 1 for every arm at
    every epoch. The informative quantity is the *paired* difference, which the callback stores
    directly -- the same pair, scored under each solver's choice.

    It carries the one result that is genuinely surprising, and the figure exists to say it
    plainly rather than bury it: **individual swapped pairs move a lot** -- one solver's choice
    can have IoU 1 where the other's has 0 -- **but the movement is symmetric, so the population
    is untouched.** That is a sharper statement than "the means agree", and it is the honest
    reason the two trainings end up equivalent.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (_, draw) in zip(axes, DELTA_PANELS, strict=True):
        draw(ax, d)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "What a disagreement actually costs: large for one pair, exactly nothing for the population",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print("wrote", out)


def fig_pair_delta_split(d: dict, prefix: str, out_dir: Path) -> None:
    """The same three panels standalone, one per slide, with the text beside them on the slide.

    Portrait-ish rather than wide: these slides put the explanation in a column to the LEFT of
    the plot, so the plot gets about 1020 x 765 pt. Title and annotation box are dropped, the
    slide says all of that in its own words.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in DELTA_PANELS:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        draw(ax, d, standalone=True)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = out_dir / f"{prefix}_{name}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out-prefix", default="shadow")
    p.add_argument("--slides-dir", type=Path, default=OUT.parent / "slides" / "figures")
    args = p.parse_args()

    d = load(args.run_dir / "matcher_shadow.jsonl")
    print(f"{len(d['epoch'])} shadow steps, epochs {d['epoch'][0]:.1f}-{d['epoch'][-1]:.1f}")

    fig_three_layers(d, OUT / f"{args.out_prefix}_three_layers.png")
    fig_split(d, args.out_prefix, args.slides_dir)
    fig_mechanism(d, OUT / f"{args.out_prefix}_mechanism.png")
    fig_pair_delta(d, OUT / f"{args.out_prefix}_pair_delta.png")
    fig_pair_delta_split(d, args.out_prefix, args.slides_dir)

    late = d["epoch"] >= 150
    print("\nheadline numbers, averaged over epochs 150-199:")
    print(f"  worst optimality gap        {d['cost_gap_q1'][late].max():.2e}   (fp32 eps = {FP32_EPS:.2e})")
    print(f"  pairs disagreeing           {d['pairs_disagreeing'][late].mean():.4%}")
    print(f"  problems matched identically {d['problems_identical'][late].mean():.3%}")
    print(f"  mean assignment IoU         {d['assignment_iou_mean'][late].mean():.5f}")
    dev, host, ctl = (d[f"mask_{k}_mean"][late].mean() for k in ("device", "host", "control"))
    print(f"  mask IoU  GPU / CPU / ctl   {dev:.4f} / {host:.4f} / {ctl:.4f}")
    print(f"  median solve speedup        {np.median(d['host_solve_s'] / d['device_solve_s']):.1f}x")


if __name__ == "__main__":
    main()
