"""The full `performance.ipynb` plot set for the paper-tag size ablation, all arms overlaid.

`plot_size_ablation_jet_iqr.py` answers one question -- jet-E response median and IQR -- and is
the Tier-1 reading. This script is the *breadth* check that sits behind it: the same five trained
models put through every other plot the performance notebook draws, because shrinking a model
could in principle leave the jet-energy resolution alone and still damage something else.

Specifically, the jet-E IQR integrates over particle classes. Photon and neutral-hadron efficiency
and fake rate do not: these are the classes inferred from calorimetry alone, with no track to
anchor them, and they are the most physically interpretable signal the model produces. A shrink
that hurt only photons would be invisible in the IQR and obvious in `plot_eff_fr_purity`. The pass
criterion is qualitative and comparative -- curves must not develop a new turn-over, and per-class
splits must degrade proportionally rather than one class collapsing -- which is why every arm is
drawn on one canvas rather than in per-arm figures: a collapse is recognised by its shape against
its neighbours, not in isolation.

Ported from the head-based study on `main`, paths and arm table repointed at this branch. The
helper-restyling below is that script's and is unchanged; it exists because the plot helpers are
shared with `performance.ipynb` and must not be restyled for this study's sake.

ARMS live in `arm_sets.py`, shared with the other plot scripts, each at its own lowest-val_loss
checkpoint -- which for C4 and C3 is not the last epoch. In the default `all` set the arms are the
three pairs C5/C4/C3 and the triple C1 of the 2^3 factorial over {A2, A3, A4}. The breadth check
matters more for C1 than for any pair: three shrinks that each left the photon and neutral-hadron
curves alone need not do so together, and on head C1 was the arm that broke.

Every arm trained with the GPU matcher and every one evaluated with the host solver, so the eval
path contributes nothing to any difference seen here.

ONE FIGURE SET PER CONVENTION. `mpflow` uses the regression head's kinematics; `mpflow_proxy`
bypasses it and rebuilds them from the incidence head. They differ by more than most effects being
looked for, so a comparison is only meaningful *within* a branch. Mixing them on one axis would
invite exactly the cross-convention reading that rule forbids. Pandora is carried into both as the
classical reference the notebook always plots.

These figures carry no significance test at all -- unlike the jet-IQR script they never did, on
head either. They are read as shapes. That suits this branch, where sigma_repro is unmeasured and
even the IQR script's verdicts have been withheld.

Run from the clic experiment dir (via submit_size_plots.sh -- jet clustering and particle matching
both want cores):
    pixi run -e clic python studies/model_size/plot_size_ablation_performance.py
"""

import hashlib
import os
import pickle  # noqa: S403 - reads only this script's own cache, never external input
import sys
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn-paper/src")
from hepattn.experiments.clic.performance.performance import NetworkType, Performance, PerformanceConfig
from hepattn.experiments.clic.performance.plot_helper import PlotHelper

# The pipeline injects Pandora under this name. It is `pandora` on this branch and `Pandora`
# on head, so the string is taken from the enum rather than written out: hardcoding it is what
# made every figure in this set fail with KeyError('pandora') on the first run (job 42119839).
PANDORA = NetworkType.PANDORA.value

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CACHE = OUT / "cache"
TRUTH = "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root"

# The arm table -- which runs, which checkpoints, which colours -- lives in `arm_sets.py`, shared
# with `plot_size_ablation_jet_iqr.py` and `plot_size_ablation_training_curves.py`. ARM_SET picks
# the set; see that module for what each one asks.
#
# The figure this study leans on hardest is `eff_fr_purity`, the one that splits by particle
# class. On head, C1's failure was in the final classification head specifically, and a
# classification failure is precisely what the jet-E IQR is least able to see and what a per-class
# efficiency curve shows immediately. C1 trained on this code, so that figure is where to look
# first for a failure that the loss curves and the jet energy both miss.
#
# Line weights are the helpers' own, deliberately. Heavier strokes were tried across every figure
# and read as a resolution change -- thick lines against unchanged tick and legend text. Only
# `plot_eff_fr_purity` needs thickening (it hardcodes linewidth=1 and its curves sit on top of one
# another), and that is done there alone, via `_restyle(bump_lw=...)`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_sets import PANDORA_COLORS, select  # noqa: E402

ARM_SET, SET = select()
PREFIX = SET.prefix
ARMS = {a.key: a.root for a in SET.arms}
COLORS = {a.key: a.color for a in SET.arms}
# Hue alone does not separate five overlapping step histograms -- at these line widths the arms
# differ by less than the stroke in most bins. Dash pattern is the second, redundant channel, and
# it survives greyscale printing and the common forms of colour blindness.
LINESTYLES = {a.key: a.dash for a in SET.arms}
LINEWIDTHS = {a.key: (2.2 if a.heavy else 1.4) for a in SET.arms}
LABELS = {a.key: a.label for a in SET.arms} | {PANDORA: "Pandora"}
# Photon gets a lightened companion to its arm's hue, so a class split reads as one arm's pair
# rather than as a dozen unrelated curves.
EFF_FR_COLORS = {a.key: {"neut had": a.color, "photon": a.photon} for a in SET.arms} | {PANDORA: PANDORA_COLORS}

PT_BINS = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


def style_for(name):
    """Per-arm style, splatted straight into the helpers' `hist` calls.

    `plot_evt_res`, `plot_jet_residuals` and `plot_residuals*` splat `style_dict[name]` into
    `ax.hist`, so linewidth and linestyle take effect there directly. `plot_eff_fr_purity` and
    `plot_jet_response` read only `linestyle` from it and hardcode `linewidth=1`; `plot_jet_res_boxplot`
    reads only `color`. Those three are corrected post-hoc in `_restyle` instead.
    """
    d = {"histtype": "step", "linewidth": 1.4, "linestyle": "-"}
    if name in COLORS:
        d |= {"color": COLORS[name], "linestyle": LINESTYLES[name], "linewidth": LINEWIDTHS[name]}
    if name == PANDORA:
        # Pandora is the classical reference, not an arm: filled and receding, so it reads as the
        # backdrop the arms are compared against rather than one more competing line.
        d |= {"color": "#888888", "alpha": 0.3, "histtype": "stepfilled", "linewidth": 1, "linestyle": "-"}
    return d


def _cache_path(convention):
    """Cache file for one convention, keyed to the exact arm set.

    The key must cover every arm's resolved path, because `reorder_and_find_intersection`
    restricts all arms to the events they ALL have: adding, removing or repointing one arm
    changes the event set and therefore every other arm's numbers. A cache keyed only by
    convention would silently serve stale, differently-intersected data.
    """
    key = hashlib.sha256("|".join(f"{a}={ARMS[a]}" for a in sorted(ARMS)).encode()).hexdigest()[:12]
    return CACHE / f"perf_{PREFIX}_{convention}_{key}.pkl"


def build(convention):
    """Return the analysed `Performance` for one convention, from cache when possible.

    THE PIPELINE IS THE ENTIRE COST OF THIS SCRIPT. Jet clustering, jet matching and particle
    matching take about three minutes per convention and want 16 cores; drawing the figures
    afterwards takes seconds. Without a cache, every cosmetic change -- a line width, a legend
    position -- meant re-running the whole reconstruction analysis as a batch job, which is what
    this cache exists to stop.

    So: the first run for a given arm set pays the pipeline once and pickles the result; every
    later run loads it and goes straight to drawing. Restyling is then fast enough to run
    directly on a login node with one core:

        pixi run -e clic python studies/model_size/plot_size_ablation_performance.py

    Set REBUILD_PERF_CACHE=1 to force recomputation (needed only if the underlying .root files
    or the pipeline parameters change -- the arm paths themselves are already in the cache key).

    Raises:
        SystemExit: If any arm's evaluation .root is missing. All are required -- a partial
            figure set would silently change the event intersection and so change every other
            arm's numbers relative to the jet-IQR figure.
    """
    cache = _cache_path(convention)
    if cache.exists() and not os.environ.get("REBUILD_PERF_CACHE"):
        print(f"loading cached pipeline: {cache} ({cache.stat().st_size / 1e9:.2f} GB)", flush=True)
        with cache.open("rb") as f:
            return pickle.load(f)  # noqa: S301 - our own cache, written by this script

    networks = []
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"MISSING {arm}: {path}", flush=True)
            continue
        networks.append({"name": arm, "path": str(path), "network_type": convention, "ind_threshold": 0.65})
    if len(networks) != len(ARMS):
        raise SystemExit("every arm is needed: a dropped arm shifts the shared-event intersection")

    perf = Performance(PerformanceConfig.from_dict({"truth_path": TRUTH, "networks": networks}))
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=16)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    # Particle-level matching is what plot_residuals* and plot_eff_fr_purity read; it is separate
    # from the jet matching above and is the half the jet-E IQR never touches.
    perf.hung_match_particles(flatten=True, return_unmatched=True)

    CACHE.mkdir(exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(perf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cached pipeline -> {cache} ({cache.stat().st_size / 1e9:.2f} GB)", flush=True)
    return perf


def _headroom(ax, factor):
    """Grow the y-axis upward so an in-axes legend stops sitting on the data.

    Log and linear axes need different arithmetic: on a log axis `factor` is a number of decades,
    on a linear one it is a fraction of the current span. The helpers call a bare `ax.legend()`,
    which is `loc='best'` -- and with histograms that peak in the middle, "best" is nearly always
    on top of the peak. Making room is more robust than fighting the placement.
    """
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        if lo > 0 and hi > 0:
            ax.set_ylim(lo, hi * (10**factor))
    else:
        ax.set_ylim(lo, hi + factor * (hi - lo))


def _legend_contents(ax):
    """Handles and labels for an axes' existing legend, however it was built.

    `ax.get_legend_handles_labels()` is NOT enough. `plot_jet_res_boxplot` builds proxy patches
    and passes them positionally (`ax.legend(labels, ...)`, plot_helper_event.py:218) because
    boxplot artists carry no labels of their own -- so that call returns empty there and rebuilding
    from it silently yields an EMPTY legend box. Read the existing legend's own handles first and
    fall back to the axes only when there is no legend to read.
    """
    leg = ax.get_legend()
    if leg is not None:
        handles = list(getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []))
        labels = [t.get_text() for t in leg.get_texts()]
        if handles:
            return handles, labels
    return ax.get_legend_handles_labels()


def _restyle(fig, *, size=None, ncol=1, fontsize=None, headroom=0.0, bump_lw=None, shared_below=False, shared_ncol=4):
    """Re-lay out one helper figure: size, line weights, and legend placement.

    Everything here is post-hoc on the returned Figure, because the helpers are shared with
    `performance.ipynb` and must not be restyled for this study's sake.

    `bump_lw` repairs the two plots that ignore `style_dict`'s linewidth (`plot_eff_fr_purity`
    hardcodes 1, `plot_jet_res_boxplot` reads only colour) by thickening the drawn artists
    directly. Pandora is left alone -- it is the filled backdrop and should stay recessive.

    `shared_below` replaces per-axes legends with ONE figure-level legend underneath. That is the
    fix for `eff_fr_purity`, where five arms plus Pandora at two particle classes each make
    twelve entries repeated across three stacked panels: no in-axes placement fits, and widening
    the figure to make room is what collapsed the axes to zero height previously.
    """
    if size is not None:
        fig.set_size_inches(*size)

    if bump_lw:
        for ax in fig.axes:
            for ln in ax.get_lines():
                if ln.get_linewidth() <= 1.5 and ln.get_color() not in {"#888888", "#999999", "#cccccc"}:
                    ln.set_linewidth(bump_lw)

    if shared_below:
        handles, labels = [], []
        for ax in fig.axes:
            h, ls = _legend_contents(ax)
            for hh, ll in zip(h, ls, strict=False):
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        # Place the legend BELOW the figure (negative y in figure coords) rather than reserving a
        # strip inside it. `tight_layout(rect=...)` is not honoured for these axes -- matplotlib
        # reports them as "not compatible with tight_layout", so the reserved strip never appeared
        # and the legend landed on the bottom panel's x-label. Every figure is saved with
        # bbox_inches="tight", which expands the output to include artists placed outside the
        # figure, so this needs no reserved space and cannot collide with anything inside.
        # Pull the bottom margin in first, or the legend sits an inch below the last panel: the
        # helper's default margin plus the x-label plus the legend's own offset add up to a wide
        # band of white. This figure is built without constrained_layout, so subplots_adjust works.
        fig.subplots_adjust(bottom=0.16)
        fig.legend(handles, labels, loc="upper center", ncol=shared_ncol, fontsize=fontsize, frameon=False, bbox_to_anchor=(0.5, 0.085))
        return

    for ax in fig.axes:
        if headroom:
            _headroom(ax, headroom)
        if ax.get_legend() is None:
            continue
        handles, labels = _legend_contents(ax)
        if not handles:
            continue
        ax.get_legend().remove()
        ax.legend(handles, labels, loc="upper right", ncol=ncol, fontsize=fontsize, framealpha=0.85, borderpad=0.4, labelspacing=0.3)
    # Only re-flow when this function actually resized the figure, and never on a figure the
    # helper already built with constrained_layout: calling tight_layout there swaps the layout
    # engine ("The figure layout has changed to tight") and silently re-proportions panels and
    # margins that were fine. That swap, not the line widths alone, is what made these figures
    # look re-rendered at a different resolution.
    if size is not None and fig.get_layout_engine() is None:
        fig.tight_layout()


def draw(perf, convention):
    style = defaultdict(dict)
    for name in perf.network_names:
        style[name] = style_for(name)
    helper = PlotHelper(perf, style_dict=style, labels=LABELS, plot_path=str(FIG))

    # (filename stem, callable, fixup). Each is wrapped so one failing plot cannot lose the rest
    # -- they are expensive to regenerate, the pipeline above being the costly part.
    #
    # The helpers size every figure off FIG_W = 10 in a single wide row, which suits the notebook's
    # 2-network comparisons and is far too letterboxed for five. Each entry below re-proportions its
    # figure toward squarer panels and gives the legend somewhere to live that is not on the data.
    # Sizes are left at the helpers' own defaults everywhere except eff_fr_purity. Re-proportioning
    # every figure changed the ratio of stroke and text to canvas and read as a resolution change;
    # the only real complaint on these plots was the legend sitting on the data, so that is all
    # that is corrected. Headroom is a fraction of the span on linear axes, decades on log ones,
    # and is kept just large enough to clear the legend rather than to reshape the plot.
    fix_hist = partial(_restyle, headroom=0.16)
    # The one genuine sizing fix: ncol=3 rather than the helper's ncol=len(networks), which put
    # five long labels in a single row wider than the canvas (plot_helper_event.py:218).
    fix_box = partial(_restyle, ncol=3, headroom=0.12)
    fix_resp = partial(_restyle, ncol=2, headroom=0.12)
    fix_log = partial(_restyle, headroom=0.7)
    # eff_fr_purity is the exception on every count: it hardcodes linewidth=1, its twelve curves
    # lie on top of one another, and no in-axes legend fits twelve entries across three panels.
    fix_eff = partial(_restyle, size=(7.6, 6.4), fontsize=8, bump_lw=1.9, shared_below=True, shared_ncol=3)

    plots = [
        ("evt_res", lambda: helper.plot_evt_res(), fix_hist),
        ("jet_residuals", lambda: helper.plot_jet_residuals(pt_relative=True), fix_hist),
        ("jet_res_boxplot_pt", lambda: helper.plot_jet_res_boxplot(var="pt", bins=PT_BINS), fix_box),
        ("jet_response_e", lambda: helper.plot_jet_response(pt_bins=PT_BINS, use_energy=True), fix_resp),
        ("particle_residuals", lambda: helper.plot_residuals(pt_relative=True, log_y=True), fix_log),
        ("particle_residuals_neutrals", lambda: helper.plot_residuals_neutrals(pt_relative=True, log_y=True), fix_log),
        ("eff_fr_purity", lambda: helper.plot_eff_fr_purity(EFF_FR_COLORS), fix_eff),
    ]

    written, failed = [], []
    for stem, fn, fixup in plots:
        name = f"{PREFIX}_{convention}_{stem}.png"
        try:
            fig = fn()
        except Exception:  # noqa: BLE001 - one bad plot must not cost the whole pipeline
            print(f"  FAILED {name}", flush=True)
            traceback.print_exc()
            failed.append(name)
            continue
        # Some helpers save internally and return None; some return a Figure; one returns a list.
        figs = fig if isinstance(fig, list | tuple) else [fig]
        for i, f in enumerate(figs):
            if not isinstance(f, plt.Figure):
                continue
            if fixup is not None:
                fixup(f)
            out = FIG / (name if len(figs) == 1 else name.replace(".png", f"_{i}.png"))
            f.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(f)
            written.append(out.name)
        if fig is None:
            print(f"  {name}: helper returned no Figure (it may have saved itself)", flush=True)
    return written, failed


FIG.mkdir(exist_ok=True)
all_written, all_failed = [], []
# CONVENTIONS=mpflow restyles against one convention only, halving the pickle read that dominates
# a cached run. Both are always regenerated for a real result -- 4.1's rule is that a comparison
# is only meaningful within a branch, so a figure set is only complete with both.
for convention in os.environ.get("CONVENTIONS", "mpflow,mpflow_proxy").split(","):
    print(f"\n=== {convention} ===", flush=True)
    perf = build(convention)
    w, f = draw(perf, convention)
    all_written += w
    all_failed += f

print("\nwrote:")
for n in all_written:
    print(f"  {n}")
if all_failed:
    print("\nFAILED:")
    for n in all_failed:
        print(f"  {n}")
