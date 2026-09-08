"""The full `performance.ipynb` plot set for the v7 GPU-matcher vs CPU-matcher A/B.

`plot_v7_matcher_ab_jet_iqr.py` answered one question — jet-E response median and IQR — and
answered it quantitatively, with bootstrap errors and a threshold test. It is the verdict. This
script is the *breadth* check that sits behind it: the same two trained models put through every
other plot the performance notebook draws, because a matcher change could in principle leave the
jet-energy resolution alone and still damage something else.

Specifically, the jet-E IQR integrates over particle classes. §4.4's metric 5 does not: photon and
neutral-hadron efficiency and fake rate are "the most physically interpretable signal; these are
the classes inferred from calorimetry alone, with no track to anchor them". A change that hurt
only photons would be invisible in the IQR and obvious in `plot_eff_fr_purity`. Tier 2 of §4.3
asks for exactly this for finalists, with a pass criterion that is qualitative and comparative:
curves must not develop a new turn-over, and per-class splits must degrade proportionally rather
than one class collapsing.

ARMS -- the converged pair, each at its own lowest-val_loss checkpoint:

  device  clic_v7_cudamatch_b200_b2048_20260827-T130314  job 40405423  device_solver=jv   7 h 40 m
  host    clic_v7_maskfix_b200_b2048_20260827-T120002    job 40400036  lap1015_late      23 h 15 m

Their resolved configs differ in the matcher block and nothing else, and **both are scored with
the host solver at evaluation time**, so what is compared is the two trained models.

ONE FIGURE SET PER CONVENTION. `mpflow` uses the regression head's kinematics; `mpflow_proxy`
bypasses it and rebuilds them from the incidence head. They differ by ~0.03 in median and
0.01-0.02 in IQR — more than most effects being looked for — so §4.1's rule is that a comparison
is only meaningful *within* a branch. Mixing them on one axis would invite exactly the
cross-convention reading that rule forbids. Pandora is carried into both as the classical
reference the notebook always plots.

Run from the clic experiment dir (via submit_physics_plots.sh -- jet clustering and particle
matching both want cores):
    pixi run -e clic python studies/b200_utilization/cuda_matcher/plot_v7_matcher_ab_performance.py
"""

import sys
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/blue/avery/m.mazza/projects/fastml/hepattn/src")
from hepattn.experiments.clic.performance.performance import Performance, PerformanceConfig
from hepattn.experiments.clic.performance.plot_helper import PlotHelper

L = Path("/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs")
OUT = Path(__file__).resolve().parent
TRUTH = "/blue/avery/m.mazza/projects/fastml/hepattn/data/clic/test_clic_common_raw.root"

DEVICE, HOST = "device", "host"
ARMS = {
    DEVICE: (L / "clic_v7_cudamatch_b200_b2048_20260827-T130314/ckpts/epoch=197-val_loss=4.05102__test.root"),
    HOST: (L / "clic_v7_maskfix_b200_b2048_20260827-T120002/ckpts/epoch=195-val_loss=4.03413__test.root"),
}
# Same hues as the jet-IQR figure, so device and host keep one identity across the study.
COLORS = {DEVICE: "#bc5090", HOST: "#003f5c"}
LABELS = {DEVICE: "device (jv)", HOST: "host (lap1015)", "Pandora": "Pandora"}
PT_BINS = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])


def style_for(name):
    d = {"histtype": "step", "linewidth": 1.4}
    if name in COLORS:
        d["color"] = COLORS[name]
    if name == "Pandora":
        d |= {"color": "#888888", "alpha": 0.3, "histtype": "stepfilled", "linewidth": 1}
    return d


def build(convention):
    """Run the shared pipeline once for one kinematic convention.

    Raises:
        SystemExit: If either arm's evaluation .root is missing. Both are required -- a
            single-arm figure set would not be a comparison.
    """
    networks = []
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"MISSING {arm}: {path}", flush=True)
            continue
        networks.append({"name": arm, "path": str(path), "network_type": convention, "ind_threshold": 0.65})
    if len(networks) != len(ARMS):
        raise SystemExit("both arms are needed for a paired comparison")

    perf = Performance(PerformanceConfig.from_dict({"truth_path": TRUTH, "networks": networks}))
    perf.reorder_and_find_intersection()
    perf.compute_jets(n_procs=8)
    perf.hung_match_jets()
    perf.compute_event_features()
    perf.compute_jet_res_features(dr_cut=0.1, leading_n_jets=2, pt_min=10)
    # Particle-level matching is what plot_residuals* and plot_eff_fr_purity read; it is separate
    # from the jet matching above and is the half the jet-E IQR never touches.
    perf.hung_match_particles(flatten=True, return_unmatched=True)
    return perf


def _unclutter_legends(fig):
    """Move legends outside the axes and widen the figure.

    `plot_eff_fr_purity` is built at half width for a 4-entry comparison. Two arms plus Pandora
    at two particle classes each makes six, and the default in-axes legend then covers the y-axis
    label and a good part of the data. The helper is shared with the notebook, so this is fixed
    here rather than there.
    """
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * 2.1, h)
    for ax in fig.axes:
        if ax.get_legend() is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)


def draw(perf, convention):
    style = defaultdict(dict)
    for name in perf.network_names:
        style[name] = style_for(name)
    helper = PlotHelper(perf, style_dict=style, labels=LABELS, plot_path=str(OUT))

    # (filename stem, callable). Each is wrapped so one failing plot cannot lose the rest --
    # they are expensive to regenerate, the pipeline above being the costly part.
    eff_fr_colors = {
        DEVICE: {"neut had": "#bc5090", "photon": "#ff9db0"},
        HOST: {"neut had": "#003f5c", "photon": "#4a90b8"},
        "Pandora": {"neut had": "#999999", "photon": "#cccccc"},
    }
    plots = [
        ("evt_res", lambda: helper.plot_evt_res(), None),
        ("jet_residuals", lambda: helper.plot_jet_residuals(pt_relative=True), None),
        ("jet_res_boxplot_pt", lambda: helper.plot_jet_res_boxplot(var="pt", bins=PT_BINS), None),
        ("jet_response_e", lambda: helper.plot_jet_response(pt_bins=PT_BINS, use_energy=True), None),
        ("particle_residuals", lambda: helper.plot_residuals(pt_relative=True, log_y=True), None),
        ("particle_residuals_neutrals", lambda: helper.plot_residuals_neutrals(pt_relative=True, log_y=True), None),
        ("eff_fr_purity", lambda: helper.plot_eff_fr_purity(eff_fr_colors), _unclutter_legends),
    ]

    written, failed = [], []
    for stem, fn, fixup in plots:
        name = f"v7_matcher_ab_{convention}_{stem}.png"
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
            out = OUT / (name if len(figs) == 1 else name.replace(".png", f"_{i}.png"))
            f.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(f)
            written.append(out.name)
        if fig is None:
            print(f"  {name}: helper returned no Figure (it may have saved itself)", flush=True)
    return written, failed


all_written, all_failed = [], []
for convention in ("mpflow", "mpflow_proxy"):
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
