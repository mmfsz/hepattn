"""One table of trainings, shared by every plot script in this study.

Ported from the head-based study on `main`
(`/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/model_size/arm_sets.py`).
The *structure* is that module's; the runs, the parameter counts and the checkpoints are this
branch's own and none of head's numbers carry over -- the code differs (SwiGLU vs SiLU
feed-forwards, no q/k/v norms, the paper's incidence head) and so does the training-to-training
scatter. Do not compare a number from a figure here against one from a figure there.

Four scripts draw these runs -- `plot_size_ablation_jet_iqr.py` (the Tier-1 verdict),
`plot_size_ablation_performance.py` (the breadth check), `plot_size_ablation_training_curves.py`
(Tier 0, the per-epoch curves) and `plot_decoder_layers.py` (the per-layer profile). This module
is the single copy of paths, parameter counts, colours and set membership; the scripts hold only
the drawing.

WHICH RUNS ARE COMPARED is chosen by the `ARM_SET` environment variable, and every script honours
it identically:

  all   (default) round 1 -- the 2^3 factorial over {A2, A3, A4} at fixed decoder depth: the
        reference (000), the three pairs C5/C4/C3, and the triple C1.

In every set the FIRST arm is the baseline all deltas are taken against.

⚠️ C1 IS ON THIS CANVAS, and that is the one deliberate departure from head's arm table. There C1
is excluded from every ablation set because it failed to train twice over -- val
`final_classification_object_ce` 1.92 and 2.13 against the reference's 0.67, a jet-E IQR near 0.4,
and a factor of five in y-range on every canvas it touched. On the paper tag it trained: 200
epochs, val_loss 4.96997, the bottom of a monotone ordering in parameter count with no outlier.
WHETHER THAT HOLDS IN THE PHYSICS is exactly the question this directory was opened to answer, so
C1 is drawn with the others rather than quarantined. If its jet-E IQR turns out to blow the
y-range the way head's did, split it into its own set then -- not pre-emptively.

⚠️ NO THRESHOLDS ARE DEFINED HERE, and the plot scripts publish no REAL / not-detectable verdicts.
sigma_repro has not been measured on this code and head's numbers (0.0029 `mpflow`, 0.0007 proxy,
n = 4 seeds) do not transfer. The scripts report every delta with its bootstrap sigma_stat, which
is the smaller half of the error and settles nothing on its own: a difference inside it is
certainly not real, one outside it is merely not excluded. Read the deltas as ordering, not as
verdicts, until a seed set is trained here.
"""

import os
from dataclasses import dataclass
from pathlib import Path

LOGS = Path("/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/logs")


@dataclass(frozen=True)
class Arm:
    """One trained run, and everything the plot scripts need to find and draw it.

    `stem` is always that run's own LOWEST-val_loss checkpoint, the selection rule head's study
    used throughout -- which for C4 and C3 is NOT the last epoch (189 and 197, not 199). `params`
    is the `trainable_params` the run wrote to its own `metadata.yaml`, not an estimate from a
    config comment.
    """

    key: str  # short id; also the network name inside the Performance pipeline
    label: str  # what a legend shows
    folder: str  # run directory under logs/
    stem: str  # checkpoint stem, without the .ckpt / __test.root suffix
    params: int
    color: str
    photon: str  # lightened companion hue, for the per-class figure's photon curves
    marker: str  # jet-IQR figure
    dash: object  # performance figures; hue alone does not separate overlapping step histograms
    heavy: bool = False  # drawn at reference weight, because it is a line to be read not scanned
    # Which evaluation of that checkpoint to read. "test" is the plain one every arm uses; the
    # suffix exists so one training can appear as more than one arm (head used it for the
    # truncated-decoder re-evaluation) without either overwriting the other's .root.
    test_suff: str = "test"

    @property
    def root(self) -> Path:
        """The evaluation output the physics figures read, written by `submit_eval_l4.sh`."""
        return LOGS / self.folder / "ckpts" / f"{self.stem}__{self.test_suff}.root"

    @property
    def metrics(self) -> Path:
        """The CSVLogger file the training-curve figures read."""
        return LOGS / self.folder / "csv_metrics" / "metrics.csv"


# Dash patterns, named so the set tables below read as intent rather than as tuples.
SOLID = "-"
DASH = (0, (5, 2))
DOT = (0, (1, 1.2))
DASHDOT = (0, (7, 2, 1.5, 2))
DASHDOTDOT = (0, (3, 1.5, 1, 1.5))
LONGDASH = (0, (6, 1.5))

_NAVY, _NAVY_LT = "#003f5c", "#4a90b8"


@dataclass(frozen=True)
class ArmSet:
    prefix: str  # figure and cache filenames start with this, so the sets never overwrite
    title: str  # suptitle of the jet-IQR figure
    arms: list[Arm]

    @property
    def baseline(self) -> Arm:
        return self.arms[0]

    def by_key(self, key: str) -> Arm:
        return next(a for a in self.arms if a.key == key)


# One line per arm, so this reads as a table: ruff format would give each field its own line and
# the columns that make arms comparable at a glance would be lost. The E501s that costs are the
# same ones head's copy carries, for the same reason.
# fmt: off
_ROUND1_ARMS = [
    Arm("reference", "reference 820k", "clic_paper_small_b200_jv_20260911-T141520", "epoch=199-val_loss=4.40887", 819_683, _NAVY, _NAVY_LT, "o", SOLID, heavy=True),  # noqa: E501
    Arm("C5_a2a4", "C5 a2a4 616k", "clic_paper_small_C5_a2a4_20260911-T141825", "epoch=199-val_loss=4.62587", 616_219, "#556b2f", "#9fb36b", "*", DASH),  # noqa: E501
    Arm("C4_a3a4", "C4 a3a4 443k", "clic_paper_small_C4_a3a4_20260911-T141823", "epoch=189-val_loss=4.70576", 443_435, "#6a4c93", "#a894c9", "X", DASHDOTDOT),  # noqa: E501
    Arm("C3_a2a3", "C3 a2a3 369k", "clic_paper_small_C3_a2a3_20260911-T141823", "epoch=197-val_loss=4.83713", 369_089, "#17a2b8", "#7fd4e0", "P", LONGDASH),  # noqa: E501
    Arm("C1_a2a3a4", "C1 a2a3a4 352k", "clic_paper_small_C1_a2a3a4_20260911-T141821", "epoch=199-val_loss=4.96997", 352_331, "#d1893a", "#f0c58f", "^", DOT, heavy=True),  # noqa: E501
]
# fmt: on


SETS = {
    # ---------------------------------------------------------------------------------------
    # ROUND 1 -- the 2^3 factorial over {A2, A3, A4}, minus the three singles, which were never
    # trained on this branch. What exists here is the reference (000), the three pairs and the
    # triple, all at 1x B200 / batch 2048 / GPU matcher (`device_solver: jv`) / 200 epochs, and
    # all evaluated with the host solver by `submit_eval_l4.sh` -- so the eval path is identical
    # across arms and contributes nothing to any difference seen.
    #
    # Hues follow head's table where an arm name matches, so a reader who knows those figures
    # keeps the same colour associations: C3 teal, C4 purple, C5 olive, reference navy.
    # ---------------------------------------------------------------------------------------
    "all": ArmSet(
        prefix="size_ablation",
        title=(
            "Paper-tag size ablation — the 2³ factorial over {A2, A3, A4} at fixed decoder depth, 820k reference\n"
            "thick navy = reference, thick orange = C1 (all three changes)   |   trained with the GPU matcher, evaluated with the host solver"
        ),
        arms=_ROUND1_ARMS,
    ),
}

# Pandora is the classical reference the performance notebook always plots. It is not an arm:
# filled and receding, so it reads as the backdrop rather than one more competing line.
PANDORA_COLORS = {"neut had": "#999999", "photon": "#cccccc"}


def select() -> tuple[str, ArmSet]:
    """The arm set named by ARM_SET, or `all`. Every script in this study calls this.

    Raises:
        SystemExit: if ARM_SET names a set that is not defined above.
    """
    name = os.environ.get("ARM_SET", "all")
    if name not in SETS:
        raise SystemExit(f"ARM_SET={name!r} unknown; pick one of {sorted(SETS)}")
    s = SETS[name]
    print(f"=== ARM_SET={name}  baseline={s.baseline.label}  arms={[a.label for a in s.arms[1:]]}", flush=True)
    return name, s
