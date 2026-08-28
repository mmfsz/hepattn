"""How square are the assignment problems the matcher actually poses?

The study has been quoting "~50 valid targets into 150 query slots" as the production
geometry, which is the mean of the *per-event* particle count. But `Matcher._prepare_costs`
crops the target axis to ``max(num_valid_targets)`` over the **whole batch**, so the shape the
solver is handed is set by the densest event in the batch, not the average one. That matters a
great deal for the device auction, whose cost is bimodal in the aspect ratio (README §7 risk 8).

This reads the particle counts straight out of a CLIC file and reports both: the per-event
distribution, and the crop a batch of a given size would produce. No GPU and no model needed,
which is why it lives outside the SLURM benchmarks.

**This is an upper bound, not the crop itself.** The matcher crops to `object_valid_mask`,
which is `indicator_truth` (`pflow_data.py:462,552`) -- the particles that are *not*
resonances -- and that is a strict subset of the `particle_pdgid` length counted here. On the
batch measured by job 40228586 this script's method predicts a crop of ~146 where the real one
was 132. Treat the numbers below as a ceiling and take the authoritative figure from
`MatcherCostDump`'s manifest (`targets_max`), which reports the valid count directly.

Usage:
    python crop_distribution.py [--file <path.root>] [--batch 2048]
"""

import argparse

import numpy as np
import uproot

# pflow_data.PflowDataset drops events with >= num_objects particles, so this is a hard
# ceiling on the crop rather than an observed maximum.
NUM_OBJECTS = 150
DEFAULT_FILE = "/cmsuf/data/store/user/mmazza/hepattn_clic_data/val_clic_fix.root"


def particle_counts(path: str) -> np.ndarray:
    """Return the per-event particle count, after the dataset's own event selection."""
    with uproot.open(path) as f:
        tree = f[f.keys()[0].split(";")[0]]
        counts = np.array([len(x) for x in tree["particle_pdgid"].array(library="np")])
    return counts[counts < NUM_OBJECTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--batch", type=int, default=2048, help="Batch size for the crop estimate.")
    parser.add_argument("--draws", type=int, default=200, help="Batches to sample for the crop estimate.")
    args = parser.parse_args()

    counts = particle_counts(args.file)
    print(f"{args.file}\n{len(counts)} events after the >= {NUM_OBJECTS} particle cut\n")

    print("per-event particle count -- an UPPER BOUND on the valid target count (see docstring):")
    print(f"  mean {counts.mean():.1f}   median {np.median(counts):.0f}   max {counts.max()}")
    for q in (90, 99, 99.9):
        print(f"  p{q}: {np.percentile(counts, q):.0f}")

    # The crop is a max over the batch, so it is governed by the tail above, not the mean.
    rng = np.random.default_rng(0)
    print(f"\ncrop = max(num_valid_targets) over a batch, {args.draws} draws:")
    print(f"  {'batch':>7} {'median':>8} {'p10':>6} {'>=140':>7}")
    sizes = sorted({32, 64, 128, 256, 512, 1024, 2048, args.batch})
    for size in sizes:
        crops = np.array([counts[rng.choice(len(counts), size=size, replace=False)].max() for _ in range(args.draws)])
        mark = "  <-- production" if size == args.batch else ""
        print(f"  {size:>7} {np.median(crops):>8.0f} {np.percentile(crops, 10):>6.0f} {(crops >= 140).mean():>7.2f}{mark}")

    print(f"\nnum_queries is {NUM_OBJECTS}: a crop at or near that is a square problem.")
    print("Upper bound only -- the real crop excludes resonance particles. See the docstring.")


if __name__ == "__main__":
    main()
