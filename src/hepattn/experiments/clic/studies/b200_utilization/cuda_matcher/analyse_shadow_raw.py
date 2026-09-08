"""Why do the two solvers' mask-IoU distributions coincide exactly? Answered from the raw dump.

`plot_shadow_matcher.py` draws, for the pairs the GPU and CPU matchers assign DIFFERENTLY, the
mask IoU of each solver's choice. The two curves land on top of one another, which looked too
good: 66% of individual swapped pairs move by more than 0.01, so with independent signs the
per-step mean should sit around 2.4e-2. It sits at ~1e-9 -- seven orders of magnitude too exact
to be chance, and identical bitwise at every quantile the .jsonl stores.

Quantiles cannot settle that, because it is a claim about the JOINT distribution. Job 40705257
resumed the converged epoch-198 checkpoint for three steps with `raw_dump_steps: 3`, writing the
per-pair arrays. This script reads them, and the answer is structural:

  * every disagreement is a clean TRANSPOSITION -- two targets exchange their two queries;
  * the two targets involved are INDISTINGUISHABLE to the matcher: each query scores identically
    against either of them, in mask IoU and in matching cost alike;
  * so the swap exchanges two identical values, leaving the multiset -- and therefore every
    quantile, the mean, and the total cost -- exactly unchanged.

These are truth particles that own the same set of constituents (typically one shared
topocluster; the IoU values are small-denominator fractions like 1/3, 1/2, 1, so the masks hold
one to three constituents). Nothing at the level of the mask can separate them, which is why
both assignments are exactly optimal rather than merely close.

⚠️ The remaining caveat, which this dump cannot close: v6 matches on `mask_dice` ALONE, so two
particles with the same mask are tied in the matching cost -- but the LOSS also carries class
and regression terms, and two particles sharing a topocluster may differ in energy or identity.
The permutation the solver picks therefore still changes the classification/regression gradient.
That is exactly what the Phase-4 physics A/B tests end to end, and passes.

Run from the clic experiment dir:
    pixi run -e clic python studies/b200_utilization/cuda_matcher/analyse_shadow_raw.py
"""

import argparse
from pathlib import Path

import numpy as np

# particle_class after pflow_data.py's observability relabelling (trackless charged -> neutral).
CLASSES = {0: "charged hadron", 1: "electron", 2: "muon", 3: "neutral hadron", 4: "photon", 5: "null"}
# configs/clic_var_transform.yaml: e is min_max_sym on sqrt(E), min 0.020 max 15.136.
E_SHIFT, E_SCALE = (15.136 + 0.020) / 2, (15.136 - 0.020) / 2
# 10,240 problems = 2,048 events x 5 cost sets, stacked layer-major.
BATCH, COST_SETS = 2048, ("layer_0", "layer_1", "layer_2", "layer_3", "final")


def energy_gev(scaled: np.ndarray) -> np.ndarray:
    """Undo the sqrt + symmetric min-max transform the data module applies to particle energy."""
    return (scaled * E_SCALE + E_SHIFT) ** 2


def what_are_they(z, transpositions: list[tuple[int, int]]) -> None:
    """What ARE the two targets a disagreement swaps? The question the queries alone cannot answer.

    Requires the extended dump (job 40779831 onward), which carries the bit-packed truth
    constituent mask and the per-particle scalars. Older dumps lack them and this is skipped.
    """
    if "truth_mask" not in z.files:
        print("  (no truth columns in this dump -- rerun with the extended callback)")
        return
    tm, ts = z["truth_mask"], z["truth_size"]
    cls, e = z["target_particle_class"], energy_gev(z["target_particle_e"])

    same_mask = np.array([np.array_equal(tm[a], tm[b]) for a, b in transpositions])
    print(f"  the two targets own the SAME constituents            : {same_mask.mean():.1%}")
    sizes = np.array([ts[a] for a, _ in transpositions])
    hist = dict(zip(*np.unique(sizes, return_counts=True), strict=True))
    print(f"  constituents per target                             : {{{', '.join(f'{k}: {v}' for k, v in hist.items())}}}")

    real = [(a, b) for (a, b), s in zip(transpositions, sizes, strict=True) if s > 0]
    ca, cb = np.array([cls[a] for a, _ in real]), np.array([cls[b] for _, b in real])
    print(f"  same particle class                                 : {(ca == cb).mean():.1%}")
    pairs = [tuple(sorted((int(x), int(y)))) for x, y in zip(ca, cb, strict=True)]
    for pair, n in sorted({p: pairs.count(p) for p in set(pairs)}.items(), key=lambda r: -r[1]):
        print(f"      {CLASSES[pair[0]]:15s} + {CLASSES[pair[1]]:15s} : {n:4d} ({n / len(pairs):5.1%})")

    ea, eb = np.array([e[a] for a, _ in real]), np.array([e[b] for _, b in real])
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    print(f"  energy [GeV]: softer median {np.median(lo):.2f}, harder median {np.median(hi):.2f}, ratio median {np.median(hi / lo):.2f}")
    print(f"  pairs within 10% in energy                          : {np.mean((hi - lo) / hi < 0.10):.1%}")


def by_cost_set(z) -> None:
    """Which of the five matched cost sets the disagreements come from.

    Load-bearing: the four in-loop layers cost `2*class_CE + 1*mask_dice`, which is blind to
    energy, so two same-mask same-class neutrals are exactly tied there. The final cost set adds
    `10*regression_L1 + KL`, which sees the energy difference and breaks the tie -- so the
    assignment that produces the actual OUTPUT is essentially always unique.
    """
    lay = z["problem"] // BATCH
    u, c = np.unique(lay, return_counts=True)
    print("  disagreeing pairs by cost set: " + ", ".join(f"{COST_SETS[int(a)]} {int(b)}" for a, b in zip(u, c, strict=True)))


CLIC = Path(__file__).resolve().parents[3]
DEFAULT_GLOB = "logs/clic_v6_cudamatch_shadow_raw_*/matcher_shadow_raw_raw_step*.npz"


def analyse(path: Path) -> None:
    """Report, for one step, whether the swap is value-preserving and why."""
    z = np.load(path)
    k, qd, qh = z["problem"], z["query_device"], z["query_host"]
    idev, ihost, cd, ch = z["iou_device"], z["iou_host"], z["cost_device"], z["cost_host"]

    print("=" * 78)
    print(f"{path.name}: {k.size} disagreeing pairs across {np.unique(k).size} problems")
    print(f"  per-pair |delta IoU| >= 0.01        : {(np.abs(idev - ihost) >= 0.01).mean():.1%}")
    print(f"  mean delta IoU                      : {(idev - ihost).mean():+.3e}")
    print(f"  sorted(IoU_device) == sorted(IoU_host)? {np.array_equal(np.sort(idev), np.sort(ihost))}")

    per_problem = [np.array_equal(np.sort(idev[k == p]), np.sort(ihost[k == p])) for p in np.unique(k)]
    print(f"  ...and per problem                  : {np.mean(per_problem):.1%} of {len(per_problem)}")

    # The mechanism, on the problems where exactly two targets disagree (the median case).
    pairs, transp, exchange, cost_exchange = 0, 0, 0, 0
    tpairs: list[tuple[int, int]] = []
    for p in np.unique(k):
        m = k == p
        if m.sum() != 2:
            continue
        a, b = np.flatnonzero(m)
        pairs += 1
        if not (qd[a] == qh[b] and qd[b] == qh[a]):
            continue
        transp += 1
        tpairs.append((a, b))
        # idev[a] is IoU(q1, t_a) and ihost[b] is IoU(q1, t_b): the SAME query on both targets.
        exchange += idev[a] == ihost[b] and idev[b] == ihost[a]
        cost_exchange += cd[a] == ch[b] and cd[b] == ch[a]
    if pairs:
        print(f"  two-target disagreements            : {pairs}, of which transpositions {transp / pairs:.1%}")
        print(f"  each query scores the same on BOTH targets — IoU  : {exchange / max(transp, 1):.1%}")
        print(f"  each query scores the same on BOTH targets — cost : {cost_exchange / max(transp, 1):.1%}")
    by_cost_set(z)
    what_are_they(z, tpairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default=DEFAULT_GLOB)
    args = p.parse_args()

    files = sorted(CLIC.glob(args.glob))
    if not files:
        raise SystemExit(f"no raw dumps matched {args.glob}")
    for f in files:
        analyse(f)


if __name__ == "__main__":
    main()
