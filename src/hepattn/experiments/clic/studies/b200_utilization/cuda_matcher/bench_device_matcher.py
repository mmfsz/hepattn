"""Phase 1 of the CUDA matching study: is the device solver exact, and is it faster?

Correctness first. For every configuration the device solver's assignment is scored against
scipy's optimum in float64, and the pass criterion is *equal total cost*, not an identical
permutation -- tied assignments are genuinely degenerate and two exact solvers may disagree
about which optimum they return. Only then is anything timed.

The timing compares like for like. The host path's cost is not just its solve: it has to move
the whole cost tensor device->host first (~921.6 MB at the B200 config), which is exactly what
the device path exists to delete, so that transfer is timed as part of it.

Usage:
    python bench_device_matcher.py                    # synthetic sweep
    python bench_device_matcher.py --costs FILE.pt    # replay real cost matrices

``--costs`` takes a file written by ``--dump-from`` during a training run, or any ``.pt``
holding a ``[batch, num_preds, num_targets]`` tensor plus an optional ``object_valid_mask``.
Synthetic uniform costs are far better conditioned than real mask-BCE costs, so a clean
synthetic result is necessary but not sufficient.
"""

import argparse
import time

import numpy as np
import scipy.optimize
import torch

from hepattn.models.matcher import Matcher

# The production B200 geometry: batch 2048 x (4 decoder layers + the final head), 150 queries.
CLIC_BATCH = 2048 * 5
CLIC_QUERIES = 150


def optimal_cost(cost: np.ndarray) -> float:
    row, col = scipy.optimize.linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def score(cost: np.ndarray, perm: np.ndarray, n_valid: int) -> float:
    """Total cost the given permutation achieves on the first ``n_valid`` targets."""
    return float(cost[np.arange(n_valid), perm[:n_valid]].sum())


def make_costs(batch: int, num_queries: int, num_targets: int, seed: int, device: str):
    """Synthetic problems shaped like the matcher's, with a realistic spread of event sizes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    costs = torch.rand(batch, num_queries, num_targets, generator=generator)
    lengths = torch.randint(1, num_targets + 1, (batch,), generator=generator)
    object_valid = torch.arange(num_targets)[None, :] < lengths[:, None]
    return costs.to(device), object_valid.to(device)


def check_exactness(costs: torch.Tensor, object_valid: torch.Tensor, sample: int, seed: int) -> tuple[float, float]:
    """Compare the device solver against scipy on a random sample of the batch.

    Returns:
        Tuple of the fraction of problems solved to the exact optimum and the worst excess cost.
    """
    matcher = Matcher(default_solver="scipy", adaptive_solver=False, device_solver="auction")
    perms = matcher(costs, object_valid).cpu().numpy()

    rng = np.random.default_rng(seed)
    picks = rng.choice(costs.shape[0], size=min(sample, costs.shape[0]), replace=False)
    host_costs = costs.detach().float().cpu().numpy()
    lengths = object_valid.sum(dim=1).cpu().numpy()

    exact, worst = 0, 0.0
    for b in picks:
        n = int(lengths[b])
        if n == 0:
            exact += 1
            continue
        cost_b = host_costs[b].T[:n].astype(np.float64)  # solver layout is [target, pred]
        excess = score(cost_b, perms[b], n) - optimal_cost(cost_b)
        worst = max(worst, excess)
        exact += excess <= 1e-9 * max(1.0, abs(optimal_cost(cost_b)))
    return exact / len(picks), worst


def time_path(matcher: Matcher, costs: torch.Tensor, object_valid: torch.Tensor, repeats: int) -> float:
    """Median seconds per call, synchronising so that async device work is actually counted."""
    times = []
    for _ in range(repeats + 1):
        if costs.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        matcher(costs, object_valid)
        if costs.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times[1:]))  # drop the first, which pays warm-up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--costs", type=str, default=None, help="replay a saved cost tensor instead of synthetic ones")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=5, help="timed calls per configuration")
    parser.add_argument("--sample", type=int, default=256, help="problems per configuration checked against scipy")
    parser.add_argument("--n-jobs", type=int, default=16, help="host solver threads, matching the production config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-timing", action="store_true", help="correctness only")
    args = parser.parse_args()

    print(f"device: {args.device}", flush=True)
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    if args.costs:
        blob = torch.load(args.costs, map_location=args.device)
        costs = blob["costs"] if isinstance(blob, dict) else blob
        object_valid = blob.get("object_valid_mask") if isinstance(blob, dict) else None
        if object_valid is None:
            object_valid = torch.ones(costs.shape[0], costs.shape[2], dtype=torch.bool, device=costs.device)
        if isinstance(blob, dict) and blob.get("query_valid_mask") is not None:
            # Both solver paths mask padded queries themselves, and dropping the mask here
            # would compare two solvers on a problem neither of them is actually given.
            raise NotImplementedError("this dump carries a query_valid_mask, which the replay does not thread through yet")
        label = f"real n={costs.shape[0]} q={costs.shape[1]} t={costs.shape[2]}"
        cases = [(label, costs, object_valid)]
    else:
        cases = []
        for batch, queries, targets in [
            (1024, 50, 50),
            (1024, 150, 50),
            (1024, 150, 150),
            (CLIC_BATCH, CLIC_QUERIES, 50),
            (CLIC_BATCH, CLIC_QUERIES, CLIC_QUERIES),
        ]:
            label = f"b={batch} q={queries} t={targets}"
            cases.append((label, *make_costs(batch, queries, targets, args.seed, args.device)))

    host = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=True, n_jobs=args.n_jobs)
    device = Matcher(default_solver="scipy", adaptive_solver=False, device_solver="auction")

    header = f"{'case':26s} {'exact':>8s} {'worst excess':>13s} {'host (s)':>10s} {'device (s)':>11s} {'speedup':>8s} {'fallbacks':>10s}"
    print(f"\n{header}\n{'-' * len(header)}", flush=True)

    for label, costs, object_valid in cases:
        frac_exact, worst = check_exactness(costs, object_valid, args.sample, args.seed)

        if args.skip_timing:
            print(f"{label:26s} {frac_exact:7.2%} {worst:13.2e} {'-':>10s} {'-':>11s} {'-':>8s} {device.device_fallbacks:10d}", flush=True)
            continue

        host_s = time_path(host, costs, object_valid, args.repeats)
        device.device_fallbacks = 0
        device_s = time_path(device, costs, object_valid, args.repeats)
        speedup = host_s / device_s if device_s else float("nan")
        print(
            f"{label:26s} {frac_exact:7.2%} {worst:13.2e} {host_s:10.4f} {device_s:11.4f} {speedup:7.2f}x {device.device_fallbacks:10d}",
            flush=True,
        )

    print(
        "\nExactness is the gate: anything below 100% needs explaining before the timings mean\n"
        "anything. 'fallbacks' counts problems the auction did not converge on, which the host\n"
        "solver then re-solved -- a high count makes the device path a host path in disguise.",
        flush=True,
    )


if __name__ == "__main__":
    main()
