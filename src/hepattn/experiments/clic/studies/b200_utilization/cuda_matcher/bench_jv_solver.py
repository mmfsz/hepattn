"""Phase 1, re-run for a Jonker-Volgenant device solver instead of the auction.

The auction failed Phase 1 on real cost matrices (job 40228586: 89.01% exact, 21.6%
non-convergence, 350x slower than the host path), for a reason that is structural rather than
incidental -- it is pseudo-polynomial, so its cost depends on the *values* in the matrix and not
only on its shape. See NOTES.md, 2026-08-25.

``torch-linear-assignment`` implements Crouse (2016), the same algorithm scipy uses, batched on
the GPU. It is strongly polynomial, so neither the aspect ratio nor the degeneracy of real
mask-BCE costs can provoke the auction's failure mode. This script asks the same two Phase-1
questions of it, against the same bar: exact total cost versus scipy in float64, and a solve
faster than the host path's 0.538 s on this tensor.

Not installed in the environment. Build per NOTES.md (2026-08-25) and point PYTHONPATH at it;
nothing in hepattn imports it, and matcher.py is deliberately untouched until Phase 1 clears.

Usage:
    python bench_jv_solver.py --costs phase1_logs/<run>/clic_b200_matcher_costs.pt
    python bench_jv_solver.py --costs <...> --dtype float64   # exactness diagnostic
"""

import argparse
import time

import numpy as np
import scipy.optimize
import torch

from hepattn.models.matcher import Matcher

try:
    # The private backend, because the public wrapper silently downcasts float64 on CUDA;
    # see solve_jv. Both are imported so the float32 arm stays the call Phase 1 measured.
    import torch_linear_assignment._backend as tla_backend  # noqa: PLC2701
    from torch_linear_assignment import batch_linear_assignment
except ImportError as exc:  # pragma: no cover - depends on an out-of-env build
    raise SystemExit(
        "torch_linear_assignment is not importable. It is built but not installed; see\n"
        "NOTES.md (2026-08-25) for the recipe, then set PYTHONPATH to the built tree and\n"
        "LD_LIBRARY_PATH to the pixi env's lib (CXXABI errors mean the latter is missing).\n"
        f"Original error: {exc}"
    ) from exc


def optimal_cost(cost: np.ndarray) -> float:
    row, col = scipy.optimize.linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def prepare(costs: torch.Tensor, object_valid: torch.Tensor, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop, sanitise and pad the costs into the fixed-shape batch a batched JV solver needs.

    Mirrors ``Matcher._match_on_device``'s preparation, then handles the one thing a batched
    solver cannot do that ``match_individual`` can: vary the row count per problem.

    Two numerical decisions carry the correctness of the whole arm:

    * **Forbidden entries** (non-finite costs) are normalised onto ``num_rows + 1`` rather than
      the ``float32_max / 10`` sentinel ``_prepare_costs`` uses for the host solvers. Every
      complete assignment sums exactly ``num_rows`` normalised entries, each at most 1, so
      ``num_rows + 1`` beats any feasible assignment while staying small enough to keep JV's
      duals (``cost - u - v``) well conditioned in fp32. A 3.4e37 sentinel would not.
    * **Padded rows** get a constant cost across every column. They contribute the same total
      whichever columns they take, so the solver is free to park them wherever suits the real
      rows, and the optimum over the real rows is unchanged. This is the standard dummy-row
      reduction, and it is exact -- unlike padding with a large sentinel, which would make the
      padded rows compete for the cheap columns.

    ``dtype`` selects the arithmetic the solver runs in. float32 is the production case and what
    Phase 1 measured; float64 is the diagnostic that separates fp32 rounding from a systematic
    error, since JV is exact in exact arithmetic and any excess it shows is precision.

    Returns:
        Tuple of the [batch, max_len, num_pred] prepared costs, the [batch] valid target counts,
        and the [batch, max_len] bool marking real rows.
    """
    costs = costs.detach().to(dtype)
    lengths = object_valid.detach().bool().to(costs.device).sum(dim=1)
    max_len = int(lengths.max())

    # [batch, target, pred] -- the solver layout, cropped to the batch's largest event.
    costs_t = costs.transpose(1, 2)[:, :max_len].contiguous()
    row_valid = torch.arange(max_len, device=costs.device)[None, :] < lengths[:, None]

    allowed = torch.isfinite(costs_t) & row_valid[:, :, None]
    inf = torch.inf
    hi = torch.where(allowed, costs_t, torch.full_like(costs_t, -inf)).amax(dim=(1, 2))
    lo = torch.where(allowed, costs_t, torch.full_like(costs_t, inf)).amin(dim=(1, 2))
    degenerate = ~torch.isfinite(hi) | ~torch.isfinite(lo)
    lo = torch.where(degenerate, torch.zeros_like(lo), lo)
    scale = torch.where(degenerate, torch.ones_like(hi), hi - lo).clamp_min(torch.finfo(costs.dtype).tiny)

    normalised = (costs_t - lo[:, None, None]) / scale[:, None, None]
    prepared = torch.where(allowed, normalised, torch.full_like(normalised, float(max_len + 1)))
    # Padded rows: constant across columns, so they never distort the real rows' optimum.
    return torch.where(row_valid[:, :, None], prepared, torch.zeros_like(prepared)), lengths, row_valid


def solve_jv(prepared: torch.Tensor) -> torch.Tensor:
    """Assign every row a distinct column. Returns [batch, max_len] column indices.

    The float64 arm calls the backend directly. ``batch_linear_assignment`` casts anything that
    is not a *CPU* float or double tensor down to float32 -- ``isinstance(cost, DoubleTensor)``
    is False for a CUDA tensor whatever its dtype -- which would silently turn the diagnostic
    back into the float32 run it exists to be compared against. The kernel itself is templated
    and dispatches over both, so this is the same solve, minus the downcast.
    """
    if prepared.is_cuda and prepared.dtype == torch.float64:
        if prepared.shape[2] < prepared.shape[1]:
            raise NotImplementedError("more targets than predictions; the wrapper's transpose branch is not reproduced here")
        col4row, _ = tla_backend.batch_linear_assignment(prepared.contiguous())
        return col4row.long()
    return batch_linear_assignment(prepared)


def check_exactness(costs, assignment, lengths, sample: int, seed: int) -> tuple[float, float, list[float]]:
    """Score the JV assignment against scipy's float64 optimum on a random sample.

    Returns the exact fraction, the worst absolute excess, and the *relative* excess of every
    problem that disagreed -- the last so a handful of near-ties at fp32 epsilon can be told
    apart from a systematic bias, which is the question 99.76% left open.
    """
    rng = np.random.default_rng(seed)
    picks = rng.choice(costs.shape[0], size=min(sample, costs.shape[0]), replace=False)
    host_costs = costs.detach().float().cpu().numpy()
    assign = assignment.cpu().numpy()
    lengths_np = lengths.cpu().numpy()

    exact, worst, misses = 0, 0.0, []
    for b in picks:
        n = int(lengths_np[b])
        if n == 0:
            exact += 1
            continue
        # Score on the *raw* costs, not the normalised ones, so this is a like-for-like
        # comparison with the host solver rather than a check of our own arithmetic -- which
        # means applying the host's own treatment of non-finite entries (_prepare_costs'
        # nan_to_num), not this script's. The two differ: the host maps -inf to a hugely
        # attractive -big, while prepare() above treats any non-finite entry as forbidden. On
        # the CLIC dumps the question is moot because the non-finite entries sit in padded
        # target rows that [:n] excludes, but a dump where they do not would score as a
        # disagreement here, and that would be a real one worth seeing rather than hiding.
        big = float(np.finfo(np.float32).max / 10)
        cost_b = np.nan_to_num(host_costs[b].T[:n].astype(np.float64), nan=big, posinf=big, neginf=-big)
        cols = assign[b][:n]
        if len(set(cols.tolist())) != n or cols.min() < 0:
            worst = float("inf")
            misses.append(float("inf"))
            continue
        optimal = optimal_cost(cost_b)
        excess = float(cost_b[np.arange(n), cols].sum()) - optimal
        worst = max(worst, excess)
        relative = excess / max(1.0, abs(optimal))
        if relative <= 1e-9:
            exact += 1
        else:
            misses.append(relative)
    return exact / len(picks), worst, misses


def time_call(fn, costs, repeats: int) -> float:
    """Median seconds per call, synchronising so async device work is actually counted."""
    times = []
    for _ in range(repeats + 1):
        if costs.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if costs.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--costs", type=str, required=True, help="cost tensor written by MatcherCostDump")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="solver arithmetic; float64 is the diagnostic that separates rounding from bias",
    )
    parser.add_argument("--sample", type=int, default=2048, help="problems checked against scipy")
    parser.add_argument("--n-jobs", type=int, default=16, help="host solver threads, matching production")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"device: {args.device}", flush=True)
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    blob = torch.load(args.costs, map_location=args.device)
    costs = blob["costs"] if isinstance(blob, dict) else blob
    object_valid = blob.get("object_valid_mask") if isinstance(blob, dict) else None
    if object_valid is None:
        object_valid = torch.ones(costs.shape[0], costs.shape[2], dtype=torch.bool, device=costs.device)
    if isinstance(blob, dict) and blob.get("query_valid_mask") is not None:
        raise NotImplementedError("this dump carries a query_valid_mask, which this replay does not thread through")

    dtype = getattr(torch, args.dtype)
    prepared, lengths, _ = prepare(costs, object_valid, dtype)
    print(
        f"costs {tuple(costs.shape)} -> prepared {tuple(prepared.shape)} (crop = {prepared.shape[1]}), solved in {args.dtype}",
        flush=True,
    )

    assignment = solve_jv(prepared)
    checked = min(args.sample, costs.shape[0])
    frac_exact, worst, misses = check_exactness(costs, assignment, lengths, args.sample, args.seed)

    host = Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=True, n_jobs=args.n_jobs)
    host_s = time_call(lambda: host(costs, object_valid), costs, args.repeats)
    # Time the whole device path, preparation included -- the host arm's transfer is timed as
    # part of its call, so charging JV only for its kernel would not be like for like.
    device_s = time_call(lambda: solve_jv(prepare(costs, object_valid, dtype)[0]), costs, args.repeats)

    header = f"{'case':34s} {'exact':>8s} {'worst excess':>13s} {'host (s)':>10s} {'device (s)':>11s} {'speedup':>8s}"
    print(f"\n{header}\n{'-' * len(header)}", flush=True)
    label = f"real n={costs.shape[0]} q={costs.shape[1]} t={prepared.shape[1]} {args.dtype}"
    speedup = host_s / device_s if device_s else float("nan")
    print(f"{label:34s} {frac_exact:7.2%} {worst:13.2e} {host_s:10.4f} {device_s:11.4f} {speedup:7.2f}x", flush=True)

    if misses:
        finite = [m for m in misses if np.isfinite(m)]
        spread = f"median {np.median(finite):.2e}, max {max(finite):.2e}" if finite else "every one an invalid assignment"
        print(
            f"\n{len(misses)} of {checked} sampled problems disagreed with scipy: relative excess {spread},\n"
            f"against a float32 epsilon of {np.finfo(np.float32).eps:.2e} and a float64 epsilon of "
            f"{np.finfo(np.float64).eps:.2e}.\nDisagreements at a few eps of the arithmetic in use are "
            "near-ties; a bias that survives\n--dtype float64 is not.",
            flush=True,
        )
    else:
        print("\nNo disagreements: every sampled problem matched scipy's float64 optimum.", flush=True)

    print(
        "\nBar for this arm (README §6, unchanged from the auction's): 100% exact, and a solve\n"
        "faster than the host path. Amdahl on the Phase-0 step: 50 ms -> 2.45x throughput,\n"
        "200 ms -> 1.85x, 500 ms -> 1.23x, ~710 ms -> break-even.",
        flush=True,
    )


if __name__ == "__main__":
    main()
