"""Why is the matcher slower with 32 threads than with 16 (fix experiment 5 follow-up).

Job 38452194 measured `n_jobs: 32` + `--cpus-per-task=32` as 22% slower end-to-end than 16
threads (3.72 vs 3.05 s/step), and the first explanation offered — that a 32-thread pool
must straddle the node's two sockets — does not hold up: a socket has 56 cores, so 32 fits.

This script gathers the evidence directly on a compute node:
  1. What cores SLURM actually gave us, and how they map onto NUMA domains. If the
     allocation is already fragmented across domains at 16 threads, placement cannot
     explain a regression that only appears at 32.
  2. How the solve itself scales with thread count, on cost matrices shaped like the real
     ones. If the solve alone regresses past 16 threads on a *contiguous* allocation, the
     cause is contention inside the matcher (GIL-held Python work per event, memory
     bandwidth), not where the threads sit.

Run with more cores than you intend to test, e.g. --cpus-per-task=48.
"""

import os
import time
from pathlib import Path

import numpy as np

from hepattn.models.matcher import SOLVERS, match_parallel

# Shaped like one real CLIC training step: 5 matched decoder outputs (4 layers + final
# head) x 2048 events, each a [num_true, 150] cost matrix. That is 10,240 solves/step.
N_PROBLEMS = 5 * 2048
N_PRED = 150
N_TRUE_MAX = 150
THREAD_COUNTS = [1, 2, 4, 8, 16, 24, 32, 48]
REPEATS = 3


def describe_placement() -> None:
    print("=" * 72)
    print("PLACEMENT")
    print("=" * 72)
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
            print(line.rstrip())
    print(f"os.sched_getaffinity: {len(os.sched_getaffinity(0))} cores")
    print(f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}")
    allowed = sorted(os.sched_getaffinity(0))
    print(f"allowed cores: {allowed}")

    # Map the allocation onto NUMA domains, so we can see whether it is contiguous.
    numa = {}
    node_dir = Path("/sys/devices/system/node")
    if node_dir.is_dir():
        for entry_path in sorted(node_dir.iterdir()):
            entry = entry_path.name
            if not entry.startswith("node") or not entry[4:].isdigit():
                continue
            cpulist_path = entry_path / "cpulist"
            if not cpulist_path.exists():
                continue
            spec = cpulist_path.read_text().strip()
            cpus = set()
            for part in spec.split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    cpus.update(range(int(lo), int(hi) + 1))
                elif part:
                    cpus.add(int(part))
            overlap = sorted(cpus & set(allowed))
            if overlap:
                numa[entry] = overlap
    print(f"\nallocation spans {len(numa)} NUMA domain(s):")
    for node, cpus in numa.items():
        print(f"  {node}: {len(cpus)} of our cores -> {cpus}")
    print()


def make_problems(rng):
    """Cost matrices with realistic per-event target counts, in solver layout."""
    lengths = rng.integers(1, N_TRUE_MAX + 1, size=N_PROBLEMS).astype(np.int32)
    costs_t = rng.standard_normal((N_PROBLEMS, N_TRUE_MAX, N_PRED)).astype(np.float32)
    return np.ascontiguousarray(costs_t), lengths


def main() -> None:
    describe_placement()

    rng = np.random.default_rng(0)
    costs_t, lengths = make_problems(rng)
    print(f"{N_PROBLEMS} problems, cost buffer = {costs_t.nbytes / 1e6:.1f} MB\n")

    solvers = [s for s in ("lap1015_late", "scipy") if s in SOLVERS]
    max_cores = len(os.sched_getaffinity(0))

    print("=" * 72)
    print(f"THREAD SCALING (median of {REPEATS} runs, seconds per full step's worth of solves)")
    print("=" * 72)
    header = f"{'n_jobs':>8}" + "".join(f"{s:>18}" for s in solvers)
    print(header)

    baseline = {}
    for n_jobs in THREAD_COUNTS:
        if n_jobs > max_cores:
            print(f"{n_jobs:>8}" + "".join(f"{'(> alloc)':>18}" for _ in solvers))
            continue
        row = f"{n_jobs:>8}"
        for solver in solvers:
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                match_parallel(SOLVERS[solver], costs_t, lengths, N_PRED, n_jobs=n_jobs)
                times.append(time.perf_counter() - t0)
            t = float(np.median(times))
            baseline.setdefault(solver, t)
            speedup = baseline[solver] / t
            row += f"{f'{t:.2f}s ({speedup:.1f}x)':>18}"
        print(row, flush=True)

    print("\nReading this: if the best time is at 16 and 32 is worse even though all 32 cores")
    print("sit in the same allocation, the regression is contention inside the matcher, not")
    print("thread placement. Compare against the end-to-end numbers in NOTES.md fix 5.")


if __name__ == "__main__":
    main()
