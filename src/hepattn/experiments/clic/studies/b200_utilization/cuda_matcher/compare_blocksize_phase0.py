"""Compare the block-size cells of a Phase-0 run, in-training rather than offline.

`bench_jv_blocksize.py` measured `TLA_BLOCK_SIZE=32` at 10% under the default 128 by replaying
a dumped tensor on an idle GPU. This reads what `submit_phase0_blocksize_b200.sh` writes, which
is the same question asked inside a real training step, where the solver shares the GPU with the
model. That distinction has mattered before: JV measured 10.5x offline and 2.3-3.1x in training.

⚠️ **Read the step time, not the `device` bucket.** `MatcherTimer` brackets the matcher with
`torch.cuda.synchronize()`, but the JV path deliberately skips the post-solve sync -- the
`solved=None` contract, since JV cannot fail to converge and syncing would hand back the host
stall the device path exists to remove. So on a device arm the `device` bucket records only
kernel *launch* time (~1.5 ms against a ~170 ms kernel) and the real cost lands in `other` by
subtraction. The bucket attribution is only meaningful on host arms.

Significance comes from bootstrapping the difference of per-step medians, because the cells are
50-step samples whose spread (std ~11-13 ms) is comparable to the effect (~20 ms).

Usage, from the clic experiment dir:
    python studies/b200_utilization/cuda_matcher/compare_blocksize_phase0.py <jobid> [<jobid> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

LOGS = Path(__file__).resolve().parent / "phase0_logs"
CELLS = {"bs128_default": "block 128 (default)", "bs32": "block 32"}
BOOTSTRAP = 2000


def load(job):
    """-> {cell label: (summary dict, per-step arrays)}; missing cells are skipped."""
    out = {}
    for key, label in CELLS.items():
        j, n = LOGS / f"phase0_blocksize_{key}_{job}.json", LOGS / f"phase0_blocksize_{key}_{job}.npz"
        if not (j.exists() and n.exists()):
            print(f"  MISSING cell {key} for job {job}")
            continue
        out[label] = (json.loads(j.read_text()), np.load(n))
    return out


def report(job):
    cells = load(job)
    if len(cells) != 2:
        print(f"job {job}: need both cells, got {len(cells)}\n")
        return None

    (la, (sa, za)), (lb, (sb, zb)) = cells.items()
    print(f"=== job {job} ===")
    for label, summary, z in ((la, sa, za), (lb, sb, zb)):
        step = z["step"]
        print(
            f"  {label:<22} step median {np.median(step) * 1e3:7.2f} ms  std {step.std() * 1e3:5.2f}  "
            f"n={len(step):3d}  {summary['samples_per_s']:7.0f} samples/s  "
            f"fallbacks={summary['device_fallbacks']}"
        )

    delta = np.median(za["step"]) - np.median(zb["step"])
    rng = np.random.default_rng(0)
    boot = np.array([
        np.median(rng.choice(za["step"], len(za["step"]))) - np.median(rng.choice(zb["step"], len(zb["step"]))) for _ in range(BOOTSTRAP)
    ])
    gain = 100 * (sb["samples_per_s"] - sa["samples_per_s"]) / sa["samples_per_s"]
    print(
        f"  {'128 - 32':<22} {delta * 1e3:+7.2f} ms  bootstrap sd {boot.std() * 1e3:.2f} ms  "
        f"z = {boot.mean() / boot.std():+.1f}  sign reversed in {(boot <= 0).mean():.1%} of draws"
    )
    # The device bucket is not a valid attribution here; print it only to show it is inert.
    dev = [s["buckets"]["device"]["median_s"] * 1e3 for s in (sa, sb)]
    oth = [s["buckets"]["other"]["median_s"] * 1e3 for s in (sa, sb)]
    print(f"  {'device bucket':<22} {dev[0]:.2f} -> {dev[1]:.2f} ms   (inert -- see the docstring)")
    print(f"  {'other bucket':<22} {oth[0]:.2f} -> {oth[1]:.2f} ms   ({oth[1] - oth[0]:+.2f}, where the kernel lands)")
    print(f"  throughput {gain:+.2f}%\n")
    return gain


jobs = sys.argv[1:]
if not jobs:
    raise SystemExit(__doc__)
gains = [g for g in (report(j) for j in jobs) if g is not None]
if len(gains) > 1:
    print(f"across {len(gains)} allocations: " + ", ".join(f"{g:+.2f}%" for g in gains) + f"   mean {np.mean(gains):+.2f}%")
