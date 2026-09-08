"""Does the extension now pick block size 32 on a B200 without being told to?

The block-size win (+4.4% end to end, 2026-08-28) was switched on by `TLA_BLOCK_SIZE=32` in the
environment of individual submit scripts, which fails open: a script that omits it silently gets
the old block-128 geometry and loses the 4.4%, with no error and no warning. The fix is a
compute-capability major-10 case in `SMPCores()`, built by `build_tla_smpcores10.sh`, so the
device decides instead of the environment.

That fix is a change to a CUDA source that cannot be checked by reading the binary -- `strings`
can confirm the *override* is compiled in, but not what the default resolves to on real
hardware. So it is checked the only way that means anything: time the solve with the variable
unset and compare it against the two geometries it could have picked.

Passes when, on the real dumped cost tensor:

- unset matches `TLA_BLOCK_SIZE=32` to within `--tol` (default 2%), and
- unset is faster than `TLA_BLOCK_SIZE=128` by at least `--min-gain` (default 5%), and
- all three return identical assignments -- thread i solves problem i whatever the block size,
  so any difference on a valid row would be a latent race and outranks every timing here.

A build *without* the fix fails the first two: unset would sit on top of the 128 row instead.

Run from the clic experiment dir (see submit_verify_smpcores10_b200.sh):
    python studies/.../verify_smpcores10.py --costs <dump.pt>
"""

import argparse
import os

import torch
from bench_jv_solver import prepare, solve_jv, time_call

EXPECTED_DEFAULT = 32  # what SMPCores() should now return for compute capability major 10
SLOW_DEFAULT = 128  # what it returned before, via `return 128; // Unknown device`


def set_block_size(value: int | None) -> None:
    if value is None:
        os.environ.pop("TLA_BLOCK_SIZE", None)
    else:
        os.environ["TLA_BLOCK_SIZE"] = str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs", required=True, help="cost tensor written by MatcherCostDump")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--tol", type=float, default=0.02, help="allowed |unset - 32| / 32")
    ap.add_argument("--min-gain", type=float, default=0.05, help="required (128 - unset) / 128")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this check is about what the kernel picks on a GPU; none is visible")

    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {props.name}   SMs: {props.multi_processor_count}   capability: {props.major}.{props.minor}")
    if props.major != 10:
        print(f"⚠️  this GPU is major {props.major}, not the major-10 case under test; the result says nothing about a B200.")

    blob = torch.load(args.costs, map_location="cuda")
    costs = blob["costs"] if isinstance(blob, dict) else blob
    object_valid = blob.get("object_valid_mask") if isinstance(blob, dict) else None
    if object_valid is None:
        object_valid = torch.ones(costs.shape[0], costs.shape[2], dtype=torch.bool, device=costs.device)

    prepared, _lengths, row_valid = prepare(costs, object_valid, dtype=torch.float32)
    print(f"costs {tuple(costs.shape)} -> prepared {tuple(prepared.shape)}\n", flush=True)

    cases = {"unset": None, str(EXPECTED_DEFAULT): EXPECTED_DEFAULT, str(SLOW_DEFAULT): SLOW_DEFAULT}
    times, assignments = {}, {}
    for label, value in cases.items():
        set_block_size(value)
        assignments[label] = solve_jv(prepared).clone()  # also warms this configuration up
        times[label] = time_call(lambda: solve_jv(prepared), prepared, args.repeats) * 1e3
    set_block_size(None)

    print(f"{'TLA_BLOCK_SIZE':>15}{'solve (ms)':>12}{'vs 128':>9}  assignment")
    print("-" * 56)
    for label in cases:
        same = torch.equal(assignments[label], assignments[str(SLOW_DEFAULT)])
        if same:
            verdict = "identical"
        else:
            on_valid = bool(((assignments[label] != assignments[str(SLOW_DEFAULT)]) & row_valid).any())
            verdict = "DIFFERS ON VALID ROWS" if on_valid else "differs, padded rows only"
        print(f"{label:>15}{times[label]:>12.2f}{times[label] / times[str(SLOW_DEFAULT)]:>8.2f}x  {verdict}", flush=True)

    unset, fast, slow = times["unset"], times[str(EXPECTED_DEFAULT)], times[str(SLOW_DEFAULT)]
    drift = abs(unset - fast) / fast
    gain = (slow - unset) / slow
    print(f"\nunset vs {EXPECTED_DEFAULT}: {drift:+.2%} (tolerance {args.tol:.0%})")
    print(f"unset vs {SLOW_DEFAULT}: {gain:+.2%} faster (need {args.min_gain:.0%})")

    failures = [
        f"block size {label} changed the assignment on valid rows -- latent race, outranks any timing"
        for label in cases
        if not torch.equal(assignments[label], assignments[str(SLOW_DEFAULT)])
        and bool(((assignments[label] != assignments[str(SLOW_DEFAULT)]) & row_valid).any())
    ]
    if drift > args.tol:
        failures.append(f"unset does not match block {EXPECTED_DEFAULT} ({drift:.2%} > {args.tol:.0%}) -- SMPCores() fix is not in this build")
    if gain < args.min_gain:
        failures.append(f"unset is only {gain:.2%} faster than block {SLOW_DEFAULT} (need {args.min_gain:.0%})")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"\nPASS: an unset TLA_BLOCK_SIZE now gives the block-{EXPECTED_DEFAULT} geometry on this device.")


if __name__ == "__main__":
    main()
