"""Two occupancy questions for the JV device solver, on the real dumped cost matrices.

The solver's CUDA kernel is **one thread per problem**: each thread runs a complete sequential
Jonker-Volgenant solve on its own matrix (see the vendored
``src/torch_linear_assignment_cuda_kernel.cu``). The training step hands it 10,240 problems
(batch 2048 x 5 decoder layers), so it launches 10,240 threads = **320 warps**. A B200 has 148
SMs x 64 warps = 9,472 warp slots, so the kernel occupies about **3.4% of the machine** -- and
the torch profiler still shows it as the single largest kernel in the step, 172.7 ms and 39.4%
of all CUDA time (job 40449541).

Two independent axes follow from that, and they answer different questions. **Neither is a
batch-size experiment**: a real one means training at a larger batch, which changes the number
of optimiser updates, needs the LR re-tuned and needs its own physics validation.

**Axis A -- block size.** ``solve_cuda_batch`` sets ``blockSize = SMPCores(device_index)``, whose
switch statement covers compute capability majors 2-9. The B200 is major 10, so it falls through
to ``return 128; // Unknown device``: our block size is a fallback for hardware the library has
never heard of. At 128 that is ceil(10240/128) = 80 blocks for 148 SMs, so 46% of SMs get no
work. **But the warp count is fixed by the problem count, not the block size** -- this axis only
redistributes 320 warps over more SMs, it creates none. Expect a small effect or none. It is
measured because it is nearly free and because "we checked" beats "we assumed".

**Axis B -- problem count.** The load-bearing one. At 3.4% occupancy every added warp should find
an empty slot, so solve time should stay roughly *flat* as problems are added. If it does, then
handing the solver more problems is nearly free, which is the solver-side input to whether a
larger training batch is worth pursuing. If instead it comes out linear, the residency reasoning
is wrong and the whole occupancy argument needs revisiting.

Axis A needs the patched extension (``TLA_BLOCK_SIZE``, see ``build_tla_blocksize.sh``). Axis B
runs against any build; with an unpatched one the env var is ignored and axis A collapses to a
single row, which the output states rather than hides.

⚠️ These are offline replays on an idle GPU. This study already has one hard lesson about
promoting such a number: JV measured 10.5x faster than the host path offline and 2.3-3.1x inside
training, where it contends with the model for the same SMs. Nothing here is a throughput claim.

Run from the clic experiment dir (see submit_bench_jv_blocksize_b200.sh):
    python bench_jv_blocksize.py --costs <dump.pt> --dtype float32
"""

import argparse
import os

import torch
from bench_jv_solver import prepare, solve_jv, time_call

# Block sizes to sweep. Multiples of 32 spanning the range the patched kernel accepts; 128 is
# what the unpatched library picks on this hardware, so it is the baseline row.
BLOCK_SIZES = [32, 64, 128, 256, 512]
DEFAULT_BLOCK_SIZE = 128


def set_block_size(value: int | None) -> None:
    """Set or clear the runtime override the patched kernel reads."""
    if value is None:
        os.environ.pop("TLA_BLOCK_SIZE", None)
    else:
        os.environ["TLA_BLOCK_SIZE"] = str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs", required=True, help="cost tensor written by MatcherCostDump")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    if args.device != "cuda":
        raise SystemExit("this benchmark measures GPU occupancy; --device must be cuda")

    props = torch.cuda.get_device_properties(0)
    n_sm = props.multi_processor_count
    print(f"gpu: {props.name}   SMs: {n_sm}", flush=True)

    blob = torch.load(args.costs, map_location=args.device)
    costs = blob["costs"] if isinstance(blob, dict) else blob
    object_valid = blob.get("object_valid_mask") if isinstance(blob, dict) else None
    if object_valid is None:
        object_valid = torch.ones(costs.shape[0], costs.shape[2], dtype=torch.bool, device=costs.device)

    prepared, _lengths, row_valid = prepare(costs, object_valid, dtype=dtype)
    n_problems = prepared.shape[0]
    print(f"costs {tuple(costs.shape)} -> prepared {tuple(prepared.shape)}, solved in {args.dtype}\n", flush=True)

    # ---------------------------------------------------------------- axis A: block size
    # Correctness first. Thread i handles problem i however i decomposes into (block, thread),
    # so every block size must return the SAME assignment. This is asserted rather than argued:
    # a difference would mean a latent race in the kernel, which is worth more than any timing.
    print("=== axis A: block size, at the full problem count ===")
    print(f"{'TLA_BLOCK_SIZE':>15}{'blocks':>9}{'blocks/SM':>11}{'solve (ms)':>12}{'vs 128':>9}  assignment")
    print("-" * 74)

    # Time and collect every configuration first, then compare. Comparing inside the loop would
    # leave the sizes measured before the 128 baseline permanently unchecked.
    results, assignments = {}, {}
    for bs in BLOCK_SIZES:
        set_block_size(bs)
        assignments[bs] = solve_jv(prepared).clone()  # also warms this configuration up
        results[bs] = time_call(lambda: solve_jv(prepared), prepared, args.repeats) * 1e3

    baseline_assignment = assignments[DEFAULT_BLOCK_SIZE]
    baseline_ms = results[DEFAULT_BLOCK_SIZE]
    mismatched_on_valid = []
    for bs in BLOCK_SIZES:
        if torch.equal(assignments[bs], baseline_assignment):
            verdict = "identical"
        else:
            # Padded rows carry a constant cost, so which column they take is arbitrary; only a
            # difference on a valid row is a correctness failure.
            on_valid = bool(((assignments[bs] != baseline_assignment) & row_valid).any())
            verdict = "DIFFERS ON VALID ROWS" if on_valid else "differs, padded rows only"
            if on_valid:
                mismatched_on_valid.append(bs)
        blocks = -(-n_problems // bs)
        print(
            f"{bs:>15}{blocks:>9}{blocks / n_sm:>11.2f}{results[bs]:>12.2f}{results[bs] / baseline_ms:>8.2f}x  {verdict}",
            flush=True,
        )

    set_block_size(None)
    unpatched = solve_jv(prepared)
    same_as_default = torch.equal(unpatched, baseline_assignment)
    print(f"\nTLA_BLOCK_SIZE unset reproduces the block-128 assignment: {same_as_default}")
    if mismatched_on_valid:
        raise SystemExit(
            f"STOP: block sizes {mismatched_on_valid} changed the assignment on valid rows. Thread i "
            "handles problem i whatever the block size, so this means a latent race in the kernel. "
            "That is a correctness finding and outranks any timing result -- do not quote the table above."
        )
    if len({round(v, 3) for v in results.values()}) == 1:
        print("⚠️  every block size gave an identical time -- the override is probably not in this")
        print("    build, so axis A is meaningless here. Check build_tla_blocksize.sh.")

    best = min(results, key=results.get)
    print(f"fastest block size: {best} ({results[best]:.2f} ms vs {results[DEFAULT_BLOCK_SIZE]:.2f} ms at the default 128)")

    # ------------------------------------------------------- axis B: problem-count scaling
    # Slice the PREPARED tensor so the matrix shape is untouched and the problem count is the
    # only thing that varies. Re-preparing a subset could change the crop and confound the two.
    print(f"\n=== axis B: problem-count scaling, at block size {best} ===")
    print("flat per-problem time => the GPU has spare slots => more problems are nearly free")
    print(f"{'problems':>10}{'warps':>8}{'% of slots':>12}{'solve (ms)':>12}{'us/problem':>12}{'vs full':>9}")
    print("-" * 63)

    set_block_size(best)
    warp_slots = n_sm * 64
    full_ms = None
    for frac in (8, 4, 2, 1):
        n = n_problems // frac
        subset = prepared[:n].contiguous()
        solve_jv(subset)
        ms = time_call(lambda s=subset: solve_jv(s), subset, args.repeats) * 1e3
        if frac == 1:
            full_ms = ms
        warps = -(-n // 32)
        rel = f"{ms / full_ms:.2f}x" if full_ms else "-"
        print(f"{n:>10}{warps:>8}{100 * warps / warp_slots:>11.1f}%{ms:>12.2f}{ms * 1e3 / n:>12.2f}{rel:>9}", flush=True)

    print(
        "\nRead the us/problem column: constant => cost scales with problem count (saturated);\n"
        "falling => added problems ride along in idle slots (the 3.4%-occupancy prediction)."
    )
    set_block_size(None)


if __name__ == "__main__":
    main()
