# Profiling study — running notes

## Phase 1 — SimpleProfiler (job 38122563, 2026-07-27) — DONE

Setup: 1× B200, 16 CPU, `base.yaml` + `configs/profile.yaml`
(`profiler: simple`, `max_steps: 200`, `limit_val_batches: 0`), batch 2048.
Ran 16m47s total; raw table in
[`profile_logs/phase1_simpleprofiler_slurm-38122563.out`](profile_logs/phase1_simpleprofiler_slurm-38122563.out).

| Action | Total (s) | % of 962 s | Per step |
|---|---|---|---|
| `run_training_batch` | 885.1 | 92.0% | 4.43 s |
| `training_step` | 752.9 | **78.2%** | 3.76 s |
| `backward` | 125.5 | **13.0%** | 0.63 s |
| `[_TrainingEpochLoop].train_dataloader_next` | 4.0 | **0.42%** | 20 ms |
| `PflowDataModule.setup` (ROOT → RAM, one-off) | 67.1 | 7.0% | — |
| `batch_to_device` | 0.14 | 0.01% | 0.7 ms |

Steady-state throughput after compile warmup: ~0.27 it/s (steps 50→200 from the
tqdm timestamps; the first 50 steps took 5:29 due to torch.compile warmup).

**Verdict: NOT data-bound.** The dataloader wait is 0.4% of wall time, so
`persistent_workers`/`prefetch_factor` tuning (Phase 1b) is pointless — **skipped
per the plan's decision rule.** Essentially all time is inside
`training_step` (78%) + `backward` (13%). SimpleProfiler cannot see inside the
step, so it cannot distinguish real GPU kernels from CPU stalls (the scipy
Hungarian matcher runs inside `model.loss`, i.e. inside `training_step`).
→ Proceed to Phase 2 (PyTorchProfiler) to split GPU-busy vs GPU-idle inside the step.

Note: the expected `ModelCheckpoint(monitor='val/loss') could not find the
monitored key` warning fired (validation disabled); harmless, `save_last` worked.

## Phase 2 — PyTorchProfiler (op-level, Chrome trace)

Config: `configs/profile_phase2.yaml` layered on `base.yaml`.
Deviations from the README's sketch, and why:

- **No `schedule:` key.** `PyTorchProfiler` forwards `schedule` verbatim to
  `torch.profiler.profile`, which requires a *callable*
  (`torch.profiler.schedule(...)`) — a YAML dict would crash at runtime. Instead
  the window is bounded with `max_steps: 12` and no schedule (profile all steps);
  ignore the first few warmup steps when reading the trace timeline.
- **`Compile` callback removed** (callbacks list overridden minus
  `hepattn.callbacks.Compile`) per the README gotcha — first pass is eager-mode
  so op-level attribution is clean. A compiled-mode pass can follow if needed.
- `record_shapes: false` to keep the trace small.
- **No `with_stack:` key** — jsonargparse rejects it for `PyTorchProfiler`
  (first attempt, job 38127129, failed at config parse with
  "Key 'with_stack' is not expected"; it's False by default anyway).
  Validate future overlay configs cheaply on the login node with
  `pixi run python main.py fit --config ... --print_config` before submitting.

View the exported Chrome trace at https://ui.perfetto.dev. Look for: gaps between
kernels during loss computation (CPU matcher stall), `cudaStreamSynchronize` /
`Memcpy DtoH` around the matcher's `.cpu().numpy()` calls (`matcher.py:261`),
and whether flash-attn kernels actually fill the step.

### Phase 2 results (job 38127863, 2026-07-27) — DONE

Ran 3m47s. Lightning applied its default profiler schedule (wait 1, warmup 1,
active 3) → 3 profiled steps at ~2.85 s wall each. Outputs in `profile_logs/`:
`fit-clic_profile_phase2.txt` (op table), `*.pt.trace.json` (46 MB Chrome trace,
view at https://ui.perfetto.dev), `phase2_pytorchprofiler_slurm-38127863.out`.
Trace analysis script: [`analyze_trace.py`](analyze_trace.py) (interval-union of
kernel/memcpy events vs. `ProfilerStep*` spans; note the trace contains duplicate
thread-level `ProfilerStep` spans — 6 spans = 3 real steps).

- **GPU busy 70.6% / idle 29.4%.** Idle is host-bound: `model.matcher` = 1.24 s
  CPU per step (scipy Hungarian + numpy), plus eager-Python overhead.
- GPU busy-time breakdown: **~53% fused triton loss kernels** (mask BCE/dice
  cost/loss reductions, fp32, per decoder layer — these are `torch.compile`d loss
  fns, present even with the Compile callback off), **23.5% `Memcpy DtoH
  (Device → Pageable)`** (2 × ~236 ms/step — cost matrices → CPU for the matcher,
  non-pinned), 11% elementwise, and only **~7% actual model** (attention 4.1% +
  GEMM 2.5%).

## Fix experiment 1 — lap1015_late matcher solver (job 38143939, 2026-07-27)

First fix candidate under test: `default_solver: scipy` → `lap1015_late`. This is a
**pure optimization** — `lap1015` is an exact LAP solver already registered in
`matcher.py` (and installed in the env), so the optimal assignment (and hence the
loss) is unchanged; only speed differs. Single-variable change vs. the Phase 1
baseline: same `base.yaml` + `configs/profile.yaml`, plus
`configs/profile_lap1015.yaml` (validated with `--print_config` — only
`default_solver` differs). Submitted via `submit_profile_lap1015_1gpu.sh`.
Compare `training_step` / `run_training_batch` per-step times against job
38122563 (baseline: 3.76 s / 4.43 s over 200 steps incl. compile warmup).

**First attempt (job 38143939) FAILED** at step 0 with `AssertionError: Matcher
error!` — negative indices from the solver. Root cause (reproduced on the login
node): for an event with **zero valid targets** the cost matrix is empty
(`shape (0, n_pred)`) and `lap1015.lap_late` returns uninitialised memory,
including negatives; the scipy branch survived only because its padding logic
rebuilds the permutation. Likely why `base.yaml` pins `scipy` and disables
`adaptive_solver`. Fixed in `models/matcher.py` `match_individual`: empty cost →
return the identity permutation (`default_idx`), solver-agnostic and
behavior-identical for scipy. Verified directly (empty events → identity; scipy
and lap1015_late assignments identical on all valid rows) and with
`tests/matching/`. Resubmitted as **job 38205109**.

**Second attempt (job 38205109) FAILED** differently: CUDA device-side assert
(`index out of bounds`) at `maskformer.py:359` when permuting outputs — lap1015
returned indices ≥ num_queries. Root cause: real cost matrices contain **-inf
entries** (padded-hit mask logits are `masked_fill`ed with -inf in
`loss.py:292,322` before cost computation). scipy treats ±inf as
forbidden/mandatory assignments; `lap1015.lap_late` has **undefined behaviour on
non-finite input** (segfaults / returns garbage — reproduced on the login node,
where a NaN/inf fuzz dumped core). Fixes in `models/matcher.py`, both
scipy-equivalent:
1. `Matcher.forward` now sanitizes costs **on-device** before the DtoH copy:
   `torch.nan_to_num(costs, nan=big, posinf=big, neginf=-big)` with
   `big = float32_max / 10` (the sentinel `compute_matching` already uses).
   A huge finite cost is equivalent to inf for an argmin assignment; ~free as a
   GPU elementwise op.
2. `match_individual` validates that non-scipy solvers return a true permutation
   (length/range/`bincount` uniqueness) and **falls back to scipy per event**
   otherwise — a residual lap1015 bug now degrades gracefully instead of
   corrupting training. (The fallback fires e.g. for all-constant cost matrices,
   which also break lap1015.)
Equivalence re-verified on the login node: on 64 events with -inf/all-inf/empty
degeneracies, both solvers return valid permutations and **identical per-event
assignment costs in float64** (tie-break-level differences only).
Resubmitted as **job 38206804**, which COMPLETED (28m10s; log archived as
`profile_logs/fix1_lap1015_slurm-38206804.out`).

**VERDICT: NEGATIVE — do not promote. lap1015_late is ~1.9× SLOWER end-to-end**
(`training_step` 7.14 s/step vs 3.76 s scipy baseline; only 1 scipy-fallback
event in 200 steps, so the fallback is not the cause). Login-node benchmark
(2000 events, [n×150] fp32) explains it:

| backend × solver | time |
|---|---|
| thread × scipy (**current config**) | **113 ms** |
| thread × lap1015_late | 326 ms |
| process × scipy | 201 ms |
| process × lap1015_late | 191 ms |

Single-threaded, lap1015_late is 3.5× *faster* than scipy (310 vs 1084 ms) — but
its pybind11 binding (`src/lap1015/src/main.cpp`) **never releases the GIL**, so
the 16-way thread parallelism serializes; scipy releases the GIL and scales ~11×.
The process backend dodges the GIL but its shared-memory IPC overhead loses to
threaded scipy anyway. So the current `scipy + parallel_solver + thread` config
is already the fastest available combination — evidently pinned deliberately.

**Future option (not done):** add `py::gil_scoped_release` around the solve in
`main.cpp` and rebuild the extension — projected ~4× on the solve
(~20-30 ms benchmark-scale vs scipy's 113 ms). Requires rebuilding the
hepattn/lap1015 C++ in the pixi env; left as a deliberate decision.
Side benefit of the attempt: two real matcher bugs fixed (empty-cost UB,
non-finite-cost UB) + a permutation-validation safety net, all accuracy-neutral.

## Fix experiment 2 — pinned-memory DtoH transfer of cost matrices

Fix candidate 1 from the report: route `Matcher.forward`'s
`costs.cpu()` through a cached, grow-only **pinned** host buffer
(`matcher.py`) instead of a fresh pageable allocation each step
(Phase 2 showed 2 × ~236 ms/step of `Memcpy DtoH (Device -> Pageable)`).
Bit-identical numerics — only the copy route changes. Measured with the same
Phase-1 protocol (`base.yaml` + `profile.yaml`, scipy solver, 200 steps) as
**job 38209457** (log: `profile_logs/fix2_pinned_slurm-38209457.out`).

**VERDICT: KEEP — 11.6% faster end-to-end, exactly the predicted DtoH saving.**

| 200 steps, 1× B200, batch 2048 | baseline 38122563 | pinned 38209457 | Δ |
|---|---|---|---|
| `run_training_batch` | 4.43 s/step | **3.92 s/step** | −0.51 s (−11.6%) |
| `training_step` | 3.76 s/step | **3.25 s/step** | −0.51 s |
| `backward` (control) | 0.627 s/step | 0.627 s/step | identical |

The change lives in shared `models/matcher.py` (grow-only cached pinned staging
buffer in `Matcher.forward`), so it applies to all experiments automatically —
"promotion" = keeping it. `tests/matching/` passes (118 tests, both solvers).

## Fix experiment 3 — lap1015 with GIL released (job 38212448, 2026-07-28)

Follow-up to fix experiment 1's future option: added `py::gil_scoped_release`
around the solve in `src/lap1015/src/main.cpp` and rebuilt `_core` manually with
the CMake flags (but `-march=x86-64-v3` instead of `-march=native` — compiled on
the login node, must run on B200 nodes; original `.so` backed up as
`_core.*.so.bak` in site-packages). pybind11 headers came from a scratch clone
(env has no pip/pybind11). Verified before submitting: 300/300 optimal-cost
agreement with scipy; 4 concurrent solves in ~1.2× single-solve wall time
(GIL genuinely released).

**VERDICT: WINNER — `run_training_batch` 3.53 s/step** (log:
`profile_logs/fix3_lap1015_gilrelease_slurm-38212448.out`; 1 scipy-fallback
event in 200 steps; `backward` control 0.629 s unchanged).

| cumulative progression | s/step | vs original |
|---|---|---|
| scipy baseline (38122563) | 4.43 | — |
| + pinned DtoH copy (38209457) | 3.92 | −11.6% |
| + lap1015_late with GIL release (38212448) | **3.53** | **−20.3%** |

To promote: flip `default_solver: scipy` → `lap1015_late` in `base.yaml`
(config), keep the `main.cpp` change committed, and note that the extension
must be rebuilt for the fix to take effect (`pip`-reinstalling hepattn rebuilds
it from source; the site-packages `.so` was hot-patched this time).

Known issue (pre-existing, unrelated to these changes): `pytest tests/matching/`
dies at `test_solvers.py::test_lap1015[100]` — that test calls
`lap1015.lap_early`, which **hangs indefinitely** on a 100×100 matrix on this
machine (reproduced standalone). `lap_early` is already commented out of the
`SOLVERS` registry in `matcher.py:85`, so nothing in production uses it; the
test just exercises it anyway. The rest of `tests/matching/` (matcher +
equivalence tests, both solvers) passes with the new sanitize/fallback code.

## Fix experiment 4 — device-side cost preparation (job 38451000, 2026-07-31)

Two exact, non-approximating fixes in one run, on branch `matcher-perf` (commit
27b483a), which also **promotes fix 3** (`base.yaml`: `default_solver: scipy` →
`lap1015_late`, so this run measures fixes 1+2 on top of the 3.53 s/step fix-3
result — single variable).

1. **On-device cost prep.** Phase 2 only counted the DtoH bytes; re-reading
   `compute_matching` showed the host also paid *two* full-size numpy copies of
   the ~1 GB stacked cost tensor every step (batch 2048 × 150 queries × 150
   targets × 5 matched decoder outputs, fp32 — the 4 decoder layers plus the final
   head, all stacked into one matcher call; 921.6 MB exactly, which is what the
   Phase-2 trace's DtoH copy reports):
   `np.ascontiguousarray(costs.swapaxes(1, 2))` (a strided single-threaded
   transpose into solver layout) and the `np.where` masking padded queries. Both
   moved into a new `Matcher._prepare_costs`, which sanitises, masks, transposes
   and stages the pinned copy — so the DtoH now lands already contiguous in
   `[batch, true, pred]`.
2. **Crop to `max(num_valid_targets)`** before the copy: `match_individual` only
   ever reads `cost[: lengths[k]]`, so target rows past the largest event were
   pure transfer + solve overhead.

**VERDICT: KEEP — a further −13.7% (3.53 → 3.05 s/step); −31.2% cumulative.**

| 200 steps, 1× B200, batch 2048, 16 CPU | fix3 38212448 | +dev prep 38451000 | Δ |
|---|---|---|---|
| `run_training_batch` | 3.53 s/step | **3.05 s/step** | −0.48 s (−13.7%) |
| `training_step` | 2.868 s/step | **2.387 s/step** | −0.48 s |
| `backward` (control) | 0.629 s/step | 0.628 s/step | identical |

Steady state after compile warmup ~2.29 s/step (steps 50→200 from the tqdm
timestamps), vs ~3.7 s/step at the original baseline. 1 scipy-fallback event in
200 steps, same as fix 3. Log: `profile_logs/fix4_deviceprep_slurm-38451000.out`.

Equivalence: assignments unchanged. `scratchpad/check_equiv.py`-style check over
556 events spanning `-inf`, NaN, all-constant, empty and query-masked cost
matrices at three shapes — both solvers agree with the old host-side prep +
scipy on optimal assignment cost in float64. `tests/matching` 118 passed
(`--deselect tests/matching/test_solvers.py::test_lap1015`, the pre-existing hang).

One deliberate behaviour change: when `object_valid_mask is None`, per-event
lengths now come from the *target* dimension (`costs.shape[2]`) rather than the
*pred* dimension (`costs.shape[1]`, the old default). Equivalent for square cost
matrices, which is the CLIC case at 150×150; the old form would have silently
truncated targets when `num_true > num_pred`.

**CPU budget for the thread pool (checked 2026-07-31):** `hpg-b200` nodes are 112
physical cores (2 sockets × 56, `ThreadsPerCore=1`) with 8 B200s → 14 cores/GPU
fair share, so the current `--cpus-per-task=16` already exceeds it. The `avery`
QoS limits the *account* (`cpu=430, gres/gpu=34`), not the job, so a 1-GPU run at
`--cpus-per-task=32` + `n_jobs: 32` is legal — untested, and it would queue
slower and consume shared budget.

## Fix experiment 5 — 32 matcher threads (job 38452194, 2026-07-31)

Tests whether the solve is still thread-starved at 16. `configs/profile_cpu32.yaml`
(`n_jobs: 32`) + `--cpus-per-task=32` in `submit_profile_cpu32_1gpu.sh`; single
variable vs job 38451000.

**VERDICT: NEGATIVE — 32 threads are 22% SLOWER than 16.**

| 200 steps, 1× B200, batch 2048 | 16 CPU (38451000) | 32 CPU (38452194) | Δ |
|---|---|---|---|
| `run_training_batch` | **3.05 s/step** | 3.72 s/step | +0.67 s (+21.9%) |
| `training_step` | **2.387 s/step** | 3.051 s/step | +0.66 s |
| `backward` (control) | 0.628 s/step | 0.632 s/step | identical |

### Follow-up benchmark (job 38458388): the NUMA explanation was WRONG

The first explanation recorded here — that a 32-thread pool must straddle the node's
two sockets — does not survive scrutiny: a socket has 56 cores, so 32 fits inside one
with room to spare, and if placement were arbitrary enough to scatter 32 threads it
would scatter 16 too. [`bench_matcher_threads.py`](bench_matcher_threads.py) tested it
directly on a B200 node with `--cpus-per-task=48`:

- **Placement is not the cause.** `Cpus_allowed_list: 60-71,76-111` — all 48 cores
  landed inside a *single* NUMA domain (`node1` = cores 56-111). `hpg-b200` nodes
  expose exactly 2 NUMA domains, one per socket. A 32-core allocation fits in one.
- **The solve saturates at ~16 threads**, on cost matrices shaped like the real ones
  (10,240 problems, 921.6 MB, the same as one training step):

| n_jobs | lap1015_late | scipy |
|---|---|---|
| 1 | 3.32 s (1.0×) | 1.23 s (1.0×) |
| 8 | 0.41 s (8.0×) | 0.29 s (4.3×) |
| **16** | **0.23 s (14.2×)** | **0.27 s (4.5×)** |
| 24 | 0.24 s (13.6×) | 0.26 s (4.7×) |
| 32 | 1.67 s (2.0×) | 1.91 s (0.6×) |
| 48 | 1.16 s (2.8×) | 1.41 s (0.9×) |

So there was never any headroom past 16 to buy: 24 threads are no faster than 16, and
at ≥32 both solvers fall off a cliff — which reproduces the end-to-end regression but
in isolation, with placement ruled out.

**The mechanism at ≥32 is still not identified.** It is not NUMA, and it is not
solver-specific (both regress by a similar factor). It is also not a clean monotonic
trend — 48 threads are *faster* than 32 — which is a warning sign. This benchmark has
a known confound: `_get_thread_pool` caches pools forever, so by the time it measures
n_jobs=32 the pools from every earlier count are still alive (~55 idle threads), and
the counts are measured in ascending order in one process. Before anyone draws a
mechanistic conclusion, re-run with a fresh process per thread count and a randomised
order. Candidate mechanisms to test then: GIL contention on the per-event Python work
in `match_individual` (array conversion, permutation validation, `np.bincount`), which
no amount of GIL release in the C++ solve can parallelise; and memory-bandwidth
saturation on a 921.6 MB working set.

**Operationally this does not matter: keep `n_jobs: 16` and `--cpus-per-task=16`.**
16 is at the saturation point by direct measurement, so the tuning question is closed
even though the cliff is not explained.

## Phase 2 re-run — post-fix trace (job 38452195, 2026-07-31)

Same protocol as the original Phase 2 (eager mode, `Compile` callback off, 12 steps,
`configs/profile_phase2.yaml`) but with all four fixes in place, to re-aim before
attempting the remaining candidates. Artifacts: `profile_logs/phase2_postfix_*`
(the pre-fix ones are preserved as `phase2_prefix_*`).

| share of GPU-busy time | pre-fix (38127863) | post-fix (38452195) |
|---|---|---|
| `Memcpy DtoH` (cost matrices) | **23.5%** | **0.9%** |
| fused triton loss kernels | ~53% | **~69%** |
| model (attention + GEMM) | ~7% | ~8.6% |
| GPU busy / idle | 70.6% / 29.4% | 54.5% / 45.5% |

The DtoH copy is gone as a cost centre — 23.5% → 0.9% confirms fix 2 + fix 4 at the
trace level, not just end-to-end. What remains is a two-part problem, and the
proportions have flipped: **the loss kernels are now ~69% of everything the GPU
does**, and the GPU is idle a *larger* fraction of the step than before, because we
removed GPU work (the copy) and host work without removing the serialisation between
them.

Caveats on the idle number: this is eager mode with profiler overhead, so 45.5% is
not the production idle fraction and is not comparable to the 3.05 s/step figure —
only the *composition* transfers. The script also reports 6 `ProfilerStep` spans for
3 real steps (duplicate thread-level spans), so per-step rows in its output are not
independent measurements.

Implication for what to try next: raising thread count is dead (fix 5), and the
transfer is solved. The two live candidates both target the remaining serialisation
— per-decoder-layer pipelining of DtoH+solve, or an exact GPU LAP solver — and a
third now looks more attractive than it did, namely attacking the loss kernels
themselves, since they are where the GPU actually spends its time.

**Study conclusion:** not data-bound; not model-compute-bound. The step is
dominated by the loss/matcher pipeline: memory-bound loss reductions + pageable
DtoH cost-matrix transfer + CPU Hungarian stall. This explains the B200≈1.76×L4
result — only ~5% of the step scales with GPU FLOPS. Phase 3 (nsys) not needed.
Full write-up + fix candidates: "Profiling results" section in
[`../training_runs_report.md`](../training_runs_report.md).
