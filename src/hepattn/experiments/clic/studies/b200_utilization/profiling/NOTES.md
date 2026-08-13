# Profiling study — running notes

> ## ►► START HERE — status as of 2026-08-03
>
> **Read this block, then §"Mask-fix training A/B — RESULTS" and §"B200 re-baseline" at the
> bottom. The middle of this file is a chronological log; do not read it top-to-bottom
> unless you need the derivation of a specific number.**
>
> ### What this study is
>
> It started as "why is 1× B200 slower than 3× L4 on CLIC particle flow?" The answer:
> neither data-bound nor model-compute-bound — the step is owned by the loss/matcher
> pipeline. Along the way it found a **real correctness bug** in the mask losses
> (a silent broadcast doing `batch_size`× too much work *and* distorting the objective).
> That bug is now the headline result; the performance work is secondary.
>
> ### State of play
>
> | thing | status |
> |---|---|
> | 4 matcher/transfer fixes (`matcher.py`) | done, kept, **but perf attribution is n=1 and unconfirmed** |
> | mask-loss bug found + analysed | done — `LOSS_BUG_ANALYSIS.md`, `PROFILING_EXPLAINED.md` §11 |
> | mask fix implemented as **opt-in** v2 losses | done — `loss.py`, `configs/clic_v6_maskfix.yaml` |
> | mask fix trained 200 epochs (3× L4, job 38469247) | **done, COMPLETED 2026-08-01** |
> | mask fix evaluated (speed / IQR / quality) | **done 2026-08-03** — see the RESULTS section |
> | B200 re-baseline after the fix | **done 2026-08-03** — mask fix = 1.70× on B200; 3× L4 still ahead 1.16× but the platforms now overlap |
> | `mask_focal_loss` / `mask_kl_div_loss` | **fixed 2026-08-03** as opt-in `mask_focal_v2` / `mask_kl_div_v2` (no CLIC config uses either) |
> | Phase 3 re-profile (post-fix, B200 + L4) | **done 2026-08-03** — loss kernels gone; **L4 is 85% GPU-busy, B200 is 69.5% idle** |
> | full 200-epoch B200 training (timing test) | **RUNNING** — job 38598204, tests the extrapolated ~37 h (honest range 29-50 h) |
> | making v2 the default | **not done, deliberate** — breaks comparability with published numbers |
>
> ### The three results a new session must not re-derive
>
> 1. **The mask fix is worth +25.4% throughput in the production L4 config** (1.81 → 2.27 it/s),
>    which is ~2.7× more than all four matcher fixes combined *on that hardware*.
> 2. **It does not change the physics.** Jet-E IQR differences are all |z| < 2 and the two
>    output branches disagree on sign. **The rising-IQR trend survives the fix**, so the bug is
>    definitively not the `glow_jet_iqr` cause.
> 3. **3× L4 still edges out 1× B200** after the fix: medians 1735 vs 1497 samples/s (1.16×, down
>    from 1.43×). But the B200 spans 1122–1911 across three nodes while the L4 is pinned to 0.6%,
>    so the honest verdict is "comparable on average, far less predictable" — not "slower".
> 4. **The L4 is now GPU-bound (85% busy) and the B200 is host-bound (69.5% idle).** So for the
>    production L4 config **there is no large win left in the loss/matcher pipeline** — the
>    shelved candidates (per-layer pipelining, GPU LAP solver) target idle time the L4 does not
>    have. Do not restart optimisation work without reading the Phase 3 caveats (eager mode).
>
> ### What to do when the running jobs finish
>
> Done — see "B200 re-baseline — RESULTS". Use
> `python parse_throughput.py <logs> --batch <2048|768>` for any future throughput log; it reads
> steady state from the tqdm stamps and splits paired logs on their `ARM:` markers.
>
> **The most valuable single follow-up** is a paired `scipy` vs `lap1015_late` run — same
> `submit_paired_b200_1gpu.sh` trick, since `default_solver` is also just a config key. That
> would settle the one change in this study whose deployment is genuinely unsafe (see the audit
> section). A 4th paired job with `ORDER=maskfix_first` would also close the order/node
> confound noted in the results.
>
> ### Operational gotchas that will bite you
>
> - **NOTHING IS COMMITTED.** All of this — `loss.py`, `matcher.py`'s fixes, `pflow_data.py`,
>   every config and script and doc — is uncommitted working-tree state on branch `matcher-perf`
>   (HEAD `6af8682`). The 200-epoch training run's provenance points at a working tree, not a
>   commit. **Ask the user before committing; do not push unprompted.**
> - **`lap1015` is hot-patched, not rebuilt.** `base.yaml` promotes `default_solver: lap1015_late`,
>   which is only a win with the GIL-releasing build. The `.so` in `.pixi/envs/default` was patched
>   **by hand**; `lap1015.releases_gil` still reports `False`, so a spurious `RuntimeWarning` fires
>   in every log — ignore it. **Reinstalling hepattn, or running `pixi run -e clic` (whose env still
>   has the stock `.so`), silently makes the matcher ~2× slower than scipy.** See "Are the fixes safe
>   to keep?" below.
> - `pytest tests/matching/` hangs at `test_solvers.py::test_lap1015` (pre-existing, `lap_early`
>   never returns on HPG). Deselect it.
> - Validate config overlays with `--print_config` on the login node before submitting;
>   jsonargparse rejects unknown kwargs and you will otherwise burn a queue slot.
> - The login node is cgroup-limited to **one core**, so any threading/parallelism benchmark run
>   there is meaningless. Submit it.
>
> ### Document map
>
> - **`NOTES.md`** (this file) — chronological log + results. The record of what was measured.
> - **`PROFILING_EXPLAINED.md`** — the pedagogical version, no prior knowledge assumed. §11 is the
>   mask bug: what it was, **how it was found**, and what fixing it changed.
> - **`LOSS_BUG_ANALYSIS.md`** — the correctness analysis of the bug on real CLIC data
>   (loss deltas, gradient cosine, batch-size dependence, `glow_jet_iqr` verdict).
> - **`README.md`** — the original study plan and phase structure.
> - `../training_runs_report.md` — the broader hardware report this study feeds.


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

## DECISION 2026-07-31: both remaining speed options ON HOLD

Per-layer pipelining and the GPU LAP solver are **not being attempted**. −31.2%
(4.43 → 3.05 s/step, ≈ +45% throughput) with the physics unchanged is being taken
as sufficient; neither will be revisited until real training runs show step time is
actually a constraint. Fix 5 (thread count) is closed permanently. The loss kernels
are now the biggest GPU cost, but the effective attacks on them (bf16, matched pairs
only) are modelling changes that this study ruled out of scope — no approximations in
the computations. A deliberate stop, with the analysis for each option recorded above
so nobody has to re-derive it.

**Study conclusion:** not data-bound; not model-compute-bound. The step is
dominated by the loss/matcher pipeline: memory-bound loss reductions + pageable
DtoH cost-matrix transfer + CPU Hungarian stall. This explains the B200≈1.76×L4
result — only ~5% of the step scales with GPU FLOPS. Phase 3 (nsys) not needed.
Full write-up + fix candidates: "Profiling results" section in
[`../training_runs_report.md`](../training_runs_report.md).

## Head-to-head rerun (B200 vs 3× L4) with all fixes — submitted 2026-07-31 13:34 EDT

**Purpose.** Every number in the fix experiments above comes from one hardware
configuration (1× B200, batch 2048), which is exactly the configuration the fixes were
designed against. Two questions the study never answered:

1. **Does one B200 now beat three L4s?** The whole premise of this study came from
   June's embarrassment — 3× L4 (1.81 it/s) out-ran 1× B200 (0.40 it/s) by ~1.7× in
   wall clock. −31.2% on the B200 arm is ≈ +45% throughput, which would put it at
   roughly 0.58 it/s ⇒ ~1180 samples/s vs the L4 trio's ~1393. So on paper the gap
   narrows from ~1.7× to ~1.2× but does **not** close. That prediction assumes the L4
   config gains nothing, which is the second question.
2. **Does −31.2% hold in the production L4 config?** Almost certainly **not**, and this
   is the point to keep in view when reading the result: at **batch 256/GPU the cost
   matrices are 8× smaller per GPU** (256 × 150 × 150 × 5 × 4 B = 115 MB, not 921.6 MB).
   Every one of the four fixes scales with that tensor — the pinned DtoH staging, the
   two eliminated host-side numpy copies, the target crop, and the parallel solve. An 8×
   smaller transfer and an 8× smaller solve are a much smaller share of an L4 step, so
   the L4 arm is expected to improve **much less than 31%**, possibly barely at all.
   **The interesting quantity is therefore the B200-vs-L4 gap, not the L4's own
   speedup.** If the L4 arm also gained ~30% the honest reading would be that something
   other than the matcher moved, and the single-variable claim above would need
   re-examining.

**Protocol.** Both arms run the *same* commit of branch `matcher-perf`
(`6af8682`, the current HEAD — all four fixes plus `default_solver: lap1015_late`),
the same SimpleProfiler overlay, and the June production SLURM geometry for their
hardware, so each arm is directly comparable to its own June baseline.

| | **arm A — 1× B200** | **arm B — 3× L4, 1 node** |
|---|---|---|
| SLURM job | **38461125** | **38461126** |
| script | [`submit_rerun_b200_1gpu.sh`](submit_rerun_b200_1gpu.sh) | [`submit_rerun_l4_3gpu.sh`](submit_rerun_l4_3gpu.sh) |
| configs | `base.yaml` + `profile.yaml` | `base.yaml` + `profile.yaml` + `profile_l4.yaml` |
| partition / GRES | `hpg-b200`, `gpu:b200:1` | `hpg-turin`, `gpu:l4:3` |
| tasks × CPU / mem / time | 1 × 16 / 60 G / 1 h | 3 × 16 / 150 G / 1 h |
| batch/GPU (global) | 2048 (2048) | 256 (768) |
| `max_steps` | 200 | 800 |
| June reference job | 33954039 | 33954040 |

The only three parameters that differ between the two config stacks are `devices`,
`batch_size` and `max_steps` — verified by diffing the two `--print_config` dumps on
the login node before submitting (they differ in exactly those three lines). The L4 arm
gets 800 steps rather than 200 because its steps are ~8× cheaper, so 200 steps would be
mostly `torch.compile` warmup; 800 steps ≈ 7 min at June's 1.81 it/s and leaves a long
steady-state tail. `profile_l4.yaml` is a new 2-key overlay and touches no run config.

**Numbers to compare against.**

| | June 2026 baseline | July post-fix (B200 only) | this rerun |
|---|---|---|---|
| 1× B200 | 0.40 it/s (~818 samples/s) | 3.05 s/step `run_training_batch`, steady state ~2.29 s/step (job 38451000) | job 38461125 |
| 3× L4 | 1.81 it/s (~1393 samples/s) | never measured | job 38461126 |

Read from each log: the SimpleProfiler `run_training_batch` / `training_step` /
`backward` per-step rows (`backward` is the control — it should be unchanged from June,
since no fix touched it) and the steady-state it/s from the tqdm timestamps after the
compile warmup. For the L4 arm the profiler table is per-rank; rank 0's is the one
printed. Convert to samples/s as `it/s × global batch` before comparing hardware.

**lap1015 GIL-release verification (precondition, checked before submitting).**
`default_solver: lap1015_late` is only a win with the rebuilt extension; the stock build
is ~2× *slower* than scipy. Status in this environment:

- The rebuilt `.so` **is in place** in the env the jobs use:
  `.pixi/envs/default/lib/python3.12/site-packages/lap1015/_core.cpython-312-x86_64-linux-gnu.so`
  (264,560 B, 2026-07-28), with the stock build preserved beside it as `.so.bak`
  (219,848 B). `pixi run` resolves to the `default` env — confirmed, the interpreter is
  `.pixi/envs/default/bin/python`. (Note the **`clic` env still has the stock `.so`**;
  nothing here uses it, but a job that ran `pixi run -e clic` would silently lose the fix.)
- **It genuinely releases the GIL — verified behaviourally.** The ThreadPool wall-time
  test from fix experiment 3 is **not runnable on this login node**: the node is
  cgroup-limited to a *single* core (`Cpus_allowed_list: 71`), so 4 concurrent solves
  take 3.93× a single solve no matter what the GIL does. Substituted an equivalent
  single-core test: spin a pure-Python counter thread while the solve runs in the main
  thread, and compare its tick rate against a baseline where the main thread merely
  sleeps. A held GIL starves the counter; a released one lets it timeshare.
  Result: **lap1015 53.2% of baseline, scipy (positive control) 53.5%** — the ~50%
  signature of two threads sharing one core, identical for both. Verdict: GIL released.
- **The build-time flag reports `False`** — `lap1015.releases_gil` is `False`, so the
  `RuntimeWarning` added in 97b10db **will fire in both job logs**. This is the known
  false alarm called out in that commit message: the flag was added *after* this `.so`
  was built, and the flag is compiled in, so a correct-but-older build cannot advertise
  it. **Ignore that warning in these logs** — it is contradicted by the direct
  measurement above. It will stop firing once the env is reinstalled from source.
- No fallback to `default_solver: scipy` was needed, so both arms measure the promoted
  configuration as intended.

Deliberately not waited on: queue times on `hpg-b200` have ranged from minutes to days.
Both jobs were `PENDING` at submission (`38461125` on Priority, `38461126` unqueued
reason). Results go in a follow-up section here and in the report's cumulative table.

## Loss-kernel efficiency benchmark (H1 dynamic=True / H2 boolean-mask sync) — job 38463563, 2026-07-31

Follow-up to the lead recorded at the end of `PROFILING_EXPLAINED.md` §10.4: four Triton
loss kernels are 67.8% of GPU-busy time in the post-fix trace, and `mask_bce_loss`'s
forward appeared to move bytes ~400× less efficiently than the dice-cost `bmm` measured
in the same trace. Script: [`bench_loss_kernels.py`](bench_loss_kernels.py),
submitted by [`submit_bench_loss.sh`](submit_bench_loss.sh). Log archived as
`profile_logs/bench_loss_slurm-38463563.out`, raw numbers in
`profile_logs/bench_loss_38463563.json`. 1× B200 (c1004a-s5), 2 min 48 s, torch 2.9.1+cu128.

### Hypotheses tested

- **H1** — `torch.compile(fn, dynamic=True)` (`loss.py:350-370`) stops Inductor
  specialising on the fixed 160-long reduction axis and makes it emit a bad kernel.
- **H2** — the boolean-mask indexing at the head of each mask loss
  (`pred_logits = pred_logits[object_valid_mask]`) has a data-dependent output shape,
  forcing `nonzero()` + a device→host sync that the profiler mis-attributes to the kernel.

**Both are real effects, and neither is the cause.** The cause is a third thing that
neither hypothesis anticipated, found by measurement: an **accidental broadcast that
inflates the work by a factor of `batch_size` (2048×)** — and it is a *correctness* bug,
not only a performance bug.

### Method

Synthetic inputs shaped exactly like one CLIC decoder layer's mask-task output:
`pred_logits [2048, 150, 160]` bf16 under `torch.autocast(bf16)` (matching
`precision: bf16-mixed`), padded constituents overwritten with `finfo.min` as
`task.py:562` does, `sample_weight = target + 1.0*(1-target)` as `task.py:617` does, and
`object_valid_mask` tuned to `N_valid = 102 448`. That target is not a guess: the
post-fix trace's `triton_red_fused_sum_0` kernel (which is `targets.sum(-1)` inside
`mask_dice_loss`, one output per surviving object) launches grids of 100 580 / 100 999 /
102 312 blocks on the three profiled steps — i.e. ~⅓ of the 2048×150 = 307 200 object
slots survive, ~50 real particles per event. Timing: `torch.cuda.synchronize()` around
each call, warmup excluded and reported separately, median of 8 iterations; backward
reported as (fwd+bwd) − fwd.

### Results — `mask_bce_loss`, [2048, 150, 160] bf16

| variant | fwd ms | bwd ms | fwd+bwd ms | warmup s | value |
|---|---|---|---|---|---|
| eager | 184.06 | 140.94 | 325.01 | 1.6 | 1.58053 |
| **compile `dynamic=True` (PRODUCTION)** | **88.73** | **29.83** | **118.54** | 17.3 | 1.58053 |
| compile `dynamic=False` | 58.81 | 11.72 | 70.53 | 9.8 | 1.58053 |
| compile `dynamic=None` (auto) | 58.81 | 11.72 | 70.52 | 0.9 | 1.58053 |
| eager, `object_valid_mask=None` † | 1.15 | 0.59 | 1.75 | 0.0 | 1.74979 |
| compile `dyn=True`, mask `None` † | 0.20 | 0.34 | 0.56 | 4.3 | 1.74979 |
| compile `dyn=False`, mask `None` † | 0.20 | 0.46 | 0.68 | 0.3 | 1.74979 |
| mulmask ‡ eager | 1.17 | 0.56 | 1.73 | 0.0 | 1.75101 |
| mulmask ‡ compile `dynamic=True` | 0.21 | 0.66 | 0.88 | 1.6 | 1.75101 |
| **mulmask ‡ compile `dynamic=False`** | **0.20** | **0.49** | **0.71** | 0.3 | 1.75101 |
| mulmask ‡ compile `dyn=False` max-autotune | 0.20 | 0.47 | 0.69 | 14.0 | 1.75101 |

### Results — `mask_dice_loss`, same shapes

| variant | fwd ms | bwd ms | fwd+bwd ms | warmup s | value |
|---|---|---|---|---|---|
| eager | **OOM — "tried to allocate 62.53 GiB"** | | | | |
| **compile `dynamic=True` (PRODUCTION)** | **51.00** | **48.21** | **100.55** | 17.3 | 0.89594 |
| compile `dynamic=False` | 31.70 | 23.21 | 56.19 | 9.8 | 0.89594 |
| compile `dynamic=None` (auto) | 31.70 | 23.18 | 56.15 | 0.7 | 0.89594 |
| eager, `object_valid_mask=None` † | 0.55 | 0.51 | 1.06 | 0.0 | 0.89617 |
| compile `dyn=True`, mask `None` † | 0.15 | 0.50 | 0.67 | 3.2 | 0.89617 |
| compile `dyn=False`, mask `None` † | 0.15 | 0.67 | 0.84 | 0.2 | 0.89617 |
| mulmask ‡ eager | 0.57 | 0.51 | 1.09 | 0.0 | 0.89623 |
| mulmask ‡ compile `dynamic=True` | 0.15 | 0.67 | 0.83 | 1.4 | 0.89623 |
| **mulmask ‡ compile `dynamic=False`** | **0.16** | **0.70** | **0.87** | 0.2 | 0.89623 |
| mulmask ‡ compile `dyn=False` max-autotune | 0.15 | 0.50 | 0.67 | 7.5 | 0.89623 |

† `object_valid_mask=None` **changes the computed number** (it averages over all 307 200
object slots, not the 102 448 valid ones). Included purely to attribute *time*.
‡ "mulmask" = a rewrite that keeps the `[B, N, C]` rank and applies `object_valid_mask` as
a multiplicative weight with explicit normalisation. It is mathematically equivalent to
what the docstrings *say* the functions compute — which, per below, is **not** what they
currently compute.

The production forward times reproduce the trace almost exactly (88.73 vs 86.7 ms for BCE,
51.00 vs 51.3 ms for dice), so the microbenchmark is measuring the same thing the profiled
training step was measuring.

### What is actually wrong: a 2048× accidental broadcast

`mask_bce_loss` (`loss.py:211-220`) does, in order:

```python
pred_logits = pred_logits[object_valid_mask]    # [2048, 150, 160] -> [N_valid, 160]   RANK 3 -> 2
loss = F.binary_cross_entropy_with_logits(...)  # [N_valid, 160]
loss = loss * input_pad_mask.unsqueeze(1)       # [N_valid, 160] * [2048, 1, 160]
```

Boolean-mask indexing with a 2-D mask **collapses the batch and object axes into one**.
The pad mask, still `[2048, 1, 160]`, therefore no longer broadcasts against the batch
axis — it broadcasts against the *flattened object* axis. The product is
**`[2048, N_valid, 160]`**: every surviving object is paired with every event's pad mask.
The subsequent `.sum(-1)`, `/ valid_counts` and `.mean()` all ride on that shape, so the
reduction does 2048× the intended element visits.

Four independent confirmations, all from this job or the existing trace:

1. **The OOM message is the exact number.** Eager `mask_dice_loss` fails with *"Tried to
   allocate 62.53 GiB"*, and `2048 × 102 448 × 160 × 2 B (bf16) = 62.5293 GiB`.
2. **The trace's grid arithmetic is exact.** The forward BCE kernel launches
   3 218 560 / 3 231 968 / 3 273 984 blocks on the three profiled steps, and
   `2048 × N_valid / 64` for `N_valid ∈ {100 580, 100 999, 102 312}` gives
   3 218 560 / 3 231 968 / 3 273 984 — i.e. `xnumel = batch × N_valid`, XBLOCK 64. (The
   backward grids match `N_valid × 160 / 64` for two of the three steps; the third,
   252 143, does not divide cleanly and is presumably a different autotuned XBLOCK.)
3. **The value is wrong**, on this synthetic batch by ~11% for BCE (1.58053 vs 1.75101)
   and ~0.03% for dice (0.89594 vs 0.89623).
4. **Reproduced at toy scale** on the login node with `B, N, C = 3, 4, 5`: the internal
   product is `torch.Size([3, 6, 5])` where `[6, 5]` was intended, and
   `mask_bce_loss` returns 0.815040 where per-event masking gives 0.838600.

What the buggy loss computes, in words: instead of masking each object with *its own
event's* pad mask and normalising by *its own event's* hit count, it masks every object
with *every* event's pad mask and averages. Because padded constituents already carry
`logit = finfo.min` (so their BCE is 0 regardless), the damage is not "padding leaks in"
but a **positional reweighting of the constituent axis** plus a wrong normaliser — a
smeared, batch-composition-dependent loss.

**The kernel itself is fine.** Counting the work it is actually forced to do —
`2048 × 102 448 × 160 = 3.36e10` element visits × 3 bf16 operands = 201 GB of operand
traffic in 88.73 ms — it sustains ~2.3 TB/s of element throughput, ~35% of this GPU's
measured copy ceiling. Inductor generated a reasonable kernel for an unreasonable tensor.

**Same bug, same shape, in four functions** — every mask loss that boolean-indexes and
then uses `input_pad_mask.unsqueeze(1)`:
`mask_dice_loss` (`loss.py:77-83`), `mask_focal_loss` (`loss.py:145-156`),
`mask_bce_loss` (`loss.py:211-220`), `mask_kl_div_loss` (`loss.py:287-293`).
It fires only when **both** `object_valid_mask` and `input_pad_mask` are passed —
which `ObjectHitMaskTask.loss` (`task.py:600-624`) always does. Every experiment using
that task is affected, not just CLIC.

### Bandwidth: confirming/correcting the "400×" claim

- **Measured ceiling on this GPU**: a 2 GiB device-to-device copy (2 GiB read + 2 GiB
  written) sustains **6.54 TB/s**. B200 HBM3e datasheet peak is ~8 TB/s, and a d2d copy
  typically reaches 75–85% of peak, so 6.54 TB/s is the right practical ceiling to compare
  against. (§10.3's 2.9 TB/s for the dice-cost `bmm` was therefore ~44% of achievable,
  not "near the hardware limit".)
- **Production `mask_bce_loss` forward, useful-work accounting**: the reduction only
  *needs* to read 3 operands over 102 448 rows × 160 = **98.8 MB**. 98.8 MB / 88.73 ms =
  **1.11 GB/s = 0.017% of the 6.54 TB/s ceiling**. So the shortfall is worse than the
  ~400× quoted in §10.4 — nearer **5900×** — but it is *entirely* work amplification,
  not code generation.
- **After the rewrite**: BCE forward reads the full 3 × `[2048,150,160]` bf16 = 294.9 MB
  in 0.20 ms = **1.47 TB/s (22% of ceiling)**; dice reads 196.6 MB in 0.15 ms =
  **1.31 TB/s (20%)**. Ordinary, healthy memory-bound kernels.

### Verdicts

**H1 — `dynamic=True` is a real but second-order cost, and flipping it is NOT a usable
fix on its own.** `dynamic=False` is 1.51× faster forward for BCE (88.73 → 58.81 ms) and
1.61× for dice (51.00 → 31.70 ms), with the same value. But probe C measured what happens
when `N_valid` varies from step to step, which it does in real training (the trace's own
three steps give 100 580 / 100 999 / 102 312):

| variant | per-batch wall over 8 batches with different `N_valid` (ms) | unique graphs |
|---|---|---|
| `mask_bce` `dynamic=True` (production) | 85.2, 86.8, 87.4, 87.4, 89.2, 89.9, 90.3, 91.5 | 4 |
| `mask_bce` `dynamic=False` | 57.3, **3839, 3615, 3888, 4152, 3684, 3974, 3992** | **25** |
| `mask_bce` mulmask `dynamic=False` | 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 | 1 |

`dynamic=False` recompiles on **every** step and costs ~3.9 s of compilation per step —
catastrophically worse end to end. `dynamic=True` at `loss.py:350-370` is the correct
setting *given the boolean indexing*. It only becomes safe to drop once the
data-dependent shape is gone, at which point it no longer matters (mulmask is 0.20 ms
either way).

**H2 — the mechanism named in the hypothesis is confirmed to exist but is not where the
time goes.** `torch.cuda.set_sync_debug_mode("error")` raises inside `mask_bce_loss`
(3 syncs — one per indexed tensor) and inside `mask_dice_loss` (2 syncs) when
`object_valid_mask` is set, and raises nothing when it is `None` or when the mulmask
rewrite is used. Dynamo also breaks the graph three times, resuming at `loss.py:212`,
`213`, `214` — exactly the three indexing lines — so the "compiled" loss is 4 separate
graphs. But the sync costs ~1 ms against an 88.73 ms call: the boolean indexing matters
because of the **rank collapse it causes**, not because of the sync it forces. Anyone
reading the profiler's 4.1 s of `cudaStreamSynchronize` as wasted time would be wrong;
that is the host waiting for the (2048× oversized) kernel it just launched.

### Recommended production change — NOT made here

In `src/hepattn/models/loss.py`, in all four affected functions, stop boolean-indexing and
keep the `[B, N, C]` rank; apply `object_valid_mask` as a multiplicative weight with
explicit normalisation. For `mask_bce_loss` (`loss.py:198-228`) that is:

```python
loss = F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight, reduction="none")
if input_pad_mask is not None:
    loss = loss * input_pad_mask.unsqueeze(1)                        # now correctly [B,1,C] vs [B,N,C]
    loss = loss.sum(-1) / input_pad_mask.sum(-1, keepdim=True).clamp_min(1.0)
else:
    loss = loss.mean(-1)
if object_valid_mask is None:
    return loss.mean()
w = object_valid_mask.to(loss.dtype)
return (loss * w).sum() / w.sum().clamp_min(1.0)
```

**Exactness status — this is a BUG FIX, not a numerically-exact optimisation.** It changes
the value the model trains on (1.58053 → 1.75101 for BCE on the synthetic batch). It is
exact only with respect to the functions' documented intent. It also removes the sync and
the graph breaks, so `dynamic=True` becomes optional rather than load-bearing.

A strictly minimal alternative that keeps the current style and fixes only the broadcast is
to index the pad mask alongside everything else —
`ev = torch.arange(B, device=d).unsqueeze(1).expand(B, N)[object_valid_mask]; pad = input_pad_mask[ev]`
then use `pad` with no `unsqueeze`. Same value as the form above, but it keeps the
`nonzero()` sync and the graph breaks, so it is worse. Recorded only because it isolates
the broadcast from everything else if someone wants to A/B the two effects.

**Expected impact if promoted.** 5 supervised decoder outputs per step × (BCE 118.54 +
dice 100.55) ms = **1.10 s/step** today, → 5 × (0.71 + 0.87) = **7.9 ms/step**. That is a
~1.09 s/step saving, which agrees closely with the trace's own 3.154 s of these four
kernels over 3 profiled steps (1.05 s/step) and is set against a steady state of ~2.29
s/step (job 38451000). It would be by far the largest single win found in this study.

**Validation required before promotion** (do not skip — the value changes):
1. Unit tests in `tests/` comparing each rewritten loss against a plain per-event Python
   reference loop, on small shapes, with and without each mask.
2. NaN check: the rewrite evaluates BCE on invalid object rows before zeroing them, so a
   row with an infinite logit would give `0 * inf = NaN`. Padded constituents carry
   `finfo.min`, whose BCE is finite (0), so this is safe as written — but it must be
   asserted, not assumed.
3. A short training run comparing loss curves against the current code. Existing CLIC
   checkpoints were trained with the buggy loss and their metrics are **not** comparable
   across this change.
4. The other experiments using `ObjectHitMaskTask` (trackml, itk, tide, pixel,
   atlas_muon, cld, colliderml) are affected identically and need the same check.
5. Flagged for the separate `glow_jet_iqr` investigation: a mask loss whose effective
   per-constituent weighting depends on the *batch composition* is exactly the kind of
   thing that could interact with a metric drifting over training. Not investigated here.

> **Correctness follow-up (2026-07-31):** the physics impact of this bug — buggy-vs-intended
> loss values, gradients, batch-size dependence and the glow_jet_iqr connection, all measured
> on real CLIC data — is analysed in [`LOSS_BUG_ANALYSIS.md`](LOSS_BUG_ANALYSIS.md).

### What is proven vs what is inferred

**Proven by direct measurement in this job (or by exact arithmetic on the existing trace):**

- The internal tensor in the masked path is `[batch, N_valid, num_constituents]`. Proven
  three ways: the eager OOM figure matches to five significant figures; the trace's
  forward grid sizes equal `batch × N_valid / 64` for all three profiled steps; and the
  toy-scale reproduction prints the shape directly.
- The four production kernel times reproduce the trace (88.73 vs 86.7; 51.00 vs 51.3 ms).
- Removing the broadcast makes the forward 444× (BCE) and 340× (dice) faster, and changes
  the returned value.
- A device→host sync fires inside the masked path and does not fire without the mask.
- `dynamic=False` recompiles every step when `N_valid` moves, at ~3.9 s per recompile.
- This GPU's achievable copy bandwidth is 6.54 TB/s.

**Inferred, not measured:**

- The ~1.09 s/step end-to-end saving. It is `per-call × 5 calls`, cross-checked against
  the trace's kernel sum, but **no end-to-end training run has been done with the fix**.
  If some other stage is partly overlapping these kernels the realised saving is smaller.
- That the rewrite is what the authors intended. It matches the docstrings and the toy
  reference, but the intent is not documented anywhere authoritative.
- That fixing this does not change model quality. It certainly changes the loss; whether
  the corrected loss trains better, worse, or the same is unknown and untested.

**Confounds I could not eliminate** (recorded in the spirit of fix experiment 5's wrong
NUMA explanation and §10.4's unchecked premise):

1. **Synthetic data.** Valid-object counts and hit occupancy are modelled, not read from
   the CLIC ROOT files; `N_valid = 102 448` was tuned to match the trace, and the
   per-event distribution is Gaussian rather than real. Timing depends only on shapes, so
   this is weak for the *time* results — but the specific loss *values* (1.58 vs 1.75) are
   illustrative of the size of the error, not the production loss values.
2. **Isolated microbenchmark.** One kernel at a time, no contention with the rest of the
   model, no NCCL, one GPU. Step-level extrapolation assumes serialisation.
3. **Backward times are a difference** ((fwd+bwd) − fwd), so they carry both measurements'
   noise.
4. **`warmup s` is not a clean compile time** — the first compiled variant of each
   function absorbs one-time Inductor/Triton startup (hence 17.3 s for the first row and
   0.9 s for an identical later one).
5. **The `fwd GB/s` column in the raw log is wrong for the mulmask rows**: the script fed
   it the masked-row byte count (`N_valid` rows) although mulmask reads all `B × N` rows,
   understating those rows by 3×. The corrected figures are in the bandwidth section
   above; the log column is left as produced.
6. **`max-autotune` was only tried on the already-fast rewritten form**, where it showed
   no benefit. It was not tried on the broken form, where it would not have helped anyway.
7. **No end-to-end run.** Everything above is per-kernel. The next step, if this is
   picked up, is a 200-step run with the rewrite in place against job 38451000's baseline.

## Head-to-head results + variance controls (jobs 38461125/38461126/38463188/38463189, 2026-07-31)

Answers the question the head-to-head was submitted to answer, and adds two B200 control
runs because the first result did not reproduce the July figure.

### Results — steady state (steps 50→200 from tqdm timestamps, warmup excluded)

| job | arm | protocol | node | s/step | samples/s |
|---|---|---|---|---|---|
| 38451000 (July) | 1× B200 | profiler on | — | 2.29 | 894 |
| 38461125 | 1× B200 | profiler on | c0904a-s25 | 2.90 | 706 |
| 38463188 | 1× B200 | profiler on | c1001a-s25 | 2.66 | 770 |
| 38463189 | 1× B200 | **profiler off** | c1003a-s15 | 2.14 (1.97 over 200→300) | 957 (1040) |
| 38461126 | 3× L4 | profiler on | c0606a-s20 | 0.517 (0.65 avg) | **1485** |

June production reference: B200 0.40 it/s = 2.47 s/step = 819 samples/s;
3× L4 1.81 it/s = 1390 samples/s.

### PROVEN

1. **3× L4 still beats 1× B200.** Best observed B200 (1040 samples/s, profiler off) vs
   3× L4 (1485 samples/s, profiler *on*, so its true figure is higher). Gap ≥ 1.43×,
   narrowed from June's 1.70× but not closed. The study's goal of making the B200 the
   better buy for this workload was not achieved.
2. **The L4 arm gained ~7%** (1.93 vs 1.81 it/s), as predicted: at batch 256/GPU the cost
   matrices are 115 MB not 921.6 MB, so the matcher fixes have little to bite on.
3. **B200 run-to-run variance on identical code is ~27%** (2.29 / 2.66 / 2.90 s/step,
   three runs, same commit, same config, three different nodes).
4. **`backward` is flat across every run** (0.628 / 0.630 / 0.632 s). The variance is
   entirely host-side, inside `training_step`.

### The methodological problem this exposes

`backward` is a good control for GPU compute but **does not control for host-side
variance — which is exactly what every fix in this study targeted.** The per-fix
increments were each measured once, on different nodes, at different times:
−11.6% (fix 2), −10% (fix 3), −13.7% (fix 4). All three are at or below the 27%
run-to-run spread now measured. The cumulative direction survives (4.43 → ~3.0-3.4 s/step
avg is larger than the spread), but **the individual attributions are not supported by
n=1 measurements** and should not be quoted as precise.

Same caveat retroactively weakens fix 5's "22% slower at 32 threads" end-to-end number,
though that one has independent support from the isolated thread benchmark (job 38458388).

### UNRESOLVED

- **Profiler overhead vs node variance cannot be separated at n=1.** 38463189 (profiler
  off) is the fastest run observed, but its 2.14 s/step is only 7% below the best
  profiler-on run (2.29), well inside the spread. `profiler: simple` should be cheap
  (`time.monotonic()` around actions), so a 0.5 s/step cost is implausible — node
  variance is the more likely explanation for most of the difference. Not proven either way.
- **Absolute gain vs June is +17% to +27%** (957–1040 vs 819 samples/s), not the +45%
  the report claims. The +45% was measured against this study's own protocol baseline
  (4.43 s/step avg / ~3.7 s/step steady), which was itself far slower than June's
  production run (2.47 s/step). The percentages are internally consistent; they do not
  transfer to production.

### Recommendation for any future measurement in this study

Repeats, not single runs: ≥3 runs per configuration, and report the spread. Prefer
profiler off with tqdm-derived steady state (the profiler's own numbers agree with tqdm
anyway). Note that the loss-broadcast bug found in job 38463563 changes the denominator
for every percentage above — re-baseline after fixing it rather than trusting the
progression table.

Artifacts: `configs/profile_noprof.yaml`, `submit_noprof_b200_1gpu.sh`; logs in
`../../../slurm_logs/slurm-384631{88,89}.*.out` and `slurm-38461125/6.*.out`.

## ~~RESUME HERE~~ (2026-07-31) — SUPERSEDED, kept for the record

> **This was the resume point for the 2026-07-31 session. Its "NEXT STEPS" are now done
> (steps 1 and 2) or in flight (step 3) — see the START HERE block at the top of this file
> and the RESULTS sections below. Retained because it documents what was built and why.**

### What was done (2026-07-31)

### What was done

The mask-loss broadcasting bug (see [`LOSS_BUG_ANALYSIS.md`](LOSS_BUG_ANALYSIS.md)) has been
fixed **as an opt-in alternative, not as an in-place edit**, so the published/paper behaviour
remains the default and stays runnable.

- `src/hepattn/models/loss.py`: added `mask_bce_loss_v2` and `mask_dice_loss_v2` (plus the
  shared helper `_mean_over_valid_objects`), registered in `loss_fns` as **`mask_bce_v2`** and
  **`mask_dice_v2`**. The legacy `mask_bce_loss` / `mask_dice_loss` are **untouched** and remain
  the default everywhere. `mask_focal_loss` and `mask_kl_div_loss` share the same bug but were
  left alone — no CLIC config uses them.
- `configs/clic_v6_maskfix.yaml` — copy of `base.yaml`, **3 lines differ**: run name + the two
  loss keys. `configs/clic_v7_maskfix.yaml` — same edit against `clic_v7.yaml` (not submitted,
  provided for convenience).
- The `costs:` blocks are deliberately **unchanged** — the cost/matching path never had the bug.

**To revert to the paper behaviour: run `base.yaml` instead of `clic_v6_maskfix.yaml`.**
That is the whole revert. No code needs touching.

### Verification (done, passing)

`verify_mask_loss_v2.py` (in this directory) checks both v2 losses against an explicit
per-event Python reference loop:

- v2 == reference to **1e-9** in float64, both losses. Legacy differs (the bug).
- **Uniform-padding check**: with every event the same length, legacy == v2 to *exactly* 0.0 —
  confirming the bug is precisely the variable-padding effect and v2 is a strict generalisation.
- No-mask path identical to legacy.

Run it with `pixi run python studies/b200_utilization/profiling/verify_mask_loss_v2.py`
from the clic experiment dir.

### Jobs

- **38468860** — smoke test, 1× B200, 25 steps, `clic_v6_maskfix.yaml` at batch 256. Purpose:
  confirm the new loss keys resolve and the run is stable end-to-end before a long job.
- **Full training** — 3× L4, `clic_v6_maskfix.yaml` at batch 256/GPU, matching the geometry of
  the June 3×L4 v6 run (job 33954040), which is the **plotted baseline model** in the
  `glow_jet_iqr` study (`logs/clic_v6_20260605-T113014`, best ckpt epoch 195). That makes the
  comparison a clean A/B: same config, same hardware, same batch — only the loss differs.
  **Job 38469247**, submitted 2026-07-31, `submit_training_maskfix_l4.sh`, 168 h limit,
  200 epochs. Smoke test 38468860 passed first (`max_steps=25 reached`, validation ran,
  no errors) before this was submitted.

### NEXT STEPS when you return

1. **Check the training finished** and compare its loss curves to a v6 baseline run. Note the
   metric keys are now `mask_bce_v2` / `mask_dice_v2`, so plotting scripts keyed on the old
   names need updating. Absolute loss values are **not** comparable across the fix (the
   objective changed by construction) — compare *shapes* and the physics metrics, not values.
2. **Evaluate the jet-E IQR** on the new checkpoint the same way `glow_jet_iqr` does, and
   compare against the v6 baseline. This is the open question: the bug provably changed the
   training signal (gradient cosine 0.90), but whether it moves reconstruction quality is
   **unmeasured**.
3. **Re-baseline the performance numbers.** Every percentage in this file was measured against
   the buggy loss, which computed a 2048× oversized tensor. The fix should cut roughly half the
   step time on its own. Re-run the head-to-head (`submit_rerun_b200_1gpu.sh`,
   `submit_rerun_l4_3gpu.sh`) with the maskfix config — and use **≥3 repeats**, since
   run-to-run variance on B200 is ~27%.
4. **Decide whether to fix `mask_focal_loss` / `mask_kl_div_loss`** for the other experiments
   (trackml, itk, tide, cld, colliderml, atlas_muon all use `ObjectHitMaskTask` and are
   affected identically).

### Answer to "could my IQR difference be due to my different choice of batch size?"

**No — your own study already ruled this out, twice over.**

- `00_cross_run/jet_iqr_discrepancy.md` (recapped in `glow_jet_iqr/README.md`): the high-E IQR
  climb is *"universal across all four v6 runs (3×L4, 1×B200, 6×L4, 4×B200) **regardless of
  hardware / batch size / val_loss**"*. Those runs span per-GPU batch 256 → 2048 and all rise.
- The paper-tag retrain used batch **170** — *inside* that range — and its IQR **falls**. So the
  variable that flipped the trend was the code version, not the batch size.
- Independently, the loss-bug analysis measured the bug's own batch-size sensitivity as
  negligible (−11.16% / −11.30% / −11.50% at B=32/256/2048), so batch composition cannot
  rescue a batch-size explanation through that route either.

Batch size is excluded. The refactor remains the cause, and the bisect (task 5 follow-up in
`glow_jet_iqr`) is still the live thread there — note this loss bug is **not** the culprit, as
it is present on both sides of that comparison.

## Mask-fix training A/B — RESULTS (2026-08-03)

Closes steps 1 and 2 of the previous section. Training job **38469247 COMPLETED** on
2026-08-01 23:20 — full 200 epochs (`max_epochs=200 reached`), 31 h 55 m, 200 checkpoints,
best `epoch=199-val_loss=3.36249`. Val loss was still improving at the end.

### Speed: the fix is worth ~25% throughput in the production L4 config

Same geometry, same node family (`hpg-turin`, 3× L4, batch 256/GPU, 200 epochs, 1295
steps/epoch), so this is a like-for-like wall-clock comparison against the June v6 run.

| | June v6 baseline (33954040) | maskfix (38469247) | Δ |
|---|---|---|---|
| wall clock, 200 epochs | 39 h 46 m | **31 h 55 m** | **−19.8%** |
| steady-state | 1.81 it/s (11:55/epoch) | **2.27 it/s (9:31/epoch)** | **+25.4%** |
| node | c0609a-s13 | c0609a-s5 | — |

Decomposition against the head-to-head rerun (job 38461126, matcher fixes only, same
geometry): 1.81 → 1.93 it/s from the four matcher fixes (+6.6%), 1.93 → 2.27 from the
mask-loss fix (**+17.6%**). So on L4 the mask-loss fix is worth ~2.7× more than every
matcher fix combined — the opposite of the ordering on the B200 protocol, and consistent
with the loss kernels (not the cost-matrix transfer) being the dominant cost at batch 256.

**Caveat:** n=1 per arm, and this study measured ~27% run-to-run variance on B200. The L4
spread is unmeasured. The per-epoch time is averaged over 200 epochs so within-run noise is
negligible; the residual risk is node-to-node variation. The effect is large and the
per-epoch times are stable, but it should not be quoted to better than ~±5%.

### Physics: no significant change to jet-E IQR — the rising trend survives the fix

Eval job **38589352** (1 m 47 s, 1× L4) wrote
`logs/clic_v6_maskfix_20260731-T152547/ckpts/epoch=199-val_loss=3.36249__test.root`.
Analysis: [`plot_maskfix_jet_iqr.py`](plot_maskfix_jet_iqr.py) → `maskfix_jet_iqr.png`,
per-jet residuals cached in `maskfix_jet_residuals.npz` (re-runs skip the ~7 min clustering).
Same pipeline as `glow_jet_iqr`: gen-kt R=0.7, ≤2 jets, pT>10 GeV, dR<0.1, `ind_threshold` 0.65.
Both the `mpflow` and `mpflow_proxy` branches are shown, since the branch choice moves the
IQR by more than the effect being measured.

| IQR of jet-E response | 20–40 GeV | 160–180 GeV | rise |
|---|---|---|---|
| baseline (buggy loss), `mpflow` | 0.0666 | 0.0987 | +0.0322 ± 0.0034 |
| maskfix (v2 loss), `mpflow` | 0.0679 | 0.1040 | +0.0360 ± 0.0037 |
| baseline (buggy loss), `mpflow_proxy` | 0.0654 | 0.0843 | +0.0189 ± 0.0027 |
| maskfix (v2 loss), `mpflow_proxy` | 0.0638 | 0.0794 | +0.0156 ± 0.0024 |

Bootstrap over matched jets (400 resamples, per-bin IQR). maskfix − baseline:

| branch | IQR @ 160–180 | rise |
|---|---|---|
| `mpflow` | +0.0048 ± 0.0043 (z = +1.10) | +0.0035 ± 0.0049 (z = +0.72) |
| `mpflow_proxy` | −0.0052 ± 0.0028 (z = −1.86) | −0.0033 ± 0.0035 (z = −0.94) |

**Verdict: no detectable effect on jet-E IQR.** Every |z| < 2, and the two branches disagree
on the *sign* — so the small differences are branch-choice and sampling noise, not a physics
change. **The rising-IQR trend is untouched: both runs rise ~+0.032 (`mpflow`) while Pandora
falls.** This independently re-confirms the `LOSS_BUG_ANALYSIS.md` conclusion that the mask
bug is not the `glow_jet_iqr` cause — previously argued from "both arms trained on the buggy
loss", now demonstrated directly by training an arm *without* it. **The bisect in
`glow_jet_iqr` (task 5 follow-up) remains the live thread; this rules out one more candidate.**

Note the bootstrap resamples the two models independently even though they are evaluated on
the same events, which makes the difference test *conservative* (a paired test would have
smaller errors). Since the conclusion is "not distinguishable", conservative is the safe
direction — a paired test could only sharpen a null, not overturn it.

### Quality metrics: the fixed model is slightly better on mask quality, equal elsewhere

Both checkpoints re-scored on the **same** val set under the **same** (v2) objective —
jobs **38590337** (baseline ckpt) / **38590338** (maskfix ckpt), via the new
[`submit_validate_run.sh`](../../../submit_validate_run.sh) with `CONFIG=logs/_v2_scoring/config.yaml`.
This removes every confound: one yardstick, one dataset, one code version.

| val metric (v2 objective) | baseline ckpt | maskfix ckpt | Δ |
|---|---|---|---|
| `final_loss` | 1.04402 | **1.03735** | −0.64% |
| `final_mask_mask_bce_v2` | 0.145673 | **0.143084** | −1.78% |
| `final_mask_mask_dice_v2` | 0.131772 | **0.128654** | −2.37% |
| `final_mask_purity` | 0.828284 | **0.836363** | +0.98% |
| `final_mask_exact_match` | 0.565893 | **0.575383** | +1.68% |
| `final_mask_recall` | 0.895602 | 0.896536 | +0.10% |
| `final_eff` | 0.905829 | 0.905474 | −0.04% |
| `final_pur` | 0.905541 | 0.903840 | −0.19% |
| `final_incidence_kl_div` | 0.0094252 | 0.0093168 | −1.15% |
| `final_regression_e_abs_res` | 0.0127884 | 0.0127292 | −0.46% |
| `final_regression_pt_abs_res` | 0.0096216 | 0.0095504 | −0.74% |
| `final_regression_eta_abs_res` | 0.0147716 | 0.0143907 | −2.58% |
| `final_obj_class_accuracy_macro` | 0.851260 | 0.852026 | +0.09% |
| `final_classification_object_ce` | 0.620988 | 0.621078 | +0.01% |

The v2-trained model wins on the v2 objective (expected — it optimised it) and on every mask
metric, is marginally better on the regression residuals and incidence KL, and is a hair worse
on event-level eff/pur (−0.04% / −0.19%, i.e. nothing). **The fix is not harmful and is mildly
beneficial for mask quality; it does not change reconstruction performance in any way that
matters.** ~5,000 val events, so ~1% differences are near the noise floor — the consistent
*direction* across the mask metrics is the signal, not any single number.

> **Retracted:** an earlier version of this comparison used `clic_v6_fp32_20260707-T150453` as
> the reference (the only pre-fix run with a full `csv_metrics` history — the June baseline
> predates the CSVLogger) and reported the regression residuals as 4–6% *worse* under the fix.
> That was the fp32/6×L4/global-batch-1536 confound, not the loss. The confound-free
> re-scoring above reverses the sign. Curves: [`plot_maskfix_curves.py`](plot_maskfix_curves.py)
> → `maskfix_loss_curves.png`, `maskfix_quality_metrics.png`; the fp32 panels are kept as a
> secondary reference but the table above supersedes them.

### Convergence

`maskfix_loss_curves.png`: train and val track each other for all 200 epochs with no
divergence, and the self-normalised val curves of maskfix / baseline / fp32 are
indistinguishable. The fix does not change convergence behaviour or training stability.

### Code change made while doing this

`pflow_data.py` `PflowDataModule.setup` built `val_dset` only for `stage == "fit"`, so
`main.py validate` crashed with `AttributeError: 'PflowDataModule' object has no attribute
'val_dset'` for every CLIC run. Now `stage in {"fit", "validate"}`. Enables re-scoring a
checkpoint on the val set; touches no training path.

### What is still open

1. **Re-baseline the B200 performance numbers** (step 3 of the previous section) — still not
   done. The L4 number above is the first post-fix production measurement; the B200 protocol
   runs (`submit_rerun_b200_1gpu.sh`) have not been re-run with the maskfix config, and every
   percentage earlier in this file predates the fix. Use ≥3 repeats.
2. **Decide whether to fix `mask_focal_loss` / `mask_kl_div_loss`** (step 4) — unchanged; the
   other `ObjectHitMaskTask` experiments are affected identically.
3. **Whether to make v2 the default.** The evidence now says the fix is ~25% faster on L4 and
   physics-neutral-to-mildly-positive. It is still opt-in, and existing CLIC checkpoints/paper
   numbers were trained with the legacy loss.

## B200 re-baseline after the mask fix — submitted 2026-08-03

Everything in this file above the A/B section was measured against the buggy loss, so the
progression table's denominators are all wrong. This re-measures the two things that
actually matter, and deliberately does **not** try to repair the per-fix increments (they
were n=1 below the noise floor; see "The methodological problem this exposes").

**Q1 — does 1× B200 beat 3× L4 now?** The founding question. Unpaired across hardware, so it
needs repeats: report **median and full spread of 3 runs per arm**, never mean±sd of three.

**Q2 — what did the mask fix buy on B200?** Paired within-node, so the ~27% node variance
cancels in the difference.

Why this is worth one round: the bug inflated work by a factor of `batch_size`, and the B200
protocol runs at batch 2048 vs the L4's 256 — so the B200 is the arm with the most to gain,
and this is the first fix in the study that could plausibly flip the ranking. From the
figures already here: the four mask kernels are ~1.09 s/step of the B200's 2.29 s/step
steady state ⇒ predicted ~1.2 s/step ≈ 1700 samples/s, against the L4's newly measured
**1743 samples/s**. That is a coin flip; starting from the profiler-off 1.97 s/step instead
puts the B200 near 2300 and winning outright. Falsifiable either way.

| job | arm | script | node |
|---|---|---|---|
| 38596036 | 1× B200 paired, `ORDER=legacy_first` | [`submit_paired_b200_1gpu.sh`](submit_paired_b200_1gpu.sh) | c0904a-s5 |
| 38596037 | 1× B200 paired, `ORDER=maskfix_first` | same | c0906a-s15 |
| 38596038 | 1× B200 paired, `ORDER=legacy_first` | same | c0910a-s5 |
| 38596039/40/41 | 3× L4, maskfix | [`submit_maskfix_l4_3gpu.sh`](submit_maskfix_l4_3gpu.sh) | c0610a-s13, +2 |

**Design notes.**

- **Paired is possible only because the mask fix is config-only** — the v2 losses are opt-in,
  so the two arms are `base.yaml` vs `clic_v6_maskfix.yaml` with identical code. Verified on
  the login node with `--print_config`: the resolved configs differ in exactly three lines
  (`mask_bce`/`mask_dice` → `mask_bce_v2`/`mask_dice_v2`, plus `name`). No checkout, no rebuild.
- **Profiler off** (`configs/profile_noprof.yaml`, 300 steps, no validation). The `backward`
  control is only meaningful with the profiler on, but a paired design does not need it, and
  the profiler was itself an unseparable confound last round.
- **Separate Inductor/Triton cache per arm** (`/var/tmp/clic_<jobid>_<arm>/`). Without it the
  second arm inherits the first's compiled kernels and looks faster for a reason unrelated to
  the loss — the same leaked-state confound that produced the bogus 32-thread numbers in job
  38458388. **Arm order is alternated** across the three jobs as a second guard.
- The three B200 jobs landed on three different nodes, so the variance is genuinely sampled.
- L4 sets no cache overrides: every 1-node 3× L4 run here has been fine on the defaults, and
  cache warmth only affects compile warmup, which the steady-state window excludes.

**Read from each log:** steady-state it/s from tqdm (steps 50 → end, warmup excluded), the
per-step distribution (median + IQR, not a single number), and the node. Convert before
comparing hardware: samples/s = it/s × global batch (B200 2048, L4 768).

**Decision rule, fixed in advance.** B200 median ≥ L4 median ⇒ the founding question flips to
"yes" and the hardware recommendation changes. B200 still below ⇒ close the study
permanently, recommend L4, and **retire the cumulative percentage table** rather than
re-deriving it — one honest samples/s per hardware is more useful than a corrected
progression measured against a protocol that never matched production.

## Are the fixes safe to keep? An audit of everything this study changed (2026-08-03)

Written because "we changed a lot of code chasing performance, is any of it justified?" is
the right question to ask before committing, and because the evidence quality is **very
uneven** across the changes. Grouped by how well-supported each one is.

### Keep unconditionally — these are correctness fixes, not optimisations

`models/matcher.py`, three changes, all found while trying to make `lap1015_late` work:

1. **Empty cost matrix → identity permutation.** An event with zero valid targets gives a
   `(0, n_pred)` cost matrix, and `lap1015.lap_late` returned *uninitialised memory* for it —
   including negative indices, which is what crashed job 38143939. Solver-agnostic and
   behaviour-identical for scipy.
2. **On-device `nan_to_num` before the DtoH copy.** Real cost matrices contain `-inf`
   (padded-hit logits are `masked_fill`ed before cost computation). scipy treats ±inf as
   forbidden/mandatory assignments; `lap1015.lap_late` has **undefined behaviour** on
   non-finite input — it segfaulted under a fuzz test. Replacing ±inf with ±(float32 max/10)
   is equivalent for an argmin assignment.
3. **Permutation validation + per-event scipy fallback.** If a non-scipy solver returns
   something that is not a permutation, fall back rather than silently corrupting training.
   Fires ~1 event per 200 steps.

**Evidence:** direct equivalence checks (both solvers agree on optimal assignment cost in
float64 over hundreds of events spanning `-inf`, NaN, all-constant, empty and query-masked
degeneracies) plus `tests/matching` (118 passed). These would be worth keeping **at zero
speedup** — they fix undefined behaviour that was reachable from a supported config.

### Keep, but stop quoting the individual numbers

`models/matcher.py`: the pinned DtoH staging buffer (claimed −11.6%) and `_prepare_costs`
device-side transpose/mask/crop (claimed −13.7%).

**These are numerically exact** — bit-identical transfer, assignment-identical prep — so the
downside risk of keeping them is essentially zero even if the speedup is smaller than
claimed. And the speedup direction is not in doubt.

**But each was measured once, on a B200, at or below the ~27% run-to-run variance floor.**
The only clean measurement of this bundle is the L4 head-to-head: **+6.6% (1.81 → 1.93 it/s)**
— and that too is n=1 per arm. So: keep the code, quote "~6.6% on the production L4 config",
and **do not** quote −11.6% / −13.7% / −31.2% as though they were separately established.

### The one that is NOT safe as it stands: `default_solver: lap1015_late`

Fix 3 promoted `default_solver: scipy → lap1015_late` in `base.yaml`. The speedup is real —
but it depends on a binary that **is not reproducibly built**:

- The GIL-release patch lives in `src/lap1015/src/main.cpp`, but the `.so` actually being
  used was **hot-patched by hand** into `.pixi/envs/default/.../lap1015/_core*.so`. The stock
  build is preserved beside it as `_core*.so.bak`.
- `lap1015.releases_gil` reports **`False`** (the flag postdates the build), so the runtime
  guard added in 97b10db fires a `RuntimeWarning` on every run as a **false alarm**. A warning
  that is always wrong is a warning everyone learns to ignore — the guard is currently useless.
- The **`clic` pixi env still has the stock `.so`**. Anything run with `pixi run -e clic` gets
  it. Several plotting/analysis scripts in this repo use `-e clic`.
- `pixi install`, or any reinstall of hepattn, **reverts the patch silently**.

With the stock `.so`, `lap1015_late` is **~1.9× slower than scipy** (7.14 vs 3.76 s/step,
measured). So the current state is a config that is a large win on one machine's hand-patched
environment and a large **pessimisation** everywhere else, with a broken guard.

**Recommended, in order:**

1. **Short term / before sharing this branch:** revert `base.yaml` to `default_solver: scipy`
   and set `lap1015_late` only in configs used on an environment known to have the rebuilt
   extension. Costs ~10% on B200; removes a silent 2× regression for everyone else.
2. **Proper fix:** commit the `main.cpp` change and rebuild the extension from source in the
   env so `releases_gil` reports `True` and the guard becomes meaningful. Then promoting
   `lap1015_late` is safe and the warning does its job.

### Keep — now well-evidenced

- `models/loss.py` **v2 mask losses**: opt-in, legacy untouched, +17.6% on L4, physics-neutral,
  mildly better mask quality. Zero risk by construction — nothing uses them unless a config
  asks. See the A/B RESULTS section.
- `pflow_data.py` **`val_dset` for `stage == "validate"`**: `main.py validate` crashed for every
  CLIC run before this. Touches no training path.

### What the currently-running jobs do and do not settle

Jobs 38596036-38 are paired **only on the mask fix** (`base.yaml` vs `clic_v6_maskfix.yaml`).
They will **not** validate the matcher fixes — those need either old code (pinned buffer,
device prep) or a separate pairing. Note that `default_solver` **is** a config key, so a
paired scipy-vs-lap1015_late run is possible at the same low cost, and given the deployment
hazard above that is the most valuable follow-up measurement available. Not submitted.

## B200 re-baseline — RESULTS (jobs 38596036-41, 2026-08-03)

All six completed clean. Parser: [`parse_throughput.py`](parse_throughput.py) (steady state
from tqdm stamps, steps 50 → end, warmup excluded).

### The paired B200 measurement

| job | node | order | legacy loss | mask-fixed | ratio | saving |
|---|---|---|---|---|---|---|
| 38596036 | c0904a-s5 | legacy first | 0.431 it/s (883 sa/s) | 0.731 it/s (1497 sa/s) | **1.696×** | 0.95 s/step |
| 38596037 | c0906a-s15 | maskfix first | 0.355 it/s (727 sa/s) | 0.548 it/s (1122 sa/s) | **1.544×** | 0.99 s/step |
| 38596038 | c0910a-s5 | legacy first | 0.502 it/s (1028 sa/s) | 0.933 it/s (1911 sa/s) | **1.859×** | 0.92 s/step |

**The mask fix is worth ~1.70× on B200** (median; range 1.54–1.86×), against 1.176× on L4.
That asymmetry is the predicted one and is now measured, not inferred: the bug's cost scales
with batch size, and the B200 protocol runs at batch 2048 vs the L4's 256.

**The paired design earned its keep.** Absolute throughput varied 1.41× (legacy) and 1.70×
(maskfix) across the three nodes, but the *ratio* varied only 1.20×. Node ranking was
identical on both arms — c0910a-s5 fastest on both, c0906a-s15 slowest on both — which is
exactly the common-factor structure pairing is designed to cancel. An unpaired design at
n=3 could not have resolved a 1.7× effect out of a 1.7× spread.

**The inferred saving was roughly right.** NOTES flagged the ~1.09 s/step figure as
extrapolated from microbenchmarks and never measured end to end. Measured: **0.92–0.99
s/step**, i.e. the extrapolation was ~15% optimistic. Good enough to have been worth acting on.

**Legacy arm reproduces history**, which validates the setup: 1.99 / 2.32 / 2.82 s/step here
vs 2.14 / 2.29 / 2.66 / 2.90 s/step recorded previously.

### Q1 — does 1× B200 beat 3× L4 now? Per the pre-registered rule: still NO

| arm | median | spread |
|---|---|---|
| 1× B200, mask-fixed | **1497 samples/s** | 1122–1911 (**1.70×**) |
| 3× L4, mask-fixed | **1735 samples/s** | 1725–1735 (**1.006×**) |

**L4 is 1.16× ahead on the median** — down from 1.43× before the fix, but not overturned.
The decision rule was fixed before the data existed and it says: close the study, recommend
L4, retire the cumulative percentage table.

**But the honest reading is "too close and too noisy to call", and that is itself the finding.**
The B200 spread is enormous — the *best* B200 node (1911 sa/s) beat the L4 median, the worst
(1122) lost by 1.55×. With n=3 spanning 1.70×, the median 1497 carries wide uncertainty,
while the L4's 1735 is pinned to 0.6%. So the two platforms now overlap; which wins depends
on which B200 node you land on. Recommending L4 is still right, but the reason has shifted
from "B200 is slower" to **"B200 is comparable on average and far less predictable"**.

**The B200's variability is the unexplained result now, and it got worse, not better.**
Previously documented as ~27% on the legacy loss; here the mask-fixed arm spans 1.70×.
Removing host-side work made throughput *less* consistent across nodes, not more. Not
investigated. Anyone tempted to explain it should note that fix experiment 5's first
explanation was wrong and section "Follow-up benchmark" exists because of it.

### Caveat that limits this

**Arm order is confounded with node at n=3.** Two jobs ran legacy-first (ratios 1.696, 1.859)
and one ran maskfix-first (1.544, the lowest). That is the ordering you would expect if
running second were an advantage, but with one job in that arm it is equally consistent with
c0906a-s15 simply being the slowest node — which it also was on the legacy arm. **Not
separable from this data.** A fourth job with `ORDER=maskfix_first` would settle it; the
per-arm Inductor caches make a compile-cache explanation unlikely but not excluded.

## `mask_focal` / `mask_kl_div` fixed too (2026-08-03)

Same broadcast bug, same opt-in treatment: `mask_focal_loss_v2` and `mask_kl_div_loss_v2`
added to `loss.py` and registered as **`mask_focal_v2`** / **`mask_kl_div_v2`**. The legacy
functions are untouched and remain the default. **No CLIC config references either loss**
(checked), so this changes nothing for CLIC — it closes the bug for trackml / itk / tide /
cld / colliderml / atlas_muon.

`verify_mask_loss_v2.py` now covers all four losses against per-event reference loops.
All pass: v2 == reference to 1e-9, and legacy == v2 to ~0 under uniform padding.

**Two things worth flagging:**

1. **Legacy `mask_kl_div_loss` returns `NaN`**, not merely a wrong number, on ordinary
   variable-padding input. Its renormaliser `targets / targets.sum(-1, keepdim=True)` divides
   by zero whenever the broadcast pairs an object with an event whose pad mask zeroes all of
   that object's targets — which the `[B, N_valid, C]` broadcast makes easy to hit. So this
   loss was not usable as written, which is presumably why nothing selects it.
2. **`mask_kl_div_loss_v2` needed one deliberate deviation**, documented in its docstring: a
   `clamp_min(eps)` on the target normaliser. Keeping rank 3 means invalid object slots are
   now evaluated before being zero-weighted, and an all-zero-target slot would give 0/0 → NaN
   that survives multiplication by a zero weight. For any object with non-zero targets the
   clamp is a no-op, so valid objects are unaffected. Explicitly tested.
   The constant-V check for kl_div agrees to 4.4e-10 rather than exactly 0 (float64 reassociation),
   unlike bce/dice/focal which are exactly 0.

**Not done:** no training run uses these, so they are verified but not exercised end to end.
Anyone adopting `mask_kl_div_v2` for a real experiment should re-check the NaN behaviour on
their own data.

## Phase 3 + B200 full training — submitted 2026-08-03

Three jobs, answering two questions the re-baseline left open.

**1. Does the extrapolated B200 training time hold? (job 38598204)**
[`submit_training_maskfix_b200.sh`](submit_training_maskfix_b200.sh) — full 200 epochs,
1× B200, batch 2048, `clic_v6_maskfix.yaml`, geometry copied exactly from June's job 33954039
(1 task × 16 CPU, 60 GB, bf16-mixed) which took **68 h 18 m**.

The ~37 h figure quoted for this was **extrapolated from 300-step throughput, not measured**,
and the B200 spread is 1.7× across nodes, so the honest prediction is **29–50 h, centred ~37 h**.
This settles it. 96 h walltime requested — comfortably above the pessimistic end.

**The resulting model is not physics-comparable to the 3× L4 mask-fixed run** (job 38469247):
global batch 2048 vs 768, and the LR is not batch-scaled. This is a timing measurement.

**2. Where does the time go now? (jobs 38598205 B200, 38598206 L4)**
[`submit_profile_phase3_b200.sh`](submit_profile_phase3_b200.sh) /
[`submit_profile_phase3_l4.sh`](submit_profile_phase3_l4.sh). Same Phase-2 protocol
(`configs/profile_phase2.yaml`: PyTorchProfiler, 12 steps, eager mode with the `Compile`
callback removed, no validation), so the new traces are directly comparable to
`phase2_postfix_*`. Trace filenames overridden to `phase3_maskfix_{b200,l4}` so the arms
cannot clobber each other or the Phase-2 artifacts. Both stacks validated with
`--print_config`: they differ in exactly `batch_size` (2048 vs 256) and the trace filename.

Why re-trace: **every "where the time goes" number in this file predates the mask fix.** The
four Triton mask-loss kernels were 67.8% of GPU-busy time and the step is now ~1.7× faster on
B200 — whatever dominates now is unknown.

**Why the L4 arm is 1 GPU, not 3.** `batch_size` is per GPU, so a single L4 at 256 does exactly
a production rank's work and the kernel *composition* is the production one. What it does not
capture is the DDP gradient all-reduce; if NCCL turns out to matter that needs a separate
multi-rank trace (PyTorchProfiler writes one file per rank). This keeps the trace readable and
directly comparable to the single-GPU B200 arm.

**Why profile L4 at all** — new for this study, and the important part: the production config is
3× L4 at 256/GPU, and this study's central lesson is that batch-2048 B200 conclusions did not
transfer to it (matcher fixes −31% on protocol vs ~7% in production; mask fix 1.70× on B200 vs
1.18× on L4). Profiling only the B200 config would repeat that mistake.

Analyse with [`analyze_trace.py`](analyze_trace.py). Caveats carried over from Phase 2: eager
mode + profiler overhead means the **absolute** GPU-idle fraction is not the production one —
only composition transfers — and the script reports 6 `ProfilerStep` spans for 3 real steps.

## Phase 3 — RESULTS (jobs 38598205 B200 / 38598206 L4, 2026-08-03)

Both completed clean (3m50s / 2m28s). Artifacts `profile_logs/fit-phase3_maskfix_{b200,l4}*`.
Same eager-mode Phase-2 protocol on both, so composition is comparable across arms.

### The loss kernels are gone, and the profile is now ordinary transformer work

| share of GPU-busy | pre-fix B200 (38127863) | post-matcher-fix B200 (38452195) | **Phase 3 B200** | **Phase 3 L4** |
|---|---|---|---|---|
| fused triton loss kernels | ~53% | **~69%** | **~4.0%** | **~1.8%** |
| `Memcpy DtoH` (cost matrices) | 23.5% | 0.9% | 2.9% | 0.6% |
| model (attention + GEMM) | ~7% | ~8.6% | **26.9%** | **26.4%** |

The 2048× broadcast is confirmed removed at the trace level: the four mask-loss kernels went
from **the single dominant cost to not appearing in the top 12 on either arm**. What is left
is layer norms, elementwise copies/adds, attention and GEMM — a normal model, no pathology.
Note the model's *share* roughly tripled without the model getting slower; the denominator shrank.

### The headline: the two hardwares are now in completely different regimes

| | **B200 @ batch 2048** | **L4 @ batch 256** |
|---|---|---|
| GPU busy | **30.5%** | **85.3%** |
| GPU idle | **69.5%** | **14.7%** |
| GPU work per sample | 0.241 ms | 2.25 ms |
| wall clock per sample | 0.78 ms | 2.66 ms |

**The B200's GPU does the same work ~9.3× faster per sample than the L4's, but delivers only
~3.4× in wall clock.** The missing ~2.7× is the host stall. Meanwhile **the L4 is 85%
saturated** — it has almost no idle left to reclaim.

This is the clean explanation for the asymmetry measured earlier: the mask fix was 1.70× on
B200 and only 1.18× on L4, because removing GPU work only helps if the GPU is the constraint.
On the L4 it (nearly) is; on the B200 it is not, and never was.

### What this means for where to optimise next

**For CLIC on 3× L4 — the production config — there is no large win left in this pipeline.**
The GPU is 85% busy on ordinary transformer kernels. Further speedup requires doing less model
work (architecture, precision) or better kernels, not host-side fixes. The two shelved
candidates (per-layer pipelining of DtoH+solve, exact GPU LAP solver) target idle time the L4
does not have, and would buy it very little.

**For B200 the remaining headroom is large and entirely host-side** — 69.5% idle. Those same
two shelved candidates aim exactly there. Whether that is worth doing depends on whether
anyone actually trains on B200.

### Caveats — two of them matter a lot

1. **This is eager mode; production runs `torch.compile`.** The Phase-2 protocol removes the
   `Compile` callback so op attribution is clean. Inductor fuses many of the elementwise and
   layer-norm ops that dominate here, so **the ~45-55% "elementwise" share is an overestimate
   of the production figure**, and the busy/idle split will differ too. The *ranking* (no
   pathological kernel, model work is now a major share) is what transfers. A compiled-mode
   trace would be needed before acting on the elementwise number.
2. **Eager + profiler overhead inflates the host side**, so 69.5% idle is not the production
   B200 idle fraction. The direction is solid — the B200 is host-bound and the L4 is not — but
   the magnitude is not a production number.

Also: the analysis script may double-count kernels on overlapping streams, and reports 6
`ProfilerStep` spans for 3 real steps (the ~1.6 s / ~0.32 s alternation is that artifact, not
two kinds of step). The L4 arm is 1 GPU, so DDP all-reduce is not in this trace.

### A hypothesis for the unexplained B200 variance — NOT tested

A step that is 70% host-bound depends on CPU contention on a shared node, which varies with
whatever else is running there. That would explain why B200 throughput spans 1.7× across nodes
while the L4 — GPU-bound and 85% busy — reproduces to 0.6%. **This is a hypothesis and nothing
here tests it.** Fix experiment 5's first explanation (NUMA) was confidently wrong and had to
be retracted; do not promote this to a finding without a measurement that could refute it.
