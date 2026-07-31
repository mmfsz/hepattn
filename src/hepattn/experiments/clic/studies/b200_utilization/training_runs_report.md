# CLIC Training Runs — HiPerGator Report

All original jobs submitted **2026-06-05**; the 4× B200 resubmit ² was submitted **2026-06-09**. Training config: 200 epochs, `bf16-mixed` precision, `flash-varlen` attention, Lion optimizer.

| | **4× B200** | **4× B200 (resubmit)** ² | **1× B200** | **3× L4 (1 node)** | **6× L4 (2 nodes)** |
|---|---|---|---|---|---|
| **GPU VRAM** | 4× 192 GB | 4× 192 GB | 1× 192 GB | 3× 24 GB | 6× 24 GB |
| **Nodes** | 1 | 1 | 1 | 1 | 2 |
| **CPUs/task** | 16 | 16 | 16 | 16 | 16 |
| **Requested CPU RAM:** | 200 GB ¹ | 300 GB | 60 GB | 150 GB | 150 GB/node |
| **Time limit** | 48 h | 48 h | 72 h | 168 h | 168 h |
| **Batch size/GPU** | 2048 | 2048 | 2048 | 256 | 256 |
| **Global batch size** | 8192 | 8192 | 2048 | 768 | 1536 |
| **Steps/epoch** | — | 122 | 486 | 1295 | 648 |
| **Speed (it/s)** | — | 0.38 | 0.40 | 1.81 | 1.74 |
| **Time/epoch** | — | ~5.3 min | ~20 min | ~12 min | ~6 min |
| **Throughput (samples/s)** | — | ~3115 | ~818 | ~1393 | ~2676 |
| **Queue time** | 32 h 42 min | 3 d 0 h 37 min ² | 12 h 59 min | 26 min | 2 h 48 min |
| **Run time** | 43 min | 18 h 07 min | 2 d 20 h 17 min | 1 d 15 h 46 min | 20 h 49 min |
| **Outcome** | OOM crash ¹ | ✅ 200 epochs | ✅ 200 epochs | ✅ 200 epochs | ✅ 200 epochs |

¹ The 4× B200 job OOM-crashed at epoch 0 step 50/122. Root cause: `--mem=200G` is shared across 4 DDP tasks (50 GB/task), less than the ~60 GB/task needed. Fixed to `--mem=300G` and resubmitted as job 34185716.

² The resubmit (job 34185716) was submitted **2026-06-09 09:49** — not 2026-06-05 like the others.

## Hardware Efficiency: why 3× L4 beat 1× B200

A surprising result: **three L4 GPUs trained faster than one B200** (3× L4 ≈ 12 min/epoch and
finished in ~40 h, vs 1× B200 ≈ 20 min/epoch over ~68 h). The B200 is a flagship data-center
GPU; the L4 is a small, cheap inference card. So why did the "weaker" hardware win?

### What the numbers show
The fair way to compare two GPUs is **throughput per GPU** (training samples processed per
second, per GPU), which strips out the queue and the different batch sizes:

| | samples/s (total) | ÷ GPUs = **per GPU** |
|---|---|---|
| 1× B200 | ~818 | **~818 / GPU** |
| 3× L4 | ~1393 | **~464 / GPU** |

Per GPU, one B200 is only **~1.76× faster than one L4**. Since three L4s add up to
`3 × 464 ≈ 1393`, they collectively out-run a single B200 (~818) by ~1.7× — which is exactly
the wall-clock gap we saw.

### Why this means the B200 was underutilized
"FLOPS" (floating-point operations per second) is a GPU's raw math horsepower — how many
multiply/add calculations it can do each second. On paper (spec-sheet "peak" FLOPS in the
`bf16` number format these runs use), a **B200 has roughly 15–40× the horsepower of an L4** —
i.e. one B200 ≈ 15–40 L4s' worth of raw compute.

But in these runs it delivered only **1.76×** the real-world speed of an L4. Paying for 15–40×
the horsepower and getting ~1.8× the speed means **the GPU's math was not the bottleneck** —
the B200 spent most of its time waiting, not computing. 

This underutilization is expected for **this** workload: the model is tiny (10.1 M parameters),
which doesn't come close to filling a B200's compute, and each step does substantial host-side
work outside the model. Adding more GPUs also adds more parallel data pipelines (each DDP task
runs its own 16 workers), so more-but-weaker GPUs win here.

### What is proven vs. what was still a hypothesis
- **Proven** (from the measured throughput): the B200 is heavily underutilized on this
  workload — something other than its raw compute is the ceiling.
- **Was not proven when this section was written**: the *exact* bottleneck. The leading
  suspect was the data pipeline (CPU/disk feeding the GPU); the alternatives listed were
  launch overhead from many tiny operations on a small model, host-side Python overhead, or
  memory bandwidth. These runs had the profiler disabled (`profiler: null`) and logged no
  GPU-utilization data, so the mechanism could not be confirmed from their logs.
- **Now resolved** by the profiling study below (jobs 38122563 / 38127863, 2026-07-27):
  the leading suspect was **wrong** — the run is not data-bound (`train_dataloader_next` =
  0.42% of wall time). The bottleneck is the **loss/matcher pipeline**: memory-bound loss
  reductions, a ~0.9 GB device→host copy of the cost matrices, and the CPU Hungarian solve.
  See the **"Profiling results"** section below for the numbers and for the four fix
  experiments that followed.

### Practical takeaway
For this small model, the B200 is poor value as configured — you pay for compute you can't
feed. Several cheap L4s beat it on both wall-clock time and cost. A B200 would only pay off if
the model were scaled up, the per-GPU batch increased, or the loss/matcher pipeline made cheap
enough that the GPU itself becomes the limiting factor. **Update (2026-07-31):** the fix
experiments below recovered ~45% throughput (4.43 → 3.05 s/step) by attacking exactly that
pipeline, without touching the model or the numerics — but model compute is still only ~5% of
the step, so the conclusion about B200 value stands.

## Issues Encountered and Fixes

### 1. Truncated validation file
`val_clic_fix.root` was downloaded with only 233 MB out of an expected 296 MB. The ROOT header reported `fEND=309 MB` while the actual file was shorter, causing uproot to fail with `received 0 bytes from FSSpecSource`. Re-downloaded from CERNBox.

### 2. Comet ML requires API key
`MyCometLogger` was failing at startup with `Comet.ml requires an API key` because `COMET_API_KEY` was not set. Fixed by modifying `src/hepattn/utils/loggers.py` to default to `online=False` (offline mode) when no API key is present:
```python
if not os.environ.get("COMET_API_KEY") and kwargs.get("online") is None:
    kwargs["online"] = False
```

### 3. FlashAttention dtype error on L4 GPUs
`flash-varlen` attention was receiving fp32 tensors on L4 GPUs despite `bf16-mixed` precision, because `torch.compile` does not correctly propagate the AMP autocast context into the compiled graph. Fixed in `src/hepattn/models/attention.py` with an explicit cast:
```python
if q_flat.dtype == torch.float32:
    q_flat, k_flat, v_flat = q_flat.to(torch.bfloat16), ...
```

### 4. OOM crash on 4× B200
The 4-GPU job OOM-killed ranks 1–3 at epoch 0 step 50/122. Root cause: SLURM's `--mem` is per-node total CPU RAM. With 4 DDP ranks each spawning 16 DataLoader workers (64 workers total), the CPU RAM requirement is ~4× the single-GPU case (~240 GB), but only 200 GB was requested. Fixed by increasing `--mem` to 300 GB in `submit_training_hpg.sh`.

## Profiling results (2026-07-27 → 2026-07-31)

Diagnosis: jobs 38122563 (Phase 1) & 38127863 (Phase 2), plus a post-fix Phase-2 re-run
(job 38452195). Fix experiments: jobs 38206804, 38209457, 38212448, 38451000, 38452194.
Executed per [`profiling/README.md`](profiling/README.md);
raw tables/traces in [`profiling/profile_logs/`](profiling/profile_logs/), running notes in
[`profiling/NOTES.md`](profiling/NOTES.md), and a from-scratch explanation of the concepts in
[`profiling/PROFILING_EXPLAINED.md`](profiling/PROFILING_EXPLAINED.md).

### Confirmed bottleneck: loss/matcher pipeline, NOT the data pipeline and NOT the model

**Phase 1 — SimpleProfiler** (job 38122563: 1× B200, `base.yaml` batch 2048, 200 steps):
the leading suspect from this report was wrong — the run is **not data-bound**.
`train_dataloader_next` was 4.0 s of 962 s (**0.42%**); `training_step` (78.2%) +
`backward` (13.0%) consumed everything. Phase 1b (dataloader tuning) was skipped per the
decision rule; `persistent_workers`/`prefetch_factor` would buy nothing.

**Phase 2 — PyTorchProfiler** (job 38127863: eager mode, Compile callback off, 3 profiled
steps @ ~2.85 s): inside the step, per Chrome-trace analysis:

- **GPU busy 70.6% / idle 29.4%** of the profiled window. The idle time is host-bound —
  dominated by the CPU scipy Hungarian matcher + eager-Python overhead
  (`model.matcher` = 1.24 s CPU per step, 23% of CPU time).
- Of the GPU-busy time, the **model is almost nothing**: attention kernels 4.1% + GEMMs
  2.5% ≈ **7%**. The rest is loss/matcher machinery:
  - **~53% fused triton loss kernels** — the dense mask BCE/dice cost/loss matrices
    (`batch × queries × constituents`, fp32) for all 5 matched decoder outputs
    (`num_decoder_layers: 4` plus the final head; `maskformer.py:_compute_decoder_costs`).
    These are memory-bound reductions, so extra B200 FLOPS barely help. (These come from
    `torch.compile`d loss fns and are present even with the `Compile` callback disabled.)
  - **23.5% `Memcpy DtoH (Device → Pageable)`** — shipping the cost matrices to the CPU for
    the Hungarian matcher, into *pageable* (non-pinned) memory, at the matcher's host copy.
    Per step this is **one 921.6 MB copy taking ~472 ms** (5 stacked cost matrices ×
    2048 events × 150 queries × 150 targets × 4 B, i.e. ~1.95 GB/s effective) plus an 80 KB
    copy of the per-event target counts; the profiler's "235.9 ms mean over 6 calls" is the
    average over both sizes across the 3 profiled steps, not two equal 236 ms copies.

### Why 1 B200 ≈ 1.76× 1 L4 (mechanism)
Per ~2.85 s baseline step, only ~0.13 s is model compute that scales with GPU FLOPS. The
remaining ~95% (memory-bound loss reductions + DtoH transfer + CPU Hungarian solve + Python
overhead) is roughly GPU-independent, so a 15–40× FLOPS advantage collapses to ~1.8×
wall-clock. The fixes below shrink the GPU-independent part; they do not change the ~0.13 s.

### Fix candidates and outcomes (updated 2026-07-31; details in [`profiling/NOTES.md`](profiling/NOTES.md))
1. **Pinned-memory DtoH transfer — APPLIED, KEPT: −11.6% step time** (job 38209457 vs
   38122563: `run_training_batch` 4.43 → 3.92 s/step; `training_step` 3.76 → 3.25 s/step;
   `backward` unchanged as control). The cost-matrix copy is staged through a cached pinned
   buffer (`models/matcher.py`, now in `Matcher._prepare_costs`) — bit-identical numerics,
   applies to all experiments. **From job 38209457 onward, all runs include this change.**
2. **Faster assignment solver (`lap1015_late`) — REJECTED as shipped, then FIXED and
   PROMOTED.** As shipped it was 1.9× *slower*
   end-to-end (job 38206804: 7.14 s/step). The lap1015 algorithm is 3.5× faster than
   scipy single-threaded, but its pybind11 binding never releases the GIL, so the
   16-thread matcher parallelism serializes; scipy releases the GIL and scales ~11×.
   (Side benefit: the attempt uncovered and fixed two lap1015-exposed matcher bugs —
   undefined behaviour on empty and on non-finite cost matrices — plus added a
   permutation-validation scipy fallback; all accuracy-neutral.)
   **Follow-up — GIL release: WINNER.** Adding `py::gil_scoped_release` around the solve
   in `src/lap1015/src/main.cpp` and rebuilding `_core` makes `lap1015_late` scale across
   the matcher's 16 threads. Measured (job 38212448): `run_training_batch` **3.53 s/step**,
   i.e. a further −10% on top of the pinned fix and **−20.3% cumulative vs baseline**
   (4.43 → 3.53 s/step, ≈ +25% throughput), with `backward` unchanged as control and
   1 scipy-fallback event in 200 steps. **PROMOTED 2026-07-31** (`default_solver:
   lap1015_late` in `base.yaml`, `main.cpp` committed on branch `matcher-perf`); the
   extension must be rebuilt wherever the env is reinstalled, or lap1015 reverts to
   being ~2× slower than scipy.
3. **Device-side cost preparation — APPLIED, KEPT: a further −13.7%** (job 38451000 vs
   38212448: `run_training_batch` 3.53 → **3.05 s/step**). The host used to pay two
   full-size numpy copies of the 921.6 MB stacked cost tensor per step — the
   `np.ascontiguousarray(costs.swapaxes(1, 2))` transpose into solver layout, and the
   `np.where` that masks padded queries — and the DtoH copy landed in the wrong layout.
   `Matcher._prepare_costs` now sanitises, masks, transposes and crops to
   `max(num_valid_targets)` on-device, so the single pinned copy arrives contiguous, in
   final shape, and carrying no target rows the solver will not read. Assignments
   unchanged (556-event equivalence check across -inf/NaN/constant/empty/query-masked
   matrices, both solvers vs scipy optimal cost in float64).
4. **Overlap matching with GPU work** (~0.85 s/step GPU-idle): not attempted — invasive
   in general, but per-decoder-layer pipelining (async DtoH + solve layer *i* while the
   GPU computes layer *i+1*'s costs) is a contained version worth trying next.
5. **More matcher threads — MEASURED, REJECTED: 22% *slower*** (job 38452194:
   3.72 s/step at `n_jobs: 32` + `--cpus-per-task=32`, vs 3.05 s/step at 16).
   `hpg-b200` nodes are 112 physical cores / 8 GPUs = 14 per GPU, so `--cpus-per-task=16`
   was already above fair share; the `avery` QoS caps the account (`cpu=430`), not the
   job, so the 32-core request was legal, just counterproductive. A follow-up benchmark
   on the real hardware (job 38458388, `profiling/bench_matcher_threads.py`) **disproved
   the initial NUMA explanation** — a 48-core allocation landed entirely inside one NUMA
   domain, and `hpg-b200` has only two — and showed the solve **saturates at ~16 threads**
   (16: 0.23 s per step's worth of solves; 24: 0.24 s; 32: 1.67 s). There was no headroom
   to buy. The cliff at ≥32 is real and reproduces in both solvers but is still
   unexplained. **Keep 16.**
6. **Exact GPU LAP solver** (e.g. batched Jonker-Volgenant / auction with ε-scaling):
   would delete the DtoH and the CPU stall outright rather than shrinking them. Largest
   remaining structural win; needs a dependency and the same equivalence check.
7. **Loss-kernel cost** (~1.05 s/step): not attempted — computing mask losses in bf16 /
   only on matched pairs is a modelling change needing accuracy validation. An exact
   alternative is the algebraic rewrite of the mask-cost einsums
   (`einsum(pos, t) + einsum(neg, 1-t)` → `einsum(pos-neg, t) + neg.sum(-1)`), halving
   one GEMM; mathematically exact but not bit-identical, so it needs a loss-curve check.

### Cumulative progression (200 steps, 1× B200, batch 2048, 16 CPU)

| run | `run_training_batch` | `training_step` | `backward` (control) | vs baseline |
|---|---|---|---|---|
| baseline, scipy + pageable copy (38122563) | 4.43 s | 3.76 s | 0.627 s | — |
| + pinned DtoH staging (38209457) | 3.92 s | 3.25 s | 0.627 s | −11.6% |
| + lap1015_late w/ GIL release (38212448) | 3.53 s | 2.87 s | 0.629 s | −20.3% |
| + device-side cost prep (38451000) | **3.05 s** | **2.39 s** | 0.628 s | **−31.2%** |

≈ +45% throughput end-to-end, `backward` flat across all four runs as a control. Steady
state after compile warmup is ~2.29 s/step (steps 50→200), vs ~3.7 s/step at baseline.

**Every accepted change is assignment-identical** — no accuracy was traded for speed. The
pinned-buffer and device-prep changes only alter how and where the cost matrices are moved
and laid out; the promoted solver was checked against scipy's optimal assignment cost in
float64 over 556 events spanning `-inf`, NaN, all-constant, empty and query-masked cost
matrices, plus `tests/matching` (118 passed).

### Post-fix Phase-2 re-run (job 38452195, 2026-07-31): where the time goes now

Same protocol as the original Phase 2 (eager mode, `Compile` off, 12 steps), re-run with
all four fixes in place to re-aim before attempting anything further.

| share of GPU-busy time | pre-fix (38127863) | post-fix (38452195) |
|---|---|---|
| `Memcpy DtoH` (cost matrices) | **23.5%** | **0.9%** |
| fused triton loss kernels | ~53% | **~69%** |
| model (attention + GEMM) | ~7% | ~8.6% |
| GPU busy / idle | 70.6% / 29.4% | 54.5% / 45.5% |

The transfer is no longer a cost centre — 23.5% → 0.9% confirms the pinned-copy and
device-prep fixes at the trace level, not just end-to-end. The proportions have flipped:
the loss kernels are now ~69% of all GPU work, and the GPU is idle a *larger* fraction of
the step than before, because we removed GPU work and host work without removing the
serialisation between them. Note this is eager mode with profiler overhead, so 45.5% is
not the production idle fraction; only the composition transfers. Details and caveats in
[`profiling/NOTES.md`](profiling/NOTES.md).

Phase 3 (nsys / custom torch.profiler callback) was not needed — Phase 2 answered the
question. Profiling overlays live in `configs/profile.yaml` / `configs/profile_phase2.yaml`
and leave the run configs alone; the one deliberate change to a baseline config is
`base.yaml: default_solver: scipy → lap1015_late` (fix 2's follow-up, promoted 2026-07-31 on
branch `matcher-perf`, and only valid with the rebuilt `lap1015` extension). `clic_v7.yaml`
is untouched.
