# Step-time gap: why the paper's code steps a quarter slower than head on a B200

`studies/paper_tag_baseline/` measured the paper's small model at **311 ms/step (8 h 59)** on a
B200 at batch 2048 with the GPU JV matcher (job 41758027), where the head-based v7 model at the
same geometry did **251-255 ms/step (7 h 26-7 h 40)** (jobs 40405423, 41331284 on `main`). The
first suspect, `Dense`'s gated SwiGLU (14% more parameters), was refuted in `studies/swiglu_silu/`:
removing the parameters did not move the step. This study asks where the ~55 ms/step actually go.

## What the code comparison found (2026-09-13, before any new job)

The training-step hot path was compared file by file against the commit head's reference ran
(9d5e351): `maskformer.py`, `decoder.py`, `task.py`, `attention.py`, `transformer.py`/`encoder.py`,
`norm.py`, `dense.py`, `wrapper.py`, `lightning_module.py`, plus both runs' resolved configs.
`matcher.py`, `loss.py`, the `Compile` callback and the CLIC data reader are byte-identical after the
ports. The per-layer orchestration (costs summed per layer, one stacked matcher call, same
`has_intermediate_loss` gating), attention backends (flash-varlen encoder, SDPA with a bool mask in
the decoder), fp32 cost casts, bf16 autocast, callbacks, `log_every_n_steps`, workers, event counts
and steps per epoch are the same. Head's profile on this geometry is **69.5% GPU-idle**
(`main`'s `studies/b200_utilization/profiling/NOTES.md`, phase 3), so the step is host-bound and a
gap of this size must be host-side work, not arithmetic -- consistent with the SwiGLU result.

Paper-only work in the model, all small and host-side:

1. `maskformer.py:189` builds the loss-permutation index `torch.arange(B).unsqueeze(1)` on the
   CPU; head builds it on the device. Each of the ~12 gathers per step copies it host-to-device
   from pageable memory, which synchronises the stream. Estimate 1-5 ms/step.
2. `task.py:331` `attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False` inside the compiled
   decoder: single-argument `torch.where` is `nonzero`, a device-to-host sync and a dynamo graph
   break, four times per step. Head's `attn_mask()` has no such line (the decoder re-enables
   all-false rows itself, on both branches). Estimate 1-3 ms/step.
3. `task.py:308` `ObjectHitMaskTask.hit_net`: an extra `Dense(dim, dim)` over the hit embeddings,
   five times per step; head's run config has `constituent_net: null`. Estimate 1-3 ms/step.

Several things go the other way (head computes a softmax in the classification forward, head
expands `key_is_node` to (B, N) and indexes 49 M-element masks with it). Nothing in the model
accounts for 55 ms.

Outside the model the two runs differ in their **environment and logger**:

| | paper run 41758027 | head run 40405423 |
|---|---|---|
| lightning | 2.5.0.post0 (pinned by the tag) | 2.5.2 |
| comet_ml | 3.58.6 | 3.50.0 |
| logger | stock `CometLogger`, `log_env_gpu/cpu/network/disk: true`, `auto_output_logging: simple`, `log_graph: true` | `MyCometLogger` (2.5.2's rewrite over `comet_ml.start`) |
| numpy / scipy / lion-pytorch / torchjd | 2.5.3 / 1.18.1 / 0.2.5 / 0.17.0 | 2.4.2 / 1.17.0 / 0.2.3 / 0.7.0 |
| torch, triton, flash-attn, CUDA | 2.9.1+cu128, 3.5.1, 2.8.3 | same |

Lion 0.2.3 vs 0.2.5 differ by `sign_` -> `sgn_` only; torchjd is unused (`mtl: false`). The
logger and the lightning training loop are the live candidates: Comet's system-metric sampler
threads and stdout capture are per-step host overhead the model code cannot show, and 2.5.0.post0
predates the 2.5.1/2.5.2 logger rewrite.

## The experiment: same code, different environment

Four 300-step pre-flights on one B200 (batch 2048, GPU JV matcher, 30 min limit), submitted
together on 2026-09-13 so the day-to-day pre-flight bias cancels. Read each one's ms/step from the
progress bar between steps 200 and 300 (after compile warm-up), as `studies/swiglu_silu/` did.

| arm | code | environment | logger | job | run folder |
|---|---|---|---|---|---|
| A control | paper `clic-paper-main` | paper (`hepattn-paper/.pixi/envs/clic`) | stock CometLogger | **42000163** | `logs/pf_gap_A_paper_code_paper_env_<ts>` |
| B env | paper | **head** (`hepattn/.pixi/envs/clic`, lightning 2.5.2) | 2.5.2 CometLogger via the launcher's shim | **42000243** | `logs/pf_gap_B_paper_code_head_env_<ts>` |
| C no logger | paper | paper | `--trainer.logger=false` | **42000242** | `logs/pf_gap_C_paper_code_paper_env_nologger_<ts>` |
| D head control | head `main` (d18ceda), `clic_v7_cudamatch_b200_b2048.yaml` | head | `MyCometLogger` | **41999925** | `main`'s `logs/pf_gap_head_code_head_env_<ts>` |

A, B and C launch through `launch_paper_code.py` so that the launch path is identical; D uses
`main`'s `studies/model_size/submit_ablation_b200.sh`. A first submission of arm C (42000164) was
cancelled before it started: the script did not yet forward `EXTRA_ARGS`, so it would have been a
second control.

`launch_paper_code.py` exists because both worktrees install `hepattn` editable through
scikit-build-core, whose import finder maps `hepattn` to the environment's own checkout no matter
what `PYTHONPATH` says; the launcher strips the `hepattn` entries from that finder, puts the paper
`src` first, and prints which `hepattn` it imported so the log proves the arm. In head's
environment it also swaps `CometLogger` for a shim that accepts the paper config's argument names
(`project_name`, `experiment_name`, `save_dir`), because lightning 2.5.2 renamed them.

### Predictions

| result | reading |
|---|---|
| B ≈ D, A slow | the environment (lightning/comet versions) is the whole gap; fix = bump the tag's pins |
| C ≈ D, A slow, B slow | the stock 2.5.0.post0 CometLogger's per-step work is the gap |
| A ≈ B ≈ C, all slower than D | the gap is in the paper code after all; next is a compiled-step profile to size candidates 1-3 |
| everything within a few % | the 311 vs 251 gap was not a same-day comparison; remeasure before believing it |

## Status

### Round 1 (2026-09-13): no gap at 300 steps, in any environment

Read with `step_rate.py` (progress-bar samples at steps 200 and 300; the 150->300 window in
brackets). Arm C's first job (42000164) was cancelled unstarted; the resubmission (42001017)
failed in `SaveConfig.on_train_start`, which read `logger.save_dir` unconditionally on this
branch; `main`'s guard was ported (`logger-optional-callbacks`, merged as d493b6f) and the arm
resubmitted as 42001017, which then failed one hook later: Lightning's `LearningRateMonitor`
raises `MisconfigurationException` without a logger. **Arm C was dropped there**: round 1 had
already shown the logger cannot be the gap (A, B and D agree with two different loggers), so
running the paper config logger-free -- which needs its callback list rewritten -- would answer
nothing this study still asks.

| arm | code / env / logger | job | ms/step 200->300 (150->300) |
|---|---|---|---|
| A control | paper / paper / stock CometLogger | 42000163 | **360** (367) |
| B env | paper / **head** / 2.5.2 CometLogger | 42000243 | **370** (367) |
| D head control | **head** / head / MyCometLogger | 41999925 | **360** (367) |
| PF-ctl (swiglu_silu, same day) | paper / paper / stock | 41992197 | 360 (367) |

**All identical.** The environment (lightning 2.5.0.post0 vs 2.5.2, comet_ml, numpy, scipy) is
not the cause, the logger is not visibly the cause, and -- the real finding -- **the head model
is not faster than the paper model at 300 steps.** The 311-vs-251 gap is not present at the start
of training.

### The gap opens during training

Per-epoch wall time (seconds per 486 steps, training bar) from the two full runs:

| epoch | 1 | 2 | 3 | 4 | 5 | 10 | 14 | 20 | 30 | 40 | 50+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper 41758027 | 202 | 192 | 172 | 171 | 169 | 163 | 159 | 156 | 153 | 151 | 150-152 |
| head 40405423 | 193 | 176 | 167 | 152 | 152 | 145 | 135 | 132 | 131 | 131 | 126-130 |

Both runs get *faster* over the first ~40 epochs (paper -25%, head -35% from epoch 1), and the
head run separates from the paper run within the first five epochs, then keeps pulling away
(head drops another 8 s between epochs 13 and 14). Step time therefore depends on the *state of
the model*, and only one component of the step is data-dependent: the GPU Jonker-Volgenant
matcher, whose kernel (`vendor/torch-linear-assignment/src/torch_linear_assignment_cuda_kernel.cu`)
runs one Dijkstra-style augmentation loop per problem with an iteration count set by the cost
matrix. `MatcherTimer` measured it at 0.4-0.5% of the step (4 ms) on a *fresh* model (jobs
41755411/41755907), and head's eager profile at 9d5e351 caught `solve_cuda_kernel_batch` at
173 ms per call in the first steps -- the solver's cost is set by the cost matrices the model
produces, and the two models converge to different cost matrices (head's is 0.35 lower in
val_loss and reaches a different solution).

### Round 2: matcher share at the trained state

Resume each full run's epoch-199 checkpoint for 400 steps with a flat LR (`skip_scheduler`, so
the restored OneCycleLR state does not raise past its total steps) and `MatcherTimer`:

| arm | code | checkpoint | job | matcher median | step median |
|---|---|---|---|---|---|
| E | paper | `clic_paper_small_b200_jv_20260911-T141520`, epoch 199 | **42002583** | **1.7 ms** (0.6%) | **308 ms** |
| F | head | `main`'s `clic_v7_cudamatch_b200_b2048_20260827-T130314`, epoch 199 | **42002584** | **1.7 ms** (0.7%) | **263 ms** |

First attempts 42001381/42001382 ran zero steps: with `--trainer.max_epochs=201` Lightning
declared `max_epochs=201 reached` straight after restoring (the checkpoint's epoch progress reads
`processed: 200`, and the restored loop counts one more), so the resubmissions use
`--trainer.max_epochs=-1` and let `max_steps=97600` (= 97,200 + 400) end the run.

**The resumed runs reproduce the gap exactly** -- 308 vs 263 ms/step, against 311 vs 261 from
the full runs' steady state -- so a 400-step resume from a checkpoint is a faithful, six-minute
model of the steady state, and the comparison is now same-day, same-protocol. **The matcher is
not it:** the synchronised device-solver bucket is 1.7 ms in both, identical, and "everything
else" carries the whole 45 ms (306 vs 261). The step is state-dependent and the state-dependent
part is not the solver.

### Round 3: where the 45 ms are, at the trained state

A `torch.profiler` window (steps 250-256 after the resume, compile on, `ProfileWindow` in this
directory, added with `--trainer.callbacks+=profile_window.ProfileWindow`) on the same two resumed
runs. The chrome traces answer, per step, whether the extra 45 ms are GPU kernel time (and which
kernels) or host gaps (and around which ops); `main`'s
`studies/b200_utilization/profiling/analyze_trace.py` reads them.

| arm | code | resumed from | job | run folder |
|---|---|---|---|---|
| G | paper | E's checkpoint | **42005058** | `logs/pf_gap_G_paper_trained_profile_<ts>` |
| H | head | F's checkpoint | **42005059** | `main`'s `logs/pf_gap_H_head_trained_profile_<ts>` |

Both launch through `submit_env_ab_b200.sh` with `CODE_REPO`/`CONFIG` selecting the checkout, so
the launch path, JV build and profiler window are identical; `MatcherTimer` stays on as the
cross-check against E/F.

### Result: the gap is the JV solver kernel, and the matcher timer could not see it

`analyze_trace.py` on the six profiled steps of each arm (`MatcherTimer` medians 295 / 253 ms,
i.e. E/F again):

| per six steps | G paper | H head | difference per step |
|---|---|---|---|
| GPU window | 2.048 s | 1.714 s | 56 ms |
| GPU busy | 1.539 s | 1.334 s | 34 ms |
| `solve_cuda_kernel_batch` (JV) | **0.506 s** | **0.283 s** | **37 ms** |
| attention kernels | 0.249 s | 0.249 s | 0 |
| loss / triton fused | 0.264 s | 0.271 s | -1 |
| GEMM | 0.120 s | 0.119 s | 0 |
| elementwise | 0.148 s | 0.168 s | -3 |
| "other" minus the JV kernel | 0.133 s | 0.142 s | -2 |

Per call the JV kernel takes **76-91 ms on the paper model (84, 87, 91, 85, 76, 83)** and
**41-58 ms on head (44, 58, 50, 46, 41, 44)**. Every other kernel class agrees to within noise.
The single largest GPU op in both traces is the matcher, and it alone carries the 37 ms of the
45 ms gap; the remaining ~10 ms is extra GPU-idle around it (the paper's trace has 332
`cudaStreamSynchronize` calls in the window against head's 178).

**Why `MatcherTimer` said 1.7 ms.** The callback wraps `Matcher._match_on_device`, which launches
the JV kernel and returns; the timer synchronised *before* the matcher (so earlier kernels are
not charged to it) but not *after*, so its "device solver" bucket recorded the launch and the
kernel's 47-84 ms landed in "everything else" -- the next blocking op. Every "matcher = 0.4-0.6%
of the step" figure taken with the device solver (jobs 41755411, 41755907, E, F, and the
`studies/model_size` and `README_HPG.md` sentences built on them) is that artifact. Fixed in
`matcher-timer-device-sync` (merged 2026-09-13): the device bucket now synchronises after the call.
Validation run I (**42006692**, E's setup with the fixed timer) reads **90.4 ms median in the
device-solver bucket, 30.7% of a 294 ms step** (std 5.5 ms; the profiler's 84 ms was six
steps). The synchronised timer and the trace agree.

**Why the kernel is data-dependent.** `torch_linear_assignment_cuda_kernel.cu` runs one thread per
assignment problem (`i = blockDim.x * blockIdx.x + threadIdx.x`, `if (i >= bs) return`), each
thread executing a full Jonker-Volgenant solve with a `while (sink == -1)` Dijkstra loop whose
iteration count is set by the cost matrix. With 2048 events x 5 decoder layers = 10,240 problems of
150 x 150 stacked into one call, the kernel lasts as long as its slowest thread. Head's eager
profile at step 0 (`main`'s `profile_logs/fit-v7_cudamatch_b2048.txt`) caught the same kernel at
**173 ms per call on a freshly initialised model**; after 30 steps it is already much cheaper, and
by epoch 40 it has settled -- which is the fall in epoch time both full runs show, and why the
300-step pre-flights (A = B = D = 360 ms) see no gap: the solver's cost is set by the cost
matrices the *trained* model produces, and the paper's model converges to cost matrices that are
harder to solve than head's. The cost definitions and weights are identical on both branches
(`object_ce` 2.0, `mask_dice` 1.0, `kl_div` 1.0; both compute costs the same way in `loss.py`),
so the difference is in the trained outputs themselves: which of the post-paper model changes
(SwiGLU -> SiLU, the incidence/mask-head width, the norm rewrite) moves the solver's difficulty
is not established. `studies/swiglu_silu/`'s full SiLU run (41992199) is the first data point:
its epoch times so far (epoch 12: 155 s, against 161 s for the SwiGLU run and 145 s for head at
the same epoch) will show whether the activation alone accounts for it.

### What this means

- The paper code is not slow; **its trained model gives the GPU matcher harder problems.** No
  environment, logger, activation or model-code change on this branch is the cause, and none is
  needed to reproduce head's speed if the matcher is made insensitive to it.
- Levers, none tested here: `device_solver_eps` (1e-6; the kernel's tie tolerance sets how many
  augmentations near-equal costs cost), the cost matrix's conditioning (the shadow-matcher study
  on `main` found exact equal-cost cycles among neutrals), and overlap: a one-thread-per-problem
  kernel leaves the B200 nearly idle for 50-90 ms per step, so solving on a side stream while the
  backward of the previous step runs, or matching layer *k* while layer *k+1* is still in the
  forward, would hide most of it. Any of these applies equally to head, which pays 47 ms/step
  (18% of its step) to the same kernel.
- **Measurement protocol that worked:** resume the trained checkpoint with `skip_scheduler`,
  `max_epochs=-1` and a 300-400 step cap; six minutes reproduces the steady-state step time to
  1%. Fresh-model pre-flights measure the first 300 steps, not the run.
