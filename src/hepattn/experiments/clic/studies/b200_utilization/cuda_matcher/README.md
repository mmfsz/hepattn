# CUDA Hungarian matching — study plan

> **Status (2026-08-28): Phases 0–4 are done and all PASS. The one thing still standing between
> this study and a ship decision is the L4 arm of Phase 3, which is running now.** Phase 4 cleared
> on **both** v6 and v7: every jet-E IQR difference sits 5–9× inside §4.3's thresholds in both
> kinematic conventions, with the conventions disagreeing on sign, which is that rule's own noise
> check. Runtimes: v6 **11 h 22 m vs 33 h 03 m** (2.91×), v7 **7 h 40 m vs 23 h 15 m** (3.03×).
> The energy-*scale* question the v6 run left open is closed too — the v7 arms swap ranking on
> the global median between epoch 169 and convergence, a ~0.012 within-run wander against a
> σ_stat of 0.0004, so it was epoch scatter and not the solver. See [`NOTES.md`](NOTES.md),
> 2026-08-28.
>
> **Earlier status (2026-08-27): Phases 0–3 are done.** The matcher owns 66.8% of a B200 training step. The auction failed
> §6's kill criterion on real costs (job 40228586: 89% exact, 21.6% non-convergence, 350× slower).
> Jonker–Volgenant replaced it and **passed Phase 1 cleanly**: 100% exact in float64 and exact
> to fp32 rounding in float32 (job 40290161), with a ~180 ms solve that reproduces across
> nodes. It ships as `device_solver: jv`, off by default. **The paired A/B then passed on B200
> (jobs 40291275–40291277): the device arm delivers 4231 samples/s and reproduces to 0.8% across
> three nodes, while the host arm scatters over 32% — which is the host-bound thesis stated as a
> measurement.** Phase 3's L4 repeat has *not* been run, and the whole opt-in design rests on it.
>
> | phase | state |
> |---|---|
> | 0 — size the prize on B200 | **DONE — GO.** Matcher = **66.8%** of the step, against a ≥20% bar (job 39401280). The prize is the *solve* (63.9%), not the device→host copy (1.3%). See [`NOTES.md`](NOTES.md) |
> | 1 — offline correctness | **DONE — PASS for JV.** *Auction: NEGATIVE* (job 40228586, real costs): 89.01% exact, 21.6% non-convergence, 350× slower — §6 kill criterion met as written. *JV (`torch-linear-assignment`): POSITIVE* (jobs 40252873, 40290161, same tensor): **100.00% exact in float64, worst excess 0**; in float32, 99.76% exact with the disagreements at 2.8e-09 relative — near-ties an order below fp32 epsilon. Device solve **~180 ms**, reproducible across nodes |
> | 2 — config option | **DONE.** `device_solver: jv` → `device_lap.batched_jv`, carrying the two load-bearing numerical decisions (`num_rows + 1` for forbidden entries, constant-cost padded rows), a `require_jv()` build check that fails at construction, and a `solved=None` contract that skips the per-call device sync. Tests parametrised over both solvers, green on CPU and on a B200 (job 40290666); `Matcher().device_solver` is still `None` |
> | 3 — paired A/B throughput | **DONE — PASS on B200** (jobs 40291275, 40291276, 40291277; three nodes, both `ORDER`s, all six arms clean). Device **4231 samples/s** against a host arm of 1358–1790; within-allocation ratios 2.34×/2.77×/3.12×, mean 2.74× = **91% of Phase 0's 3.0× Amdahl ceiling**. The load-bearing observation is not the ratio: **the device arm varies 0.8% across nodes and the host arm 32%**, exactly as "host-bound" predicts, so the ratio is a function of the node the host arm drew and the absolute 4231 samples/s is the honest headline. **Still outstanding: the L4 repeat at batch 256, and the `--cpus-per-task` sensitivity check** |
> | 4 — physics equivalence | **DONE — PASS on v6 and v7.** v6 (job 40393482 vs baseline 38598204) and v7 (jobs 40405423 vs 40400036), both arms at 200 epochs, each evaluated at its own lowest-val_loss checkpoint and **both scored with the host solver** so the eval path contributes nothing. v7 converged deltas: global IQR −0.0015 / +0.0008 against a 0.0073 threshold; high-E −0.0039 / +0.0004 against 0.0164. Opposite signs across conventions on every row. The GPU matcher is physics-neutral |
>
> ### Next steps, in order (2026-08-28)
>
> 1. **The L4 arm — the last ship condition, and running now.** It is what the "opt-in" default
>    is *justified* by: the whole design assumes the L4 loses. It was never merely unscheduled —
>    it was not runnable, because every build of the extension contained only `sm_100` cubins
>    (verified with `cuobjdump --list-elf`) and an L4 is `sm_89`. The device arm would have died
>    with "no kernel image is available for execution on the device". A build carrying both
>    architectures is the prerequisite.
> 2. **Redo the `--cpus-per-task` check**, which job 40516325 got wrong: it used `taskset -c 0-15`
>    and SLURM does not allocate CPUs 0–15, so the cells measured the intersection rather than the
>    requested core counts. Vary the matcher's `N_JOBS` through the existing override in
>    `submit_phase0_matcher_share_b200.sh` instead — no affinity manipulation, and it isolates the
>    matcher from the dataloader. See NOTES.md, 2026-08-28.
> 3. **[TOMORROW] Move block size 32 from an env var into the kernel, where it belongs.**
>    Block size 32 is deployed and measured (+4.41% mean over three allocations, jobs 40533246,
>    40534109, 40538140), but it is switched on by `export APPTAINERENV_TLA_BLOCK_SIZE=32` in
>    individual submit scripts. **That is the wrong mechanism and it fails open**: the device
>    solver is enabled by *config* (`device_solver: jv`) while the block size comes from the
>    *environment*, so the two are decoupled and any new submit script — the likely thing for
>    someone to write — gets the device matcher at 128 and silently loses the 4.4%.
>
>    The fix is one line in `SMPCores()`. The only reason 128 is in play is that its switch covers
>    compute capability majors 2–9 and the B200 is major 10, so it falls through to
>    `return 128; // Unknown device`. Add the major-10 case returning 32 and every B200 run gets it
>    automatically, with nothing to forget. An L4 keeps 128 through its existing Ada branch, so the
>    hardware where 32 has never been measured is untouched, and `TLA_BLOCK_SIZE` survives as the
>    override for future sweeps. Rebuild is ~2 min (`build_tla_candidate.sh`), then re-run
>    `submit_phase0_blocksize_b200.sh` to confirm the default path now lands at ~512 ms rather
>    than ~535 ms without the variable set.
>
>    Once it is in the kernel, the per-script `TLA_BLOCK_SIZE=32` lines become redundant and
>    should be removed so there is one source of truth.
>
>    ⚠️ When reading any Phase-0 file from a *device* arm, read the step time, not the `device`
>    bucket. §6's `solved=None` contract skips the post-solve sync, so the bucket records only
>    kernel launch (~1.5 ms against a ~170 ms kernel) and the cost lands in `other`. Bucket
>    attribution is valid on host arms only.
> 3. **The vendoring decision**, now unblocked: §3's objection to this candidate was the
>    compiled-CUDA dependency, and that objection was made contingent on Phase 3 being worth it.
>    It is. Last because it is an engineering choice with no measurement attached, and Phase 4
>    could still make it moot.
>
> **One environment trap the A/B script now handles.** The extension is ABI-bound to the pixi
> env it was built against, and `main.py` trains under the **default** env (torch 2.9.1) while
> the offline benches run under **clic** (torch 2.10). So there are two builds:
> `vendor/torch-linear-assignment-default` for training and `vendor/torch-linear-assignment`
> for the benches. Loading the wrong one fails with `undefined symbol:
> _ZNK3c1010TensorImpl15incref_pyobjectEv`; neither is a fallback for the other.
>
> The open decision, now unblocked by Phase 3 and item 3 above: whether
> `torch-linear-assignment` gets **vendored into `src/`** as `lap1015` was, or stays an external
> build at `/blue/avery/m.mazza/projects/fastml/vendor/`. §3's original objection to this whole
> candidate was the compiled-CUDA dependency. Decide at ship time.
>
> **Dead ends — do not re-run.** The auction (§6 kill criterion, job 40228586). The
> `-march=native` hypothesis (job 40137096 disproved it). The `n_jobs` sweep (job 40138527:
> 16 is correct). The GIL warning (a false negative on a hand-patched `.so`). The offline
> host-vs-device *speedup* ratio: it moves with whichever node the host arm lands on (0.405 s,
> 0.538 s and 1.879 s for the same host call), so 2.25× and 10.5× are the same measurement and
> neither is the answer Phase 3 is after. **And the *paired* ratio is subject to the same
> node-dependence** — 2.34×, 2.77× and 3.12× across the three Phase-3 allocations, from a device
> arm that reproduced to 0.8% — so re-running the A/B in the hope of a cleaner ratio measures
> the node, not the solver. The ratio has no cleaner value to converge on; the device arm's
> absolute samples/s does.
>
> One result is already banked and is written up in §3: **the textbook epsilon-scaling auction
> is wrong for the rectangular problems this matcher poses.** It is exact only when
> `num_rows == num_cols`, and the matcher's problems are ~50 valid targets into 150 query slots.
>
> **Read that last clause with care** — it is the mean of the *per-event* particle count, and
> the crop is a `max` over the batch, so the tensor the solver actually receives at batch 2048
> is ~146 × 150. See `NOTES.md`, 2026-08-25.
>
> This is the follow-up to candidate 6 ("exact GPU LAP solver") that
> [`../profiling/NOTES.md`](../profiling/NOTES.md) put on hold on 2026-07-31. The hold
> condition was *"not until real training runs show step time is actually a constraint"*.
> B200 training is now the thing we want to be fast, and the Phase-3 trace says the B200 is
> **69.5% GPU-idle and host-bound**, so the condition is met.
>
> **Nothing here should change default behaviour.** The production config is 3× L4, and the
> L4 is 85% GPU-busy — it has no host stall to reclaim, and moving the solve onto an
> already-saturated GPU would make it *slower*. The device solver is therefore an opt-in
> config key that defaults to off. **This remains an expectation, not a measurement**: the L4
> arm of Phase 3 has not been run.

## 1. The problem, in numbers

All figures below come from the profiling study and are **not to be re-derived**; see
[`../profiling/NOTES.md`](../profiling/NOTES.md) and
[`../profiling/PROFILING_EXPLAINED.md`](../profiling/PROFILING_EXPLAINED.md).

Every training step, `MaskFormer._compute_decoder_costs` stacks the costs from the 4 decoder
layers plus the final head into **one** matcher call (`maskformer.py:341`):

| quantity | value at the B200 config (batch 2048) |
|---|---|
| LAP problems per step | 2048 × 5 = **10,240** |
| size of each problem | up to 150 preds × 150 targets (rectangular after cropping to `max(n_valid_targets)`) |
| cost tensor | **921.6 MB fp32**, copied device → host every step |
| CPU solve, 16 threads | 0.23 s (`lap1015_late`) / 0.27 s (`scipy`) standalone — but **712.8–783.3 ms measured inside a real training step** (Phase 0 and job 40138527), where the same 16 cores also feed 16 dataloader workers. `n_jobs=16` is the right setting: it is 7.1× a single thread and 1.14× eight |
| CPU threads needed | 16 (`--cpus-per-task=16`; measured to saturate there) |

And the hardware split after the mask-loss fix (Phase 3, jobs 38598205 / 38598206):

| | **B200 @ batch 2048** | **L4 @ batch 256** |
|---|---|---|
| GPU busy | **30.5%** | **85.3%** |
| GPU idle | **69.5%** | 14.7% |
| GPU work per sample | 0.241 ms | 2.25 ms |
| wall clock per sample | 0.78 ms | 2.66 ms |

The B200's GPU does the same work ~9.3× faster per sample but delivers only ~3.4× in wall
clock. The missing ~2.7× is host stall.

### Why the GPU should win this

10,240 independent ~150×150 assignment problems is close to the ideal case for a batched GPU
solver and close to the worst case for a CPU one:

- CPU parallelism is **already saturated** — measured, 16 threads is the plateau, and ≥32
  threads falls off a cliff (fix 5 in the profiling notes). There is no more CPU to buy.
- The problems are independent, identically shaped, and small enough that a whole batch of
  them fits in GPU memory alongside the model.
- The costs are **already on the GPU**. `Matcher._prepare_costs` sanitises, masks, transposes
  and crops on-device precisely so the DtoH lands contiguous — the device path simply stops
  before the `.numpy()` call. There is no new data movement to design.

The win is not just the 0.23 s of solve: it is the 921.6 MB DtoH, the `cudaStreamSynchronize`
that the host stall forces, and the 16-core CPU request that goes with it.

> **Superseded by Phase 0 (2026-08-15).** The DtoH is **1.3% of the step**, and deleting it
> outright would be invisible. The win is the solve and nothing else: 63.9% of the step. Read
> the paragraph above as the reason this study was opened, not as a description of the prize.

## 2. Honest expected size of the prize

**We do not yet know what fraction of the B200's 69.5% idle is the matcher**, and this is the
single most important unknown. Every "where the time goes" attribution for the matcher in the
profiling study (`model.matcher` = 1.24 s/step) is **pre-mask-fix**; the step is ~1.7× faster
now and the composition has changed completely (the mask-loss kernels went from 69% of GPU-busy
to 4%). The DtoH is only 2.9% of GPU-*busy* time post-fix, but GPU-busy is only 30.5% of the
step, so that tells us little about the idle.

So: **Phase 0 below is a gate, not a formality.** A perfect device solver caps out at whatever
fraction of the step the matcher actually owns. If that is 10%, this study is not worth
finishing.

> **Answered, 2026-08-15: 66.8%.** A free device solver would leave a ~372 ms step, i.e. a
> **3.0× throughput ceiling**; one costing 100 ms still leaves 2.4×. This is against a properly
> threaded host solver: a GIL warning in the same log briefly suggested otherwise, but a
> single-allocation sweep (job 40138527) showed a serialised solve costs 5595.7 ms against the
> 712.8 ms measured, so the host arm was threaded all along. See [`NOTES.md`](NOTES.md).

## 3. Algorithm choice

| option | exact? | new dependency | verdict |
|---|---|---|---|
| **Batched auction with ε-scaling, pure PyTorch** | ε-optimal; exact when ε < 1/n on suitably scaled costs | **none** | ~~first choice~~ — **implemented, and pseudo-polynomial: rounds ∝ C/ε, so cost depends on the matrix *values*. Collapses at the production crop (`NOTES.md`, 2026-08-25)** |
| Batched auction in Triton | same | none (triton 3.5.1 already in the env) | fallback if the torch version is kernel-launch-bound |
| Batched Jonker–Volgenant (`torch-linear-assignment`) | exact | yes — pip package, CUDA build, arch-fragile | ~~second choice~~ — **now the recommended path.** Implements Crouse (2016), the same algorithm scipy uses; natively rectangular; strongly polynomial, so immune to both the aspect ratio and cost degeneracy. Builds here (`NOTES.md`, 2026-08-25) |
| Batched Hungarian, CUDA (HyLAC, MIT) | exact | yes — C++/CUDA, no Python API | fallback if the above underperforms: its "stream-solver" (one thread block per small LAP) is this workload verbatim, 22.59× over prior work, but square-only in its docs |
| Sinkhorn / soft assignment | **no** | none | **rejected** — an approximation that changes the training objective. The profiling study's standing rule is *no approximations in the computations*. |

> **Superseded, 2026-08-25.** The rationale below weighs "naturally batched and data-parallel"
> as the deciding factor. That is the right criterion for **one** large LAP; this workload is
> **10,240 independent** ones, where inter-problem parallelism is already ample and the
> auction's intra-problem parallelism buys nothing — in exchange for giving up strong
> polynomiality. See `NOTES.md`, 2026-08-25, and the GPU LAP literature in §8.

**Recommendation: pure-PyTorch batched auction with ε-scaling.** Rationale:

- Zero new dependency. `lap1015` already demonstrates the cost of an arch-fragile compiled
  extension in this repo — it silently degrades to ~2× *slower* than scipy on a build that does
  not release the GIL, and that failure mode is invisible without the explicit warning that had
  to be added for it.
- Auction is naturally batched and data-parallel: every unassigned query bids simultaneously,
  which is a `max` and a `scatter_reduce` over the batch, not a sequential augmenting-path
  search. Jonker–Volgenant is the faster serial algorithm but the harder one to batch.
- ε-scaling gives a principled exactness knob rather than a quality/speed tradeoff we have to
  defend to physics.

**Exactness is a claim to be tested, not assumed.** Auction terminates ε-optimal: the returned
assignment is within `n·ε` of optimal. For exactness we need ε below the minimum gap between
distinct achievable assignment costs, which for float costs is not bounded a priori. The
acceptance criterion is therefore empirical and matches what the study already did for
`lap1015`: **equal total assignment cost against scipy in float64** on real cost matrices, not
an identical permutation (ties are genuinely degenerate). Any event that fails gets a CPU
fallback.

### FINDING (2026-08-13): ε-scaling is wrong for rectangular problems

Testing that claim immediately caught a real error, and it is the kind that would have shipped
silently — the solver converged, returned valid permutations, and was simply *matching the
wrong things*.

The textbook auction runs a sequence of phases with a shrinking ε, **carrying the prices
forward** and restarting the assignment; that carry-over is what makes ε-scaling fast. It is
invalid here. The dual of the rectangular assignment problem admits `p_j > 0` only for columns
that are actually assigned. A forward auction maintains that within a phase, because a column
is priced only by being won and is never released — but a restart breaks it, since a column can
enter a phase carrying a price and end that phase unassigned. When `num_rows == num_cols` every
column ends up assigned and the bug cannot appear, which is exactly why it would survive a test
suite built on square matrices.

Measured on uniform random costs, excess over the scipy optimum (costs normalised to [0, 1], so
the theory promises `num_rows · ε` ≈ 1e-5):

| shape | scaling, carry prices | scaling, reset prices | single phase at ε=1e-6 |
|---|---|---|---|
| 50 × 150 | **1.6e+00** ✗ | 0.0 (479 rounds) | 0.0 (**55 rounds**) |
| 100 × 150 | **6.5e-01** ✗ | 0.0 (1136 rounds) | 0.0 (**141 rounds**) |
| 140 × 150 | **2.9e-01** ✗ | 0.0 (3856 rounds) | 0.0 (**710 rounds**) |
| 150 × 150 | 0.0 (3007 rounds) | 0.0 (16320 rounds) | 0.0 (4238 rounds) |

Two things follow. First, resetting the prices each phase restores exactness — so the
implementation does that, for anyone who sets a schedule anyway. Second, and more usefully,
**a single phase at the final ε is both exact and the fastest option on every rectangular
shape**, by 5–9× in rounds. ε-scaling only earns its keep in the square case, which is not the
case we have. Hence `device_solver_eps` is a single value and there is no schedule to tune.

(Round counts are for a sequential reference implementation, used because it isolates the
algorithm from the batching. The batched Jacobi form bids for all unassigned rows at once, so
its round count is far lower; the *ranking* is what transfers.)

## 4. Config design

The device solver is a new opt-in key on `Matcher`. Default `None` reproduces today's
behaviour exactly.

Implemented as specified. `configs/clic_v6_cudamatch.yaml` is the mask-fixed production config
with only this block changed, which is what makes the paired A/B in Phase 3 possible:

```yaml
matcher:
  class_path: hepattn.models.matcher.Matcher
  init_args:
    adaptive_solver: false
    # --- opt-in: solve on the GPU instead. null (the default) => today's host path ---
    device_solver: auction
    device_solver_eps: 1.0e-6        # bidding increment, in units of each problem's cost range
    device_solver_max_iters: 10000   # cap before a problem is declared unsolved
    device_solver_fallback: true     # re-solve those on the host with `default_solver`
    # --- host solver, now only used for the fallback ---
    default_solver: scipy
    parallel_solver: false
```

Design notes:

- **`device_solver: null` is the default and must remain so.** On the L4 the GPU is the
  constraint; adding solver kernels to it is expected to be neutral at best. Phase 3 measures
  this rather than assuming it.
- The device path is a separate `Matcher._match_on_device`, not a branch inside
  `_prepare_costs`: it needs none of the sentinel fill, pinned staging or `.numpy()` that only
  exist to serve a host solver, and the auction treats non-finite and disallowed entries as
  forbidden directly.
- `forward` returns a CPU tensor on the host path (`torch.from_numpy`) and a device tensor on
  the device path. `maskformer.py:359` indexes device tensors with it, so this **removes** an
  implicit host-to-device copy rather than adding one.
- The device path bypasses `adaptive_solver`: the host solvers it would be timed against sit on
  the far side of the transfer this exists to avoid.
- **Three syncs survive**, and Phase 0/1 should confirm they are negligible: reading
  `max(num_valid_targets)` to crop the padded target rows (4 bytes, against the ~1 GB the host
  path moves), the completion test inside the auction loop (once every `check_interval`
  rounds), and one `solved.all()` to decide whether the fallback is needed. The
  `assert torch.all(pred_idxs >= 0)` that the host path ends with is *not* run on the device
  path — `assignment_to_permutation` constructs a permutation by construction, and the
  `solved` flag already carries the failure signal.
- The solver has a data-dependent `while` loop, so it stays outside any `torch.compile` region.
- `Matcher.device_fallbacks` counts problems the auction failed to converge on. It is public
  precisely so a run can be audited: a high count means the device path is a host path wearing
  a disguise.

## 5. Test plan

Six steps, in order, each gating the next.

### Phase 0 — size the prize (GATE) — **DONE, GO** (job 39401280, ran 2026-08-15)

One job on B200 with the current mask-fixed config, attributing the step's wall clock to
(a) matcher device→host copy, (b) matcher solve, (c) device-side prep, (d) everything else.

`hepattn.callbacks.MatcherTimer` does the attribution, driven by
[`../../../configs/profile_phase0.yaml`](../../../configs/profile_phase0.yaml) and submitted
with [`submit_phase0_matcher_share_b200.sh`](submit_phase0_matcher_share_b200.sh). It uses
explicit `torch.cuda.synchronize()`-bracketed timers, because the profiler's idle attribution
lumps the matcher in with every other host cost in the step. The leading sync on entry to
`Matcher.forward` is the load-bearing part: without it, the host path's first blocking
operation is charged for all the GPU work queued earlier in the step.

**This does *not* reuse the Phase-2/3 trace protocol**, contrary to the original sketch here.
That protocol runs eager with `Compile` removed, under `PyTorchProfiler` — which inflates the
host side, and the host side is the thing being measured. Phase 0 runs production settings
instead: compiled, no profiler, batch 2048, 90 steps with 40 discarded as warmup. See
[`NOTES.md`](NOTES.md) for the bucket definitions and the instrumentation's own cost.

- **Go** if the matcher owns ≳20% of B200 step time.
- **Reconsider** at 10–20% — the ceiling is small but the CPU-core saving may still justify it.
- **Kill** below 10%, and record that in `NOTES.md` as the answer to candidate 6.

**Result: 66.8%** — matcher 745.6 ms of a 1116.2 ms step, of which the host solve is 712.8 ms
(63.9%) and the device→host copy only 14.9 ms (1.3%). Go. The full attribution, the Amdahl
ceiling, and the GIL caveat that makes the host arm a degraded baseline are in
[`NOTES.md`](NOTES.md).

### Phase 1 — offline solver, correctness first (no training)

[`bench_device_matcher.py`](bench_device_matcher.py), submitted with
[`submit_bench_device_matcher.sh`](submit_bench_device_matcher.sh) — there is no GPU on the
login node, so even the correctness runs go through SLURM.

1. **Correctness on synthetic problems** — **DONE, on CPU.** Random uniform, degenerate (costs
   rounded to a few levels so most entries tie), rectangular, scale-swept over 1e-3…1e6,
   padded rows, forbidden columns, non-finite entries, and empty-target events. Every case
   reaches scipy's optimum in float64 with zero fallbacks. This is necessary, not sufficient:
   synthetic uniform costs are far better conditioned than the real matcher costs.
2. **Correctness on real cost matrices** — **still to do, and it is the one that matters.**
   Dump one training step's real stacked cost tensor (10,240 × 150 × 150) from a B200 run and
   replay it with `--costs`.
3. **Speed** — **still to do**; needs a GPU. Solve time vs `scipy`/`lap1015_late` at 16
   threads, over the shapes in the script's sweep, timed *including* the DtoH the host path
   needs and the device path does not. For reference, on CPU the device solver is ~33× slower
   than the host one, exactly as expected: the auction buys nothing without thousands of
   parallel lanes, which is also why this option must stay opt-in.

Deliverable: a table of (exactness pass rate, fallback rate, speedup) per configuration.

### Phase 2 — integration behind the config key — **DONE**

- `src/hepattn/models/device_lap.py` — the batched auction and the permutation builder.
- `src/hepattn/models/matcher.py` — `DEVICE_SOLVERS`, the four `device_solver*` init args, and
  `Matcher._match_on_device`.
- `tests/matching/test_device_solver.py` — 18 pass on CPU; the same checks are parametrised
  over `cuda` behind the `gpu` marker, so they need a GPU run to have been exercised.
- The host default is untouched: `tests/matching/` still passes 118, and `Matcher().device_solver`
  is asserted to be `None` so the option cannot regress anyone by accident.

Still to do: a 20-step smoke run on B200 asserting the per-step losses match the host path to
within fp32 noise, plus the `gpu`-marked tests.

### Phase 3 — paired throughput A/B — **DONE on B200, PASS; L4 arm NOT RUN**

`device_solver` is a single config key, so the study's existing paired-run trick applies
directly. [`submit_paired_device_matcher_b200.sh`](submit_paired_device_matcher_b200.sh) runs
both arms back-to-back **in one SLURM allocation** and splits the log on its `ARM:` markers.
This is the only way to avoid the node-to-node confound — the B200 throughput spans
1122–1911 samples/s across nodes, which is larger than any effect we expect to measure.

- **Arm `host`**: `configs/clic_v6_maskfix.yaml` — current production behaviour.
- **Arm `device`**: `configs/clic_v6_cudamatch.yaml` — identical but for the matcher block.
- Run both orders (`ORDER=` flag) to close the ordering confound the mask-fix study left open.
- Parse with [`../profiling/parse_throughput.py`](../profiling/parse_throughput.py) `--batch 2048`.
- **Repeat on 3× L4** at batch 256 with `submit_maskfix_l4_3gpu.sh` as the template. The
  expectation is neutral-to-slightly-negative; this is what justifies "opt-in", and a surprise
  win there would be a much bigger result than the B200 one. **NOT RUN as of 2026-08-27** — and
  it is the third of §6's three ship conditions, so it is an open item rather than a formality.

Also record `--cpus-per-task` sensitivity: if the device path works, arm B should be
insensitive to dropping from 16 cores to 4, which is a real scheduling win on top of throughput.
**Also not done**: none of the three Phase-3 allocations varied the core count.

**Result (B200), 2026-08-27** — jobs 40291275 / 40291276 / 40291277, three nodes, both `ORDER`s,
all six arms clean (no solver warnings, no host fallbacks, 300 steps reached):

| arm | samples/s across the three nodes | spread |
|---|---|---|
| host | 1790 / 1528 / 1358 | **32%** |
| **device** | 4197 / 4231 / 4231 | **0.8%** |

Within-allocation ratios 2.34×, 2.77×, 3.12×; mean 2.74×, which is 91% of Phase 0's 3.0× Amdahl
ceiling. The device arm wins in both orders. **The node-invariance of the device arm against the
32% scatter of the host arm is the load-bearing observation** — it is the host-bound thesis
stated as a measurement — and it also means the ratio is not a stable quantity. Quote the
absolute 4231 samples/s. Full write-up in [`NOTES.md`](NOTES.md), 2026-08-27.

### Phase 4 — physics equivalence

If Phase 1 shows exact assignment costs, the trained model *should* be bit-comparable modulo
tie-breaking, and this phase is confirmation rather than investigation. Same protocol as the
mask-fix validation:

- Loss curves for the two arms ([`../profiling/plot_maskfix_curves.py`](../profiling/plot_maskfix_curves.py)).
- Jet-E IQR ([`../profiling/plot_maskfix_jet_iqr.py`](../profiling/plot_maskfix_jet_iqr.py)),
  with the same |z| < 2 criterion used for the mask fix.

Only needed at full length if Phase 3 says we want to actually deploy this.

## 6. Success and kill criteria

Written down now, before any measurement, so they cannot drift:

| | criterion |
|---|---|
| **Ship** | exact assignment cost on real matrices (fallback rate < 0.1%) **and** ≥ 10% B200 throughput **and** no L4 regression beyond noise |
| **Ship as opt-in anyway** | exact, ≥ 10% on B200, but a measurable L4 regression — this is the expected outcome and the reason for the config key |
| **Kill** | Phase 0 shows the matcher owns < 10% of the B200 step |
| **Kill** | assignment cost differs from scipy, or the fallback rate is high enough that the CPU path dominates anyway |

> **TRIGGERED 2026-08-25 for `device_solver: auction`** (job 40228586): 89.01% exact against a
> required 100%, and a 21.6% fallback rate against a required < 0.1%. Applied as written. The
> criteria are left unchanged here so that the next solver is judged by the same bar.

> **PARTIALLY MET 2026-08-27 for `device_solver: jv`**, against the same bar. *Exactness*: met —
> 100.00% exact against scipy in float64, worst excess 0, zero fallbacks against a required
> < 0.1% (job 40290161). *B200 throughput*: met, with a very large margin — 4231 against 1358–1790
> samples/s, i.e. +134% to +212% against a required ≥ 10% (jobs 40291275–40291277). *L4
> regression*: **untested** — the Phase-3 L4 repeat has not been run, so the third condition is
> neither met nor failed and the "Ship" row is not yet satisfied as written. On the evidence so
> far the outcome is at worst the "Ship as opt-in anyway" row, which is what §4 already builds.
> The criteria themselves are again left unchanged.

## 7. Known risks

1. ~~**The `big` sentinel will break ε-scaling.**~~ **HANDLED.** `_prepare_costs` fills invalid
   queries and non-finite costs with `float32_max / 10` ≈ 3.4e37, which would have destroyed
   the fp32 dynamic range, since auction prices accumulate in units of the cost difference.
   The device path never sees it: it takes the raw costs and normalises each problem's allowed
   entries onto [0, 1], writing `num_rows + 1` into the forbidden ones — enough to beat any
   feasible assignment (which costs at most `num_rows`) without being astronomically large.
   The affine normalisation is exact because every permutation sums exactly `num_rows` entries.
   Covered by `test_auction_is_scale_invariant` over 1e-3…1e6.
2. **Degenerate cost matrices cause auction thrashing.** The real matcher costs have many
   near-equal entries. Synthetic ties are handled (`test_auction_handles_degenerate_costs`,
   costs rounded to 4 levels, exact with no fallbacks), and the tie-break is deterministic:
   the highest-indexed of the tied bidders wins the column, so exactly one row is seated even
   when the bids are bit-identical. Still, **the fallback rate on real matrices is the number
   to watch in Phase 1**, not the synthetic one. `max_iters` plus the host fallback bounds the
   damage to a slowdown rather than a wrong answer.
3. **fp32 precision.** The host path solves in fp32 too, so this is not a regression, but the
   float64 reference in the equivalence test may disagree on genuinely near-tied assignments.
   The acceptance criterion (equal *cost*, not equal permutation) is chosen for exactly this.
4. **Added GPU work on a saturated GPU.** On the L4 this is expected to be a small net loss.
   That is the whole reason for the opt-in default; Phase 3 quantifies it.
5. **GPU memory.** Each bidding round materialises a `[batch, max_targets, num_queries]`
   value tensor and its top-2, alongside the cost tensor itself. Cropping to
   `max(num_valid_targets)` keeps that well under the 921.6 MB of the uncropped costs at the
   expected ~50 targets, but it scales with the *largest* event in the batch, not the mean —
   so a single dense event sets the peak. Worth watching at batch 2048.
6. **Residual syncs.** Three remain (listed in §4). Under
   `torch.cuda.set_sync_debug_mode("error")` the device path should show only those; that is a
   cheap and decisive test that the stall is really gone.
7. **The auction's round count is data-dependent, and only measured on synthetic costs.**
   Rounds scale roughly with 1/ε and with how competitive the problem is. Real costs could be
   much worse than uniform random ones, and a solver that needs 10× the rounds is a slower
   host path with extra steps. Phase 1 on real matrices settles it.
8. **The auction collapses on square problems, and the crop is what decides squareness.**
   **CONFIRMED, 2026-08-15** (job 40078074, synthetic, B200). The speedup is not a property of
   the solver but of the *aspect ratio* of the problem it is given:

   | shape | device vs host |
   |---|---|
   | 10,240 × 150 preds × 50 targets — *(mislabelled "the production geometry"; it is not)* | **6.4× faster** |
   | 1,024 × 150 × 50 | 8.9× faster |
   | 1,024 × 50 × 50 (square) | 7× **slower** |
   | 1,024 × 150 × 150 (square) | 36× **slower** |
   | 10,240 × 150 × 150 (square, production batch) — **the closest row to production** | **107× slower — 213 s per call, 6 non-convergences** |

   The mechanism is the auction's: with 150 slots for 50 bidders there is slack, and rounds
   converge almost immediately; at 150-into-150 every bidder contends for every seat and the
   price war runs long. Exactness holds throughout (≥99.8%, worst excess 1.2e-7, zero
   fallbacks) — this is a *speed* cliff, not a correctness one.

   Why this is a live risk and not a curiosity: `_prepare_costs` crops the target axis to
   `max(num_valid_targets)` **over the batch**, so a single dense event drags all 10,240
   problems toward square. **MEASURED 2026-08-25 and it is worse than "a live risk": the median
   crop at batch 2048 is 146 against 150 queries, and the lowest in 200 draws was 141**
   (`crop_distribution.py`). CLIC averages ~50 targets, but the tail is what sets the crop, and
   the penalty for landing there is 200×, not 2× — a 1.1 s step becomes a 213 s one, which
   presents as a hung run rather than a slow one. The square production row is also the only
   case in the sweep the auction failed to converge on (6 fallbacks), so `max_iters` is doing
   real work there rather than merely being generous.

   **A guard is therefore a precondition for deploying this, not a refinement**: measure the
   distribution of `max_targets` per batch — the dumped tensor's manifest (`targets_max`)
   reports it — and route batches whose crop approaches `num_queries` back to the host.

## 8. Related work

Absent from this plan until 2026-08-25, which is why the auction's failure mode arrived as a
surprise rather than a prediction. GPU LAP is a mature literature.

- Bertsekas, *Auction Algorithms* — the auction is pseudo-polynomial without ε-scaling;
  bidding rounds go as `C/ε`. With scaling, `O(nm log(nC))`.
  <https://web.mit.edu/dimitrib/www/Auction_Encycl.pdf>
- Crouse, *On implementing 2D rectangular assignment algorithms*, IEEE TAES 52(4), 2016 —
  the algorithm behind `scipy.optimize.linear_sum_assignment` and behind
  `torch-linear-assignment`'s CUDA kernel.
- Date & Nagi, *GPU-accelerated Hungarian algorithms for the Linear Assignment Problem*,
  Parallel Computing, 2016. <https://doi.org/10.1016/j.parco.2016.05.012>
- Lopes et al., *Fast block distributed CUDA implementation of the Hungarian algorithm*,
  JPDC, 2019. <https://doi.org/10.1016/j.jpdc.2019.03.014>
- Kawtikwar & Nagi, *HyLAC: Hybrid linear assignment solver in CUDA*, JPDC, 2024 — its
  coarse-grained "stream-solver" (one thread block per small LAP) is this workload exactly.
  <https://github.com/researchgroup-zx93920/HyLAC>
- `torch-linear-assignment` — batched CUDA LAP with a PyTorch API, natively rectangular.
  <https://github.com/ivan-chai/torch-linear-assignment>

## 9. Operational notes

- **No GPU on the login node** (`torch.cuda.is_available()` is `False` there). Everything —
  including correctness tests — runs under SLURM.
- Available in the env today: `triton` 3.5.1, `torch` 2.10.0+cu128, `scipy` 1.17.0, `lap` 0.9.4,
  `lap1015` (vendored, `src/lap1015`). **Not** available: `cupy`, `numba`.
- `torch_linear_assignment` is **built but not installed**, at
  `/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment` — put it on
  `PYTHONPATH` and set `LD_LIBRARY_PATH` to the pixi env's `lib`. Build recipe and its three
  traps are in `NOTES.md`, 2026-08-25.
- Read [`../profiling/NOTES.md`](../profiling/NOTES.md)'s "Operational gotchas" block before
  submitting anything.

## 10. Files in this directory

| file | purpose |
|---|---|
| `README.md` | this plan |
| `NOTES.md` | chronological running log — what was actually measured |
| `submit_phase0_matcher_share_b200.sh` | Phase 0 gate: the matcher's share of a B200 step |
| `bench_device_matcher.py` | Phase 1 offline correctness + speed benchmark |
| `crop_distribution.py` | the per-event target distribution and the crop it produces per batch — no GPU needed. **Reports a ceiling**, not the crop: it counts particles, the matcher crops to non-resonance particles |
| `bench_jv_solver.py` | Phase 1 for `torch-linear-assignment`: exactness and speed on a dumped tensor |
| `submit_bench_jv_b200.sh` | the above on one B200, replaying an existing dump (no training stage) |
| `.gitignore` | keeps the ~923 MB dumped tensor and the incidental checkpoints out of git |
| `submit_bench_device_matcher.sh` | Phase 1 on one B200 |
| `submit_real_cost_replay_b200.sh` | Phase 1 step 2: dump one step's real costs, then replay them |
| `submit_paired_device_matcher_b200.sh` | Phase 3 paired A/B, both arms in one allocation |
| `phase0_logs/` | Phase 0 outputs, stamped with the SLURM job id |
| `phase1_logs/` | the dumped real cost tensor and its manifest |

And what it touches outside this directory:

| file | purpose |
|---|---|
| `src/hepattn/models/device_lap.py` | the batched auction solver and the permutation builder |
| `src/hepattn/models/matcher.py` | `DEVICE_SOLVERS`, the `device_solver*` config keys, `_match_on_device` |
| `src/hepattn/callbacks/matcher_timer.py` | `MatcherTimer` — the Phase-0 sync-bracketed attribution |
| `src/hepattn/callbacks/matcher_cost_dump.py` | `MatcherCostDump` — lifts one step's real cost tensor out for offline replay |
| `src/hepattn/experiments/clic/configs/profile_phase0.yaml` | Phase 0 overlay: production settings, no profiler |
| `src/hepattn/experiments/clic/configs/dump_matcher_costs.yaml` | Phase 1 overlay: dump one step's costs, then stop |
| `tests/callbacks/test_matcher_cost_dump.py` | the dump's contract with the replay reader |
| `tests/matching/test_device_solver.py` | exactness vs scipy, CPU and (marked) GPU |
| `src/hepattn/experiments/clic/configs/clic_v6_cudamatch.yaml` | Phase 3 arm B — `clic_v6_maskfix.yaml` with the matcher block swapped |
