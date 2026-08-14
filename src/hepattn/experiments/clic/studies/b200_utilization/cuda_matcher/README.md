# CUDA Hungarian matching — study plan

> **Status (2026-08-13): solver implemented and exact on CPU; nothing measured on GPU yet.**
>
> | phase | state |
> |---|---|
> | 0 — size the prize on B200 | **not started — this is the gate, see §5** |
> | 1 — offline correctness | **done on CPU**: exact vs scipy on every synthetic shape, zero fallbacks. GPU + real cost matrices still to run (`submit_bench_device_matcher.sh`) |
> | 2 — config option | **done**: `Matcher(device_solver="auction")`, default `None`, `tests/matching/test_device_solver.py` (18 pass), full host suite unchanged (118 pass) |
> | 3 — paired A/B throughput | not started (`submit_paired_device_matcher_b200.sh`) |
> | 4 — physics equivalence | not started |
>
> One result is already banked and is written up in §3: **the textbook epsilon-scaling auction
> is wrong for the rectangular problems this matcher poses.** It is exact only when
> `num_rows == num_cols`, and the matcher's problems are ~50 valid targets into 150 query slots.
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
> config key that defaults to off.

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
| CPU solve, 16 threads | 0.23 s (`lap1015_late`) / 0.27 s (`scipy`) |
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

## 3. Algorithm choice

| option | exact? | new dependency | verdict |
|---|---|---|---|
| **Batched auction with ε-scaling, pure PyTorch** | ε-optimal; exact when ε < 1/n on suitably scaled costs | **none** | **first choice** |
| Batched auction in Triton | same | none (triton 3.5.1 already in the env) | fallback if the torch version is kernel-launch-bound |
| Batched Jonker–Volgenant (`torch-linear-assignment`) | exact | yes — pip package, CUDA build, arch-fragile | second choice; the dependency was the original objection to this whole candidate |
| Sinkhorn / soft assignment | **no** | none | **rejected** — an approximation that changes the training objective. The profiling study's standing rule is *no approximations in the computations*. |

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

### Phase 0 — size the prize (GATE) — **SUBMITTED** (job 39401280, 2026-08-14)

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

### Phase 1 — offline solver, correctness first (no training)

[`bench_device_matcher.py`](bench_device_matcher.py), submitted with
[`submit_bench_device_matcher.sh`](submit_bench_device_matcher.sh) — there is no GPU on the
login node, so even the correctness runs go through SLURM.

1. **Correctness on synthetic problems** — **DONE, on CPU.** Random uniform, degenerate (costs
   rounded to a few levels so most entries tie), rectangular, scale-swept over 1e-3…1e6,
   padded rows, forbidden columns, non-finite entries, and empty-target events. Every case
   reaches scipy's optimum in float64 with zero fallbacks. This is necessary, not sufficient:
   synthetic uniform costs are far better conditioned than real mask-BCE costs.
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

### Phase 3 — paired throughput A/B

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
  win there would be a much bigger result than the B200 one.

Also record `--cpus-per-task` sensitivity: if the device path works, arm B should be
insensitive to dropping from 16 cores to 4, which is a real scheduling win on top of throughput.

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

## 7. Known risks

1. ~~**The `big` sentinel will break ε-scaling.**~~ **HANDLED.** `_prepare_costs` fills invalid
   queries and non-finite costs with `float32_max / 10` ≈ 3.4e37, which would have destroyed
   the fp32 dynamic range, since auction prices accumulate in units of the cost difference.
   The device path never sees it: it takes the raw costs and normalises each problem's allowed
   entries onto [0, 1], writing `num_rows + 1` into the forbidden ones — enough to beat any
   feasible assignment (which costs at most `num_rows`) without being astronomically large.
   The affine normalisation is exact because every permutation sums exactly `num_rows` entries.
   Covered by `test_auction_is_scale_invariant` over 1e-3…1e6.
2. **Degenerate cost matrices cause auction thrashing.** Real mask-BCE costs have many
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

## 8. Operational notes

- **No GPU on the login node** (`torch.cuda.is_available()` is `False` there). Everything —
  including correctness tests — runs under SLURM.
- Available in the env today: `triton` 3.5.1, `torch` 2.9.1+cu128, `scipy` 1.17.0, `lap` 0.9.4,
  `lap1015` (vendored, `src/lap1015`). **Not** available: `cupy`, `numba`,
  `torch_linear_assignment`.
- Read [`../profiling/NOTES.md`](../profiling/NOTES.md)'s "Operational gotchas" block before
  submitting anything.

## 9. Files in this directory

| file | purpose |
|---|---|
| `README.md` | this plan |
| `NOTES.md` | chronological running log — what was actually measured |
| `submit_phase0_matcher_share_b200.sh` | Phase 0 gate: the matcher's share of a B200 step |
| `bench_device_matcher.py` | Phase 1 offline correctness + speed benchmark |
| `submit_bench_device_matcher.sh` | Phase 1 on one B200 |
| `submit_paired_device_matcher_b200.sh` | Phase 3 paired A/B, both arms in one allocation |
| `phase0_logs/` | Phase 0 outputs, stamped with the SLURM job id |

And what it touches outside this directory:

| file | purpose |
|---|---|
| `src/hepattn/models/device_lap.py` | the batched auction solver and the permutation builder |
| `src/hepattn/models/matcher.py` | `DEVICE_SOLVERS`, the `device_solver*` config keys, `_match_on_device` |
| `src/hepattn/callbacks/matcher_timer.py` | `MatcherTimer` — the Phase-0 sync-bracketed attribution |
| `src/hepattn/experiments/clic/configs/profile_phase0.yaml` | Phase 0 overlay: production settings, no profiler |
| `tests/matching/test_device_solver.py` | exactness vs scipy, CPU and (marked) GPU |
| `src/hepattn/experiments/clic/configs/clic_v6_cudamatch.yaml` | Phase 3 arm B — `clic_v6_maskfix.yaml` with the matcher block swapped |
