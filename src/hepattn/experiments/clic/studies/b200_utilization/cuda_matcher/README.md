# CUDA Hungarian matching — study plan

> **Status: planned, nothing measured yet (2026-08-13).**
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

## 4. Config design

The device solver is a new opt-in key on `Matcher`. Default `None` reproduces today's
behaviour exactly.

```yaml
matcher:
  class_path: hepattn.models.matcher.Matcher
  init_args:
    # --- existing CPU path, unchanged, still the default ---
    default_solver: lap1015_late
    adaptive_solver: false
    parallel_solver: true
    n_jobs: 16
    # --- new, opt-in: solve on the GPU instead ---
    device_solver: auction        # null (default) => CPU path, byte-identical to today
    device_solver_max_iters: 1000 # iteration cap before falling back
    device_solver_fallback: true  # per-event fallback to `default_solver` on non-convergence
```

Design notes:

- **`device_solver: null` is the default and must remain so.** On the L4 the GPU is the
  constraint; adding solver kernels to it is expected to be neutral at best. Phase 3 measures
  this rather than assuming it.
- When `device_solver` is set, `_prepare_costs` skips the pinned-buffer staging and the
  `.numpy()` entirely and hands `_solve` a CUDA tensor. `lengths` stays on device too, so
  there is **no** `.cpu()` anywhere on the step's critical path.
- `forward` currently returns a CPU tensor (`torch.from_numpy`). The device path returns a CUDA
  tensor. `maskformer.py:359` indexes CUDA output tensors with it, so this removes an implicit
  HtoD rather than adding one — but `adaptive_solver` timing and the `assert torch.all(...)`
  both need auditing for new syncs.
- `adaptive_solver` must **not** try to time the device solver against CPU solvers in the same
  loop — different devices, and the timing would need syncs. The device path bypasses adaptation.
- The solver has a data-dependent `while` loop, so it stays outside any `torch.compile` region.

## 5. Test plan

Six steps, in order, each gating the next.

### Phase 0 — size the prize (GATE)

One profiling job on B200 with the current mask-fixed config, instrumented to attribute the
step's wall clock to (a) matcher DtoH, (b) matcher solve, (c) everything else. Reuse the
Phase-2/3 protocol in [`../profiling/README.md`](../profiling/README.md) and
[`../profiling/analyze_trace.py`](../profiling/analyze_trace.py); add explicit
`torch.cuda.synchronize()`-bracketed timers around `Matcher.forward` since the profiler's idle
attribution alone will not separate the matcher from other host work.

- **Go** if the matcher owns ≳20% of B200 step time.
- **Reconsider** at 10–20% — the ceiling is small but the CPU-core saving may still justify it.
- **Kill** below 10%, and record that in `NOTES.md` as the answer to candidate 6.

### Phase 1 — offline solver, correctness first (no training)

`bench_device_matcher.py` in this directory, run under SLURM (no GPU on the login node).

1. **Correctness on synthetic problems**: random uniform, degenerate (many equal costs),
   rectangular (n_pred > n_true), all-padded, and empty-target events. Compare *total
   assignment cost* against `scipy.optimize.linear_sum_assignment` in float64. Extend the
   existing `tests/matching/test_match_equivalence.py` pattern rather than inventing one.
2. **Correctness on real cost matrices**: dump one training step's real stacked cost tensor
   (10,240 × 150 × 150) from a B200 run and replay it. This is the test that matters — synthetic
   uniform costs are far better conditioned than real ones.
3. **Speed**: solve time vs `lap1015_late`/`scipy` at 16 threads, swept over batch (256 … 2048)
   and n (50 … 250), reported *including* the DtoH the CPU path needs and the device path does
   not.

Deliverable: a table of (exactness pass rate, fallback rate, speedup) per configuration.

### Phase 2 — integration behind the config key

Implement in `src/hepattn/models/matcher.py`. Add `tests/matching/test_device_solver.py`
(marked `gpu`). CI is CPU-only, so the CPU default path must be unaffected — verify by running
`pytest -m 'not gpu and not requiresdata'` unchanged.

Then a 20-step smoke run on B200 asserting the per-step losses match the CPU path to within
fp32 noise.

### Phase 3 — paired throughput A/B

`device_solver` is a single config key, so the study's existing paired-run trick applies
directly: copy [`../profiling/submit_paired_b200_1gpu.sh`](../profiling/submit_paired_b200_1gpu.sh),
which runs both arms back-to-back **in one SLURM allocation** and splits the log on its `ARM:`
markers. This is the only way to avoid the node-to-node confound — the B200 throughput spans
1122–1911 samples/s across nodes, which is larger than any effect we expect to measure.

- **Arm A**: `device_solver: null` (CPU, current production behaviour)
- **Arm B**: `device_solver: auction`
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

1. **The `big` sentinel will break ε-scaling.** `_prepare_costs` fills invalid queries and
   non-finite costs with `float32_max / 10` ≈ 3.4e37. Auction's bidding increments are relative
   to cost differences; a sentinel 30+ orders of magnitude above the real costs destroys the
   dynamic range and will either stall convergence or lose all precision in fp32. The device
   path needs a *tight* sentinel — e.g. `max(finite cost) + n · (cost range)` — which is
   sufficient to forbid an assignment without being astronomically large. **This must be
   handled before any correctness test is believable**, and it is the most likely source of a
   silent wrong answer.
2. **Degenerate cost matrices cause auction thrashing.** Real mask-BCE costs have many
   near-equal entries. Mitigation: ε-scaling schedule plus an iteration cap plus the CPU
   fallback; measure the fallback rate in Phase 1 on real matrices, not synthetic ones.
3. **fp32 precision.** The CPU path solves in fp32 too, so this is not a regression, but the
   float64 reference in the equivalence test may disagree on genuinely near-tied assignments.
   The acceptance criterion (equal *cost*, not equal permutation) is chosen for exactly this.
4. **Added GPU work on a saturated GPU.** On the L4 this is expected to be a small net loss.
   That is the whole reason for the opt-in default; Phase 3 quantifies it.
5. **GPU memory.** Auction needs price/bid vectors alongside the 921.6 MB cost tensor. Small
   relative to the costs, but the B200 config is already at batch 2048.
6. **New syncs sneaking in.** `assert torch.all(pred_idxs >= 0)` in `forward` is a device→host
   sync. Under `torch.cuda.set_sync_debug_mode("error")` the device path should show *no* syncs
   in the matcher; that is a cheap and decisive test that the stall is really gone.

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
| `NOTES.md` | chronological running log — created when Phase 0 starts |
| `bench_device_matcher.py` | Phase 1 offline correctness + speed benchmark |
| `submit_*.sh` | SLURM submissions, one per phase |
