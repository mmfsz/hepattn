# CUDA matching — running log

Chronological. The plan and the decision rules live in [`README.md`](README.md); this file is
the record of what was actually measured, in the order it happened.

---

## 2026-08-13 — Phases 1a and 2, offline

Solver written and integrated. Summary of what the README already records in detail:

- `src/hepattn/models/device_lap.py` — batched Jacobi auction, pure torch, no new dependency.
- `Matcher(device_solver="auction", ...)`, opt-in, default `None`.
- Exact against scipy on every synthetic shape tested, zero fallbacks, on CPU.
- **ε-scaling with carried prices is wrong for rectangular problems** (README §3). A single
  phase at the final ε is both exact and 5–9× cheaper in rounds, and is the default.

Nothing measured on GPU: there is none on the login node, so everything goes through SLURM.

---

## 2026-08-14 — Phase 0 submitted (GATE)

**The question.** What fraction of a B200 training step does the matcher own? A device solver
cannot buy back more than that fraction, and no current number for it exists — every
attribution in [`../profiling/NOTES.md`](../profiling/NOTES.md) predates the mask-loss fix,
which made the step ~1.7× faster on this hardware and so changed the denominator.

**Why not just read a trace.** The profiler reports GPU-idle time, but idle time is the sum of
*all* host-side cost in the step: the matcher, the Python of the loss loop, the optimiser, the
logger. The gate turns on the matcher's share alone, so it needs its own timer.

**Method.** `hepattn.callbacks.MatcherTimer` (commit `914a9d8`), driven by
[`configs/profile_phase0.yaml`](../../../configs/profile_phase0.yaml) and submitted with
[`submit_phase0_matcher_share_b200.sh`](submit_phase0_matcher_share_b200.sh).

Explicit `torch.cuda.synchronize()`-bracketed timers, bucketing each step into:

| bucket | what it is |
|---|---|
| `prep` | device-side sanitising, query masking, transpose, crop — plus the 4-byte read of the per-event target counts, which is the sync that drains those kernels |
| `dtoh` | the pinned-staged device→host copy of the cost tensor (~1 GB/step). **The transfer the device solver removes outright** |
| `solve` | the host LAP solve (`lap1015_late`, 16 threads) |
| `device` | the whole device-solver path, when running the `cudamatch` arm |
| `other` | the rest of the step, by subtraction |

The leading sync on entry to `Matcher.forward` is the part that makes the numbers mean
anything. The host path's first blocking operation drains every kernel queued earlier in the
step, so without it the matcher's transfer is charged for all the GPU work preceding it. With
it, the GPU is caught up before the timer starts, and everything after is time the GPU has
nothing to run — which is exactly what a device solver would be competing for.

**Protocol deviation, deliberate.** The README sketched "reuse the Phase-2/3 protocol". That
protocol runs eager (Compile removed) under `PyTorchProfiler`, which inflates the host side —
fine for reading the *composition* of GPU work, wrong for a gate on a host-side *fraction*.
Phase 0 runs production settings instead: compiled encoder/decoder, no profiler, batch 2048,
16 CPUs, 90 steps with the first 40 discarded as `torch.compile` warmup.

Cost of the instrumentation: two syncs per step plus one per matcher call, which serialise host
and device and stretch absolute step times slightly. Read the fractions from this run; take
throughput from an uninstrumented one.

**Decision rule, fixed before the measurement** (README §6):

| matcher share of step | action |
|---|---|
| ≥ 20% | go — proceed to the real-cost replay, then the paired A/B |
| 10–20% | reconsider — small ceiling, but the CPU-core saving may still justify it |
| < 10% | kill — and record it here as the answer to profiling candidate 6 |

**Job 39401280**, submitted 2026-08-14, host arm (`clic_v6_maskfix.yaml`). Queued behind a
largely drained `hpg-b200` partition. Results below when it lands.

### Result

_pending._

---

## Follow-up parked: batched Jonker–Volgenant

Raised 2026-08-14, to be tried regardless of how the auction performs. JV is exact by
construction rather than exact-when-tested, and its `O(n³)` runtime is independent of cost
conditioning — which is the auction's one genuinely untested weakness, since real mask-BCE
costs are far more degenerate than the uniform-random ones the solver has been validated on.
Against it: it parallelises only across problems, not within one, and `torch-linear-assignment`
is a compiled CUDA dependency of exactly the kind `lap1015` already demonstrated the cost of
(a bad build degraded silently to ~2× slower than scipy).

Two things to check before spending a queue slot on it: whether the package handles rectangular
problems natively or wants square padding with a sentinel — the numerical trap the auction had
to be rescued from — and whether it builds in this env at all (`torch_linear_assignment` is not
currently installed). Sequenced after Phase 0 either way: if the matcher owns too little of the
step, neither solver is worth deploying.
