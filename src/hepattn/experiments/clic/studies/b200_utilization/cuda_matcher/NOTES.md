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

### Result — **GO**, and by a wide margin

Job 39401280 ran 2026-08-15 on `c0904a-s5` at commit `3e2d69b`, 6m18s wall, 90 steps, 50
recorded after 40 discarded. Full output in
[`phase0_logs/clic_v6_maskfix_20260815-T150307/`](phase0_logs/clic_v6_maskfix_20260815-T150307/).

| bucket | median | % of step |
|---|---|---|
| **matcher, total** | **745.6 ms** | **66.8%** |
| — device-side prep + length sync | 0.4 ms | 0.04% |
| — device→host copy | 14.9 ms | 1.3% |
| — host LAP solve | 712.8 ms | 63.9% |
| everything else | 371.1 ms | 33.2% |
| *step* | *1116.2 ms* | *(1835 samples/s at batch 2048)* |

The gate wanted ≳20%. The matcher owns **66.8%**, three times that, and the share is stable to
±3% across the 50 recorded steps. Proceed.

Amdahl's ceiling, taking the measurement at face value: a free device solver would leave a
~372 ms step, i.e. **3.0× throughput**. A device solver that costs 100 ms still leaves 2.4×.

**The prize is not the transfer — it is the solve.** This inverts the study's framing. §1 and
§7 of the README lead with the 921.6 MB device→host copy; it costs **14.9 ms, 1.3% of the
step**, and deleting it outright would be invisible. The 63.9% is the host solve itself. The
device path therefore has to be *fast*, not merely copy-free, and the pinned-buffer staging
that `_stage_to_host` exists for is already doing its job well enough not to matter.

### Caveat found in the same log: the host baseline was degraded

The run emitted the warning `matcher.py` exists to emit:

> The installed lap1015 extension does not release the GIL while solving, so the
> `lap1015_late` solver cannot run in parallel: threaded matching will serialise […]
> Rebuild the extension from this repo's vendored source in `src/lap1015`.

Confirmed on the login node: `lap1015.__init__` is the vendored wrapper, but `_core` is an
older compiled `.so` in `site-packages` with no `releases_gil` flag, so `lap1015.releases_gil`
is `False`. `Matcher(parallel_solver=True, n_jobs=16)` was therefore running a 16-thread pool
over a solver that holds the GIL — **the 712.8 ms is an effectively serial solve**, not the
16-thread one the config asks for. That is very close to the 3.1× gap between it and the
0.23 s the README §1 table quotes for `lap1015_late` at 16 threads.

Two consequences, and only one of them matters for the gate:

1. **The gate stands.** Substituting the README's healthy 0.23 s solve gives a matcher share of
   ≈ (0.23 + 0.015) / (0.371 + 0.245) ≈ **40%** — still double the go threshold. The decision is
   insensitive to the bug, which is the only reason this number is being reported at all.
2. **Phase 3 is blocked on it.** An A/B run today would score the device solver against a
   crippled host arm and bank a headline that evaporates the moment anyone rebuilds the
   extension. Rebuild `src/lap1015` (or move the host arm to `default_solver: scipy`, which the
   warning itself suggests) *before* the paired run, and re-take Phase 0 afterwards — it is a
   six-minute job — so the ceiling quoted above is a real one.

### Fixed, 2026-08-24 — and the root cause was worse than a stale file

The rebuild did not just fail to happen; it *could not* happen. `pixi reinstall hepattn` errors
with `Unrecognized options in config-settings: pixi-conda-environment` — pixi passes that
setting to the build backend and scikit-build-core rejects unknown settings by default. So
**`pixi install` on a fresh clone has been failing to build the extension outright**. This
environment only worked because it was created on 2026-07-28 with an older pixi; the GIL-release
commit landed 2026-07-31, three days later, and nothing has rebuilt it since.

Fixed with `strict-config = false` in `pyproject.toml` (plus a `scikit-build-core>=0.10` floor).
`pixi.lock` is untouched, so `pixi install --locked` still works. The environment is rebuilt and
`lap1015.releases_gil` is now `True`; constructing a `Matcher(parallel_solver=True, n_jobs=16)`
no longer warns.

Guarded so it cannot recur silently: `tests/matching/test_solvers.py::test_lap1015_releases_gil`
fails with an actionable message if the installed extension holds the GIL, and README.md's
`lap1015` section documents the one-line check and the fact that a plain `pixi install` will not
rebuild an already-installed package. `editable.rebuild = true` was tried as the guard and backed
out — its import hook runs under the system `cmake`, which cannot find the Python development
headers, so it turns a stale extension into an unimportable one.

**Consequence for this study: Phase 3 is unblocked, and Phase 0 is being re-taken.** The 66.8%
was measured against a serialised host solve, so it is an upper bound on a degraded baseline.
**Job 40111528**, submitted 2026-08-24, is the same job against the rebuilt extension; the
submit script now also prints `lap1015.releases_gil` into the log so every future run says which
kind of number it produced. Quote no A/B result until that lands.

Expected: the matcher's share falls from 66.8% to somewhere near 40%, and the Amdahl ceiling
with it, from 3.0× to ~1.7×.

### It went the other way — the rebuild halved throughput

Job 40111528, same config, same instrumentation, `lap1015.releases_gil: True` confirmed in the
log:

| | job 39401280 (GIL held) | job 40111528 (GIL released) |
|---|---|---|
| host LAP solve | 712.8 ms | **1760.7 ms** (std 453, was 33) |
| device→host copy | 14.9 ms | 14.9 ms |
| everything else | 371.1 ms | 359.1 ms |
| step | 1116.2 ms | 2147.7 ms |
| throughput | 1835 samples/s | **954 samples/s** |

`everything else` is unchanged, which rules out the node and the GPU side as the confound: the
solve itself got 2.47× slower. The prediction above was simply wrong, and the reason is a
second trap hiding behind the first.

**`CMakeLists.txt` compiles with `-march=native`.** That targets whichever CPU runs the build.
The login nodes — the only place `pixi reinstall` can be run interactively — are **AMD EPYC 7702
(Zen 2, no AVX-512)**; the B200 nodes are **Intel Emerald Rapids (AVX-512)**. So the rebuild
fixed the GIL and simultaneously retargeted the vector code at the wrong microarchitecture, and
the detuning cost more than the GIL release bought. The old `.so` was 264 KB against the new
224 KB, consistent with the previous one having been built on an Intel node.

Corroborating numbers on the login node, same CLIC geometry, rebuilt extension:

| solver | 1 thread | 4 | 16 |
|---|---|---|---|
| `lap1015_late` | 14.17 s | 8.86 s | 7.46 s |
| `scipy` | 4.87 s | 3.23 s | 3.33 s |

The GIL release does work — threading now scales, where before it could not — but only 1.9× on
16 threads, and the rebuilt `lap1015` is ~3× *slower than scipy in absolute terms*, which it
should never be.

This is the failure mode `matcher.py`'s `check_import_safe` comment already warns about
("compiled for the wrong arch"), and it is fatal for the study's headline claim: **any host-arm
number depends on where the extension was compiled**, which is not a property anyone would think
to record. Two things follow:

1. `-march=native` is the wrong default for a repo run across heterogeneous nodes, and
   especially for anyone cloning the fork: a fresh clone bakes in whatever CPU happened to run
   `pixi install`, then runs somewhere else.
2. Neither Phase-0 number is yet the honest baseline. 66.8% was GIL-bound; 83.3% is
   architecture-detuned. The real one needs an extension built on the node type that trains.

### The `-march` hypothesis was wrong — it is the thread count

Job 40137096, on a B200 node (`INTEL(R) XEON(R) PLATINUM 8570`, AVX-512 present), same source
built three ways and timed in one place, 10,240 problems of 150 preds × ≤50 targets:

| arm | lap:1 | lap:4 | **lap:8** | lap:16 | scipy:1 | scipy:4 | scipy:8 | scipy:16 |
|---|---|---|---|---|---|---|---|---|
| `installed` (built on AMD) | 1.192 | 0.383 | **0.241** | 1.067 | 0.354 | 0.313 | 0.339 | 1.415 |
| `native` (built on this Intel node) | 1.417 | 0.434 | **0.268** | 1.739 | 0.350 | 0.311 | 0.345 | 1.353 |
| `x86-64-v3` (portable AVX2) | 1.346 | 0.460 | **0.273** | 2.017 | 0.357 | 0.320 | 0.355 | 0.558 |

**`-march` is second-order and the AMD-built binary is not the problem.** The three arms agree
to within ~10% at 1, 4 and 8 threads, and if anything the "wrongly" built `installed` arm is the
*fastest*. The story about AVX-512 detuning does not survive contact with the measurement.

**The first-order variable is `n_jobs`, and 16 is off a cliff.** Every arm is 4–8× slower at 16
threads than at 8, and the same cliff appears in the *scipy* column, which shares no code with
lap1015 — so it is the allocation, not the solver. `--cpus-per-task=16` running a 16-thread pool
leaves nothing for anything else; the scipy control at 1–8 threads is stable to ±3% across all
three arms and then swings by 2.5× at 16, which is what saturation looks like.

That fully explains the Phase-0 regression without any appeal to architecture:

- job 39401280: GIL held → the 16-thread pool serialised to *one* effective thread → 712.8 ms.
- job 40111528: GIL released → 16 threads genuinely ran → over the cliff → 1760.7 ms.

The GIL fix was correct. It simply exposed a thread count that was never actually being used,
and that turns out to be the wrong one.

**And this may be the most consequential number in the study so far.** At 8 threads the host
solve is ~0.24–0.27 s, against the 712.8 ms that Phase 0 measured and the study has been treating
as the prize. If that carries into training, the matcher's share falls to roughly 40%, the step
to ~0.6 s, and the device solver — which took 0.174 s at this geometry in job 40078074 — would be
competing for a much smaller margin than the 3.0× ceiling in the Phase-0 write-up. **A one-line
config change may recover most of the prize with no GPU solver at all.** That has to be measured
in training, not inferred from a bench: `--cpus-per-task=16` is shared with 16 dataloader
workers, so the in-training optimum may be lower than 8 again.

Caveat on the table: the 16-thread column is noisy (the scipy control varies 0.56–1.42 across
arms that run identical code), and the node was shared. The *direction* is consistent across all
six columns and both solvers; the exact optimum is not settled by this run.

### Second correction: the cross-job `n_jobs` sweep was confounded too

Submitting one job per `n_jobs` value reproduced, in miniature, the exact mistake the study's
paired-A/B design exists to prevent. Every arm landed on a different shared B200 node:

| job | GIL | n_jobs | solve | `other` | step | samples/s | node |
|---|---|---|---|---|---|---|---|
| 39401280 | held | 16 | 712.8 | 371.1 | 1116.2 | 1835 | c0904a-s5 |
| 40111528 | free | 16 | 1760.7 | 359.1 | 2147.7 | 954 | c1100a-s5 |
| 40137764 | free | 8 | 1024.1 | 370.2 | 1425.0 | 1437 | c0910a-s15 |
| 40137765 | free | 4 | 1514.3 | 358.5 | 1900.1 | 1078 | c1010a-s5 |

The `other` bucket — everything that is not the matcher — is flat at 358–371 ms across all four,
so the GPU side really is comparable node to node. The *solve* swings 2.5× and is
**non-monotonic in `n_jobs`** (16 → 1761, 8 → 1024, 4 → 1514). A thread-count effect cannot be
non-monotonic like that; shared-node CPU contention can, and these nodes are shared (one showed
26 of 112 CPUs allocated to other jobs while we held 16).

So the "8 threads is the sweet spot, a one-line config change recovers the prize" reading of job
40137764 is **withdrawn**. It was one point on one node, over-read. What survives from the bench
is narrower and still worth having: at a *fixed* allocation on a *single* node, 16 threads was
4–8× worse than 8 for both lap1015 and scipy, so 16 is very likely wrong — but by how much, and
what is right, is not established.

Job 40138527 (`submit_phase0_njobs_sweep_b200.sh`) sweeps 1/2/4/8/16 **inside one allocation**,
which is the only way this question can be asked honestly here. Until it lands, the only Phase-0
number that should be quoted is the original 66.8%, with the caveat that its host arm was
GIL-serialised.

### RESOLVED: the warning was a false negative, and Phase 0's 66.8% stands

Job 40138527, all five arms in one allocation on `c0910a-s15` (Intel Xeon 8570),
`lap1015.releases_gil: True`:

| n_jobs | solve (ms) | `other` (ms) | step (ms) | speedup vs 1 thread |
|---|---|---|---|---|
| 1 | 5595.7 | 369.9 | 5977.6 | 1.00× |
| 2 | 2873.1 | 360.5 | 3252.3 | 1.95× |
| 4 | 1643.2 | 361.8 | 2030.8 | 3.41× |
| 8 | 894.4 | 363.7 | 1282.6 | 6.26× |
| **16** | **783.3** | 366.4 | **1187.2** | **7.14×** |

Monotonic, `other` flat to ±1.5%, and near-linear to 8 threads. Two things follow, and the
second one undoes most of this day's detour.

**`n_jobs: 16` is right.** There is no cliff in training — the bench's 16-thread collapse was an
artifact of benchmarking a solve loop with nothing else running, not of the training workload.
16 beats 8 by 14%; 8 already captures 88% of the benefit, which is worth knowing if CPU cores
ever need to be traded for dataloader workers, but the config as written is correct.

**The original Phase-0 host arm was never degraded.** Job 39401280 solved in **712.8 ms with
`n_jobs=16`**. A genuinely GIL-serialised solve on this hardware costs **5595.7 ms** — the
`n_jobs=1` row. 712.8 ms is a *threaded* number, right alongside this sweep's 783.3 ms for the
same setting on a different node. So the extension in that environment was releasing the GIL all
along.

Why it warned anyway is written in the commit that added the warning, `97b10db` (2026-07-31):

> the flag is build-time, so it reports false on an extension that was **patched in place rather
> than rebuilt**, which is the case in the current environment until it is reinstalled.

That is exactly what was there: a hand-patched `.so` dated 2026-07-28, with its pre-patch backup
(`_core...so.bak`) sitting beside it. The patch made the binary release the GIL; it could not add
a compile-time attribute. `Matcher` saw a missing flag and warned. The warning was accurate about
the *flag* and misleading about the *behaviour*, and this log read it as the latter.

**Consequences.**

1. **Phase 0's 66.8% stands as measured**, and so does the 3.0× Amdahl ceiling. The "≈40% against
   a healthy host solver" caveat recorded earlier in this file is **withdrawn** — there was no
   sick host solver.
2. **Phase 3 was never blocked.** The A/B can proceed on the original protocol.
3. The rebuild was still worth doing, but for reproducibility rather than speed: the environment
   now runs an extension built from `src/lap1015` instead of a hand-patched binary nobody could
   reproduce, and it performs the same (783.3 vs 712.8 ms, inside node-to-node scatter).
4. The genuinely valuable fix of the day is unrelated to any of this: `strict-config = false`,
   without which `pixi install` cannot build the extension at all on a fresh clone.

**Superseded — kept because the reasoning was wrong in an instructive way.** The paragraphs
above about `-march=native` retargeting the vector code were the hypothesis this job was built to
test, and it failed. `-march=native` remains a genuine reproducibility wart for anyone cloning
the fork — a fresh clone still bakes in whatever CPU ran `pixi install` — but it is not worth a
performance-motivated change, and it did not cause the regression.

**Measuring before changing anything** (`bench_lap1015_march.py`,
`submit_bench_lap1015_march_b200.sh`): one B200 job builds the same source three ways — the
installed `.so`, a fresh `-march=native` meaning *that* node, and a portable
`-march=x86-64-v3` AVX2 baseline every node supports — and times each over a 1/4/8/16-thread
sweep against scipy. `installed` vs `native` prices building in the wrong place; `native` vs
`x86-64-v3` prices portability. Each arm compiles into scratch and is loaded by file path, so
the shared environment is not modified and the job is safe to run alongside other work.

Also worth recording, since it cost time to rule out: the `LAP solver returned an invalid
permutation` warning appears exactly **once** in each of the two Phase-0 logs, so it is not a
regression and not the cause. Its origin is narrow — `lap_late` returns an invalid result on
about 1.5% of *single-row* (one valid target) problems and is exact on everything else, all
shapes from 2×150 to 150×150 and square up to 2450×2450. Synthetic benchmarks that draw target
counts from 1 will trip it thousands of times and mistake solver fallbacks for solver slowness;
`bench_lap1015_march.py` draws from 2 for that reason.

Separately, the 64 `tests/matching/test_solvers.py::test_lap1015` failures were **not** the same
root cause and survived the rebuild. `lap_late` — the only entry point hepattn exposes — is exact
against scipy at every size from 3 to 2450. `lap_early`, the OpenMP variant, returns an all-`-1`
(unassigned) solution at *every* size tried, 3×3 through 1000×1000: it does not work at all. That
is presumably why `SOLVERS["lap1015_early"]` is commented out in `matcher.py`, so nothing in
hepattn can select it and production is unaffected. The test was asserting a path the library
does not expose; it now pins `lap_late` and carries the OMP path as an `xfail` with that reason,
so the suite is green on a fresh clone and a build that ever fixes OpenMP reports `XPASS`.

---

## 2026-08-24 — Phase 1 on GPU: exact, and 6.4× at the production geometry

Job 40078074, `bench_device_matcher.py --n-jobs 16 --sample 512` on one B200 (`c1006a-s25`).
Synthetic uniform costs. `host (s)` is the whole host matcher call on GPU-resident costs —
prep, the device→host copy, and the 16-thread scipy solve — so the two columns are like for
like.

| case | exact | worst excess | host (s) | device (s) | speedup | fallbacks |
|---|---|---|---|---|---|---|
| b=1024 q=50 t=50 | 100.00% | 0 | 0.1126 | 0.8245 | **0.14×** | 0 |
| b=1024 q=150 t=50 | 100.00% | 0 | 0.1231 | 0.0139 | **8.86×** | 0 |
| b=1024 q=150 t=150 | 99.80% | 1.19e-07 | 0.1962 | 7.0086 | **0.03×** | 0 |
| **b=10240 q=150 t=50 — the production geometry** | **100.00%** | **0** | **1.1102** | **0.1735** | **6.40×** | **0** |
| b=10240 q=150 t=150 | 100.00% | 0 | 1.9988 | 213.1335 | **0.01×** | **6** |

**Exactness holds.** Zero fallbacks everywhere, and the only sub-100% cell misses by 1.19e-07
on costs of order 1 — fp32 rounding against a float64 reference, which is the disagreement the
acceptance criterion was written to tolerate (README §7 risk 3). The auction is not returning
worse assignments; it is returning the same ones in fp32.

**Speed is bimodal in the aspect ratio, and that is the real result.** The device solver is not
uniformly faster or slower — it is 6–9× faster when queries outnumber targets and 7–107× slower
when they do not. The mechanism is the auction's own: 150 slots for 50 bidders leaves slack and
the price war ends almost immediately; 150-into-150 makes every bidder contend for every seat.

The last row is the one to be frightened of. At the production *batch size* with a square
problem the device solver takes **213 seconds** against the host's 2.0 — and it is the only case
in the sweep that failed to converge at all, hitting `max_iters` on 6 of 10,240 problems (the
first non-zero fallback count this solver has ever produced). A step that costs 1.1 s today
would cost 213 s. That is not a slow arm in an A/B; that is a training run that appears to hang.

CLIC sits on the good side of the cliff (~50 targets into 150 queries), which is why the
production row wins. But `_prepare_costs` crops the target axis to `max(num_valid_targets)`
**over the whole batch**, so a single dense event drags all 10,240 problems toward square — and
the penalty for landing there is 200×, not 2×. **A guard that routes square-ish batches back to
the host is now a precondition for deploying this, not a refinement**, and the distribution of
per-batch `max_targets` is the number that decides how often it would fire. The cost dump's
manifest reports it (README §7 risk 8).

**Cross-check on Phase 0.** The host path here takes 1.11 s at the production geometry with
scipy on 16 threads and an otherwise idle node — *slower* than the 712.8 ms the Phase-0 training
run spent with a GIL-bound `lap1015_late`. Whichever host solver is used, the matcher is a
dominant fraction of the step, so the gate's verdict does not rest on the degraded build.

### Still to do in Phase 1

Real cost matrices — the case that actually settles risks 2 and 7, since uniform-random costs
are far better conditioned than mask-BCE ones. Nothing implemented that, so:

- `hepattn.callbacks.MatcherCostDump` snapshots one step's real cost tensor to a `.pt` in the
  schema `bench_device_matcher.py --costs` already expected (the `--dump-from` its docstring
  referred to never existed), plus a `.json` manifest of shapes, target counts and cost spread.
  Tests in `tests/callbacks/test_matcher_cost_dump.py` pin that contract.
- [`configs/dump_matcher_costs.yaml`](../../../configs/dump_matcher_costs.yaml) drives it and
  stops training as soon as the file is written.
- [`submit_real_cost_replay_b200.sh`](submit_real_cost_replay_b200.sh) does both stages in one
  allocation — dump, then replay — because the queue is the expensive part. The `.pt` outlives
  the job, so every later re-run of the replay is free.
---

## 2026-08-25 — The "production geometry" is not the production geometry

Prompted by a question that should have been asked on 2026-08-13: if the matcher's problems are
rectangular, where do the 150 × 150 matrices in the sweep come from?

They come from the crop. `_prepare_costs` crops the target axis to `max(num_valid_targets)`
**over the whole batch** (`matcher.py:330-331`), so the shape handed to the solver is set by the
densest event in the batch, not the average one. The study has been quoting "~50 valid targets
into 150 query slots" throughout — README §1, §3, and the label on the winning row of job
40078074. That is the mean of the *per-event* particle count. It is not the shape of anything
the solver is ever given at batch 2048.

Measured with [`crop_distribution.py`](crop_distribution.py) on `val_clic_fix.root`, 24,966
events surviving the dataset's own `>= 150 particles` cut:

| statistic | particles/event |
|---|---|
| mean | 50.5 |
| median | 47 |
| p90 | 79 |
| p99 | 117 |
| p99.9 | 140 |
| max | 149 |

The mean is where the "~50" came from and it is correct. The tail is the problem, and it runs
right up to the query count — 149, and only because `pflow_data.py:120` *drops* events with
≥ 150 particles, so the ceiling is an artifact of the selection rather than of the physics.

Sampling batches from that distribution, the crop is:

| batch | median crop | p10 | fraction ≥ 140 |
|---|---|---|---|
| 32 | 106 | 85 | 0.04 |
| 128 | 123 | 108 | 0.14 |
| 256 | 133 | 118 | 0.26 |
| 1024 | 143 | 136 | 0.64 |
| **2048 — production** | **146** | **141** | **0.92** |

**At batch 2048 this is not a tail risk, it is a certainty.** The median crop is 146 against 150
queries; the *lowest* crop in 200 draws was 141. Drawing 2048 events from a distribution whose
p99.9 is 140 guarantees several near-ceiling events in every batch.

### What this does to job 40078074

The row labelled **"the production geometry"** — `b=10240 q=150 t=50`, the 6.4× win — is a shape
that never occurs at batch 2048. The row that describes production is `b=10240 q=150 t=150`:
**107× slower, 213 s, and the only case in the sweep that failed to converge.** The study's
headline and its worst-case row are the wrong way round.

One thing could have rescued this, and does not. Padded target rows are inert: `row_valid`
(`matcher.py:383`) marks only real targets and `unassigned = row_valid & ...`
(`device_lap.py:172`) keeps the padding from bidding, so a 47-particle event inside a 146-row
tensor still solves as 47-into-150 and still converges fast. The crop does not make every
*problem* square. It doesn't help, for two reasons that are both in the code:

1. **Per-round work is dense over the full tensor.** `value` is built as
   `[batch, num_rows, num_cols]` every round regardless of how many rows are still bidding
   (`device_lap.py:180`), and `topk(2)` runs over all of it. Crop 146 costs ~3× crop 50 in every
   round, real rows or not.
2. **Termination is `unassigned.any()` over the entire batch** (`device_lap.py:174`). The batch
   loops until the *last* problem finishes, and the last problem is the densest event — the
   near-square, slow-converging one. 10,239 fast problems wait for it.

Dense work per round, times a round count set by the batch's densest event. That is the same
arithmetic as the all-square bench row, which is why the 213 s is probably a fair proxy for a
real batch rather than the pessimistic stress test it was filed as.

### Measured, and not

Measured: the particle distribution and the crop (above), and the bench timings at fixed
synthetic shapes (job 40078074). **Inferred:** that a real batch costs what the square bench row
costs. The reasoning above is solid but the round count for a realistic mix — one dense event
among 10,239 sparse ones, on degenerate mask-BCE costs rather than uniform random ones — has
never been measured.

Two caveats on the numbers themselves. This is the **val** file, not train; same sample, but
unconfirmed. And production crops to ~146, not to exactly 150 — README §3's round-count table
puts 140 × 150 at 710 rounds against 150 × 150's 4238 and 50 × 150's 55, so 146 is well into the
steep region but its magnitude is genuinely uncertain between "much worse than the host" and
"213 seconds".

### Consequences

1. **The guard in §7 risk 8 is not a guard, it is a kill switch.** It was scoped as protection
   against an occasional dense batch. At a 92% hit rate it would route essentially every batch to
   the host, which is the host path with a wasted device pass in front of it.
2. **Batch size is now a variable in this study, not a fixed setting.** The crop table is a
   steep function of it: 256 gives a median crop of 133 and 32 gives 106. Whether a smaller
   batch buys back the aspect ratio faster than it loses GPU efficiency is a real question, and
   nobody has asked it.
3. **The replay job changes character.** It was filed as a correctness check for risks 2 and 7.
   It is now the measurement that decides whether the device solver has a case at all, because
   it is the first time the auction will see a real crop and a real cost distribution together.
   Its manifest reports `targets_max`, which confirms the table above on train rather than val.

**Job 40228586** — `submit_real_cost_replay_b200.sh`, submitted 2026-08-25. **Landed; see below.**
The actual crop was 132, not the 146 predicted here — the val-based estimate was too pessimistic.

---

## 2026-08-25 — Phase 1 on real costs: the auction fails its own kill criterion

Job 40228586, 28 minutes on one B200. Stage 1 dumped step 45 of a real `clic_v6_maskfix` run
(923.1 MB, `[10240, 150, 150]` fp32, 30,889,950 non-finite entries); stage 2 replayed it.

| | synthetic, job 40078074 | **real, job 40228586** |
|---|---|---|
| exact vs scipy | 100.00% | **89.01%** |
| worst excess | 1.19e-07 | **1.81e-05** |
| host (scipy, 16 threads) | 1.1102 s | **0.5379 s** |
| device (auction) | 0.1735 s | **188.5029 s** |
| speedup | 6.40× | **0.003× — 350× slower** |
| non-convergences | 0 | **13,260 over 6 calls = 2,210/call, 21.6% of problems** |

Both of §6's pre-registered ship criteria fail, and its kill criterion is met:

> **Ship** — exact assignment cost on real matrices (**fallback rate < 0.1%**) …
> **Kill** — assignment cost differs from scipy, or the fallback rate is high enough that the
> CPU path dominates anyway

Measured: 89.01% exact against a required 100%, and a 21.6% fallback rate against a required
< 0.1% — **216× the threshold.** This was written down on 2026-08-13 precisely so it could not
drift afterwards, so it is applied as written. **The auction is killed.**

### What each number means

**The 11% inexact are not fallbacks.** Fallback is on, so the 2,210 non-converged problems are
re-solved exactly by scipy and cannot be the source of the disagreement. The gap comes from
problems the auction *did* converge on, to an ε-optimal rather than optimal assignment. The
worst excess of 1.81e-05 sits inside the theoretical bound (`n·ε` = 132 × 1e-6 = 1.3e-4) and is
~150× the fp32-rounding disagreement seen on synthetic costs. This is the ε-optimality gap
becoming visible for the first time, exactly where README §7 risks 2 and 7 said to look: real
mask-BCE costs are degenerate in a way uniform-random costs are not.

**The host arm got *faster* on real data** — 0.538 s against 1.110 s on the synthetic sweep.
Real problems have a median of 46 valid targets and a long thin tail; scipy exploits per-problem
sparsity that the synthetic all-dense sweep never gave it. The comparison was therefore *more*
favourable to the device path in the synthetic run than reality warrants, on both sides at once.

**21.6% non-convergence is the qualitative finding.** The auction is not merely slow on real
costs, it fails outright on one problem in five, and by the study's own phrasing that makes the
device path "a host path wearing a disguise". Note it also spends 188 s *before* handing those
back.

### Correction to yesterday's crop estimate

The manifest reports `targets per problem: min 0, median 46, max 132`. The median matches the
val-file estimate almost exactly (46 vs 47), so the distribution's body transfers. **The tail
does not: the actual crop was 132, where the val-based simulation predicted a median of 146 and
a 10th percentile of 141.** 132 is well below anything that simulation produced.

So the previous section's headline number was too pessimistic, and the caveat attached to it —
"this is the val file, not train" — was the right caveat and it mattered. The *conclusion* is
unaffected and if anything strengthened: at a crop of 132 rather than 146 the auction is still
350× slower and fails one problem in five. It did not need the worst case to break.

**Root cause of the gap, and it is a bug in the estimate.** Run against the *train* file directly,
`crop_distribution.py` gives mean 50.4, median 47, p99.9 141, and a batch-2048 crop of median 146
/ p10 141 — statistically identical to val. So the discrepancy is not train-vs-val and not
sampling noise; the script is measuring the wrong quantity.

It counts the `particle_pdgid` length. The matcher crops to `object_valid_mask`, which is
`indicator_truth` — set only for particles that are **not resonances** (`pflow_data.py:462,552`)
— and that is a strict subset. The script therefore reports a **ceiling** on the crop, not the
crop. Docstring and output now say so, and the authoritative number is the manifest's
`targets_max`, which reports the valid count directly.

Worth keeping in mind for anything downstream: the *body* of the two distributions agrees closely
(median 46 valid against 47 raw), so resonances are rare in typical events and concentrated in
dense ones — which is precisely where the crop is decided.

### Status

Phase 1 is complete and its verdict is negative for the auction specifically. **Phase 0 is
untouched** — the matcher still owns 66.8% of the step, the host solve is still 63.9% of it, and
the 3.0× Amdahl ceiling still stands. The prize is intact; the auction is not the way to it.

Phases 3 and 4 are moot for `device_solver: auction` and should not be run for it. The next
measurement is the same replay against `torch-linear-assignment` — the dumped `.pt` is on disk
and costs nothing to re-run, which is exactly what `submit_real_cost_replay_b200.sh` was
structured for:

```
phase1_logs/clic_v6_maskfix_20260825-T165908/clic_b200_matcher_costs.pt
```

---

## 2026-08-25 — The literature was never checked, and it has already solved this

Asked directly whether the square-problem collapse is a known problem and what the literature
says. It is, it does, and **this study cites no GPU LAP work at all** — §3 compares four
algorithm options without a single reference. That gap is the reason the previous section's
finding was a surprise instead of a prediction.

### The real distinction is not square vs rectangular

The host solvers (scipy's modified Jonker–Volgenant, `lap1015_late`) are **strongly
polynomial**: `O(n^2 m)`, a function of the matrix *dimensions* only. Aspect ratio moves them
smoothly — measured, 1.11 s at 150 × 50 against 2.00 s at 150 × 150, i.e. 3× the columns for
1.8× the time.

The auction is **pseudo-polynomial**. From Bertsekas' own encyclopedia entry: the number of
bidding rounds is *proportional to C/ε*, where C is the range of object values. Its runtime
depends on the **values in the matrix**, not only on its size. Squareness is not the root cause;
it is one way of forcing prices to climb far enough for C/ε to dominate. Degenerate costs are
another — which means README §7 risks 2 and 7 and risk 8 are not three risks but one, and the
replay job may find the real matrices trip it even at a favourable aspect ratio.

### And ε-scaling — the standard fix — is the thing this study switched off

With ε-scaling the auction is polynomial, `O(nm log(nC))`. Without it, pseudo-polynomial. The
2026-08-13 finding correctly established that carrying prices across phases is *invalid* on
rectangular problems, and concluded that a single phase at the final ε is "both exact and the
fastest option on every rectangular shape". True on the shapes tested — and it hard-codes the
pseudo-polynomial regime at ε = 1e-6.

§3's own table already contained the warning, in the column nobody needed at the time:

| shape | scaling, carry prices | single phase |
|---|---|---|
| 50 × 150 | **wrong** (1.6e0 excess) | 0.0, 55 rounds |
| 150 × 150 | 0.0, **3007 rounds** | 0.0, 4238 rounds |

Carry-price scaling is *both valid and fastest* when the problem is square. The two defects are
complementary: the price-carry bug needs `rows < cols`, the round-count explosion needs
`rows ≈ cols`. **~146 × 150 is the one place where both bite at once** — rectangular enough to
invalidate the carry, square enough to blow up the single phase. The crop puts production
exactly there, which is about the worst luck available.

### The design error, stated plainly

§3 justifies the auction: "Auction is naturally batched and data-parallel: every unassigned
query bids simultaneously … Jonker–Volgenant is the faster serial algorithm but the harder one
to batch."

That is the right argument for **one** large LAP. Here there are **10,240 independent** ones.
Inter-problem parallelism is already 10,240-way, which saturates a B200 by itself; intra-problem
parallelism buys nothing that is needed, and it is the only thing the auction offers in exchange
for surrendering strong polynomiality. One thread block per problem running an exact
Hungarian/JV is immune to the aspect ratio *and* to cost degeneracy at once, because its runtime
does not depend on the values.

This is the standard design in the literature, not a novel idea:

| work | what it is |
|---|---|
| Date & Nagi, *Parallel Computing* 2016 | the foundational GPU Hungarian; parallel augmenting-path search |
| Lopes et al., *JPDC* 2019 | block-distributed CUDA Hungarian |
| **Kawtikwar & Nagi, HyLAC, *JPDC* 2024** | state of the art, MIT-licensed. Two granularities, and the coarse one is **this workload verbatim**: a "stream-solver" for a *list of small LAPs, each solved by a single thread block* — reported **22.59× faster than prior solutions**. Docs are square-only |
| `torch-linear-assignment` (ivan-chai) | batched CUDA solver with a PyTorch API. Implements **Crouse (2016)** — *the same algorithm scipy uses* — and is **natively rectangular** |

### Built it: `torch-linear-assignment` compiles here

Against the parked follow-up's two open questions (`NOTES.md`, "Follow-up parked"): it handles
rectangular problems **natively**, with no square padding and so none of the sentinel trap that
risk 1 had to rescue the auction from; and it **builds in this environment**. Both answered.

The wrapper transposes when `rows > cols` and returns `-1` for unassigned rows, so both
orientations work. Verified on the login node and *inside `pixi.sif`*, which is what the SLURM
jobs actually run:

```
4x3 (rows > cols) -> [-1, 1, 2, 0]      # one row necessarily unassigned
3x5 (rows < cols) -> [0, 1, 4]          # our orientation: every target seated
```

Build recipe, into `/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment` so the
shared pixi env is **not** modified (same discipline as the `-march` arms):

```bash
module load cuda/12.8.1
git clone --depth 1 https://github.com/ivan-chai/torch-linear-assignment.git
cd torch-linear-assignment
CUDA_HOME=/apps/compilers/cuda/12.8.1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=10.0 \
  pixi run -e clic python setup.py build_ext --inplace
```

Three traps worth recording, all of the same family that cost this study a day on `lap1015`:

1. **`FORCE_CUDA=1` is mandatory.** `setup.py` gates the CUDA extension on
   `torch.cuda.is_available()`, which is `False` on the login node — the only place the build can
   be run interactively. Without it the build **silently succeeds** and produces the CPU-only
   extension, which loops over scipy per problem. Exactly the `lap1015` failure mode: a working
   import that is quietly the wrong binary.
2. **`TORCH_CUDA_ARCH_LIST=10.0`** targets B200 (sm_100). There is no GPU on the build host to
   detect it from. Note the kernel's own `SMPCores()` switch has no `case 10` for Blackwell and
   falls through to a 128 default — a block-sizing heuristic, so a performance question rather
   than a correctness one, but it means the tuning has never seen this hardware.
3. **`LD_LIBRARY_PATH` must point at the pixi env's `lib`**, or the import dies with
   ``CXXABI_1.3.15 not found`` — the system `libstdc++` gets loaded ahead of the env's newer
   one. Inside apptainer this has to be passed as `APPTAINERENV_LD_LIBRARY_PATH`.

Nothing is installed into the environment and nothing in `hepattn` imports it yet; the built
tree is used by putting it on `PYTHONPATH`.

### Where this leaves the study

The auction is very likely the wrong algorithm for this workload, for a reason that is textbook
and that a literature check on day one would have caught. That does not touch Phase 0: the
matcher still owns 66.8% of the step and the host solve is still 63.9% of it, so **the prize is
intact and only the route to it is in question**.

Revised order of business:

1. Read job 40228586 when it lands — it now sizes the gap a JV solver would have to close,
   rather than deciding whether to ship the auction. Its `targets_max` also confirms the crop on
   train rather than val.
2. Bench `torch-linear-assignment` against the host path and the auction on one B200, at the
   *real* crop, with the dumped real costs. This is the measurement that matters and the tooling
   for it now exists on both sides.
3. HyLAC only if `torch-linear-assignment` is not enough: better published numbers for this exact
   granularity, but a C++/CUDA class with no Python API and square-only docs. Padding 146 → 150
   with constant-cost dummy rows is exact and standard for Hungarian/JV — a complete assignment
   gains the same constant either way — and is *not* the numerical trap that padding was for the
   auction.

---

## 2026-08-25 — Jonker–Volgenant on GPU: 2.25× the host path, and exact to fp32 rounding

Job 40252873, 36 seconds on one B200 (`bench_jv_solver.py`, replaying the same dumped tensor as
job 40228586 — no training stage, which is what the dump was for).

| | auction (40228586) | **JV, `torch-linear-assignment` (40252873)** |
|---|---|---|
| exact vs scipy | 89.01% | **99.76%** |
| worst excess | 1.81e-05 | **4.77e-07** |
| host (scipy, 16 threads) | 0.5379 s | 0.4048 s |
| device | 188.5029 s | **0.1802 s** |
| speedup | 0.003× | **2.25×** |
| non-convergences | 2,210/call (21.6%) | **none possible** — JV is strongly polynomial, there is no iteration cap to hit |

Same tensor, same crop (132 × 150), same scoring code. The two arms differ only in the solver,
and the difference is three orders of magnitude in time and 38× in worst excess.

The host column differs between the two jobs (0.538 vs 0.405 s) because they ran on different
nodes; this is the node-to-node scatter the paired-A/B design exists to control for, and it is
why the 2.25× here is a within-job ratio rather than a cross-job one.

### The exactness needs one more look before it is called a pass

99.76% is not 100%, and §6 asks for exact assignment cost. Two reasons to think this is the
benign case rather than the auction's:

1. **The excess is 4.77e-07 on costs of order 1–9**, i.e. ~5e-7 relative — squarely fp32
   rounding. The auction's 1.81e-05 was 38× larger and sat inside its *ε-optimality* bound,
   which is a different and worse thing: an approximation the algorithm is entitled to make. JV
   has no such entitlement; it is exact in exact arithmetic, so any gap here is arithmetic
   precision, not algorithm.
2. This is what README §7 risk 3 predicted and what the acceptance criterion was written for —
   "equal total assignment cost against scipy in float64 … not an identical permutation". The
   bench's threshold is `1e-9` *relative*, which is tighter than fp32 can deliver and tighter
   than §7 risk 3 intends.

**Not resolved, and not to be waved through.** The check that settles it: re-score the 0.24% in
float64 through the same preparation, and confirm the disagreements are near-ties where two
optima exist rather than a systematic bias. Until that is done, quote this as "exact to fp32
rounding", not as "exact".

### Amdahl, and what it does not tell us

Substituting the 180 ms solve into the Phase-0 step (1116.2 ms, of which 712.8 ms is the host
solve) gives a ~583 ms step, i.e. **~1.9× throughput**. Two reasons that is an estimate and not
a result:

- The offline host solve here is 405 ms against the 712.8 ms measured *inside* training, where
  the same 16 cores also feed 16 dataloader workers. The in-training host number is the right
  denominator and it is the larger one, which flatters the substitution.
- The 180 ms is measured on an **idle** GPU. In training the same GPU is running the model, so
  JV contends for it in a way the host solver never did. This is the one structural advantage
  the host path keeps, and it cannot be measured offline.

Only Phase 3 settles it. But 2.25× on a like-for-like offline comparison, against an arm that
was 350× the wrong way, is the first result in this study that makes Phase 3 worth a queue slot.

### Where the study stands

- **The auction is dead** (job 40228586, §6 kill criterion).
- **JV clears Phase 1** on speed, and on exactness modulo the fp64 re-score above.
- **Phase 2 is now the work**: `torch-linear-assignment` behind a `device_solver: jv` key, with
  the two numerical decisions from `bench_jv_solver.prepare()` carried into `matcher.py` —
  `num_rows + 1` for forbidden entries rather than the `float32_max / 10` sentinel, and
  constant-cost padded rows. Both are load-bearing and neither is obvious from the host path.
- **Phase 3** then runs the paired A/B exactly as designed, with `clic_v6_cudamatch.yaml`
  repointed at the new key.

Open, and worth deciding before Phase 2: the dependency question §3 raised and never resolved.
`torch-linear-assignment` is a compiled CUDA extension of the same class as `lap1015`, built
out-of-env at `/blue/avery/m.mazza/projects/fastml/vendor/`, and three of its build traps are
recorded above. Vendoring it into `src/` the way `lap1015` was is the obvious move if this ships.

---

## 2026-08-26 — float64 settles the exactness question: Phase 1 is a clean pass

Job 40290161, 72 seconds on one B200. Both arithmetic arms on the same node, replaying the same
dumped tensor as jobs 40228586 and 40252873.

| | float32 | float64 |
|---|---|---|
| exact vs scipy | 99.76% | **100.00%** |
| worst excess | 4.77e-07 | **0.00e+00** |
| device solve | 0.1789 s | 0.2382 s |
| host (scipy, 16 threads) | 1.8791 s | 1.6551 s |

**The 0.24% was fp32 rounding, and the disagreements were near-ties.** The five disagreeing
problems of 2048 sat at a *relative* excess of 2.80e-09 median, 3.93e-09 max — one to two orders
below float32's 1.19e-07 epsilon — and they vanish entirely in float64. That is what README §7
risk 3 predicted and what its "equal total cost, not an identical permutation" criterion was
written for. §6 is met. Nothing systematic, so **Phase 2 proceeds**.

Also worth knowing: **float64 costs only 1.33× the solve** (0.238 s vs 0.179 s). This kernel is
branch- and latency-bound rather than flop-bound, so double precision is affordable here in a way
it is not in the model. Production stays float32 — it is exact to rounding on real costs, and the
matcher's job is to pick *an* optimum, of which ties have several — but if a future cost function
ever produces genuinely ill-conditioned matrices, the fix is one flag and 60 ms, not a redesign.

### The diagnostic nearly measured nothing

`batch_linear_assignment` does this before dispatching:

```python
if not isinstance(cost, (torch.FloatTensor, torch.DoubleTensor)):
    cost = cost.to(torch.float)
```

Those legacy tensor types are **CPU**-keyed, so *any* CUDA tensor fails the check — a CUDA
float64 tensor included, which is then silently downcast to float32. Run through the public API,
the float64 arm would have reproduced the float32 numbers exactly and "confirmed" the rounding
hypothesis by construction. The float64 path in `bench_jv_solver.solve_jv` therefore calls
`torch_linear_assignment._backend` directly; the kernel is templated and its
`AT_DISPATCH_FLOATING_TYPES` covers double, so it is the same solve without the cast. The float32
arm still goes through the public wrapper, where the cast is a no-op, so it stays the call job
40252873 measured.

Third silent degradation this dependency family has produced, after the `FORCE_CUDA=1` build and
the hand-patched `.so` GIL warning. The pattern to keep: **anything in this stack that can quietly
do the wrong thing must be made to fail loudly instead.**

### Do not quote the 10.5×

The host arm on this node took 1.879 s (float32), against 0.405 s in job 40252873 and 0.538 s in
40228586 — same script, same tensor, same `n_jobs=16`, different node. The device arm, by
contrast, reproduced to three digits (0.1789 s vs 0.1802 s).

So the offline "speedup" column is a property of the node the host arm lands on, not of the
solvers: 2.25× and 10.5× are the same measurement. **The device solve is ~180 ms and that number
is solid; the host solve is 713 ms measured inside training, and that is the only host number
worth substituting into Amdahl.** This is exactly what the paired A/B design exists to control
for, and one more reason Phase 3 is the measurement that decides the study.

---

## 2026-08-26 — Phase 2: `device_solver: jv`

The JV solver is now a config key rather than a benchmark script.

- **`device_lap.batched_jv`** — same call signature and the same `(assigned, solved)` contract as
  `batched_auction`, so `Matcher._match_on_device` needed no restructuring. It carries
  `bench_jv_solver.prepare()`'s two load-bearing decisions verbatim: forbidden entries at
  `num_rows + 1` after the per-problem affine normalisation (reusing `_normalise`, which the
  auction already needed for the same reason — a `float32_max / 10` sentinel destroys JV's fp32
  duals just as it destroyed the auction's prices), and **constant-cost padded rows**, because a
  batched solver assigns every row it is handed and cannot let padding abstain the way the
  auction's non-bidding rows do.
- **`solved=None` means "cannot come back short".** The auction's path ends in
  `bool(solved.all())`, which is a device sync on every matcher call — acceptable when the
  alternative is trusting an epsilon-optimal solver blindly, but pure loss for a solver that is
  exact by construction. Returning `None` skips both the check and the sync, and the sync is
  precisely the host stall the device path exists to remove.
- **The dependency fails loudly.** `require_jv()` runs from `Matcher.__init__` via
  `DEVICE_SOLVER_CHECKS`, so a missing build raises at construction with the build recipe rather
  than mid-training, and a `backend.has_cuda()` check refuses a CPU-only extension instead of
  letting `batch_linear_assignment` warn and quietly solve GPU costs on the host. After the
  FORCE_CUDA trap and the downcast above, this is the third guard of the same kind and it is not
  paranoia.
- **JV needs `num_cols >= num_rows` including padding**, which the auction does not, so it raises
  rather than mangling. At CLIC (150 queries, ≤150 targets) this cannot trigger.
- `tests/matching/test_device_solver.py` is parametrised over both solvers and both device-solver
  keys; the `jv` arms skip where `torch_linear_assignment` is absent, since it is not a hard
  dependency. `Matcher().device_solver is None` still holds — **the default must not move.**
- `clic_v6_cudamatch.yaml` now asks for `device_solver: jv`.

### The extension is ABI-bound to a pixi env, and training uses a different one

`main.py` runs under the **default** env (torch 2.9.1+cu128); the offline benches run under
**clic** (torch 2.10.0). The `.so` built for one does not load under the other:

```
ImportError: .../_backend.cpython-312-x86_64-linux-gnu.so:
undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
```

So the Phase-3 A/B could not have run against the existing build at all, and the failure would
have arrived two hours and one allocation later. There are now two trees, built from the same
source with the same recipe, differing only in which env's torch they link:

| tree | env | used by |
|---|---|---|
| `vendor/torch-linear-assignment` | clic (2.10) | `submit_bench_jv_b200.sh` and the offline replays |
| `vendor/torch-linear-assignment-default` | default (2.9.1) | `submit_paired_device_matcher_b200.sh`, i.e. training |

`submit_paired_device_matcher_b200.sh` now exports `APPTAINERENV_PYTHONPATH` at the `-default`
tree and `APPTAINERENV_LD_LIBRARY_PATH` at the default env's `lib`. Neither tree is a fallback
for the other; `require_jv()` turns the wrong one into a construction-time error rather than a
mid-training one.

Open and deferred to ship time, as before: whether `torch-linear-assignment` gets vendored into
`src/` the way `lap1015` was, or stays an external build reached by `PYTHONPATH`. Phase 3's
verdict decides whether the question needs answering at all.

---

## 2026-08-26 — The device arm trains, and Phase 3 is in the queue

Job 40290986: the Phase-3 device arm on its own, 300 steps under `profile_noprof.yaml`,
completed clean. No solver warnings, no host fallbacks, losses logged as usual. So the `jv`
path works end to end inside the container, on a real training step, and not only in the
offline replay and the unit tests.

`parse_throughput.py --batch 2048` reads **2.066 it/s, 4231 samples/s** over the 250 steps after
compile. The profiling study's B200 host-arm range is 1122–1911 samples/s across nodes, so this
looks like better than 2×. **It is not a result.** It is one arm, on one node, unpaired — which
is the exact comparison the paired design exists to refuse. Quoted here only as the reason to
believe the A/B is worth its allocation.

One dead end worth not repeating: shortening the smoke run with `--trainer.max_steps=20` dies in
`configure_optimizers` with `ZeroDivisionError` from `OneCycleLR` (`wrapper.py:153`) — the
schedule's phase boundaries collapse at that length. Nothing to do with the matcher. Run the
protocol's own 300 steps instead.

**Phase 3 submitted**: jobs 40291275 (`host_first`), 40291276 (`device_first`), 40291277
(`host_first`), three allocations with alternating order exactly as
`submit_paired_device_matcher_b200.sh` prescribes. Parse each with
`../profiling/parse_throughput.py --batch 2048`, which splits on the `ARM:` markers.

---

## 2026-08-27 — Phase 3 passes: the device arm is node-invariant, and the ratio is not the result

The three paired allocations submitted at the end of the previous entry all finished, both arms
clean: jobs 40291275, 40291276, 40291277, on three different B200 nodes, both `ORDER`s, no
solver warnings, no host fallbacks, `max_steps=300` reached in all six arms.

**Provenance, because the dates do not line up.** The jobs landed on 2026-08-26 at 12:52–12:53;
the entry above was written at 12:32 that day, i.e. *before* they finished, which is why the
docs said "Phase 3 is in the queue" for a day after Phase 3 was over. The logs were parsed on
2026-08-27 with the protocol's own tool,
`../profiling/parse_throughput.py --batch 2048`, which splits each log on its `ARM:` markers.
Logs at `../../../slurm_logs/slurm-<jobid>.clic-paired-devmatch-b200.out`.

| job | node | order | arm | it/s | samples/s | 250 steps in | per-interval median | per-interval range |
|---|---|---|---|---|---|---|---|---|
| 40291275 | c0903a-s25 | `host_first` | host | 0.874 | 1790 | 286 s | 0.893 | 0.820–0.909 |
| 40291275 | c0903a-s25 | `host_first` | **device** | **2.049** | **4197** | 122 s | 2.083 | 1.852–2.273 |
| 40291276 | c1004a-s25 | `device_first` | **device** | **2.066** | **4231** | 121 s | 2.083 | 1.923–2.273 |
| 40291276 | c1004a-s25 | `device_first` | host | 0.746 | 1528 | 335 s | 0.704 | 0.667–0.877 |
| 40291277 | c1100a-s5 | `host_first` | host | 0.663 | 1358 | 377 s | 0.658 | 0.625–0.725 |
| 40291277 | c1100a-s5 | `host_first` | **device** | **2.066** | **4231** | 121 s | 2.083 | 1.852–2.273 |

Within-allocation device/host ratios: **2.34×, 2.77×, 3.12×**. The device arm wins in every
allocation and in **both orders**, so the ordering confound that the mask-fix study left open is
closed here rather than merely argued away.

### The asymmetry is the result, not the ratio

Read the two arms down the table rather than across it:

| | across the three nodes |
|---|---|
| **device** arm | 4197 / 4231 / 4231 samples/s — a **0.8% spread**, and a per-interval median of **2.083 it/s in all three** |
| **host** arm | 1358 / 1528 / 1790 samples/s — a **32% spread**, with per-interval medians of 0.658 / 0.704 / 0.893 it/s |

That asymmetry is the study's thesis stated as a measurement. A host-bound step inherits the
node's CPU, its core allocation and whatever else shares it, so it scatters; a step whose
critical path has moved onto the GPU inherits the B200, and the B200s are the same. **This is
stronger evidence that the step was host-bound than the ratio is**, because it is a structural
claim that three nodes could each have falsified and none did.

**It follows that the ratio is not a stable quantity.** 2.34×, 2.77× and 3.12× are not three
estimates of one number; they are one device number divided by whichever host node the arm
happened to draw. This is the same phenomenon the README's "Dead ends" note already records for
the *offline* comparison — 0.405 s, 0.538 s and 1.879 s for the same host call on three nodes,
which is why "2.25×" and "10.5×" are the same measurement — now reappearing inside training,
where the paired design was supposed to remove it. The pairing does remove the node confound
from the *comparison*; it cannot make a ratio whose denominator is node-dependent into a
constant.

So the honest headline is the numerator: **the device arm delivers 4231 samples/s on a B200, and
that number reproduces to 0.8% across nodes.** Quote the ratio as a range with the node
dependence attached, or not at all.

### Consistency check against Phase 0

Phase 0 (job 39401280) measured the matcher at 66.8% of a 1116.2 ms step, i.e. a **3.0× Amdahl
ceiling** for a free device solve. The mean of the three paired ratios is **2.74×, which is 91%
of that ceiling** — consistent with a ~180 ms solve replacing a ~713 ms one, and consistent with
JV keeping most of its offline speed while contending with the model for the same SMs. That
contention was the one structural advantage the host path retained and the reason the offline
numbers were never allowed to stand in for this one; it costs about 9% of the ceiling.

Two independent measurements, taken eleven days apart with different instrumentation — a
sync-bracketed attribution of one step, and end-to-end throughput over 250 — agree. Neither is
evidence for the other, but a disagreement would have meant one of them was wrong, and there
isn't one.

### Against §6, and what is still not measured

§6's ship criterion is three conditions joined by **and**: exact assignment cost on real
matrices with a fallback rate < 0.1%, **and** ≥ 10% B200 throughput, **and** no L4 regression
beyond noise.

- **Exactness: met.** 100.00% exact in float64, worst excess 0, on the real dumped tensor (job
  40290161); zero fallbacks, and `jv` has no iteration cap to fall back *from*.
- **B200 throughput: met, with a margin of two orders of magnitude over the bar.** The bar is
  ≥ 10%; the measurement is +134% to +212%.
- **L4: not tested.** The Phase-3 L4 repeat at batch 256 (§5 Phase 3's last bullet,
  `submit_maskfix_l4_3gpu.sh` as the template) **has not been run.** This is not a formality to
  be waved through on the strength of the B200 result — the entire opt-in design rests on the
  expectation that the L4 loses, and an untested arm is an open item, not an assumption. Until it
  runs, the §6 verdict is "the first two conditions are met" and nothing more.

Also asked for by §5 Phase 3 and **not done**: the `--cpus-per-task` sensitivity check. If the
critical path really has left the host, the device arm should be insensitive to dropping from 16
cores to 4, which would be a scheduling win on top of the throughput. Nothing in the three
allocations above varied the core count.

Nothing about **physics** has been measured. Phase 4 is unblocked by this result and untouched by
it; no loss curve and no jet-E IQR has been looked at for the `jv` arm.

---

## Follow-up parked: batched Jonker–Volgenant

> **UN-PARKED 2026-08-25.** Both of the "two things to check before spending a queue slot"
> below are now answered — `torch-linear-assignment` is natively rectangular and builds in
> this environment — and the section above argues it should be the primary path rather than
> the second choice. Kept as written because the reasoning that parked it is still the
> reasoning that should have promoted it.

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

---

## Phase 4 — physics equivalence. PASSED (2026-08-28)

Job **40393482**, the device arm, ran the full 200 epochs in **11 h 22 m** against the host
baseline's **33 h 03 m** (job 38598204) — the same **2.91×** the throughput A/B measured, now
over a whole training rather than 250 steps. Best checkpoint epoch 196, val_loss 3.38756,
against the baseline's epoch 198 at 3.45147. Evaluated with job 40489580, scored with
[`plot_v6_phase4_jet_iqr.py`](plot_v6_phase4_jet_iqr.py).

The two resolved configs differ in the matcher block and nowhere else — same 10,126,115
parameters, batch 2048, 1× B200, 200 epochs, seed. Both arms are scored with the **host**
solver at evaluation time, so what is compared is the two trained models, not two eval paths.

**Jet-E resolution: no detectable difference, in either convention.** Against §4.3's
thresholds of 0.0073 (global) and 0.0164 (high-E bin):

| device − host | mpflow | mpflow_proxy | threshold |
|---|---:|---:|---:|
| IQR global | −0.0037 | +0.0014 | 0.0073 |
| IQR 160–180 GeV | −0.0028 | +0.0028 | 0.0164 |
| IQR 0–20 GeV | −0.0015 | −0.0003 | ~0.017 (provisional) |

Every entry is inside threshold, and **the two conventions disagree on sign in both IQR rows**,
which is the §4.3 sign check reading as noise. 32,311 matched jets; bootstrap σ_stat is
0.0012 on the global IQR, so these are not underpowered.

### One thing that is not noise, and is not a matcher effect either

The **median** — energy scale, a different failure mode from resolution — sits lower in the
device arm in both conventions: **−0.0049 (mpflow), −0.0023 (proxy)**, roughly 10 σ_stat and
6 σ_stat. Same sign, so the sign check does *not* dismiss it. Three things about it:

1. **No σ_repro exists for the median.** §4.2 measured training-to-training scatter for the IQR
   only, so there is no threshold to test this against. It cannot be called a solver effect on
   the evidence here.
2. **It is not a degradation.** In `mpflow` the device arm's median is *nearer zero* (+0.0026 vs
   +0.0076) and in `mpflow_proxy` it is further (−0.0104 vs −0.0081). Better on one reading,
   worse on the other.
3. **The v7 A/B measured the epoch-to-epoch wander of this quantity directly, and it is as large
   as the gap.** Inside the v7 device run, the global median moved **+0.012** between epoch 169
   and epoch 197 — see [`plot_v7_matcher_ab_jet_iqr.py`](plot_v7_matcher_ab_jet_iqr.py).
   A quantity that wanders that much over 28 epochs of one run does not support a 0.005 claim
   between two runs.

**Recommended follow-up, and it is the same one §4.2 already asks for:** the seed run (P0.6 of
the model-size study) would give σ_repro for the median as a by-product. Until then the honest
statement is *resolution is unchanged; scale is within the unmeasured run-to-run floor.*

### §6 status after this

| condition | verdict |
|---|---|
| exact assignment, fallback < 0.1% | **met** (Phase 2; 0 fallbacks in 200 epochs of job 40393482 — while the *host* arm's lap1015 fell back to scipy twice in job 40400036) |
| ≥ 10% B200 throughput | **met**, +191% over a full training |
| no L4 regression beyond noise | **still not tested** — the Phase-3 L4 repeat has not been run |
| physics equivalence | **met** for resolution; scale within an unmeasured floor |

### Tier 0 for both pairs, and the one asymmetry between them

[`plot_matcher_ab_curves.py`](plot_matcher_ab_curves.py) draws the paired loss and
twelve-metric curve sets for **both** A/Bs (`--pair v6|v7|both`) — v6 with both arms at 200
epochs, v7 truncated at the epochs its host arm has so far.

**v7 (same day, same commit 9d5e351): no offset.** The paired val-loss difference over 175
common epochs has mean **−0.004** with sd 0.075; over the settled epochs (≥25) mean −0.016,
sd 0.042. It wanders across zero. Every metric agrees to under 1% except proxy-E and object CE
at ~3%, with mixed signs.

**v6 (24 days apart): a small, persistent offset in the device arm's favour.** Val loss
device − host is **−0.090 ± 0.059** over the settled epochs and does not cross zero after
epoch ~85; the last-10-epoch means are 3.393 vs 3.459. It shows up in the metrics too —
mask exact match +0.0046, class acc (macro) +0.0058, |E| −2.6%, |η| −4.3% — though
**|E| proxy goes the other way, +2.3%**.

**It is not a loss-definition artefact, and that was worth checking.** The two v6 arms ran at
commits 6af8682 and 9d5e351, and the `_v2` mask losses were only *committed* on 2026-08-13
(9b52077) — the August 3 run used them from an uncommitted tree, so "the objective drifted
between the arms" was a live possibility. `submit_validate_run.sh` re-scored **both**
checkpoints under one config and today's code (jobs 40495176 / 40495178):

| | recorded at training time | re-scored today |
|---|---:|---:|
| host, epoch 198 | 3.45147 | **3.45471** |
| device, epoch 196 | 3.38756 | **3.38767** |

Both reproduce, so the objective is the same and **the v6 gap is a genuine difference between
the two trained models.**

**It is still not a solver effect, on three independent grounds.** (1) The v7 pair is the
controlled one — same day, same commit, same node type — and it shows nothing; a solver that
systematically trained better models would show it there too. (2) The gap does not reach the
jets: the Phase-4 IQR differences above are inside threshold with opposite signs across the two
conventions. (3) `|E|` and `|E| proxy` move in opposite directions, which is the §4.3 sign check
failing. The most likely reading is ordinary run-to-run variation — which this project has
**never measured at fixed seed**, the same gap §4.2 flags — and it is in the device arm's favour
in any case, so it does not threaten the ship decision.

> The lesson for the next A/B is the one the throughput work already learned: **run the arms in
> the same allocation on the same day.** The v6 pair reused a 24-day-old baseline to save 33 h
> of queue, and the saving cost an unresolvable ambiguity that only the v7 pair could settle.
