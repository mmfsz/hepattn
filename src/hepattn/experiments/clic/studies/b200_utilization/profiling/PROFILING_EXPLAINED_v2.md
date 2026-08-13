# Where the profiling stands, explained from scratch (v2)

**Status as of 2026-08-04.** Companion to
[`PROFILING_EXPLAINED.md`](PROFILING_EXPLAINED.md) (v1) and the raw lab notebook
[`NOTES.md`](NOTES.md).

**Who this is for.** Someone who knows what a neural network is, but not how a
GPU works, what "profiling" means, or why any of this took three weeks. No GPU
background is assumed. Everything is built up from first principles, and every
number quoted is one we measured — sources are given so you can check them.

**Why a v2 rather than an edit to v1.** v1 was written *during* the
investigation and its narrative arc is "we are hunting a bottleneck". That hunt
is over: the bottleneck was found, fixed, and the fix changed the picture so
completely that v1's central diagnosis no longer describes the code we run
today. v1 remains the right document for *how we got here* — the five fix
experiments, the bug hunt, the two confident-and-wrong hypotheses. This document
describes **where we ended up**, and it is the one to read first.

---

## 1. The sixty-second version

We had an expensive GPU (a B200) that was not going noticeably faster than three
cheap ones (L4s), and we wanted to know why.

The answer turned out to be two separate things:

1. **A bug.** A tensor-shape mistake in the loss function was making the GPU do
   far more arithmetic than the maths required — by a factor equal to the batch
   size, so **2048× in the B200 configuration we profiled, and 256× in
   production**. Fixing it made the step ~1.7× faster on the B200 and less on the
   L4 (whose batch, and therefore whose waste, was 8× smaller), and did not
   change the final physics.
2. **A mismatch.** Even with the bug fixed, the B200 is *too fast for this
   workload*. It finishes its share of the work and then waits for the CPU. The
   L4, being about 9× slower, never has to wait — it is busy 85% of the time.

The practical consequence: **for our production setup (3× L4) there is no large
speedup left to find in this part of the code.** The GPU is already busy doing
ordinary, legitimate transformer maths. Going faster now means changing the
model or the numerics, not fixing plumbing.

---

## 2. Background: how a GPU actually works

Skip this section if you already know what a kernel is.

### 2.1 Two processors, one job

Every training run uses two very different processors at once:

- The **CPU** ("the host") runs the Python program. It reads data from disk,
  prepares it, and — crucially — **tells the GPU what to do**. The CPU is
  flexible and can run any code, but it is comparatively slow at bulk maths.
- The **GPU** ("the device") does the heavy arithmetic. It cannot run Python. It
  can only execute small pre-compiled programs called **kernels** — things like
  "multiply these two matrices", "add these two arrays", "compute a softmax over
  this axis". The CPU queues kernels up; the GPU works through the queue.

The mental model that makes everything else make sense:

> **The GPU is an extremely fast worker who only ever does what the CPU hands
> it.** If the CPU is slow to hand over the next piece of work, the GPU sits
> there doing nothing — and you paid for it either way.

### 2.2 The two ways a GPU wastes your money

This is the single most important distinction in the whole study.

**Waste type 1 — the GPU is idle (*starved*).** The work queue is empty. The
CPU is off doing something else: loading data, running a Python loop, or
computing something itself. The GPU has nothing to run and waits.

**Waste type 2 — the GPU is busy, but with work that shouldn't exist.** The
queue is full and the GPU is at 100% "utilization", but the kernels it is running
are pointless or wildly inefficient. `nvidia-smi` would show a happy, fully-loaded
GPU. This is the more dangerous failure, because every dashboard tells you
everything is fine.

**Our study hit both.** The mask-loss bug was type 2 — the GPU was genuinely busy,
just doing 2048× more arithmetic than necessary at the batch size we profiled at.
The B200's current state is type 1 — it is idle 69.5% of the time waiting on the
host.

### 2.3 Not all busy-work is equal: compute-bound vs memory-bound

A GPU has two resources that can run out, and every kernel is limited by one of
them:

- **Compute** — how much arithmetic per second it can do. Large matrix
  multiplications are **compute-bound**: they do a lot of arithmetic on a
  relatively small amount of data, so the arithmetic is the constraint. These are
  what GPUs are designed for and where a bigger GPU shines.
- **Memory bandwidth** — how fast it can move data between its own memory and its
  arithmetic units. Simple operations over huge arrays (add two tensors, apply a
  sigmoid, copy a tensor, convert float32→bfloat16) are **memory-bound**: the
  arithmetic is trivial and essentially all the time goes into streaming bytes
  back and forth.

Why you should care: **a bigger GPU only helps if your kernels are compute-bound.**
Memory-bound kernels are limited by how fast memory moves, which scales much less
dramatically between GPU generations than raw arithmetic does. A workload made of
memory-bound kernels will disappoint you on expensive hardware. As §6.3 shows,
a large share of what we run is memory-bound.

### 2.4 Two costs that are neither

- **Transfers.** The GPU has its own separate memory. Anything it needs must be
  copied over a comparatively slow link, and any result the CPU wants back must
  be copied the other way. Device→host is written **DtoH**, host→device **HtoD**.
  These copies are pure overhead — no maths gets done during them.
- **Synchronisation stalls.** If the CPU asks the GPU for a result *right now*
  ("give me this tensor as a NumPy array"), the CPU must stop and wait for the
  GPU to catch up. Then the GPU sits idle while the CPU does whatever it wanted
  the number for. Each such round-trip serialises the two processors, and they
  are a classic source of waste type 1. **Our Hungarian matcher does exactly this,
  once per step** — see §4.3.

---

## 3. The two GPUs, and why they behave so differently

We ran the same code on two very different pieces of hardware.

| | **NVIDIA L4** | **NVIDIA B200** |
|---|---|---|
| role | cheap, small, efficient inference card | flagship datacentre training chip |
| memory (measured from `nvidia-smi`) | **23,034 MiB** (~23 GB) | **183,359 MiB** (~180 GB) |
| power cap (measured) | **70 W** | **1000 W** |
| generation | Ada Lovelace (2023) | Blackwell (2024) |
| arithmetic throughput | ~10² TFLOP/s bf16 | ~10³ TFLOP/s bf16 |
| our batch size per GPU | 256 events | 2048 events |

The power caps alone tell the story: **one B200 draws roughly as much power as
fourteen L4s.** It is a fundamentally bigger machine — more memory, far more
memory bandwidth, and one to two orders of magnitude more arithmetic throughput.

> **A note on the arithmetic figures.** Vendor spec sheets quote these in ways
> that are hard to compare (dense vs "with sparsity", different number formats),
> so the row above is deliberately given only to an order of magnitude. Do not
> quote a precise FLOPS ratio from this document. v1 §3 states that the B200's
> *bandwidth* advantage over the L4 is "far smaller" than its FLOPS advantage;
> that relative claim depends on which spec-sheet numbers you pick and should be
> re-checked before anyone repeats it. **Nothing in this document's conclusions
> rests on it** — the conclusions rest on the measured ratio below.

### What we actually measured

Paper specs are not the point. This is what the two chips did on *our* work:

| | B200 @ batch 2048 | L4 @ batch 256 | ratio |
|---|---|---|---|
| GPU time spent per event | 0.241 ms | 2.25 ms | **B200 is 9.3× faster** |
| wall-clock time per event | 0.78 ms | 2.66 ms | **B200 is only 3.4× faster** |

The B200's *GPU* does our work 9.3× faster. But end to end it delivers only
3.4×. **The missing ~2.7× is the B200 sitting idle waiting for the CPU.** The
same host-side work (Python, data loading, the matcher round-trip) costs the same
number of milliseconds no matter which GPU is attached — but on the B200 that
fixed cost is a much larger fraction of the step, because the GPU part shrank.

This is the whole story in one sentence: **you cannot buy your way past a
host-side bottleneck.**

---

## 4. What one training step actually does

### 4.1 The six stages

Training is a loop. One iteration — one **step** — looks like this:

```
┌──────────── one training step ────────────────────────────────────────────┐
│ 1. LOAD      read a batch of collision events from disk, preprocess  CPU  │
│ 2. TRANSFER  copy the batch into GPU memory                          HtoD │
│ 3. FORWARD   run the model, producing predicted particles            GPU  │
│ 4. MATCH     decide which prediction corresponds to which truth      BOTH │
│ 5. LOSS      score how wrong the predictions are → one number        GPU  │
│ 6. BACKWARD  work out how each weight contributed to the error       GPU  │
│ 7. UPDATE    nudge every weight to reduce the error                  GPU  │
└───────────────────────────────────────────────────────────────────────────┘
```

Stage 4 is unusual and is where this project's trouble lives — see §4.3.

### 4.2 Our specific model

`hepattn` CLIC particle flow trains a **MaskFormer-style transformer**:

- **Input:** the detector hits from one collision event.
- **Encoder:** 6 transformer layers (width 256) that let every hit "look at"
  every other hit and build up context.
- **Decoder:** 4 layers holding **150 "queries"** — think of them as 150 blank
  slots, each of which will be filled in with one candidate particle.
- **Output per query:** a class ("is this a real particle, and what kind?"), some
  kinematics (energy, direction), and a **mask** — a yes/no flag for every hit in
  the event saying "this hit belongs to me".

The model is **small** — ~10 million parameters, and only **256 numbers wide** —
while the batches are **large** (2048 events per step on the B200, 256 per GPU on
the L4s). These two facts matter for different reasons, and it is worth keeping
them apart.

**The narrowness is the problem.** A large language model's matrix
multiplications are thousands of numbers on a side; that is the shape GPUs are
built for, and such multiplies are firmly compute-bound (§2.3). Ours are 256
wide, so each multiplication is small relative to the "glue" operations around it
— the normalisations, residual additions and number-format conversions catalogued
in §6.3, every one of which is memory-bound. **This ratio is set by the model's
width and does not change with batch size.** It is the reason this workload
behaves so differently from the LLM workloads GPUs are marketed for.

**The large batch is not a cause; it is a partial cure.** Each step carries a
host-side cost that does not depend on how many events are in the batch — the
weight update walks the 10M parameters regardless, and Python and kernel-launch
overhead scale with the number of operations in the model, not the data. Putting
more events into each step spreads that fixed cost over more events. See §6.5 for
what this implies about tuning the batch size.

We train in **bf16-mixed precision**: most maths is done in 16-bit "brain float"
for speed, but some parts are kept in 32-bit for numerical safety. This means the
code is constantly converting tensors between the two formats — remember this
for §6.3.

### 4.3 The matching step, and why it is special

The model outputs an *unordered set* of 150 candidate particles. The truth is
also an unordered set. Before we can score anything, we have to decide **which
prediction is supposed to correspond to which true particle** — prediction #7
might be the truth's particle #3.

This is a classic assignment problem, solved with the **Hungarian algorithm**. It
needs a **cost matrix**: a table scoring how well each of the 150 predictions
matches each true particle, computed on the GPU.

Here is the problem. **The Hungarian algorithm runs on the CPU.** So every single
step:

1. the GPU computes the cost matrices,
2. they are copied back to the CPU (a **DtoH** transfer),
3. the CPU solves 2048 assignment problems,
4. the answer goes back to the GPU.

During steps 2–4 the GPU has nothing to do. This is the synchronisation stall
from §2.4, and it was the first thing we attacked (§5).

---

## 5. How the investigation was structured

Profiling tools trade detail against distortion: the more closely you watch, the
more you slow the thing down and the less it resembles the real run. So we went
cheapest-and-coarsest first.

| phase | tool | what it can see | what we learned |
|---|---|---|---|
| **1** | `SimpleProfiler` | wall-clock time per named stage | the obvious suspect (data loading) was innocent |
| **2** | `PyTorchProfiler` | every individual GPU kernel, timestamped | GPU busy 70.6%; **~53% of that busy time was four loss kernels**; the matcher transfer cost 23.5% |
| **fixes** | A/B experiments | — | five changes tried, three kept (−31% step time on B200) |
| **the bug** | code reading | — | the real find: a broadcasting bug in the mask losses |
| **3** | `PyTorchProfiler` | as phase 2 | **re-profiled after the fixes, on both GPUs** — this document |

### A terminology warning: three different things are called "step"

This trips everyone up, so, explicitly:

- **a training step** — one pass through the seven stages in §4.1. This is the
  normal meaning.
- **`ProfilerStep`** — the profiler's label for the span it records around each
  training step. **The trace contains 6 of these for 3 real steps** (the tool
  emits a duplicate span on a second thread). If you open a trace and see six
  steps alternating between ~1.6 s and ~0.3 s, that is this artifact — *not* two
  different kinds of step. Ignore the duplicates.
- **`optimizer.step()`** — stage 7 only, the weight update. Confusingly, in our
  profiles this is one of the most *CPU*-expensive lines in the table.

Also note: the phase-3 profiles run in **eager mode**, with `torch.compile`
switched off. Compilation fuses many small kernels into bigger ones, which is
great for speed but makes the profile unreadable — you can no longer tell which
original operation cost what. We turn it off *for profiling only*. Production
runs compiled. This has consequences for how much you should trust the numbers
(§7).

---

## 6. Where we are now: the phase-3 results

Two jobs, both submitted 2026-08-03, both completed clean on the same code
(commit `6af8682`), same protocol, differing only in GPU and batch size:

- **B200**, 1 GPU, batch 2048 — Slurm job `38598205`
- **L4**, 1 GPU, batch 256 — Slurm job `38598206`

(The L4 arm uses one GPU, not the production three, because batch size is
*per GPU* — so one L4 at 256 does exactly one production GPU's work. What this
misses is the communication between the three GPUs.)

### 6.1 The pathology is gone

Same hardware (B200), same protocol, three points in the code's history:

| share of GPU-busy time | before any fixes | after matcher fixes | **now (mask fix too)** |
|---|---|---|---|
| the four mask-loss kernels | 52% | **68%** | **3.4%** |
| matcher `DtoH` transfer | 23.5% | 0.9% | 2.9% |
| actual model maths (attention + matmul) | ~7% | ~9% | **27%** |
| **total GPU work for 3 steps** | 6.02 s | 4.65 s | **1.48 s** |
| **wall clock for 3 steps** | 8.53 s | 8.54 s | **4.85 s** |

Read the last two rows together. **The amount of arithmetic the GPU performs
dropped by a factor of 4.1**, and wall clock improved 1.76×. The four loss
kernels that used to be the single largest cost in the entire training step no
longer appear in the top twelve on *either* GPU.

Note also what happened to the "actual model maths" row: it roughly **tripled**
as a share, from ~7% to 27%. The model did not get slower. The denominator
collapsed. That is what a profile looks like when a genuine pathology has been
removed — the remaining time is spent on things that *should* be there.

> **Why the middle column looks worse than the first.** After the matcher fixes,
> the loss kernels went from 52% to 68% of GPU-busy time. Nothing got worse — we
> removed the transfer overhead sitting next to them, so the same loss kernels
> became a bigger slice of a smaller pie. Percentages of a shrinking total are a
> reliable way to confuse yourself.

### 6.2 The headline: the two GPUs are now in completely different regimes

| | **B200 @ 2048** | **L4 @ 256** |
|---|---|---|
| GPU **busy** | **30.5%** | **85.3%** |
| GPU **idle** | **69.5%** | **14.7%** |

Same code, same commit, same profiling protocol. Two completely different
machines in the sense that matters.

**The L4 is nearly saturated.** It spends 85% of the wall clock actually
computing. There is at most ~15% to reclaim, and reclaiming all of it would be a
heroic effort for a 1.18× speedup.

**The B200 spends more than two thirds of its life waiting.** Not for lack of
work in principle — for lack of work *handed to it*. It finishes each batch of
kernels and then waits on Python, on data loading, and on the matcher round-trip.
The profile confirms this directly: on the B200, the weight-update stage
(`optimizer.step`) accounts for **50.9% of all CPU time** in the step, against
38.6% on the L4 — the same host-side work, occupying a much larger share.

**This retroactively explains a result that had been puzzling us.** The mask-loss
fix was worth 1.70× on the B200 but only 1.18× on the L4. That looked
inconsistent. It is not: removing GPU work only speeds you up if the GPU is what
you are waiting for. On the L4 it (nearly) is. On the B200 it is not, and never
was.

### 6.3 What the GPU spends its time on now

With the pathology removed, here is where the remaining GPU time actually goes.
(These are corrected numbers — see the box below.)

| | **B200** | **L4** | what it is |
|---|---|---|---|
| copies & dtype casts | **21.9%** | **21.6%** | moving bytes, converting bf16↔fp32 |
| elementwise (add, multiply, …) | 18.4% | 27.8% | simple maths over big arrays |
| attention (fused kernels) | 16.9% | 8.4% | the transformer's core operation |
| layer norm | 13.3% | 12.2% | normalisation between layers |
| **GEMM** (matrix multiply) | 10.0% | 18.0% | the actual "learning" maths |
| softmax (see §6.4) | 3.5% | 6.2% | attention's fallback path |
| loss kernels | 3.4% | 1.7% | *(was 52-68% before the fix)* |

The striking thing: **the single largest category on both GPUs is copying data
around and converting it between number formats** — not maths. Together,
copies + elementwise + layer norm are **~54% on the B200 and ~62% on the L4**,
and every one of those is *memory-bound* in the §2.3 sense. The genuinely
compute-bound work — GEMM and fused attention — is only ~27% on either card.

That is the deep reason a B200 disappoints on this workload. You are paying for
arithmetic throughput, and roughly three-quarters of the time is spent moving
bytes, which that money does not buy.

> **Correction to a number in `NOTES.md`.** The analysis script
> (`analyze_trace.py`) buckets kernels by matching substrings in their names, in
> order, and its "elementwise" test (`"vectorized" in name`) fires *before* its
> layer-norm test. PyTorch's layer-norm kernel is literally called
> `vectorized_layer_norm_kernel`, so **all layer-norm time was being counted as
> elementwise**, inflating that bucket to 55%. The table above uses a corrected
> classification. This does not change any conclusion — both categories are
> memory-bound and the compute-bound share is unaffected — but the "elementwise
> is 55%" figure recorded in `NOTES.md` should be read as "elementwise + layer
> norm + copies".

### 6.4 One new lead, not previously recorded

Reading the operation tables directly turned up something the earlier analysis
missed.

PyTorch has several implementations ("backends") of the attention operation. The
fast ones (*flash*, *memory-efficient*) never build the full attention matrix in
memory. There is also a slow fallback, the **math backend**, which materialises
the whole thing — correct, but far more expensive.

In our traces, **12 of the attention calls fall back to the math backend on both
GPUs.** Tracing them back to their source, they are all the same thing: the
`kv_ca` cross-attention in each of the four decoder layers — the place where the
model's predicted mask is applied as an attention mask.

They are disproportionately expensive:

| | L4 | B200 |
|---|---|---|
| math-backend time / all attention time | **44%** (266 ms of 609 ms) | **38%** (196 ms of 518 ms) |
| cost per call, math backend | 22.2 ms | — |
| cost per call, other backends | ~5.7 ms | — |

Roughly **four times the cost per call.** The likely cause is in
`src/hepattn/models/attention.py:402-405`, which converts the boolean mask into a
floating-point bias filled with `-inf`; that form of mask is one of the things
that disqualifies the fast backends.

**Why this matters more than it looks:** it is GPU work, and the L4 is
GPU-bound. Unlike every other remaining optimisation candidate — which target
idle time the L4 does not have — this one would land on the production
configuration. It is currently the most promising lead we have. It is *not* yet
established that the fast backends can be made to accept this mask; that needs
checking.

### 6.5 Would a smaller batch help the idle B200? No — the opposite

A natural reading of "the B200 is host-bound" is that we should give it less work
per step. That is backwards, and the traces say so directly.

Compare the weight-update stage across the two arms:

| | L4 @ batch 256 | B200 @ batch 2048 |
|---|---|---|
| `Optimizer.step#Lion.step`, host time per step | 239 ms | 209 ms |

An **8× difference in batch size, and essentially the same cost.** That is
expected: the Lion update walks the model's ~10 million parameters and has no
idea how many events produced the gradients. The same is true of Python
interpreter overhead and of the cost of launching each kernel — those scale with
how many *operations* the model has, not how much *data* went through it.

So a substantial part of each step's host cost is **fixed per step**. Halving the
batch halves the GPU work, but you then need twice as many steps, each paying
that same fixed toll. Writing `F` for the fixed host cost per step and measuring
from the B200's 1267 events/s at batch 2048:

| if `F` is… | throughput at batch 1024 |
|---|---|
| 0 (nothing fixed) | 1267 events/s — no change |
| 0.5 s | 968 events/s |
| 0.8 s | 848 events/s |

Any non-zero `F` makes a smaller batch strictly worse, and we have measured `F`
to be substantial. **The direction that helps a host-bound GPU is a *larger*
batch** — more events amortising the same fixed cost. The B200 has 180 GB of
memory and we are running batch 2048, so there is plausibly headroom.

Two things to be clear about before anyone acts on this:

1. **Batch size is not a free performance knob.** It changes the optimisation
   itself — effective learning rate, convergence behaviour, and potentially the
   final physics. This study needed a full 200-epoch retrain to establish that
   the mask fix was physics-neutral; a batch-size change deserves the same
   scrutiny, and the speedup would be worthless if the physics moved.
2. **A larger batch addresses the idle time, not the memory-bound share.** The
   ~54% of GPU time spent moving bytes (§6.3) comes from the model being narrow,
   and no batch size changes that.

This is untested and stated here as a prediction, not a finding. It would be
cheap to settle: one batch-size sweep on B200 (1024 / 2048 / 4096), read with the
existing `parse_throughput.py`.

---

## 7. What we should not conclude

Three honest limitations. The first two are the reason the numbers above should
not be quoted as production figures.

**1. These profiles run without `torch.compile`; production runs with it.**
Compilation fuses many small memory-bound operations into single kernels — which
is precisely the category (copies, elementwise, layer norm) that dominates the
table in §6.3. So the ~54-62% memory-bound share is an **overestimate** of what
production does, and the busy/idle split will differ too. **What transfers is the
ranking**, not the magnitudes: no pathological kernel remains, and real model
work is now a major share. Before anyone acts on the copies-and-casts number
specifically, we need a compiled-mode trace.

**2. Profiling overhead inflates the host side.** The profiler itself costs CPU
time, which is exactly the resource the B200 is short of. So "69.5% idle" is not
the production B200 idle fraction. The *direction* is solid and confirmed by
independent throughput measurements — the B200 is host-bound, the L4 is not —
but do not quote the magnitude.

**3. The L4 arm is a single GPU.** Production runs three, which must exchange
gradients every step. That communication is not in this trace and could change
the L4 picture. Measuring it needs a separate multi-GPU run.

One further caution, recorded here because we got burned by it. There is a
tempting hypothesis that the B200's poor *reproducibility* (throughput varying
1.7× between nodes, while the L4 reproduces to 0.6%) is explained by its being
host-bound: a step that waits 70% on the CPU will be at the mercy of whatever
else is running on that shared node. This is plausible and consistent with
everything above. **It is also untested.** An earlier confident explanation for a
different result (a NUMA memory-layout story) turned out to be wrong and had to
be publicly retracted. Do not promote this one to a finding without a measurement
that could have refuted it.

---

## 8. Summary for colleagues

> **CLIC particle-flow GPU profiling — summary of findings**
>
> We investigated why a single B200 barely outperformed three L4s on our CLIC
> particle-flow training, using staged profiling (coarse timers first, then full
> kernel-level traces). Each performance change was measured before/after with one
> variable altered at a time and an untouched control stage to catch confounds;
> the later measurements used a paired within-node design, because run-to-run
> variance on the B200 (~27%) exceeds most of the effects involved.
>
> **1. We found and fixed a real bug — and it was not only a speed bug.** Four
> mask-loss functions in `loss.py` performed an accidental `[B, N_valid, C]`
> broadcast, inflating the intermediate tensor by a factor equal to the **batch
> size** (2048 on the B200 config, 256 on L4). It also changed the objective the
> model was optimising: measured on real CLIC data, the BCE term was **−11.5% ±
> 0.1%** off, the dice term **+189%** late in training, and the combined gradient
> had a cosine of **0.896** against the intended one — a systematic ~26° rotation
> applied at all 5 supervised decoder outputs, every step, every epoch. The model
> demonstrably did not train on the documented objective.
>
> **2. Fixing it cut GPU work by 4.1×.** From the kernel traces (3 profiled steps,
> B200, identical protocol): total GPU-busy time fell **6.02 s → 1.48 s**, and the
> four loss kernels went from **52%** of GPU-busy time to **3.4%**. Wall clock for
> the same window fell 8.53 s → 4.85 s (1.76×).
>
> **3. In production this is worth +25.4% throughput, measured on two complete
> 200-epoch runs** — not a benchmark. Both 3× L4, `batch_size=256`, 1295
> steps/epoch, both reached `max_epochs=200`:
>
> | | June baseline (33954040) | all fixes (38469247) |
> |---|---|---|
> | total wall clock | 39 h 45 m 57 s | **31 h 54 m 42 s** |
> | final epoch (199) | 11:55, 1.81 it/s | **09:31, 2.27 it/s** |
>
> **+25.4%** on the steady-state rate, **+24.6%** on total wall clock (the latter
> includes compile warmup and validation). n=1 per arm, different nodes.
>
> **The retrain also shows the fix is physics-neutral or slightly better.** All
> jet-energy IQR differences are within 2σ and the two output branches disagree on
> the *sign*, so they are sampling noise. Re-scoring both checkpoints on the same
> validation set under the same objective, the fixed model is better on every mask
> metric (dice −2.37%, purity +0.98%, exact match +1.68%) and flat elsewhere
> (efficiency −0.04%, purity −0.19%). Importantly, the separately-tracked **rising
> jet-E IQR trend survives the fix** — both runs rise by ~+0.032 — so this bug is
> ruled out as that trend's cause.
>
> **4. That +25.4% is everything the study changed, and it does not split cleanly.**
> The only measurement of the intermediate state — matcher fixes without the mask
> fix (job 38461126, 1.93 it/s) — was taken with the **SimpleProfiler enabled**,
> whereas both endpoints above ran without it. Profiler overhead depresses that
> midpoint, so it yields only a **lower bound** on the matcher share: **≥ +6.8%**,
> with the mask fix taking **≤ +17.4%**. The L4 profiler overhead was never
> measured, so the split cannot be tightened from existing logs. Quote the total,
> not the split.
>
> **5. Along the way we fixed three latent correctness bugs** in the matcher, all
> found while trying to swap in a faster assignment solver. Events with zero valid
> targets made the solver return uninitialised memory (including negative
> indices); and non-finite entries were undefined behaviour — real cost matrices
> *do* contain `-inf`, because padded-hit logits are `masked_fill`ed before the
> cost is computed, and feeding such matrices to `lap1015.lap_late` dumped core
> when reproduced by hand on the login node. The third fix validates that a
> solver returned an actual permutation, so a bad result falls back to scipy
> instead of silently corrupting training. All three are **worth keeping at zero
> speedup**.
>
> **6. Three speed optimisations were kept**, all numerically exact or
> exactness-checked: pinned-memory staging of the matcher's device-to-host copy,
> the `lap1015_late` assignment solver, and device-side cost preparation. On the
> B200 profiling protocol (`run_training_batch`, SimpleProfiler) the progression
> was **4.43 → 3.92 → 3.53 → 3.05 s/step**, with the `backward` control flat at
> **0.627–0.632 s** across all six runs — that control is what licenses
> attributing the differences to the matcher path. Each step is n=1 on a machine
> with ~27% run-to-run variance, so the individual increments are not separately
> established.
>
> **The solver took two attempts, and this matters for deployment.** Swapped in
> as-is it was **1.76× slower** (7.80 vs 4.43 s/step): single-threaded it beats
> scipy 3.5×, but its pybind11 binding never released the GIL, so our 16-way
> threaded solve serialised while scipy scales ~11×. The fix was a **C++ change**
> — `py::gil_scoped_release` around the solve in `main.cpp`, plus a rebuild — not
> a config change. (Switching `parallel_backend` from thread to process would
> also dodge the GIL, but at 191 ms it still loses to threaded scipy's 113 ms, so
> that route was not taken.) With the GIL released it became a clear win, verified
> beforehand at 300/300 optimal-cost agreement with scipy and 4 concurrent solves
> in ~1.2× single-solve time. Only **one** candidate was actually rejected:
> raising the matcher to 32 CPU threads (3.05 → 3.72 s/step, **+21.9%** — the
> solve saturates at 16).
>
> **7. The two GPUs are now in opposite regimes.** The L4 is **85.3% GPU-busy**;
> the B200 is **69.5% idle**, waiting on the CPU. The B200's GPU does our work
> 9.3× faster per event (0.241 vs 2.25 ms) but delivers only 3.4× end-to-end
> (0.79 vs 2.65 ms) — the rest is host stall.
>
> Note this is a *consequence* of the mask fix, not an explanation of it. The
> B200 was 70.6% GPU-busy before the fix and 30.5% after: removing that much GPU
> work is what pushed it out of the GPU-bound regime. The reason the fix paid off
> more on the B200 is simply that **the bug's cost scaled with batch size**, and
> the B200 config runs batch 2048 against the L4's 256 — so there was 8× more
> waste to remove per step.
>
> **8. Recommendation: stop optimising the loss/matcher pipeline.** For the
> production L4 config the GPU is already busy with ordinary transformer work.
> The remaining shelved candidates all target idle time the L4 does not have.
> The one exception worth pursuing is the decoder's `kv_ca` cross-attention, which
> silently falls back to PyTorch's slow "math" backend on both GPUs — 12 calls
> taking **44% of all attention time on L4** (266 of 609 ms) and 38% on B200.
>
> **9. On hardware.** Three paired within-node B200 runs (legacy vs fixed, same
> allocation) gave mask-fix speedups of **1.70× / 1.55× / 1.86×**. Absolute
> throughput: 1× B200 **1497 samples/s median, spanning 1123–1910 across three
> nodes (1.70×)**; 3× L4 **1735 samples/s, spanning 1725–1735 (0.6%)**. So 3× L4
> edges out 1× B200 by 1.16× and is dramatically more reproducible. Fair verdict:
> **comparable on average, far less predictable**, not "slower".
>
> **⚠ Before anyone pulls this branch:** `base.yaml` currently sets
> `default_solver: lap1015_late`, which is only fast against a **hand-patched**
> binary in one pixi environment. With the stock build — which is what you get in
> the `clic` env, or after any `pixi install` — that setting is **1.76× slower**
> than the `scipy` default (7.80 vs 4.43 s/step, job 38206804). Revert it, or
> rebuild the extension from source,
> before sharing.
>
> *Caveat for anyone reusing these numbers: profiles were taken in eager mode
> without `torch.compile` for readability, so absolute busy/idle fractions are
> not production values. The ranking of costs is what transfers.*

---

## Where to look next

| you want | read |
|---|---|
| how we got here, blow by blow | [`PROFILING_EXPLAINED.md`](PROFILING_EXPLAINED.md) (v1) |
| the raw lab notebook, all numbers and job IDs | [`NOTES.md`](NOTES.md) |
| the bug itself, in detail | [`LOSS_BUG_ANALYSIS.md`](LOSS_BUG_ANALYSIS.md) and v1 §11 |
| how the model works | [`../../../EXPLAINER.md`](../../../EXPLAINER.md) |
| the phase-3 raw data | `profile_logs/fit-phase3_maskfix_{b200,l4}*` |
| to re-run the analysis | `python analyze_trace.py <trace.json>` |
