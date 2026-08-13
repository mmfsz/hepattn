# Why our expensive GPU isn't fast: the profiling story, explained from scratch

*A pedagogical companion to [`../training_runs_report.md`](../training_runs_report.md)
and [`NOTES.md`](NOTES.md). No prior knowledge of this framework or of GPU
performance work is assumed. If you already know what a CUDA kernel and a
Hungarian matcher are, read NOTES.md instead — it's the same story in half the words.*

> **Read §11 first if you are short of time.** The study set out to explain why a B200
> was slower than three L4s, and §§2–10 are that investigation. But the largest finding
> was not a performance one: the mask losses contained a silent broadcasting bug that
> made the GPU do `batch_size` times too much work *and* trained the model on a distorted
> objective. §11 covers what it was, how it was found, and what fixing it changed
> (+25% throughput, physics unchanged). Everything before §11 was measured against the
> buggy loss.

---

## 1. The mystery we set out to solve

We trained the same CLIC particle-flow model on two very different GPUs:

- an **NVIDIA L4** — a small, cheap datacenter GPU;
- an **NVIDIA B200** — a flagship GPU with roughly **15–40× more raw compute**
  (depending on how you count) and a price tag to match.

If training speed were set by raw compute, the B200 should crush the L4.
Instead, measured over real training runs, **one B200 was only ~1.76× faster
than one L4**. Three L4s beat one B200 on wall-clock time. Somewhere, an
enormous amount of paid-for silicon was sitting idle.

The profiling study answers one question: **where does the time actually go
during a training step?** Once you know that, you know what to fix — and,
just as importantly, what *not* to waste effort fixing.

**Spoiler:** the GPU spends almost none of its time running the neural network.
About 95% of each training step is spent on things that barely benefit from a
faster GPU. The bottleneck is not the data pipeline (our leading suspect — we
were wrong), and not the model. It's the *loss computation and the matching
step* that surround the model. Sections 5–6 unpack exactly what that means.

Acting on the diagnosis, we ran **four fix experiments** (§7). Three were kept
and together cut the step from **4.43 s to 3.05 s — −31.2%, about +45% more
training throughput** — without changing a single number the model learns from.
The one we rejected — swapping in an assignment solver that is 3.5× faster on
paper — made training *twice as slow*, and the reason why is one of the best
lessons in the whole study. Section 8 explains what "without changing a single
number" means and how we checked it.

---

## 2. Background: what happens during one training step

Training a neural network is a loop. Each iteration ("step") looks like this:

```
┌──────────── one training step ────────────────────────────────────────┐
│ 1. LOAD      read a batch of events from disk/RAM, preprocess (CPU)   │
│ 2. TRANSFER  copy the batch from CPU memory to GPU memory             │
│ 3. FORWARD   run the model on the batch (GPU)                         │
│ 4. LOSS      compare predictions to truth → a single "loss" number    │
│ 5. BACKWARD  compute gradients of the loss (GPU)                      │
│ 6. UPDATE    nudge the model weights using the gradients (GPU)        │
└───────────────────────────────────────────────────────────────────────┘
```

Two processors share this work:

- The **CPU** runs the Python program, loads and preprocesses data, and
  *issues instructions* to the GPU.
- The **GPU** executes the actual heavy math. It doesn't run Python — it runs
  small pre-compiled programs called **kernels** ("multiply these matrices",
  "sum this array"), which the CPU queues up for it.

The crucial mental model: **the GPU is an extremely fast worker that only does
what the CPU hands it, and it can sit idle in two ways**:

1. **Starved** — the CPU hasn't finished preparing the next piece of work
   (e.g. data loading is slow, or the CPU is busy computing something itself).
2. **Busy with the wrong work** — the GPU is technically running, but running
   kernels that don't benefit from its strengths (see §3).

"GPU utilization" questions are always about distinguishing these cases.

### Our specific model, in one paragraph

This experiment (`hepattn`, CLIC particle flow) trains a **MaskFormer-style
transformer**: detector hits go into an encoder–decoder transformer, which
outputs a set of candidate particles ("objects"), each with a class prediction
and a *mask* saying which hits belong to it. Because the model outputs an
unordered set, training needs an extra step that regular classifiers don't
have: for every event, we must decide **which predicted particle corresponds to
which true particle** before we can compute a loss. That assignment problem is
solved every step by the **Hungarian algorithm** (a classic combinatorial
optimization algorithm), fed by a **cost matrix** — a big table scoring how
well each predicted object matches each true object. Remember this; it becomes
the villain of the story. (For a full walkthrough of the architecture, see the
[CLIC explainer](../../../EXPLAINER.md).)

One more scale fact: the model is **tiny** — ~10 million parameters — but the
batches are **huge** (2048 physics events per step, each with up to ~160 hits).

---

## 3. Background: what makes a GPU fast (and when it isn't)

A GPU has two relevant resources, and every kernel is limited by one of them:

- **Compute (FLOPS)** — how many arithmetic operations per second it can do.
  Big matrix multiplications are **compute-bound**: the data is small relative
  to the arithmetic done on it, so a GPU with more FLOPS finishes faster.
  *This is the resource the B200 has 15–40× more of.*
- **Memory bandwidth** — how fast it can move data between its own memory and
  its compute units. Simple operations over big arrays (sums, sigmoids,
  element-wise multiplies) are **memory-bound**: the arithmetic is trivial and
  the time is spent streaming data. Extra FLOPS don't help; only faster memory
  does — and the B200's *bandwidth* advantage over the L4 is far smaller than
  its FLOPS advantage.

The distinction matters for us because of what it implies about buying
hardware: a compute-bound kernel gets the B200's full 15–40× advantage, while a
memory-bound one gets only the (much smaller) bandwidth advantage. §6 shows that
~53% of our GPU-busy time is memory-bound loss reductions, which is a large part
of why paying for 15–40× the FLOPS bought us 1.76×.

Two more costs that have nothing to do with either resource:

- **CPU↔GPU transfers.** The GPU has its own memory; anything it needs must be
  copied over a comparatively slow link (and results copied back). A
  device-to-host copy is written **DtoH**, host-to-device **HtoD**.
- **Synchronization stalls.** If the CPU asks for a result back from the GPU
  (e.g. "give me this tensor as a NumPy array"), the CPU must *wait* for the
  GPU to finish, then the GPU sits *idle* while the CPU does whatever it wanted
  the data for. Round-trips like this serialize the two processors.

### Host memory vs device memory, and why "pinned" is faster

Two separate pools of RAM are in play, and performance work is full of jargon
for them:

- **Host memory** — ordinary system RAM attached to the CPU.
- **Device memory** — the GPU's own on-board RAM (192 GB of it on a B200).

A DtoH copy moves bytes from the second to the first, and the hardware that
does it is a dedicated **copy engine** that performs **DMA** (direct memory
access: it reads and writes RAM by itself, without the CPU shuffling the bytes).

The catch is that the operating system manages ordinary host memory in **pages**
that it is free to move around, or evict to disk, whenever it likes. Such memory
is called **pageable**. A DMA engine works with physical addresses, so it cannot
safely read from a page that the OS might relocate mid-transfer. The driver's
workaround is to copy in stages: the GPU DMAs into a small internal buffer that
*is* locked in place, and the CPU then copies from that buffer into your actual
destination — so the data gets touched twice and the transfer runs at a fraction
of the link's speed.

**Pinned** (also "page-locked") memory is host memory you have asked the OS to
never move or swap out. The copy engine can DMA straight into it, with no
staging and no CPU involvement. The price is that pinned memory is a scarce,
expensive-to-allocate resource — you cannot pin everything, and allocating it
per step would cost more than it saves, so the usual pattern is one reusable
buffer.

How much this is worth is not a theoretical question — it is visible in our
trace. The baseline copied its cost matrices (921.6 MB per step, see §6) into
pageable memory at an effective **~1.95 GB/s**. Routing the same bytes through
one reusable pinned buffer took 11.6% off the entire training step (§7, fix 1).

So "buy a bigger GPU" only speeds up the compute-bound fraction of your step.
Everything else — memory-bound kernels, transfers, CPU work, Python overhead —
stays the same. That is the seed of our mystery.

---

## 4. The method: profiling in phases, cheapest first

A **profiler** is a stopwatch wired into the program. We used two, in order of
increasing detail and cost (the plan is in [`README.md`](README.md)):

- **Phase 1 — Lightning's `SimpleProfiler`.** Wall-clock timers around coarse
  phases: "waiting for data", "running the training step", "backward pass".
  Near-zero overhead, answers one big question: *data-bound or compute-bound?*
- **Phase 2 — `PyTorchProfiler`.** Records every individual GPU kernel and CPU
  operation with timestamps, and exports a **Chrome trace** — a zoomable
  timeline you can open at <https://ui.perfetto.dev> and literally *see* the
  gaps where the GPU idles. Higher overhead, so you profile a handful of steps,
  not a whole run.

Method rules we followed (worth copying for any performance study):
profile on **1 GPU** to remove multi-GPU noise; keep every profiling change in
**throwaway config overlays** (`configs/profile.yaml`, `configs/profile_phase2.yaml`)
so the real training configs stay untouched; **change one thing at a time**;
and keep runs to minutes, not epochs. Because this framework builds its
`Trainer` entirely from YAML (PyTorch Lightning), turning a profiler on is a
config change, not a code change.

The one exception to "leave the real configs alone" is **promotion**: when a
change is measured, wanted and permanent, it moves into the real config
deliberately and gets written down. That happened exactly once here — `base.yaml`
now sets `default_solver: lap1015_late` instead of `scipy` (§7, fix 3).

---

## 5. Phase 1: the prime suspect is innocent

Our leading hypothesis was the **data pipeline**: each step loads and
preprocesses 2048 physics events on the CPU — surely the mighty B200 finishes
its math and then waits, starved, for the next batch?

Phase 1 (SLURM job 38122563: 200 steps on one B200, batch 2048) says: **no.**

| Where the 962 s of wall time went | Time | Share |
|---|---|---|
| Inside the model step (`training_step`) | 752.9 s | **78.2%** |
| Gradient computation (`backward`) | 125.5 s | **13.0%** |
| One-off setup (loading the input file into RAM) | 67.1 s | 7.0% |
| **Waiting for the next batch** (`train_dataloader_next`) | **4.0 s** | **0.42%** |
| Copying each batch CPU→GPU | 0.14 s | ~0% |

The GPU waits for data **0.42%** of the time. The 16 CPU dataloader workers
keep up easily. This immediately killed a whole branch of planned work
("Phase 1b": tuning dataloader knobs like `persistent_workers` and
`prefetch_factor`) — those would have optimized 0.42% of the run. *This is why
you measure before optimizing.*

But Phase 1 has a blind spot: `training_step` is one opaque 3.8-second block.
It contains the model forward pass, the loss computation, **and the Hungarian
matching** — including any CPU work and GPU idle time hidden inside. To see
inside the block, we need Phase 2.

---

## 6. Phase 2: opening the box

Phase 2 (job 38127863) recorded every operation in 3 steady-state steps
(~2.85 s each) and exported the timeline. Two headline numbers came out of the
trace (analysis script: [`analyze_trace.py`](analyze_trace.py)).

### Finding A: the GPU is idle 30% of the time

Within the profiled window the GPU was **busy 70.6%** and **idle 29.4%** of
wall-clock.

It is worth being precise about what those two numbers mean, because "GPU
utilization" is used loosely and can mean very different things. Ours is a
wall-clock definition, computed by [`analyze_trace.py`](analyze_trace.py): take
every kernel and every memory-copy event the trace recorded on the GPU, form the
**union of their time intervals** (so overlapping work is not double-counted),
and divide by the length of the training-step span. "Busy" therefore means *at
least one thing was running on the device*; "idle" means *the device had nothing
queued at all*. Note what this does **not** measure: it says nothing about how
much of the GPU's width a running kernel actually uses. A kernel that occupies
1% of the chip still counts as "busy". So 70.6% busy is an upper bound on how
well the hardware was used, and the real figure is worse — which Finding B
confirms.

The 29.4% idle lines up with the **Hungarian matcher**, which runs *on the CPU*
(SciPy's `linear_sum_assignment`), costing ~1.24 s of CPU time per step. While
the CPU grinds through the assignment problems, the GPU — all that silicon —
does nothing. This is the "synchronization stall" pattern from §3: the loss
cannot be computed until the matching is known, so the GPU genuinely has to
wait. That is ~0.85 s of a ~2.85 s step in which the most expensive component in
the machine is switched off.

### Finding B: even when busy, the GPU is barely running the model

This was the bigger surprise. Splitting the 70% busy time by what kind of
kernel was running:

| What the GPU was actually doing | Share of GPU-busy time |
|---|---|
| Fused loss kernels (mask BCE/dice cost & loss matrices) | **~53%** |
| Copying cost matrices to the CPU (`Memcpy DtoH`, *pageable*) | **23.5%** |
| Miscellaneous element-wise ops | 11% |
| **The actual model: attention kernels** | **4.1%** |
| **The actual model: matrix multiplications (GEMM)** | **2.5%** |
| Everything else (layernorm, reductions, …) | ~6% |

Read that again: **the transformer itself — the thing we think of as "the
model" — accounts for ~7% of the GPU's busy time**, which is ~5% of the total
step. Two mechanisms eat everything else:

1. **The loss is computed on enormous matrices.** The model produces a set of
   predictions at every decoder layer, and all of them are trained (a standard
   MaskFormer trick called deep supervision), so the loss machinery runs **5
   times per step** — once for each of the 4 decoder layers plus the final head
   (`maskformer.py:_compute_decoder_costs`). Each time it builds dense tensors
   of shape roughly `batch (2048) × predicted objects × hits` in full precision
   (fp32) and runs sums/sigmoids/cross-entropies over them. These show up as the
   `triton_red_fused_...binary_cross_entropy...` kernels in the trace. They are
   textbook **memory-bound** operations (§3): simple math over huge arrays. The
   B200's FLOPS advantage is nearly useless here. A tiny 10M-parameter model
   attached to a giant set-matching loss means the loss, not the model,
   dominates.
2. **The cost matrices take a slow boat to the CPU.** The Hungarian matcher
   needs the cost matrices as a NumPy array, so every step ships them
   device-to-host. This is one **921.6 MB copy taking ~472 ms**, into *pageable*
   memory, the slow kind (§3) — about **1.95 GB/s**. (Where the 921.6 MB comes
   from: 5 stacked cost matrices × 2048 events × 150 predicted objects × 150
   target slots × 4 bytes for fp32. A second, trivial 80 KB copy carries the
   per-event count of real particles.) Nearly a quarter of all GPU activity is
   therefore a memory copy whose only purpose is to feed a CPU algorithm.

   *Reading tip:* the profiler's summary table reports this as
   "235.855 ms mean, 6 calls" — that is the mean over the big and the tiny copy
   across 3 steps, not two equal 236 ms copies. Per-event `bytes` fields in the
   Chrome trace give the real picture; summary tables average across shapes.

### Putting the step back together

Approximate anatomy of one 2.85 s training step on the B200:

```
|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■|
 ~1.05 s loss kernels      ~0.47 s DtoH   ~0.85 s GPU IDLE    ~0.5 s misc
 (memory-bound, GPU)       copy to CPU    (CPU: Hungarian     (elementwise,
                           (pageable)      matcher, Python)    model ~0.13 s)
```

Only the ~0.13 s of attention + matrix-multiply work scales with GPU
horsepower. Replace the B200 with an infinitely fast GPU and the compute-bound
part goes to zero — but the step only shrinks from ~2.85 s to ~2.7 s. That is
the whole mystery solved: **a 15–40× FLOPS advantage applied to 5% of the
workload yields the ~1.8× we observed.** (This is Amdahl's law: total speedup
is capped by the fraction of work you actually accelerate.)

---

## 7. What we did about it: five experiments, three keepers

Diagnosis and treatment are separate steps. From the fix candidates we picked
the ones that are **pure optimizations** — changes that cannot affect the
physics, only the speed (§8 explains how we made sure of that) — implemented
them, and *measured* each with the same Phase-1 protocol: identical config,
hardware and step count, **one variable changed at a time**, 200 steps each.

Before the details, note the `backward` column below. The backward pass never
touches the matcher, so if our changes do only what we claim, `backward` must
not move. It sits at 0.627–0.629 s/step across all four runs. That is the
**control**: it is what lets us attribute the differences to the matcher path
rather than to a warmer filesystem cache, a different node, or luck.

| run | job | per training batch | `training_step` | `backward` (control) | vs baseline |
|---|---|---|---|---|---|
| baseline: SciPy solver, pageable copy | 38122563 | 4.43 s | 3.76 s | 0.627 s | — |
| + pinned-memory DtoH staging | 38209457 | 3.92 s | 3.25 s | 0.627 s | **−11.6%** |
| + `lap1015_late` with the GIL released | 38212448 | 3.53 s | 2.868 s | 0.629 s | **−20.3%** |
| + device-side cost preparation | 38451000 | **3.05 s** | **2.387 s** | 0.628 s | **−31.2%** |

−31.2% on the step is **≈ +45% throughput** (you do 4.43/3.05 = 1.45× as many
steps per hour). Measured in steady state — i.e. after the first ~50 steps,
during which `torch.compile` is still compiling kernels — the step went from
~3.7 s to ~2.29 s. The rest of this section is what each row was and why it
worked.

*A numbering warning:* [`NOTES.md`](NOTES.md) and the log filenames number the
experiments in the order they were **run**, which is not the order in which they
are easiest to explain. Below, "fix 1" (pinned memory) is NOTES' experiment 2
(`fix2_pinned_*`), "fix 2" (the rejected solver swap) is NOTES' experiment 1
(`fix1_lap1015_*`), and fixes 3 and 4 keep their numbers.

### Fix 1 (KEPT): pin the memory the cost matrices land in — **−11.6% step time**

Recall from §3 that copies into ordinary ("pageable") host memory have to be
staged through a driver buffer, and from §6 that each step spent ~0.47 s copying
921.6 MB of cost matrices into exactly that slow kind of memory. The fix: the
matcher keeps **one reusable pinned buffer**, grown only when a batch needs more
room, and copies the cost matrices into it instead of allocating fresh pageable
memory every step (`matcher.py:Matcher._prepare_costs`). Reusing the buffer is
the point — pinning is expensive to *allocate*, so a fresh pinned allocation per
step would give the saving back. The numbers that come out are bit-for-bit
identical; only the route changes.

| 200 steps, 1× B200, batch 2048 | before (job 38122563) | after (job 38209457) |
|---|---|---|
| time per training batch | 4.43 s | **3.92 s (−11.6%)** |
| `training_step` | 3.76 s | **3.25 s** |
| `backward` (control — should not change) | 0.627 s | 0.627 s ✓ |

The saving (−0.51 s/step) matches the profiled DtoH cost almost exactly — the
trace told us in advance what the fix was worth, and the measurement confirmed
it. Note the control row: `backward` doesn't touch the matcher, so it should be
(and is) unchanged — that's how you check the two runs are truly comparable.
This change lives in the shared matcher code, so every experiment in the
repository now benefits from it.

### Fix 2 (REJECTED as shipped): a faster Hungarian solver that made training 2× slower

The repository ships an alternative assignment solver, `lap1015`, which our
benchmark shows is genuinely **3.5× faster than scipy per matrix**
(310 ms vs 1084 ms for 2000 events, single-threaded). Switching to it is a
one-line config change. The measured result: training slowed from 3.76 s to
**7.14 s per step** — nearly twice as slow. How can a faster algorithm lose?

**The Global Interpreter Lock (GIL).** CPython lets only one thread at a time
execute Python bytecode — that lock is the GIL, and it is why "just use threads"
does not speed up pure-Python code. Threads *can* still run in parallel, but only
while they are inside compiled code that has explicitly **released** the GIL
(one line in a C extension: hand the lock back, do the number-crunching, take it
again before touching any Python object). NumPy, SciPy and most numeric
libraries do this as a matter of course.

Our matcher solves **10,240 small assignment problems per step** (5 decoder
outputs × 2048 events) and spreads them over a 16-thread pool. SciPy's solver
releases the GIL, so those 16 threads genuinely run at once — we measured ~11×
scaling. The `lap1015` binding **never released it**, so its 16 "parallel"
threads took turns running one at a time. All the parallelism silently
evaporated, and a 3.5× faster algorithm became ~3× slower in practice:

| 2000 events, 16-way parallel | scipy | lap1015 |
|---|---|---|
| threads (the configuration training actually uses) | **113 ms** | 326 ms (GIL-serialized) |
| processes (no GIL, but data must be copied between them) | 201 ms | 191 ms |

Two morals, and they generalize well beyond this repo. First, *a microbenchmark
of a component does not predict its behaviour in the real system*: lap1015 wins
every single-threaded comparison and still loses by 1.9× where it counts, so the
number to trust is always the end-to-end one, measured in the actual
configuration. Second, *when a codebase makes a choice that looks obviously
wrong, assume there is a reason until you have measured*. The `scipy` pin — odd
next to a faster solver sitting in the same repository, with `adaptive_solver`
switched off so it could not be overridden — was in fact the fastest available
combination at the time.

The attempt still paid its way. Making lap1015 run at all exposed **two real
bugs** in the solver wrapper, both fixed and both accuracy-neutral: it returned
uninitialised memory for events with *zero* true particles (empty cost
matrix), and it has undefined behaviour — up to crashing the process — on
non-finite costs, which our loss produces legitimately (padded detector hits
get −∞ mask logits; scipy treats ±∞ as "forbidden/mandatory assignment", so it
never noticed). The matcher now sanitizes costs on the GPU before the copy and
verifies that any non-scipy solver returned a valid permutation, falling back
to scipy for that event if not — so a buggy solver can degrade gracefully
instead of corrupting training.

Note the shape of the timing argument, though: lap1015 lost for a reason that is
a property of its *binding*, not of its *algorithm*. That is a lead, not a dead
end.

### Fix 3 (KEPT): release the GIL in the solver's binding — **−20.3% cumulative**

If the only thing wrong with lap1015 was that its binding held the GIL, then the
cure is one line of C++: wrap the call to the solver in
`py::gil_scoped_release`, which hands the lock back for the duration of the
solve. The solver itself touches no Python objects while it runs, so this is
safe. It does mean rebuilding the compiled extension
(`src/lap1015/src/main.cpp` → `_core`), which is why it was initially parked.

Before spending a GPU job on it, two cheap checks on the login node: 300/300
optimal-cost agreement with scipy, and 4 concurrent solves finishing in ~1.2×
the wall time of a single solve — i.e. the lock really is released, because
under the GIL that number would have been ~4×.

The full run confirmed it: `run_training_batch` **3.53 s/step** (job 38212448),
a further −10% on top of the pinned buffer and **−20.3% cumulatively**. The
algorithm that lost by 1.9× now wins, with no change to the algorithm at all.

There is a maintenance sting worth flagging, since it is easy to lose silently:
the speedup lives in a **compiled binary**, not in the configuration. `base.yaml`
now asks for `default_solver: lap1015_late`, but if the environment is
reinstalled without rebuilding lap1015 from the patched source, that config
selects the GIL-bound binding and training gets ~2× *slower* than it was with
SciPy — with no error message. Re-check `run_training_batch` after any
environment rebuild.

### Fix 4 (KEPT): prepare the cost matrices on the GPU — **−31.2% cumulative**

This one came from re-reading the code, not from the profiler, and that is the
interesting part.

The Phase-2 trace attributed the DtoH copy honestly: 921.6 MB, ~472 ms, listed
right there in the table. What it could not show is what the host did with those
bytes *after* they landed. Reading the old `compute_matching` with the tensor's
true size in mind, the answer was: two more full-size passes over ~1 GB, in
single-threaded NumPy, on the critical path —

- `np.where(...)`, which overwrites every entry belonging to a padded query slot
  (a prediction slot that isn't a real query) with a large sentinel cost so the
  solver will never choose it. `np.where` does not edit in place; it allocates
  and fills a whole new ~1 GB array.
- `np.ascontiguousarray(costs.swapaxes(1, 2))`. The solvers want each event's
  matrix laid out as `[targets, predictions]`, but the model produces
  `[predictions, targets]`. Swapping the axes is itself free — it only relabels
  how the array is indexed — but the solver needs the elements to be in that
  order *in memory*, and `ascontiguousarray` is what makes it so: a strided,
  cache-hostile copy of the whole gigabyte.

Neither shows up as GPU time, because neither happens on the GPU. Both are pure
host overhead that the GPU waits through, and both were buried in the same
"the matcher is slow" bucket as the actual Hungarian solve.

The fix (`matcher.py:Matcher._prepare_costs`) is to do all of it *before* the
copy, on the device: sanitize non-finite costs, mask the padded queries,
transpose into solver layout, and only then copy — once — into the pinned
buffer. On the GPU each of those is a cheap bandwidth-bound kernel over data
that is already there; on the host they were single-threaded passes over a
gigabyte that had just crossed the bus. The array now arrives already
contiguous and already in final shape, so the host goes straight from the copy
to the solver.

A second change rode along in the same run. Every event is padded out to 150
target slots, but `match_individual` only ever reads `cost[: lengths[k]]` —
the rows past each event's real target count are never looked at. So the tensor
is **cropped along the target axis to the largest number of real targets in the
batch** before the copy: fewer bytes transferred, and no padded rows for the
solver to walk past.

Result: **3.05 s/step** (job 38451000), a further −13.7% and −31.2% against the
original baseline, with `backward` again unchanged. Note the one place we bent
our own rule: device-side preparation and the crop shipped in a single job, so
the −0.48 s/step is their *joint* effect and we cannot say how it splits. That
was a deliberate trade (queue time on this cluster is measured in hours), but if
one of them had turned out to be a regression we would have had to re-run both.

### Fix 5 (REJECTED): give the matcher more CPU threads — **22% slower**

The cheapest idea of all, and the one that needed no code: if 16 threads help,
try 32. The matching is 10,240 independent problems per step, so it looks like
work you can simply throw cores at. We asked SLURM for `--cpus-per-task=32`,
set `n_jobs: 32`, changed nothing else.

Result: **3.72 s/step** against 3.05 s/step at 16 threads (job 38452194) —
22% *slower* for twice the cores.

This experiment is in here mostly for the mistake that followed it. The first
explanation written down was that a 32-thread pool must span the node's two
CPU sockets, paying for memory attached to the far socket. That was wrong, and
it fell over on a one-line objection: **a socket has 56 cores, so 32 fits
inside one**, and any placement effect scattered enough to hurt 32 threads
would hurt 16 as well. A guess that sounds mechanistic is still a guess.

So we measured it instead ([`bench_matcher_threads.py`](bench_matcher_threads.py),
job 38458388), which prints the cores SLURM actually handed out and then times
the solve alone at each thread count, on 10,240 cost matrices shaped like the
real ones:

| threads | lap1015_late | scipy |
|---|---|---|
| 1 | 3.32 s | 1.23 s |
| 8 | 0.41 s | 0.29 s |
| **16** | **0.23 s** | **0.27 s** |
| 24 | 0.24 s | 0.26 s |
| 32 | 1.67 s | 1.91 s |
| 48 | 1.16 s | 1.41 s |

Two results. First, the placement story is dead: the 48-core allocation came
back as `Cpus_allowed_list: 60-71,76-111`, entirely inside **one** NUMA domain,
and an `hpg-b200` node has exactly two (one per socket). Second — and this is
the finding that actually matters — **the solve saturates at 16 threads.** Going
to 24 buys nothing (0.24 s vs 0.23 s). The premise of the experiment was simply
false: there was no unused speed to unlock, so no thread count was ever going to
help.

There is also a genuine collapse at ≥32 threads, in both solvers, that we
**cannot currently explain**. We are deliberately not offering a second theory.
The honest state: 48 threads being *faster* than 32 fits no clean contention
model, and the benchmark has a known flaw — `_get_thread_pool` caches pools
forever, so by the time it reaches 32 threads the pools from every earlier count
are still alive in the same process, measured in ascending order. That has to be
removed (fresh process per count, randomised order) before the numbers at 32 and
48 support any conclusion about cause. The leading suspect to test would be GIL
contention on the per-event *Python* work in `match_individual` — array
conversion, permutation validation, `np.bincount` — which the C++ GIL release
does not cover; but that predicts a gradual decline, not a 7× cliff, so it is a
starting point and not an answer.

None of this changes what to do. **Keep `n_jobs: 16`**, now because 16 is the
measured saturation point rather than because of any story about hardware.

The general lesson is worth stating plainly, because it is the one most likely
to save you a day: **a profile tells you where the time went, not why.** The
trace correctly reported a large memcpy; it had no way to tell us that the same
gigabyte was then walked twice more by NumPy, since that time appeared only as
undifferentiated host time next to the solve. Profile to find the region, then
read the code in that region with the actual data sizes in your head.

### Where the time goes now (post-fix Phase 2, job 38452195)

Before deciding what to do next, we re-ran the Phase-2 trace with all the kept
fixes in place — same protocol as the original (eager mode, `Compile` off), so
the two are comparable:

| share of GPU-busy time | pre-fix | post-fix |
|---|---|---|
| `Memcpy DtoH` (the cost matrices) | **23.5%** | **0.9%** |
| fused triton loss kernels | ~53% | **~69%** |
| model (attention + GEMM) | ~7% | ~8.6% |
| GPU busy / idle | 70.6% / 29.4% | 54.5% / 45.5% |

The transfer has stopped being a cost centre — 23.5% → 0.9% is the pinned copy
and the device-side prep confirmed *in the trace*, not merely end-to-end. But
notice what did **not** happen: the GPU is idle a *larger* fraction of the step
than before. That is not a regression. We removed GPU work (the copy) and host
work (the NumPy passes) without removing the **serialisation** between them —
the GPU still stops dead while the CPU solves. The remaining ideas below all
attack that, or attack the loss kernels that now dominate what the GPU does.

One caveat, since this trace is easy to over-read: it runs with `torch.compile`
off so that operations stay attributable, which inflates host time (the Lion
optimizer alone shows ~0.44 s/step of pure dispatch overhead). **45.5% is not the
production idle fraction.** Only the composition transfers.

### What remains on the table — and the current decision

**Decision (2026-07-31): both speed options below are on hold.** The study
delivered −31.2% (4.43 → 3.05 s/step, ≈ +45% throughput) with no change to the
physics, and that is being taken as sufficient for now. Nothing further will be
attempted until real training runs show whether the current speed is actually a
constraint. This is a deliberate stop, not an oversight — the analysis below is
recorded so that whoever picks it up does not have to re-derive it.

- **Overlap the matching with GPU work** (the serialisation above): solve the
  assignment for decoder layer *i* while the GPU computes layer *i+1*'s costs,
  instead of waiting for all five and then stopping. The more contained of the
  two, since the per-layer loop already exists in
  `maskformer.py:_compute_decoder_costs`.
- **An exact assignment solver that runs on the GPU** (batched
  Jonker-Volgenant, or auction with ε-scaling). This would delete the
  device-to-host copy and the host stall outright rather than making them
  cheaper — the largest structural win left. Needs a new dependency and the
  same equivalence checking as everything else here.
- **Shrink the loss kernels** — now ~69% of all GPU work, so this has become the
  biggest single line item, which it was not when the study started. Computing
  mask losses in bf16, or only on matched pairs, is a *modelling* change needing
  accuracy validation and is **explicitly out of scope** for this study: no
  approximations in the computations. The exact alternative is the algebraic
  rewrite of the mask-cost einsums, which halves one GEMM; mathematically exact
  but not bit-identical, so it still wants a loss-curve check.
- **More matcher threads**: settled and closed — see Fix 5. The solve saturates
  at 16 threads; there is nothing there.

The strategic takeaway stands, now with both a mechanism and a partial remedy:
**for this workload, a B200 is poor value.** The fixes above removed ~1.4 s/step
of overhead that never scaled with GPU FLOPS in the first place — but they did
not add any work that does. Model compute measured ~0.13 s of the profiled
baseline step, and since none of these changes touch the model (the flat
`backward` control says as much), it is the same ~0.13 s now — simply a slightly
larger slice of a smaller step, which the post-fix trace confirms (model compute
~7% → ~8.6% of GPU-busy time). The B200 only pays off if the model gets much bigger, or if the
loss/matcher pipeline is reworked further so that model compute becomes the
dominant cost.

---

## 8. "Exact" is not a figure of speech

Every change kept in this study is **assignment-identical**: the matcher returns
the same permutation it would have returned before, for every event, so the loss
the model trains on is unchanged. Nothing here trades accuracy for speed. That
was a design constraint, not an outcome — a 45% speedup would be worthless if it
quietly perturbed the physics, and worse than worthless if nobody noticed for a
month.

It is worth being clear about the two different standards involved:

- **Bit-identical** — the same floating-point numbers, bit for bit. Fix 1 and
  the cropping in fix 4 are in this class: moving bytes through a pinned buffer
  or not transferring rows nobody reads cannot change a value.
- **Assignment-identical** — possibly different arithmetic, provably the same
  *answer*. Fixes 3 and 4 need this weaker standard, because a different solver
  may break ties between two equally-good assignments differently, and because
  the sanitizing step replaces ±∞ with a large finite sentinel. Neither changes
  which assignment is optimal: for an argmin over a finite set, a sufficiently
  large finite cost and an infinite cost are the same instruction ("never pick
  this"), and tied optima are by definition equally good.

The verification was deliberately hostile — the interesting failures in this
codebase all came from degenerate inputs, not typical ones. The final check ran
**556 events** across three matrix shapes, covering `-inf` entries (which the
loss produces legitimately, from padded hits), `NaN`, all-constant matrices
(every assignment optimal — the case that breaks lap1015 badly enough to need
the scipy fallback), empty matrices (events with zero true particles) and
query-masked matrices (padded prediction slots). For each, both solvers under
the new device-side preparation were compared against the old host-side path
with SciPy, on **optimal assignment cost computed in float64**. All agreed.
`tests/matching` passes (118 tests, both solvers).

Comparing *costs* rather than *index arrays* is the point of that check: two
correct solvers can legitimately disagree on which of several optimal
assignments to return, so comparing indices would produce false alarms, and
comparing the resulting cost is the property we actually care about.

One deliberate behaviour change came out of the rewrite and is recorded here
rather than hidden: when no target-validity mask is supplied, per-event lengths
are now taken from the target axis rather than the prediction axis. For square
cost matrices — the CLIC case, 150×150 — these are identical; for
`num_true > num_pred` the old default would have silently truncated targets.
That is a bug fix, but it is a behaviour change, so it is written down.

---

## 9. Small print: pitfalls we hit (so you don't)

- **Lightning configs validate strictly.** Our first Phase 2 job died in 47 s
  at config parsing: `PyTorchProfiler` doesn't accept `with_stack` as a YAML
  argument, and a `schedule:` dict crashes at runtime (PyTorch wants a Python
  callable there). Cheap insurance: validate any config overlay on the login
  node with `pixi run python main.py fit --config ... --print_config` before
  queueing a GPU job.
- **The trace duplicates step markers.** The exported Chrome trace contains two
  `ProfilerStep` spans per real step (different threads). Naively counting
  spans gives 6 "steps" with weird alternating durations; there were 3.
- **`torch.compile` is not all-or-nothing.** We disabled the `Compile`
  *callback* (which compiles the encoder/decoder) to keep the trace clean, yet
  compiled "triton" kernels still appeared — the *loss functions* are
  independently compiled in the code. Those fused loss kernels are real and
  present in production runs too.
- **A profiler only sees what it instruments.** Phase 1's `training_step` looked
  like "GPU compute" but hid CPU matching and GPU idle gaps inside. Coarse
  timers tell you *where* to look, never *what is happening* — that's what the
  op-level trace is for. And even the op-level trace has this limit one level
  down: it priced the 921.6 MB DtoH copy correctly while saying nothing about
  the two full-size NumPy passes that followed it (§7, fix 4). **Where** is the
  profiler's job; **why** is still yours.
- **Summary tables average over shapes.** The op table reported the DtoH copies
  as "235.855 ms mean over 6 calls", which reads like two ~236 ms copies per
  step. It is actually one 921.6 MB copy at ~472 ms plus an 80 KB copy, averaged
  together. When a number matters, get it from the per-event records in the
  trace JSON (which carry a `bytes` field) rather than the summary.
- **Know your tensor's size in bytes.** Almost every finding here follows from
  one arithmetic fact — the stacked cost tensor is 921.6 MB — that nothing
  printed for us. Working it out (5 decoder outputs × 2048 events × 150 × 150 ×
  4 B) is what turned "the matcher is slow" into three specific, fixable causes.
- **A "failed" experiment can be the most informative one.** The lap1015 run
  crashed twice (empty-matrix garbage, then non-finite-cost corruption) before
  producing its negative timing verdict — and each failure exposed a latent bug
  that is now fixed with tests. Budget for this: fix attempts are experiments,
  not patches.
- **A speedup that lives in a binary can vanish silently.** Fix 3's win comes
  from a rebuilt C++ extension, not from the config that selects it. Reinstall
  the environment without rebuilding and training gets ~2× slower with no error.
  Anything whose performance depends on how a dependency was *compiled* needs a
  note in the README and a sanity check after env changes.
- **One pre-existing broken test.** `tests/matching/test_solvers.py::test_lap1015`
  calls `lap1015.lap_early`, which hangs forever on this machine (it was already
  commented out of the solver registry — a clue). Deselect that test when
  running the matching tests; the other 118 tests pass.

**Artifacts:** raw profiler tables and all SLURM logs are in
[`profile_logs/`](profile_logs/) — `phase1_simpleprofiler_slurm-38122563.out`,
`phase2_pytorchprofiler_slurm-38127863.out`, and one per fix experiment
(`fix1_lap1015_*` = §7's fix 2, `fix2_pinned_*` = §7's fix 1,
`fix3_lap1015_gilrelease_*`, `fix4_deviceprep_*`). The 46 MB Chrome trace
(the `*.pt.trace.json` in that directory) can be dropped into
<https://ui.perfetto.dev> to explore the timeline yourself — the ~472 ms
`Memcpy DtoH (Device -> Pageable)` bar and the kernel-free idle gaps are easy to
spot by eye. The raw evidence and per-job details for everything above are in
[`NOTES.md`](NOTES.md).

---

## 10. Appendix: the "algebraic einsum rewrite", explained (candidate 7 — *not implemented*)

Section 7 dismisses this idea in one line: *"the algebraic rewrite of the
mask-cost einsums, which halves one GEMM; mathematically exact but not
bit-identical, so it still wants a loss-curve check."* This appendix unpacks
what that means, because the idea is genuinely elegant and worth understanding —
and because **going and reading the actual code changed the verdict on how much
it is worth here.** The trick is real and correct. The place we assumed it would
pay off turns out not to be the place the GPU is actually spending its time.

Nothing in this section has been implemented or measured. It is a design note,
recorded so that whoever picks the study back up starts from the real code
rather than the sketch.

### 10.1 What the code actually computes

Recall the setup from §2: the model proposes 150 candidate particles per event,
each carrying a **mask** — a score for every detector hit saying "this hit is
mine". Before we can compute a loss we must decide which candidate corresponds
to which true particle, and to do that the Hungarian matcher needs a **cost
matrix**: for every (predicted object *n*, true particle *m*) pair, a number
saying how bad it would be to declare them a match.

For masks, that number is a binary cross-entropy summed over hits. The code is
`mask_bce_cost` in [`src/hepattn/models/loss.py`](../../../../../models/loss.py)
(lines 231–255), and its last line is the thing under discussion:

```python
# loss.py:245-246 — two per-hit penalty tables, both [b, n, c]
pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits),  weight=sample_weight, reduction="none")
neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), weight=sample_weight, reduction="none")

# loss.py:255 — the two big contractions
return torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum("bnc,bmc->bnm", neg, (1 - targets))
```

Reading the einsum subscripts out loud, because they carry all the information:

- `b` — the **batch** axis: 2048 physics events, all handled in parallel.
- `n` — the model's 150 **predicted** objects.
- `m` — the 150 **true** particle slots.
- `c` — the **constituents**, i.e. the detector hits (padded to 160 in CLIC,
  `pflow_data.py:max_nodes = 160`).

`"bnc,bmc->bnm"` says: for each event `b`, for each prediction `n`, for each
target `m`, **sum over the hit axis `c`** the product of the two inputs. The hit
axis disappears; it is contracted away. This is exactly a batched matrix
multiplication of a 150×160 matrix with the transpose of another 150×160 matrix.

What the two terms mean physically:

- `pos[b,n,c]` is the penalty prediction *n* pays on hit *c* **if that hit really
  does belong to the particle**. Contracting it with `targets` (which is 1 on the
  hits that belong to particle *m*, 0 elsewhere) picks out exactly those hits and
  adds up their penalties.
- `neg[b,n,c]` is the penalty *n* pays on hit *c* **if that hit does not belong**.
  Contracting it with `1 - targets` picks out the complementary set.

So `cost[b,n,m] = (penalty on m's hits) + (penalty on everything else)`. Perfectly
natural to write, and it costs **two** contractions over the hit axis.

The same shape of expression appears once more, in `mask_focal_cost`
(`loss.py:170-195`, last line 195):

```python
return torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - targets))
```

Those two functions — `mask_bce_cost` and `mask_focal_cost` — are the **only**
places in the file with this structure. That fact matters in §10.4.

Both are invoked, once per decoder output, from
`maskformer.py:_compute_decoder_costs` (line 263), which loops over the 5
supervised decoder outputs and over each task, calling `task.cost(...)` and
accumulating the per-task cost matrices into one matrix per layer
(`task.py:588-598` for the mask task). Five layers × one call each = five
evaluations of whichever mask cost functions the config switches on.

### 10.2 The identity

Here is the whole idea, in one line of school algebra. For any two numbers *A*
and *B* and any weight *t*:

```
A·t + B·(1−t)  =  A·t − B·t + B  =  (A − B)·t + B
```

Applied along the hit axis, with *t* the target mask (1 for "this hit belongs to
particle *m*", 0 otherwise), summing over hits *c*:

```
Σ_c pos[n,c]·t[m,c]  +  Σ_c neg[n,c]·(1 − t[m,c])
      =  Σ_c (pos[n,c] − neg[n,c])·t[m,c]  +  Σ_c neg[n,c]
```

Look at what happened to the second term. On the left it is a contraction: it
depends on both *n* and *m*, so you must compute 150 × 150 = 22 500 of them per
event. On the right the target mask has vanished from it, so it depends only on
*n* — it is **one number per predicted object**, 150 per event instead of 22 500,
and it is simply added to every entry of that object's row. In tensor terms it
went from a matrix multiply to a row sum plus a broadcast.

**A tiny worked example.** One event, one prediction *n*, one true particle *m*,
and only 4 hits. Say hits 1 and 2 belong to the particle and hits 3 and 4 do not,
and the model happens to be predicting that quite well:

| hit *c* | `pos` (penalty if hit is mine) | `neg` (penalty if hit is not mine) | `t` (truth) |
|---|---|---|---|
| 1 | 0.10 | 2.30 | 1 |
| 2 | 0.20 | 1.80 | 1 |
| 3 | 2.00 | 0.05 | 0 |
| 4 | 3.00 | 0.02 | 0 |

*The way the code does it now — two contractions:*

```
pos·t       = 0.10 + 0.20          = 0.30
neg·(1−t)   = 0.05 + 0.02          = 0.07
cost                               = 0.37
```

*The rewrite — one contraction plus a row sum:*

```
d = pos − neg = [−2.20, −1.60, +1.95, +2.98]
d·t           = −2.20 + (−1.60)    = −3.80
neg.sum()     = 2.30+1.80+0.05+0.02 = 4.17     ← does not depend on m at all
cost          = −3.80 + 4.17        = 0.37     ✓ same answer
```

In code the rewrite is three lines:

```python
diff = pos - neg                                    # [b, n, c], one cheap elementwise pass
cost = torch.einsum("bnc,bmc->bnm", diff, targets)  # ONE contraction instead of two
cost = cost + neg.sum(-1).unsqueeze(-1)             # [b, n, 1], broadcasts across every m
```

Two things disappear along with the second einsum: the 150×150-per-event
contraction itself, and the materialisation of `1 - targets`, which today
allocates and writes a whole extra tensor the size of the target masks purely to
hold "not". The identity is exact for real arithmetic — there is no
approximation, no dropped term, no tolerance. (§10.5 is about floating point,
which is a different matter.)

Note that `torch.compile` will **not** do this for you. Compilers are allowed to
fuse and reorder operations that provably give the same answer; they are not
allowed to change which floating-point additions happen in which order, because
that changes the result. This is a rewrite a human has to decide to make.

### 10.3 Why halving the reads roughly halves the time — and how we know

Section 3 introduced the distinction: a kernel is **compute-bound** (limited by
arithmetic) or **memory-bound** (limited by how fast data can be streamed in and
out of GPU memory). These contractions look like matrix multiplications, which
are the textbook compute-bound operation — so why would removing one of them help
by anything like a factor of two?

Because of the shapes. Work out the bytes for the CLIC configuration
(2048 events × 150 objects × 160 hits, fp32 = 4 bytes):

| tensor | shape | size |
|---|---|---|
| `pos`, `neg`, `targets`, `1 - targets` | `[2048, 150, 160]` | **196.6 MB** each |
| the cost matrix out | `[2048, 150, 150]` | **184.3 MB** |

The arithmetic done on all that is 2 × 2048 × 150 × 150 × 160 ≈ 14.7 billion
floating-point operations — which sounds enormous until you remember a B200 does
tens of *trillions* per second. The contraction depth is only 160; there is very
little arithmetic per byte fetched. So it behaves like a memory-bound kernel, and
the currency that matters is **bytes moved**, not FLOPs.

We do not have to take that on faith: it is measured in our own post-fix trace.
The one mask-cost contraction that the CLIC config *does* run (§10.4) shows up as
an `aten::bmm` — 15 calls (5 decoder outputs × 3 profiled steps) at **0.202 ms
each**. Those bytes are 196.6 + 196.6 read + 184.3 written = 577.5 MB, so the
kernel is sustaining roughly **2.9 TB/s**. That is a large fraction of the B200's
memory bandwidth and nowhere near its arithmetic limit. Confirmed memory-bound,
in this trace, at these shapes.

For a kernel in that regime, time ≈ bytes ÷ bandwidth, and bandwidth is a
property of the hardware you cannot change. So removing half the bytes really
does remove close to half the time — unlike a compute-bound kernel, where
removing half the arithmetic may just leave the memory system as the new
ceiling.

Counting honestly for `mask_bce_cost`, though, the win is a bit less than a
clean 50%, because the rewrite adds two small passes of its own:

| | reads + writes |
|---|---|
| today: `1-targets` materialised, two `bmm`s, one add | ≈ 2.1 GB |
| rewritten: one subtract, one `bmm`, one row sum, one broadcast add | ≈ 1.35 GB |

**The contraction itself halves; the cost function as a whole loses roughly a
third of its memory traffic.** Both statements are worth keeping straight,
because it is the first one that gets quoted and the second one that you would
actually measure.

### 10.4 Expected impact: the honest answer is "almost none, here"

Section 7 reports that fused triton loss kernels are now **~69% of all GPU-busy
time**, and it is tempting to read that as "the mask cost matrices are 69% of the
work, so halving one of their two contractions is worth ~15% of the GPU". That
reading is wrong, and checking it is the useful part of this appendix.

Attributing every GPU kernel in the post-fix trace
(`profile_logs/phase2_postfix_trace.pt.trace.json`) back to the compiled function
that launched it gives this:

| compiled function | what it is | share of GPU-busy time | per call |
|---|---|---|---|
| `mask_bce_loss` — forward | **loss** | **28.2%** | 86.7 ms |
| `mask_dice_loss` — forward | **loss** | **16.7%** | 51.3 ms |
| `mask_dice_loss` — backward | **loss** | **14.0%** | 43.4 ms |
| `mask_bce_loss` — backward | **loss** | **9.3%** | 28.9 ms |
| **all mask *cost* functions together** | **cost** | **0.22%** | 0.69 ms |
| — of which the einsum (`aten::bmm`) | | 0.065% | 0.202 ms |

Those four loss kernels sum to 68.2%. **The "~69% of GPU-busy time" is the mask
losses — forward and backward — not the cost matrices at all.** Every mask cost
computation in the step together accounts for 0.22%.

Two facts explain that, and both are in the config and the code rather than in
the trace:

1. **`mask_bce_cost` is switched off in CLIC.** In
   [`configs/base.yaml`](../../../configs/base.yaml) the mask task lists
   `mask_bce: 5.0` and `mask_dice: 1.0` under `losses:` (lines 186–188), but
   under `costs:` (lines 189–191) the BCE line is **commented out** and only
   `mask_dice: 1.0` survives. The function the rewrite targets is never called in
   the profiled run. `mask_focal_cost` is not used by CLIC either.
2. **The cost that *is* used has already been written in the rewritten form.**
   `mask_dice_cost` (`loss.py:91-113`) is

   ```python
   numerator   = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)   # ONE contraction
   denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)   # cheap row sums
   return 1 - (numerator + 1) / (denominator + 1)
   ```

   One contraction, plus two per-row sums that broadcast — structurally identical
   to what §10.2 produces. Dice has no `A·t + B·(1−t)` pair to collapse, because
   the Dice score never needed one: its denominator is a sum of independent totals
   rather than a joint quantity. There is nothing to fold.

So, an honest back-of-envelope for the configuration this study profiled: the
rewrite would apply to nothing, and if `mask_bce` were re-enabled as a cost it
would save roughly a third of a function that currently costs 0.22% of GPU-busy
time — call it **≲0.1%**, which is below the noise of the measurements in §7. It
is not the lever we thought it was.

**Where it *is* worth doing.** Other experiments in this repository switch the
affected cost functions on: `trackml` (`tracking.yaml`, `tracking-lite.yaml`,
`queryPE-lite.yaml`, `tracking-strip.yaml` all use `mask_focal` and/or
`mask_bce` as costs), `itk/configs/tracking.yaml`, `tide/configs/base.yaml` and
`pixel.yaml`, and `atlas_muon/config/muon_tracking.yaml`. Those detectors also
have far more constituents per event than CLIC's 160 hits, and the traffic scales
linearly with that axis. The rewrite lives in shared code, so doing it once helps
all of them — it is simply not a CLIC speedup.

**And the lead this actually turned up.** Put the two measured numbers side by
side: the dice-cost contraction moves ~577 MB in 0.202 ms (~2.9 TB/s), while the
`mask_bce_loss` kernel next door takes **86.7 ms** to reduce tensors of the same
order (at most ~0.6 GB of input) — an effective ~7 GB/s, about **400× slower per
byte than what the same GPU is demonstrably doing in the same trace**. That is
not an algebra problem; the loss is already about as simple as arithmetic gets.
It points at how the kernel was generated: the trace records it launching a grid
of 3.2 million 128-thread blocks at 38% occupancy, and these loss functions are
compiled with `torch.compile(..., dynamic=True)` (`loss.py:362-370`), which asks
the compiler to emit one kernel valid for *every* shape and therefore forbids it
from specialising on the ones we actually use. Before anyone spends effort on
saving a third of 0.22%, it is worth finding out what those four kernels are
doing with their 68%.

Two caveats on that lead, in the spirit of §7's fix 5. This trace runs in eager
mode with profiler overhead, so the absolute milliseconds are not production
numbers — only the ratio between two kernels measured in the *same* trace is
solid. And "the grid looks wrong" is a hypothesis, not a finding: it needs a
direct measurement (compile the same loss with `dynamic=False`, or hand-write the
reduction, and time it) before it is anything more than the next thing to check.
That is exactly the mistake §7 documents making about NUMA.

### 10.5 The catch: exact is not the same as bit-identical

Section 8 draws a careful line between two standards of correctness, and this
rewrite sits on the wrong side of both of them.

Floating-point addition is **not associative**: `(a + b) + c` and `a + (b + c)`
can differ in the last bits, because each addition rounds to the nearest
representable number. The two routes through the identity add up genuinely
different intermediate quantities — in the §10.2 worked example, the current code
computes `0.30 + 0.07` while the rewrite computes `−3.80 + 4.17`. Both are 0.37 in
exact arithmetic; in fp32 over 160 hits they will differ somewhere around the
seventh significant digit.

There is a second, more interesting numerical wrinkle, and it is worth flagging
because it is not merely a last-bit issue. The rewrite computes the answer as a
**difference of two larger numbers**. In the worked example the answer 0.37 comes
out as 4.17 − 3.80: about one decimal digit of precision is lost to cancellation.
In the real problem the imbalance is much stronger — each true particle owns a
handful of the 160 hits, so `neg.sum(-1)` runs over ~150 non-belonging hits while
the final cost may be small. The relative error is inflated by roughly
`neg.sum / cost`. This is still "exact mathematics, inexact arithmetic", but it
means the deviation is not guaranteed to be at the level of rounding noise, and
a rewrite of this kind deserves a quick numerical check (compare the two forms in
float64 on a real batch) before any training run.

Why that matters more than usual here: the cost matrix is not the final answer.
It is fed to an **argmin** — the Hungarian matcher picks the cheapest assignment.
Most of the time a perturbation of 1e-7 changes nothing, because the best
assignment is comfortably the best. But when two assignments are nearly tied, a
last-bit difference can flip which one is chosen, and from there the model trains
against a slightly different target for that event.

That is not necessarily *bad* — a near-tie means the two assignments are almost
equally good by our own cost function, so the loss barely changes either way, and
the earlier fixes (a different solver, §8) already accept exactly this kind of
tie-breaking freedom. But notice the difference. For fix 3 we could *prove* the
assignment was equally optimal, because two exact solvers minimising the same
matrix must return equal-cost answers. Here the matrix itself is (very slightly)
different, so there is nothing to prove: the argument has to be empirical.

Hence the requirement recorded in §7: **a loss-curve comparison**. Run the same
configuration, same seed, same data, with and without the rewrite, for enough
steps to be meaningful, and overlay the training and validation loss curves. If
they lie on top of each other, the change is behaving as intended. That is a
weaker and much more expensive form of evidence than everything else kept in this
study — the other fixes were verified in minutes on a login node against a
556-event equivalence check (§8), whereas this one needs GPU-hours and a
judgement call about what "on top of each other" means.

That asymmetry is the whole reason the candidate was left on hold, and it is a
generalisable rule for performance work: **the cost of a change is not just the
cost of writing it, it is the cost of proving it did not break anything.** A
rewrite that preserves bit-identical output is nearly free to accept. A rewrite
that is only mathematically exact buys you the same speed for a much larger
verification bill — and if, as here, the speed turns out to be ≲0.1%, the bill is
the entire story.

---

## 11. The real find: a broadcasting bug in the mask losses

Everything above was about making the step *faster*. This one is about the discovery
that the step was slow because it was computing the **wrong thing** — and it is the
most important part of the study, because it is the only part that turned out to be a
correctness bug rather than a tuning exercise.

It is also the part that most undermines the rest of the document: §7's carefully
measured percentages were all measured against a loss that was doing 2048× too much
work. They are not wrong as measurements; they are answers to a question that was
built on a broken premise.

The fix has now been trained and evaluated end to end — §11.7 has the results.

### 11.1 What a "broadcast" is, and how it bites

NumPy and PyTorch let you multiply arrays of *different* shapes by silently
stretching the smaller one. Multiplying a `[100, 5]` table by a `[1, 5]` row
applies that row to all 100 lines. That is broadcasting, it is enormously
convenient, and it is dangerous for exactly one reason: **it never raises an
error when it does something you did not intend.** If the shapes happen to line
up, the operation succeeds — just not with the meaning you had in mind.

The rule is that shapes are aligned **from the right**, and any axis of length 1
is stretched. That right-alignment is what makes the bug below possible.

### 11.2 How we found it: two confident hypotheses, both wrong, and an error message

The discovery is worth recounting, because the reasoning that *nearly* found it was
wrong twice, and what actually found it was arithmetic on an out-of-memory message.

**The lead.** After the four matcher fixes, the post-fix trace (§7) said four Triton
loss kernels were **67.8% of everything the GPU did**. §10.4 then noticed something
that should have been alarming: `mask_bce_loss`'s forward appeared to move bytes about
**400× less efficiently** than a `bmm` measured *in the same trace on the same GPU*.
Two kernels, one device, three orders of magnitude apart. Something was wrong, but the
natural reading was "Inductor generated a bad kernel".

**Two hypotheses, both plausible, both testable.**

- **H1 — `dynamic=True`.** The losses are wrapped in `torch.compile(fn, dynamic=True)`.
  That flag tells Inductor not to specialise on fixed sizes, which can stop it from
  emitting a tight kernel for the fixed 160-long reduction axis.
- **H2 — a hidden synchronisation.** `pred_logits[object_valid_mask]` has an output
  shape that depends on the *data*, so PyTorch must run `nonzero()` and stall the CPU
  until the GPU reports how many elements survived. A profiler can easily bill that
  stall to the kernel next to it.

Both were checked directly (`bench_loss_kernels.py`, job 38463563). **Both mechanisms
are real. Neither is the cause.** `dynamic=False` really is ~1.5× faster per call — but
it recompiles on *every* step once `N_valid` moves, at ~3.9 s of compilation per step,
which is catastrophically worse end to end. The sync really does fire — confirmed with
`torch.cuda.set_sync_debug_mode("error")`, three times in BCE and twice in dice — but it
costs about 1 ms against an 88.73 ms call. Two good hypotheses, two real effects, and
together they explained almost none of the gap.

**What actually found it.** The benchmark ran the same loss in eager mode as a control,
and eager *crashed*:

```
torch.OutOfMemoryError: Tried to allocate 62.53 GiB
```

That number is the whole discovery. Nothing in the intended computation is anywhere
near 62 GiB — the honest tensors here are ~100–300 MB. So the question stopped being
"why is this kernel slow" and became "what is 62.53 GiB?" And it factors exactly:

```
2048 × 102,448 × 160 × 2 bytes (bf16) = 62.5293 GiB
 ^        ^        ^
batch  N_valid  constituents
```

A `batch` that had no business being there. Once you have that shape, the profiler data
that had been sitting in the trace all along confirms it independently: the forward BCE
kernel launched 3,218,560 / 3,231,968 / 3,273,984 blocks on the three profiled steps,
and `batch × N_valid / 64` for the three known `N_valid` values reproduces all three
numbers **exactly**. A four-line toy at `B, N, C = 3, 4, 5` then printed the offending
intermediate as `[3, 6, 5]` where `[6, 5]` was intended, and showed the returned loss
value was wrong too.

**The transferable lessons.** Three:

1. **We had been optimising a kernel instead of questioning it.** The study spent weeks
   treating "the loss kernels are 69% of GPU time" as a fact about memory-bound
   reductions and reasoning about how to make them cheaper — §10 is an entire appendix
   of algebra devoted to shaving reads off a computation that should never have been
   that size. When a component is surprisingly expensive, check *what it computes*
   before optimising *how fast it computes it*.
2. **The 400× anomaly was the real signal and it was under-weighted.** It was recorded
   in §10.4 as a curiosity, and the follow-up was framed as "why is codegen bad" rather
   than "a 400× gap is not a codegen problem". Codegen does not cost you 400×; only
   doing 400× more work does. (The final accounting made it worse still — nearer 5900×,
   *entirely* work amplification.)
3. **Crashes are data.** The OOM was initially just an inconvenience in a control arm.
   Its exact figure identified the bug faster than any of the deliberate instrumentation.

### 11.3 The bug

Our mask losses receive three things:

- `pred_logits`, `targets` — shape `[event, object, constituent]`, e.g.
  `[2048, 150, 160]`: for each of 2048 events, for each of 150 candidate
  particles, a score for each of up to 160 detector hits.
- `object_valid_mask` — `[event, object]`: which of the 150 slots are real particles.
- `input_pad_mask` — `[event, constituent]`: which of the 160 hit slots are real hits.

Real events have different numbers of hits. We pad every event out to 160 so they
fit in one rectangular tensor, and `input_pad_mask` records which entries are
padding. In our CLIC data the true count is **67 ± 27 hits, ranging from 1 to 159**
— remember that spread, it is what makes this bug matter.

The code (`loss.py`, `mask_bce_loss`) did this:

```python
pred_logits = pred_logits[object_valid_mask]    # [2048,150,160] -> [N_valid,160]
loss = F.binary_cross_entropy_with_logits(...)  #                   [N_valid,160]
loss = loss * input_pad_mask.unsqueeze(1)       # [N_valid,160] * [2048,1,160] = ???
```

The first line is the trap. Indexing with a 2-D boolean mask **flattens the two
indexed axes into one**: the event axis and the object axis merge into a single
list of "all valid objects from all events", `N_valid ≈ 100,000` of them. The
tensor went from rank 3 to rank 2, and the information about *which event each
object came from* is now gone from the shape.

The third line still assumes rank 3. `input_pad_mask.unsqueeze(1)` is
`[2048, 1, 160]`. Aligning from the right against `[N_valid, 160]`:

```
loss              (rank 2):        [N_valid, 160]   ->  [1, N_valid, 160]
input_pad_mask    (rank 3):  [2048,       1, 160]
                             ----------------------
result:                      [2048, N_valid, 160]
```

No error. Just a tensor **2048× bigger than intended**, in which *every object is
paired with every event's padding mask*. It is 62 GB in single precision, which
is why the loss kernels looked like they dominated the GPU: they were doing
2048× more work than the physics required.

### 11.4 What it did to the physics

The padding entries themselves contribute exactly zero (padded hits carry a
hugely negative logit against a zero target, so their loss is ~0). So nothing
fictitious leaks in. What changes is the **weighting**.

Intended: each object's loss is divided by the number of real hits *in its own
event*, so every particle contributes equally regardless of how busy its event was.

Actual: each object's loss is divided by a blend of *all 2048 events'* hit counts.
The practical consequence, measured on real data:

| an object in an event with... | intended total weight | actual |
|---|---|---|
| 20 hits | 1.00 | 0.35 |
| 40 hits | 1.00 | 0.66 |
| 60 hits | 1.00 | 0.86 |

**Sparse, low-multiplicity events were systematically under-trained.** The
measured effect on the training signal: the BCE loss came out 11.5% low, the dice
loss up to ~3× high late in training, and — the number that matters — the
**gradient pointed about 26° away** from where the intended loss would have sent
it (cosine similarity 0.90, consistently, batch after batch).

That last number is the whole argument for why this is not merely an efficiency
bug. A loss that is wrong by a constant factor trains identically; a loss whose
*gradient points somewhere else* trains to a different model.

Full measurements: [`LOSS_BUG_ANALYSIS.md`](LOSS_BUG_ANALYSIS.md).

### 11.5 Why nobody caught it

Three reasons worth internalising, because they generalise:

1. **Broadcasting fails silently.** Had the shapes been incompatible, this would
   have crashed on day one in 2025.
2. **It is invisible when padding is uniform.** If every event had the same number
   of hits, averaging over all events' masks equals using your own — the bug is a
   perfect no-op. Our verification test confirms legacy and fixed agree to
   *exactly* zero in that case. It only bites because real events vary.
3. **The oversized tensor looked like an expected cost** — see §11.2, which is the
   detailed version of this failure: the study treated "the loss kernels are 69% of
   GPU time" as a fact to optimise around rather than a claim to check.

### 11.6 The fix, and how to undo it

The fix keeps the tensor at rank 3 throughout, so the padding mask lines up with
the event axis the way it was always meant to. It is deliberately built to be
**reversible with a one-line config change**:

- `loss.py` gains two *new* functions, `mask_bce_loss_v2` and `mask_dice_loss_v2`,
  registered under the new names `mask_bce_v2` / `mask_dice_v2`. **The legacy
  functions are byte-for-byte untouched**, and remain the default. Nothing in the
  repository changes behaviour unless a config explicitly asks for the new names.
- `configs/clic_v6_maskfix.yaml` is a copy of `base.yaml` differing in exactly
  three lines: the run name, and the two loss keys. `configs/clic_v7_maskfix.yaml`
  is the same edit against `clic_v7.yaml`.

**To revert to the published behaviour: use `base.yaml` instead of
`clic_v6_maskfix.yaml`.** That is the entire revert. This is why the fix was not
applied in place — the legacy path *is* the paper's path, and it needs to stay
runnable and unambiguous.

Correctness was verified against an explicit per-event Python loop (agreement to
1e-9 in float64 for both losses), plus two structural checks: with uniform padding
the fixed and legacy versions agree exactly, and with no masks at all they are
identical. Script: [`verify_mask_loss_v2.py`](verify_mask_loss_v2.py).

### 11.7 What the A/B actually measured

A corrected-loss model **has** now been trained: job 38469247, 3× L4, batch 256/GPU,
200 epochs, using `clic_v6_maskfix.yaml`. Its counterpart is the June v6 run
(`clic_v6_20260605-T113014`) — same config, same hardware, same batch, same epoch
count, differing only in the loss. Three results.

**Speed: +25.4%, and it is the biggest single win in the whole study.** 1.81 → 2.27
it/s; 39 h 46 m → 31 h 55 m for 200 epochs. Splitting it against the matcher-fixes-only
rerun on the same geometry: the four matcher fixes of §7 bought 1.81 → 1.93 (+6.6%),
and the mask fix alone bought 1.93 → 2.27 (**+17.6%**). On the production L4 config the
mask fix is worth roughly **2.7× all four matcher fixes combined** — the reverse of
their ranking on the B200 protocol, because at batch 256 the cost matrices are 8×
smaller while the loss kernels are not.

**Physics: no detectable change.** Jet-E response was evaluated exactly as the
`glow_jet_iqr` study does, with a bootstrap over matched jets:

| branch | IQR @ 160–180 GeV (fixed − legacy) | rise, 20-40 → 160-180 GeV |
|---|---|---|
| `mpflow` | +0.0048 ± 0.0043 (z = +1.10) | +0.0035 ± 0.0049 (z = +0.72) |
| `mpflow_proxy` | −0.0052 ± 0.0028 (z = −1.86) | −0.0033 ± 0.0035 (z = −0.94) |

Every |z| < 2, **and the two output branches disagree on the sign** — so the differences
are sampling and branch-choice noise, not physics. Note what this does *not* say: it
does not say the gradient rotation of §11.4 was imaginary. It says a 26° rotation of the
mask-loss gradient, sustained over 200 epochs, lands the model somewhere that jet-level
metrics cannot distinguish. Those are different claims.

**The rising-IQR trend survives the fix**: both runs rise ~+0.032 (`mpflow`) while
Pandora falls. This settles a question the `glow_jet_iqr` study could previously only
argue indirectly — the bug is **not** the cause of that regression, now demonstrated by
training an arm without it rather than by reasoning about which commits contained it.

**Quality: mildly better, never worse.** Scoring *both* checkpoints under the *same*
(corrected) objective — the only way to compare two models trained on different losses —
gives mask purity +0.98%, mask exact match +1.68%, dice −2.37%, η residual −2.58%, while
event-level efficiency and purity are flat (−0.04% / −0.19%).

**Verdict: keep the fix.** Substantially faster, physics-neutral, mildly better masks.

### 11.8 What is still not known

- **Whether the corrected loss is better *as an objective*.** Everything above says it
  is not *worse*, and that it is much cheaper. It does not demonstrate that training on
  the intended loss produces a better detector reconstruction — the honest summary is
  that the distortion mattered far less to the physics than to the compute bill.
- **The B200 numbers are not re-baselined.** Every percentage in §7 was measured against
  the buggy loss, i.e. against a 2048× oversized tensor. Paired B200 runs were submitted
  on 2026-08-03 to fix this (see NOTES.md); until they report, treat §7's progression as
  a record of what was measured at the time, not as current fact.
- **`mask_focal_loss` and `mask_kl_div_loss` still have the bug.** They are unused by any
  CLIC config, but the other `ObjectHitMaskTask` experiments (trackml, itk, tide, cld,
  colliderml, atlas_muon) are affected identically and have not been examined.
- **The fix is still opt-in.** Making it the default is a deliberate decision that breaks
  comparability with every existing checkpoint and with the published numbers.
