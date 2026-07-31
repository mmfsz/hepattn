# Why our expensive GPU isn't fast: the profiling story, explained from scratch

*A pedagogical companion to [`../training_runs_report.md`](../training_runs_report.md)
and [`NOTES.md`](NOTES.md). No prior knowledge of this framework or of GPU
performance work is assumed. If you already know what a CUDA kernel and a
Hungarian matcher are, read NOTES.md instead — it's the same story in half the words.*

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
Acting on the diagnosis, we then tried the two "free" fixes (§7): one delivered
an **11.6% end-to-end speedup** and was kept; the other — swapping in a solver
that is 3.5× faster on paper — made training *twice as slow*, and the reason
why is one of the best lessons in the whole study.

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

Two more costs that have nothing to do with either resource:

- **CPU↔GPU transfers.** The GPU has its own memory; anything it needs must be
  copied over a comparatively slow link (and results copied back). A
  device-to-host copy is written **DtoH**. Transfers into ordinary ("pageable")
  CPU memory are slower than into special **pinned** memory that the copy
  engine can access directly.
- **Synchronization stalls.** If the CPU asks for a result back from the GPU
  (e.g. "give me this tensor as a NumPy array"), the CPU must *wait* for the
  GPU to finish, then the GPU sits *idle* while the CPU does whatever it wanted
  the data for. Round-trips like this serialize the two processors.

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
wall-clock. The idle time lines up with the **Hungarian matcher**, which runs
*on the CPU* (SciPy's `linear_sum_assignment`), costing ~1.24 s of CPU time per
step. While the CPU grinds through the assignment problem for 2048 events, the
GPU — all that silicon — does nothing. This is the "synchronization stall"
pattern from §3: the loss cannot be computed until the matching is known, so
the GPU genuinely has to wait.

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

1. **The loss is computed on enormous matrices.** For every one of 5 decoder
   layers, the training loss builds dense tensors of shape roughly
   `batch (2048) × predicted objects × hits` in full precision (fp32) and runs
   sums/sigmoids/cross-entropies over them. These show up as the
   `triton_red_fused_...binary_cross_entropy...` kernels in the trace. They are
   textbook **memory-bound** operations (§3): simple math over huge arrays. The
   B200's FLOPS advantage is nearly useless here. A tiny 10M-parameter model
   attached to a giant set-matching loss means the loss, not the model,
   dominates.
2. **The cost matrices take a slow boat to the CPU.** The Hungarian matcher
   needs the cost matrix as a NumPy array, so each step performs ~2 large
   **device-to-host copies of ~236 ms each** (~0.47 s/step) — into *pageable*
   memory, the slow kind (§3). That's nearly a quarter of all GPU activity
   spent on a memory copy whose only purpose is to feed a CPU algorithm.

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

## 7. What we did about it: two fixes, two verdicts

Diagnosis and treatment are separate steps. From the fix candidates, we picked
the two that are **pure optimizations** — changes that cannot affect the
physics, only the speed — implemented them, and *measured* each one with the
same Phase-1 protocol (identical config, hardware, and step count; one variable
changed at a time). One won, one lost, and both taught us something.

### Fix 1 (KEPT): pin the memory the cost matrices land in — **−11.6% step time**

Recall from §3 that CPU↔GPU copies into ordinary ("pageable") CPU memory are
several times slower than copies into **pinned** memory, and from §6 that each
step spent ~0.47 s copying cost matrices into exactly that slow kind of memory.
The fix: the matcher now keeps one reusable pinned buffer and copies the cost
matrices into it, instead of allocating fresh pageable memory every step. The
numbers that come out are bit-for-bit identical — only the route changes.

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

### Fix 2 (REJECTED): a faster Hungarian solver that made training 2× slower

The repository ships an alternative assignment solver, `lap1015`, which our
benchmark shows is genuinely **3.5× faster than scipy per matrix**
(310 ms vs 1084 ms for 2000 events, single-threaded). Switching to it is a
one-line config change. The measured result: training slowed from 3.76 s to
**7.14 s per step** — nearly twice as slow. How can a faster algorithm lose?

**The Global Interpreter Lock (GIL).** Python allows only one thread at a time
to execute Python-level code — that's the GIL. Compiled extensions can
explicitly *release* it while they crunch numbers in C, letting other threads
run. The matcher solves ~10,000 small assignment problems per step across 16
threads. scipy's solver releases the GIL, so those 16 threads genuinely run in
parallel (we measured ~11× scaling). The lap1015 binding **never releases the
GIL**, so its 16 "parallel" threads take turns running one at a time — all that
parallelism silently evaporates, and a 3.5× faster algorithm becomes ~3×
slower in practice:

| 2000 events, 16-way parallel | scipy | lap1015 |
|---|---|---|
| threads (current setup) | **113 ms** | 326 ms (GIL-serialized) |
| processes (no GIL, but data must be copied between them) | 201 ms | 191 ms |

The moral, twice over: *single-threaded benchmarks don't predict parallel
behaviour*, and *always measure the end-to-end number before promoting an
"obvious" win*. The current scipy configuration — which looked like an odd
choice next to a faster solver sitting in the same repo — turns out to be the
fastest available combination. Whoever pinned it knew.

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

### What remains on the table

- **Release the GIL in lap1015's binding** — a one-line C++ change
  (`py::gil_scoped_release`) plus a rebuild of the compiled extension.
  Projected ~4× on the solve; not done because rebuilding shared binaries
  affects everyone using the environment.
- **Overlap the matching with GPU work** (~0.85 s/step of GPU idle): start the
  next batch's forward pass while the CPU matches the current one. Real
  engineering, numerics-preserving in principle.
- **Shrink the loss kernels** (~1.05 s/step): compute mask losses in bf16 or
  only on matched pairs. This changes numerics — the only candidates on the
  list with accuracy risk — so they need physics validation.

The strategic takeaway stands, now with both a mechanism and a partial remedy:
**for this workload, a B200 is poor value.** After the pinned-memory fix, ~93%
of the step still doesn't scale with GPU FLOPS. The B200 only pays off if the
model gets much bigger, or if the loss/matcher pipeline above is reworked so
that model compute becomes the dominant cost.

---

## 8. Small print: pitfalls we hit (so you don't)

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
  op-level trace is for.
- **A "failed" experiment can be the most informative one.** The lap1015 run
  crashed twice (empty-matrix garbage, then non-finite-cost corruption) before
  producing its negative timing verdict — and each failure exposed a latent bug
  that is now fixed with tests. Budget for this: fix attempts are experiments,
  not patches.
- **One pre-existing broken test.** `tests/matching/test_solvers.py::test_lap1015`
  calls `lap1015.lap_early`, which hangs forever on this machine (it was already
  commented out of the solver registry — a clue). Deselect that file when
  running the matching tests; the other 118 tests pass.

**Artifacts:** raw profiler tables and all SLURM logs (phases 1–2 and both fix
experiments, `fix1_lap1015_*` / `fix2_pinned_*`) are in
[`profile_logs/`](profile_logs/); the 46 MB Chrome trace
(`fit-clic_profile_phase2-*.pt.trace.json`) can be dropped into
<https://ui.perfetto.dev> to explore the timeline yourself — the ~236 ms
`Memcpy DtoH` bars and the kernel-free idle gaps are easy to spot by eye.
