# Task 7 — paper-tag code at SMALL (v7) width

**Status:** ✅ **DONE 2026-08-13** — smoke 39236740, training **39236741** (COMPLETED, 200/200 ep, 20 h 02 m,
1 node × 3 L4 `hpg-turin`), eval **39352893**. **VERDICT: the falling IQR SURVIVES the 14.7× shrink**
— the paper's shape is a property of its *code*, not its capacity. See **Results** at the bottom.

## Objective — close the 2×2

Tasks 1 and 5 each varied one thing and left the other fixed. This run fills the empty cell:

| | **paper-tag code** (`fb90390`) | **drifted HEAD** (`1df05cc`) |
|---|---|---|
| **large** (10–12M) | task 5: IQR **falls** 0.072 → 0.043 ✅ | task 0: IQR **rises** 0.075 → 0.096 |
| **small** (~0.7–0.8M) | **THIS RUN: IQR falls 0.078 → 0.050** ✅ | task 1: IQR **rises** ~0.077 → ~0.104 |

Reading the result:

- **IQR still falls** → the falling trend is a property of the paper's *code*, independent of
  capacity. That completes the argument that the post-paper refactor is the whole story, and
  makes the bisect (task 5 follow-up) the only remaining question.
- **IQR turns over** → capacity and code *interact*: the paper's shape needs the paper's width.
  That would partly rehabilitate the 12M-vs-10.1M lead this study set aside after task 1, and
  would mean the refactor verdict is narrower than currently written.

Either way it is informative, which is why it is worth the ~19 h.

## What was built

| File | What |
|---|---|
| `hepattn-clic-paper/.../configs/base_small.yaml` | paper-tag `base.yaml` + 5 width edits, nothing else |
| `hepattn-clic-paper/.../submit_smoke_paper_small.sh` | 2 epochs × 20 batches, checks param count |
| `hepattn-clic-paper/.../submit_training_paper_small.sh` | the real 200-epoch run |

(All three live in the **paper-tag clone**, `/blue/avery/m.mazza/projects/fastml/hepattn-clic-paper`,
because they must run against that code. See `../REPRODUCE.md` §B for the clone's four HPG patches.)

### The five edits, and why exactly five

`base.yaml` already carries a `&dim` anchor, so narrowing the model is mostly one value:

| # | Edit | Note |
|---|---|---|
| 1 | `dim: &dim 256 → 64` | propagates to input net, posenc, encoder, decoder layers, class-head input, mask task, **and** the incidence head's `[*dim, *dim]` / `[*dim]` |
| 2 | encoder `num_heads: 16 → 8` | v7's value; head_dim 16 → 8 |
| 3 | decoder `num_heads: 16 → 8` | v7's value |
| 4 | class head `[256, 128, 32] → [64, 128, 32]` | v7's value |
| 5 | regression head `input_size 518 → 134`, `[512,256,128,64,32] → [128,128,128,64,32]` | v7's values; 518 = 2×256+6 and 134 = 2×64+6 (query emb + incidence-weighted node emb + 6 raw node vars), so the formula is identical in both code versions |

Everything else is untouched: 6-enc/4-dec depth, 150 queries, 8 registers, Lion @ 8e-5, 200
epochs, bf16-mixed, flash-varlen, `batch_size 170` (global 1020 over 6 L4), same data.

**`batch_size` stays at 170** even though 0.8M params would fit far more. The LR is not
batch-scaled, so raising it would move a second variable and this would stop being a clean
capacity ablation.

## Geometry: 1 node × 3 L4 + `accumulate_grad_batches=2`

The reference run (37233919) was 2 nodes × 3 L4 × 170 = **global batch 1020**. That global batch
has to be held. On 3 GPUs there were two routes:

| | global | per-rank | verdict |
|---|---|---|---|
| `batch_size 340 × 3` | 1020 | **340** (2×) | rejected |
| `batch_size 170 × 3 × 2 accum` | 1020 | **170** (same) | **chosen** |

Accumulation wins on two counts, both about fidelity rather than convenience:

1. **Per-rank tensor shapes match the reference exactly.** That matters more than usual here: the
   mask-loss broadcast bug (`../../b200_utilization/profiling/LOSS_BUG_ANALYSIS.md`) is present
   in this tag, and its intermediate scales as `B × N_valid × C`, i.e. ~B². Doubling the per-rank
   batch would quadruple that tensor for no scientific gain.
2. **The loss is normalised per rank.** DDP over 6 ranks of 170 averages six independently
   normalised 170-sample losses; 2 accumulation steps over 3 ranks does the same. A single batch
   of 340 would not — it normalises over 340 at once. So accumulation is *closer* to the
   reference than the naive "same global batch" option.

1 node also schedules much faster than 2 and removes inter-node NCCL.

### ⛔ Why not B200

Asked for, checked, not possible. This clone's env is **torch 2.7.0+cu126** with
**flash-attn 2.7.4** — neither has Blackwell (`sm_100`) kernels; that needs CUDA ≥ 12.8. A B200
job would die with "no kernel image is available for execution on the device".

The B200 runs in the `b200_utilization` study are not a counter-example: they ran the **fork's**
apptainer container (`pixi.sif`, torch 2.9.1 / cuda 12.8.1) against HEAD code. Rebuilding the
paper-tag env on cu128 with a newer torch + flash-attn would change the software snapshot the
whole reproduction rests on, so the paper-tag arm is **L4-only** unless someone decides that
trade is worth making. Worth remembering for any future "just run it on the B200" request.

### Why v7's config could not simply be dropped onto the tag

The earlier "the format differs" answer was right, and this is precisely where:

- **`input_hit` vs `input_constituent`** — the tag's tasks take `input_hit`; HEAD renamed it in
  the terminology refactor. v7's file would fail to construct.
- **`loss_class_weights` vs `class_weights`** — same story on `ObjectClassificationTask`.
- **`hidden_layers: 2`** — v7's shorthand for the incidence head. The tag's `Dense` accepts
  `list[int] | None` **only**; there is no int branch, so this raises outright.
- `decoder:` needs no `class_path` at HEAD but the tag resolves it differently, and the posenc
  class path moved (`hepattn.models.FourierPositionEncoder` vs `...models.posenc....`).

So the port went the other way: keep the **tag's** file and change only the widths.

### ⚠️ The incidence head is deliberately NOT v7's

v7's `hidden_layers: 2` is not `[64, 64]`. At HEAD, `Dense` expands an int to
`[input_size × hidden_dim_scale] × n` = **`[128, 128]`** — the post-paper incidence *widening*
(the same mechanism that took v6 from the paper's `[256, 256]` to `[512, 512]`, flagged in
`../README.md` as a prime suspect for the IQR regression).

Copying it here would have imported the suspect change into the run meant to isolate it. The
config keeps `[*dim, *dim] → [64, 64]`, i.e. **the paper's structure at the new width**. This is
the one place where "match v7" and "match the paper" disagree, and the comment in
`base_small.yaml` says so, so nobody silently "fixes" it later.

## Validation done so far (2026-08-11)

- ✅ `--print_config` under the tag's own pixi env parses cleanly.
- ✅ Every width resolved as intended: `dim 64` throughout, `num_heads 8` enc+dec, class head
  `[64,128,32]`, incidence `64→64` via `[64,64]`, node_net `[64]`, regression `134 → [128,128,128,64,32] → 5`.
- ✅ The resolved file is structurally line-for-line identical to `../config_paper_tag.yaml`
  with only the widths differing.
- ❌ **Param count not yet measured.** A CPU `fast_dev_run` on the login node was OOM-killed
  opening the 12 GB train ROOT (exit 137) — a login-node memory cap, not a config fault. The
  count comes from the smoke job instead.

## Run order

```shell
cd /blue/avery/m.mazza/projects/fastml/hepattn-clic-paper/src/hepattn/experiments/clic
sbatch submit_smoke_paper_small.sh      # ~40 min cap; CHECK IT BEFORE THE NEXT LINE
sbatch submit_training_paper_small.sh   # 200 epochs; reference run was 19 h 05 m ON 6 L4 (2 nodes)
```

**Smoke gate — the point of the smoke is this one line.** `ModelSummary` must print roughly
**0.7–0.9 M** params (naive dim² scaling of the paper's 12.07M gives ~0.75M; v7 landed slightly
above the naive estimate against its own baseline, so treat ~0.8M as the expectation and
anything ≲1M as fine). If it prints **12.1 M** the run silently loaded full-width `base.yaml`
and the whole experiment is void — that is the failure this gate exists to catch.

Also confirm from the smoke: no `InvalidOfflineDirectory`, checkpoints written under
`logs/clic_paper_small_smoke_*/ckpts/`, and `csv_metrics/metrics.csv` carrying train + val loss.

Head-dim 8 (64/8) is new for this code path; flash-varlen should accept it, but the smoke is
where that surfaces rather than hours into the real run.

## Then

Evaluate exactly as `../REPRODUCE.md` §B, and **plot `mpflow_proxy`, not the regression branch** —
the paper's Fig. 3/4 convention. Overlay against task 5's paper-tag curve and task 1's small-HEAD
curve with `../05_reproduce_paper_tag/plot_paper_iqr_proxy.py`.

---

## Results — 2026-08-13

The mirror of task 1. Task 1 shrank the *drifted HEAD* 14× and the rising IQR stayed → capacity is
not what broke HEAD. This asks the same question of the code that actually reproduces Fig. 4:
**narrow the paper-tag model to the v7 width and does its falling IQR survive?**

**Run:** `configs/base_small.yaml` in the paper clone — `base.yaml` with 5 width edits only
(dim 256→64 anchor, enc/dec heads 16→8, class head [256,128,32]→[64,128,32], regression head
518→134 in / [512,256,128,64,32]→[128,128,128,64,32]). Incidence head deliberately left at the
PAPER's `[*dim, *dim]` structure (→[64,64]), NOT v7's `hidden_layers: 2` shorthand, which would
smuggle in the post-paper incidence widening that is itself the prime suspect.
- Job **39236741**, 3× L4 1 node, `accumulate_grad_batches=2`, batch 170 (LR deliberately not
  re-scaled). **200/200 epochs**, **20 h 02 m**, COMPLETED clean.
- **Cost, stated carefully — compare GPU-hours, not wall clock.** The reference run (37233919,
  19 h 05 m) used **6 L4 over 2 nodes**; this one used **3 L4 on 1 node**. Equal wall clock on half
  the hardware = **115 → 60 GPU-h, ~1.9× cheaper**. Do NOT read "20 h vs 19 h" as "shrinking the
  model bought nothing" — that comparison is confounded by the GPU count. (Note the small run also
  does *more* micro-steps per epoch, ~1949 vs the full run's ~975 optimizer steps, because batch
  170 was held fixed rather than raised; so the per-step gain is larger than 1.9× and a
  batch-tuned small run would be cheaper still.)
- **Also not comparable:** the `b200_utilization` L4 numbers (39 h 46 m → 31 h 55 m) are **HEAD**
  v6, 10.1M, batch 256/GPU, 1295 steps/epoch — different code, model, and batch. Only compare
  paper-tag runs to paper-tag runs.
- **819,683 trainable params** (vs 12,065,291 full) = **14.7× smaller**. NB this is *close to* but
  not equal to HEAD-v7's **702,395** — the two code versions build the same nominal widths into
  slightly different param counts, so label them separately (0.82M paper-tag-small vs 0.70M v7).
- val_loss 31.67 → **4.3716** (best epoch 196); `val/final_regression_e_abs_norm_res` 1.256→0.058.
- Eval job **39352893** → `epoch=196-val_loss=4.37156__test.root`. Plot:
  **`plot_paper_small_iqr.py`** → **`jet_iqr_paper_small.png`** (2×2: capacity × code-version,
  both conventions).

### Jet-E IQR, PROXY convention (the paper's own; compare these four)

| model | code | params | E10 | E90 | E170 | E190 |
|---|---|---|---|---|---|---|
| paper-tag full | tag | 12.1M | 0.072 | 0.053 | 0.049 | **0.043** |
| **paper-tag small** | **tag** | **0.82M** | 0.078 | 0.058 | 0.058 | **0.050** |
| HEAD v7 small (2-node) | HEAD | 0.70M | 0.080 | 0.058 | 0.064 | 0.055 |
| HEAD v7 small (1-node) | HEAD | 0.70M | 0.079 | 0.057 | 0.069 | 0.056 |
| HEAD v6 full | HEAD | 10.1M | 0.073 | 0.058 | **0.084** | 0.079 |

Regression convention, first→last bin trend: paper-full **−0.019**, paper-small **−0.008**,
HEAD-v7 2-node **+0.019**, HEAD-v7 1-node **+0.026**, HEAD-v6-full **+0.021**.

### ⚠️ First look at HEAD-v7 in the PROXY convention — and why it looks deceptively good

Task 1 (closed 2026-07-14) predates the proxy-convention discovery (2026-07-20), and
`05_reproduce_paper_tag/plot_paper_iqr_proxy.py` only ever re-plotted the paper-tag and v6 roots.
**So this plot is the first time HEAD-v7 has been seen in the paper's own convention.** Its
regression row here reproduces task 1's published table digit-for-digit
(`0.082 0.072 0.076 0.086 0.087 0.097 0.090 0.095 0.099 0.101`) — nothing has changed, the proxy
curve is simply new.

At a glance v7-proxy looks far better than task 1's write-up suggests (0.080 → 0.055). **Do not
read that as "HEAD is fine in the proxy convention."** Both v7 runs still rise across the
high-E region — 2-node 0.058→0.064 and 1-node 0.057→0.069 over E90→E170 — and both regression
curves rise outright. The negative overall trend is the same E190-bin artifact flagged above.
What *is* true and newly documented: **the proxy convention flatters HEAD substantially more than
the regression convention does**, compressing the apparent gap to the paper tag. Any future
comparison must state which convention it uses.

**Both v7 trainings agree**, which makes this reproducible rather than a one-run fluke:
`clic_v7_20260706-T181417` (job 36450143, 2-node, val 4.336) and `…-T181418` (job 36472892,
1-node, val 4.258) sit within ~0.005 of each other at every bin in the proxy convention.
Note the 1-node run has the **better val_loss** but the **worse** high-E IQR (0.069 vs 0.064 @E170,
regression trend +0.026 vs +0.019) — val_loss does not track the IQR trend, worth remembering
when picking a bisect metric.

**VERDICT: the paper-tag model keeps its falling IQR when made 14.7× smaller.** Proxy trend
E10→E190 is −0.029 vs −0.030 for the full paper model — essentially identical shape, shifted up
by a roughly constant ~+0.005–0.007 (the honest cost of 14.7× fewer params). It does NOT invert.

**Caveat on reading the trend number:** first-vs-last-bin is contaminated by the E190 bin, which
dips for every curve (low stats). The clean discriminator is the **E90→E170 segment**, where the
code split is unambiguous: paper-full 0.053→0.049 (falls), paper-small 0.058→0.058 (flat),
HEAD-v7-small 0.058→0.064 (rises), HEAD-v6-full 0.058→0.084 (rises hard). Note HEAD-v7-small's
proxy curve *does* post a negative overall trend (−0.025) purely via that last-bin dip — do not
read it as "HEAD small falls"; its high-E segment rises and its regression curve rises outright.

**Conclusion — this closes the capacity hypothesis from both sides:**
- shrink HEAD → still rises (task 1)
- shrink the paper tag → still falls (here)

Capacity is orthogonal. The falling-vs-rising IQR tracks the **code version**, not model size,
which is exactly what task 5's verdict claimed. A small model is therefore a legitimate cheap
testbed for the bisect (phase 2): it reproduces the paper's qualitative IQR behaviour at ~1/15 the
params and 20h on 3× L4 instead of a multi-node full run — so bisect commits can be trained at
this width without the capacity confound invalidating the readout.

## 2026-08-13 — Param counts VERIFIED from the checkpoints, and why the two "small" models differ

Prompted by "are you sure?" — both counts re-derived directly from the `.ckpt` files rather than
trusted from earlier notes. **Both confirmed:**

| model | trainable | source |
|---|---|---|
| HEAD v7 (both runs) | **702,395** | matches task 1's written figure |
| paper-tag small | **819,683** | matches the run's own COMET log |

Method: `state_dict` totals are 846,322 / 972,122, which are *not* the trainable counts — the task
modules are registered twice (once as `tasks.N`, once as `decoder.tasks`, sharing the same
Parameters), so the duplicate must be subtracted: 846,322 − 143,857 = 702,465 and
972,122 − 152,369 = 819,753. Both then sit exactly **70** above the reported figure — the same 70 in
both, so it cancels in any comparison. (70 unexplained; presumably a frozen/non-trainable param. It
is 0.01% and identical on both sides, so it changes nothing here.)

### Why they differ: +117,288 (+16.7%) at identical nominal width

They are **not the same architecture** — same `dim 64`, same 8 heads, same 6-enc/4-dec depth, but
the blocks themselves differ. Measured per-layer, paper minus HEAD:

| | HEAD v7 | paper small | Δ |
|---|---|---|---|
| encoder layer | 33,856 (18 tensors) | 41,536 (8 tensors) | **+7,680** ×6 |
| decoder layer | 84,864 (48 tensors) | 99,712 (20 tensors) | **+14,848** ×4 |

Two effects, in opposite directions:

1. **SwiGLU → SiLU in every transformer FFN — the dominant term.** The paper's `Dense` defaults to
   `SwiGLU()`, which is *gated* and therefore doubles the first projection: `Linear(64 → 256)`.
   HEAD's `Dense` defaults to plain `nn.SiLU()`: `Linear(64 → 128)`. Confirmed from the ckpt shapes
   (`dense.fn.net.0.weight` = `(256,64)` paper vs `(128,64)` HEAD). **Neither config overrides
   this** — the `activation: torch.nn.SiLU` lines in both YAMLs are on the *task heads* only; the
   encoder/decoder blocks take the library default. Worth +8,320 per encoder layer and +16,640 per
   decoder layer = **+107,264** of the +117,288.
2. **HEAD adds normalisation the paper doesn't have** — `q_norm`, `k_norm`, `v_norm` inside
   attention plus explicit `attn.norm` / `dense.norm` pre-norms: 10 extra tensors per encoder layer
   and 28 per decoder layer, 64 params each. Only −640 / −1,792 per layer, but it is why HEAD has
   *more tensors* while having *fewer params*.

Remainder: task-head width/ordering +8,512 and input net +1,512.

Sanity check on the ratio: paper/HEAD is **1.191** at full width (12,065,291 / 10,126,115) and
**1.167** at small width (819,683 / 702,395) — the same architectural delta scaled down, as expected.

### 🚨 NEW BISECT SUSPECT: SwiGLU → SiLU (`6488c91`, 2025-11-16, "TackML Perf fixes (#198)")

`git log -S "activation = nn.SiLU()" -- src/hepattn/models/dense.py` → **`6488c91`**, and
`git merge-base --is-ancestor 6488c91 fb90390` confirms it is **NOT** an ancestor of the paper tag,
i.e. **post-paper**. It changed the `Dense` default activation from `SwiGLU()` to `nn.SiLU()`,
silently removing the multiplicative gate from **every encoder and decoder FFN** in the model.

This is **not** on the suspect list in `../README.md` (which names the incidence-head widening and
the norm/decoder-attention rewrite). It is a genuine expressivity change to every block, it landed
in a PR titled "perf fixes", and it is trivially revertible via `dense_kwargs` — so it is a cheap,
high-value first probe for phase 2. Note it is *also* the reason HEAD is the **smaller** model
despite the incidence-head widening, which resolves the long-standing "12M vs 10.1M" oddity.

### ⚠️ Caveat this places on the 2×2

The two small cells are each "their own code at `dim 64`", not one architecture at one width. That
is unavoidable across a refactor — matching params exactly would have meant importing the very
changes under test — and it does not weaken the verdict, since the comparison is *code version* vs
*code version* at matched nominal width. But do not describe the two small models as identical.
