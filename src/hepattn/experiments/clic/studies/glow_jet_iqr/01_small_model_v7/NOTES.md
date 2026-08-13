# Task 1 — 700k small model (v7) training + IQR eval

**Status:** ✅ DONE (2026-07-14) — BOTH v7 runs trained + eval'd; **capacity is NOT the cause** (triple-checked).

## Final verdict (2026-07-14) — triple check complete
Both 700k runs are now evaluated on the common test set (19,722 events) and overlaid against the
10.1M bf16 baseline + fp32 + Pandora in `../jet_iqr_new_evals.png` (`../plot_new_evals_iqr.py`):
- **Both v7 runs (1-node job 36472892 best epoch=197 val 4.25845; 2-node job 36450143 best
  epoch=192 val 4.33644) show the SAME inverted rising IQR trend** (~0.077 @10 GeV → ~0.104 @190),
  a small constant offset above the 10.1M baseline. The two v7 runs track each other closely →
  consistent, not a fluke of one training.
- A **14× smaller** model (0.70M vs 10.1M) reproduces the exact trend → **model capacity / param
  count is NOT the driver**; the paper's 12M-vs-our-10.1M gap is a red herring for the IQR trend.
- 1-node eval = job 37145280 (state TIMEOUT but **inference succeeded** — the ROOT was written
  before the wall; only torch-inductor's atexit compile-worker shutdown hung 300s afterwards).
  The old partial epoch-105 root stays renamed `*.PARTIAL-1node-stilltraining` and is not plotted.

Reinforces task 5: the cause is the post-paper model **refactor**, not size.

## Job→folder mapping (VERIFIED via each folder's `metadata.yaml` slurm_job_id) — 2026-07-07
The two folders both start `clic_v7_20260706-T18141…` (both jobs started 18:14); the timestamp
does NOT tell them apart. The definitive mapping:
| Folder | slurm_job_id | Job | Config | State | ckpts | Best ckpt |
|---|---|---|---|---|---|---|
| `clic_v7_20260706-T181417` | 36450143 | **2-node** | v7 | **FINISHED (200 ep)** | 200 (0–199) | **epoch=192, val_loss 4.33644** |
| `clic_v7_20260706-T181418` | 36472892 | **1-node** | v7 | RUNNING (~ep 109) | 109 (0–108) | tbd when done |

> ⚠️ Earlier I mis-assigned these by folder timestamp and evaluated T181418 (the *running*
> 1-node run, partial epoch 105) — cancelled (job 36519705). There is **NO checkpoint anomaly**:
> T181418 only has epochs 0–108 because it is still at ~epoch 109. The finished 2-node run
> (T181417) has all 200 checkpoints normally.

## Plan (triple-check: 2-node v7 + 1-node v7 + v6 family, per user)
- **2-node (36450143):** eval best ckpt **epoch=192 (val_loss 4.33644)** → **job 36520192** (1×L4)
  → `logs/clic_v7_20260706-T181417/ckpts/epoch=192-val_loss=4.33644__test.root`.
- **1-node (36472892):** eval its best ckpt once it finishes 200 epochs.
- Plot: **`plot_v7_iqr.py`** (auto-discovers each run's `*__test.root`, SKIPs un-evaled ones) →
  `jet_iqr_v7_vs_v6.png`. Re-run as each eval lands:
  `pixi run -e clic python studies/glow_jet_iqr/01_small_model_v7/plot_v7_iqr.py`.
- (v7 best val_loss ~4.34 vs v6 ~3.79 — expected for a 14× smaller model.)

## Objective
Test whether the inverted high-E jet-IQR trend depends on model **capacity**. v7 is the
same architecture as v6 but **~14× smaller** (702,395 vs 10,126,115 params; `dim 256→64`,
heads `16→8`, narrower head MLPs — see `../../configs/CONFIGS.md`). Everything else
(6-enc/4-dec depth, 150 queries, 8 registers, Lion @ 8e-5, 200 epochs, bf16-mixed,
flash-varlen, same data) is identical.

**Interpretation key:** if v7's IQR trend is *the same* inverted shape → capacity is NOT
the driver, and the 12M-vs-10.1M lead weakens. If v7 is *dramatically worse at high E* →
capacity matters and the paper's larger 12M model is a plausible explanation.

## Runs (both v7, started 2026-07-06 T18:14)
| Job | Nodes×GPU | Global batch | Folder |
|---|---|---|---|
| 36472892 | 1×3 L4 | 512×3 = 1536 | `logs/clic_v7_20260706-T181417` |
| 36450143 | 2×3 L4 | 512×6 = 3072 | `logs/clic_v7_20260706-T181418` |
(submit scripts: `../../submit_training_hpg_l4.sh`, `../../submit_training_hpg_l4_2nodes.sh`)

## Method (when training finishes)
1. Confirm 200 epochs completed; pick best ckpt by `val_loss` (in filename).
2. Eval: `../../submit_eval_run.sh <run_folder> <best_ckpt>` → `__test.{h5,root}`
   (uses `eval.yaml`: `precision 32-true`, `attn_type torch`, `matmul highest`).
3. Overlay IQR vs the v6 runs + Pandora with `../00_cross_run/compare_runs_iqr.py`
   (add the v7 root paths; repoint `OUT` here). Bin `e_rel` in truth jet E, IQR = p75−p25.
4. Record the high-E IQR (~170–190 GeV bin) for both v7 runs in the table below.

## Checklist
- [ ] 1-node v7 (36472892) reached 200 epochs
- [ ] 2-node v7 (36450143) reached 200 epochs
- [ ] eval'd both → `.root` outputs
- [ ] IQR overlay produced
- [ ] verdict recorded (capacity matters? Y/N)

## Results — 2-node v7 (epoch 192) DONE 2026-07-07 → capacity is NOT the cause
Figure: `jet_iqr_v7_vs_v6.png` (v7 2-node vs v6 family vs Pandora; 1-node pending **at the time**
— both runs now appear, in both conventions, in `../07_paper_tag_small/jet_iqr_paper_small.png`).

> ### ⚠️ 2026-08-13 — EVERY NUMBER IN THIS FILE IS THE **REGRESSION** (`mpflow`) CONVENTION
> This task closed 2026-07-14, six days before the discovery that the paper plots the **PROXY**
> (`mpflow_proxy`) output (see `../05_reproduce_paper_tag/NOTES.md`). Nothing here is wrong — the
> regression numbers below were re-derived independently on 2026-08-13 and match digit-for-digit —
> but they are **not** directly comparable to the paper's Fig. 4.
> **v7's proxy curves were first computed 2026-08-13**, in `../07_paper_tag_small/`:
> 2-node `0.080 0.064 0.060 0.058 0.058 0.059 0.057 0.064 0.064 0.055`,
> 1-node `0.079 0.065 0.059 0.060 0.057 0.060 0.060 0.068 0.069 0.056`.
> The proxy convention flatters HEAD a lot — the curves look nearly flat and the naive
> first-vs-last-bin trend even comes out negative. **The conclusion below is unchanged**: both v7
> runs still rise across E90→E170 in proxy (0.058→0.064 and 0.057→0.069) and rise outright in
> regression. But quote the convention whenever you quote these numbers.

Jet-E response IQR (p75−p25) vs truth jet E [GeV]:
| E | 10 | 30 | 50 | 70 | 90 | 110 | 130 | 150 | 170 | 190 |
|---|---|---|---|---|---|---|---|---|---|---|
| v7 700k (2-node) | 0.082 | 0.072 | 0.076 | 0.086 | 0.087 | 0.097 | 0.090 | 0.095 | 0.099 | 0.101 |
| 3×L4 v6 (plotted) | 0.075 | 0.067 | 0.070 | 0.077 | 0.077 | 0.085 | 0.077 | 0.083 | 0.099 | 0.096 |
| Pandora | 0.091 | 0.076 | 0.068 | 0.064 | 0.061 | 0.063 | 0.059 | 0.060 | 0.056 | 0.054 |

**The 700k v7 shows the SAME inverted rising trend** (~0.08→0.10) as v6, just a ~0.005–0.015
constant offset worse (expected for 14× fewer params) — NOT a change in shape. Shrinking the
model 10.1M→0.7M does not flip or flatten the trend. → **Capacity is not the driver**; the
paper's 12M-vs-our-10.1M gap is a red herring. Consistent with task 4 (cause = post-paper model
refactor). Task 5 (paper-tag retrain) is the confirming test.

~~TODO: eval the 1-node v7 (36472892)~~ — **DONE** (eval 37145280, 2026-07-14); the stale partial
epoch-105 root remains renamed `*.PARTIAL-1node-stilltraining`. Table filled in 2026-08-13; the
1-node run reached epoch 197 / val_loss 4.25845.

Regression convention (`mpflow`), matching the table above:

| Run | best val_loss | IQR @ ~10 GeV | IQR @ ~90 | IQR @ ~170 | IQR @ ~190 |
|---|---|---|---|---|---|
| v7 1-node (36472892) | 4.258 | 0.078 | 0.089 | 0.102 | 0.104 |
| v7 2-node (36450143) | 4.336 | 0.082 | 0.087 | 0.099 | 0.101 |
| v6 3×L4 (ref) | 3.788 | 0.075 | 0.077 | 0.099 | 0.096 |
| Pandora (ref) | — | 0.091 | 0.061 | 0.056 | 0.054 |

**Note (2026-08-13): the better val_loss is the WORSE run on this metric.** The 1-node v7 beats the
2-node on val_loss (4.258 vs 4.336) but has the worse high-E IQR in both conventions (regression
trend +0.026 vs +0.019; proxy 0.069 vs 0.064 @E170). Val_loss does not track the IQR trend — do not
use it to pick checkpoints or to score commits in the phase-2 bisect.
