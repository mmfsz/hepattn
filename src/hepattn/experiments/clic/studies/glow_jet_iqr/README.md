b20 # GLOW Jet-Energy IQR Discrepancy — Investigation Hub

**Owner:** Maria Mazza · **Started:** 2026-07-06 · **This hub created:** 2026-07-07
**Paper:** GLOW, arXiv:2508.20092 (linked in `../../README.md`) · **Fork:** `lgray/hepattn`
(upstream original: `samvanstroud/hepattn`)

This folder collects all work on **one question**: why does our trained CLIC
particle-flow ("Glow"/`mpflow`) model give a **jet-energy IQR that RISES with jet
energy** (~0.075 → ~0.10) when the paper's Glow IQR **FALLS** (~0.10 → ~0.05)?
The trend is **inverted** and ~2× worse at high E. This is confirmed real — not a
data/plotting/eval bug (all ruled out in Phase 1, see `00_cross_run/`).

The investigation spans several days with concurrent SLURM jobs, so work is split
into numbered task folders. Each has its own `NOTES.md` (objective → method →
status → findings). **Update the dashboard below whenever a task changes state.**

---

## Status dashboard

| # | Task | Status | Folder | Blocking on |
|---|---|---|---|---|
| 0 | Cross-run consistency (v6: 4 runs vs Pandora) | ✅ DONE | `00_cross_run/` | — (inverted trend is universal) |
| 1 | Train **700k** small model (v7) → eval IQR | ✅ DONE — **capacity ≠ cause** (both runs eval'd, triple-checked) | `01_small_model_v7/` | — |
| 2 | Plot **loss curves** + FIX loss storage | ✅ DONE — training healthy, no overfit; CSVLogger fix landed | `02_loss_curves/` | — |
| 3 | Train **fp32** (32-true) + confirm paper precision | ✅ DONE — **precision ≠ cause** (only 3a: paper's precision still to confirm) | `03_precision/` | — |
| 4 | **Fork diff** vs upstream `samvanstroud/hepattn` | ✅ DONE — **reframes everything** | `04_fork_diff/` | — |
| 5 | **Reproduce from `clic-paper` tag** (paper's actual code) | ✅ DONE 2026-07-20 — **IQR trend REVERTS to falling → post-paper refactor IS the cause** | `05_reproduce_paper_tag/` | — (next: bisect the refactor) |
| 6 | **Eval attention/precision faithfulness** (train-matched eval) | ⬜ TODO | `06_eval_attn_faithfulness/` | re-eval plotted ckpt w/ flash-varlen+bf16 |
| 7 | **Paper-tag code at SMALL (v7) width** — closes the code×capacity 2×2 | ✅ DONE 2026-08-13 — **falling IQR SURVIVES the 14.7× shrink (proxy 0.078→0.050, trend −0.029 vs −0.030 full)** → capacity is orthogonal; trend tracks CODE VERSION | `07_paper_tag_small/` | — (unlocks small-width bisect) |

> **⚡ 2026-07-07 — investigation reframed by task 4.** The fork's committed code is byte-identical
> to `upstream/main`; the real divergence is **temporal**: we train from `HEAD`, which is **100
> commits + a model refactor past the paper's `clic-paper` tag** (`fb90390`). We are not training
> the paper's model. The decisive experiment is now **task 5** (retrain from the tag). Tasks 1 & 3
> are ablations on the *drifted* HEAD model — still informative, but secondary to task 5.

> **🏁 2026-07-20 — VERDICT (task 5).** The paper-tag (fb90390, 12.1M) retrain on our data/cluster
> shows a jet-E IQR that **FALLS with energy (0.084 → 0.065)** — same shape as paper Fig. 4 —
> while every HEAD run rises to ~0.10. **The post-paper model refactor is confirmed as the cause.**
> Overlay: `05_reproduce_paper_tag/jet_iqr_paper_vs_head.png`. Next phase: **bisect** which
> refactor change (**NEW 2026-08-13: SwiGLU→SiLU in every enc/dec FFN, commit `6488c91`, confirmed
> post-tag — the largest single architectural delta and trivially revertible; see
> `07_paper_tag_small/NOTES.md`** vs incidence-head width [256,256]→[512,512] vs norm/decoder-attention rewrite)
> is responsible. Task 6 (train-matched eval) is now optional/low-priority.

> **📌 2026-07-20 (later) — PAPER REPRODUCED EXACTLY + convention discovery.** The paper's Fig. 3/4
> GLOW curves are the **PROXY** output (`network_type: "mpflow_proxy"` in the tag's own
> `notebooks/performance.ipynb`), NOT the regression-refined `mpflow` output this study has been
> plotting. Re-plotting the paper-tag eval as proxy reproduces Fig. 4 **exactly** (median within
> ±0.01, below Pandora; IQR 0.072→0.043 vs paper's ~0.072→0.042) — full setup audit passed
> (dataset, recipe, eval flags, jets, threshold). On the paper tag, the regression head actually
> *degrades* the proxy (+0.03 median, +0.01–0.02 IQR). The refactor verdict is unchanged — HEAD's
> IQR turns up at high E in BOTH conventions (proxy 0.057→0.084; regression 0.075→0.096) — but
> **all future comparisons to paper figures must use `mpflow_proxy`.** See
> `05_reproduce_paper_tag/NOTES.md` + `jet_iqr_proxy_vs_regression.png`.

Legend: ✅ done · 🟡 running/in-progress · ⬜ not started · 🔴 blocked

---

## The core finding so far (from Phase 1 / task 0)

Full write-up: **`00_cross_run/jet_iqr_discrepancy.md`** (§1–§8). One-paragraph recap:

- The IQR climb to ~0.10 at high jet E is **real and reproducible** from stored
  predictions, and **universal across all four v6 runs** (3×L4, 1×B200, 6×L4, 4×B200)
  regardless of hardware / batch size / val_loss. The **plotted** model is the 3×L4 run
  `logs/clic_v6_20260605-T113014`, best ckpt **epoch 195, val_loss 3.788** (lowest of all runs).
- **Ruled out** (do not re-open): `reader.py` indicator threshold, input corruption /
  truth-pred misalignment, the neutral-pt write bug (real bug, immaterial: |ΔIQR|<0.0012),
  and eval/L4 precision handling (eval is already `32-true` full fp32; the trend reproduces
  on native-bf16 B200 hardware where the L4 fix is a no-op).
- **Paper check:** the paper's energy-dependent plot (Fig. 4) baseline is **HGPflow, not
  Pandora**; paper Glow's IQR **falls** with energy. Ours rises → genuine, ~2× discrepancy.
- **Leading lead:** model-vs-paper gap. Paper says **12M** params; every v6 run is **10.1M**.

## How the four new tasks attack the lead

Task 0 showed the trend is model-inherent. Tasks 1–4 probe **what about the model/recipe**:

- **Task 1 (700k v7):** capacity ablation *in the opposite direction* — if a 14× smaller
  model shows the **same** inverted trend, param count / capacity is **not** the driver
  (weakens the 12M-vs-10.1M lead). If it's dramatically worse, capacity matters and the
  10.1M→12M gap is plausible. Either result is informative.
- **Task 2 (loss curves):** did the plotted run actually converge, or is train/val
  divergence (overfitting, plateau, LR-schedule issue) hiding behind a "good" final val_loss?
- **Task 3 (fp32):** isolate precision as a cause — retrain at `32-true` and compare the IQR
  trend against the `bf16-mixed` runs; separately, confirm what precision the **paper** used.
- **Task 4 (fork diff):** the repo is a fork of `samvanstroud/hepattn`. If a local change to
  the model/loss/matcher/config drifted from the paper's code, that could produce the gap
  directly — the cleanest possible explanation.

---

## Key run inventory (quick reference)

| Run | Config | Folder | Params | Best ckpt | Eval outputs? |
|---|---|---|---|---|---|
| 3×L4 **(plotted)** | v6 | `logs/clic_v6_20260605-T113014` | 10.1M | epoch=195 val_loss 3.788 | ✅ `__test.{h5,root}` |
| 1×B200 | v6 | `logs/clic_v6_20260606-T000306` | 10.1M | epoch=199 val_loss 3.836 | ✅ |
| 6×L4 (2 nodes) | v6 | `logs/clic_v6_20260605-T143447` | 10.1M | epoch=199 val_loss 3.922 | ✅ |
| 4×B200 | v6 | `logs/clic_v6_20260612-T102707` | 10.1M | epoch=190 val_loss 3.975 | ✅ |
| v7 2-node **(task 1)** | v7 | `logs/clic_v7_20260706-T181417` | 0.70M | **36450143 ✅ epoch=192 val_loss 4.336** | ✅ `__test.{h5,root}` |
| v7 1-node **(task 1)** | v7 | `logs/clic_v7_20260706-T181418` | 0.70M | 36472892 running (~ep 110) | ⏳ |
| **paper-tag (task 5)** | paper base.yaml | `hepattn-clic-paper/.../logs/clic_paper_20260715-T173417` | 12.1M | epoch=194 val_loss 4.012 (not comparable to HEAD losses) | ✅ `__test.{h5,root}` |
| **paper-tag small (task 7)** | paper base_small.yaml | `hepattn-clic-paper/.../logs/clic_paper_small_20260811-T212339` | 0.82M | 39236741 ✅ epoch=196 val_loss 4.372 (200/200 ep, 20 h 02 m, 3×L4) | ✅ `__test.{h5,root}` (39352893) |

> ⚠️ v7 folder↔job mapping VERIFIED via each folder's `metadata.yaml` slurm_job_id (the two
> `T18141{7,8}` folders can't be told apart by timestamp): T181417=36450143 (2-node, finished),
> T181418=36472892 (1-node, running).

Full details: `../../TRAINING_LOG.md`, `../b200_utilization/training_runs_report.md`, `../../configs/CONFIGS.md`.

**To share with collaborators:** [`REPRODUCE.md`](REPRODUCE.md) — self-contained instructions for
reproducing both result sets (current 10.1M model @ `lgray/hepattn` `1df05cc`, and the paper
reproduction @ tag `clic-paper`/`fb90390`), with the two resolved configs
(`config_v6_baseline.yaml`, `config_paper_tag.yaml`) and the HPG patch for the paper clone.

## Shared conventions

- **Env:** run analysis in the clic pixi env — `pixi run -e clic python <script>`.
- **Plots:** each analysis script writes to a hard-coded `OUT` at its top. Point `OUT` at
  the relevant task folder here (not the old scratchpad) before running.
- **Eval:** `../../submit_eval_run.sh` (parameterized; already carries the `--trainer.num_nodes=1` fix).
- **Never overwrite** original eval `.root`/`.h5`; write new files with a descriptive suffix.
- **Reproduce the IQR curve** with `00_cross_run/compare_runs_iqr.py` (overlay) — the shared tool.
