# Task 5 — Reproduce from the `clic-paper` tag (THE decisive experiment)

**Status:** ✅ DONE 2026-07-20 — **VERDICT: IQR trend REVERTS to falling on the paper tag.
The post-paper model refactor IS the cause of the rising-IQR discrepancy.** See Results below.

## Why this is now the priority
Task 4 proved we are **not training the paper's model**: our `HEAD` is 100 commits + a model
refactor past the paper's `clic-paper` tag (`fb90390`, 2025-08-11). The rising-vs-falling jet-E
IQR is most likely caused by that refactor (leading suspect: incidence-head width [256,256]→
[512,512]; plus norm/decoder-attention rewrite). The one experiment that cleanly settles it:
**train the actual paper code and see if the IQR trend reverts to falling.**

## Setup ACTUALLY DONE (2026-07-07) — separate clean clone (per user: independent, off the fork)
- **Location:** `/home/m.mazza/blue/projects/fastml/hepattn-clic-paper`
  (= `/blue/avery/m.mazza/projects/fastml/hepattn-clic-paper`), a **fresh clone of the ORIGINAL
  `samvanstroud/hepattn`** (NOT the `lgray` fork), checked out at tag **`clic-paper` (fb90390)**.
  Kept fully separate so it can't disturb the fork workflow.
- **Env:** built the paper tag's OWN pixi env from its `pixi.lock` via `pixi install` (torch 2.7.0
  cu126, py3.12, flash-attn 2.7.4 prebuilt wheel). Runs via **direct `pixi run`** (no apptainer),
  so model code + deps match the paper snapshot exactly. `pixi` = `/home/m.mazza/.pixi/bin/pixi`.
- **Config:** patched only `src/.../clic/configs/base.yaml` data paths → our
  `/blue/avery/.../data/clic/{train_clic_fix,val_clic_fix,test_clic_common_infer}.root`
  (same dataset/filenames as the paper's `/share/gpu1/...`), and `batch_size 512 → 170`
  (170×6 L4 = global 1020 ≈ paper's 512×2=1024; L4 can't hold 512/GPU). Everything else is the
  pristine paper recipe (dim 256, enc6/dec4, flash-varlen, bf16-mixed, Lion @8e-5, 200 epochs).
- **Submit:** `src/.../clic/submit_training_hpg_l4_2nodes.sh` — 6×L4 (2 nodes), `COMET_MODE=offline`
  (paper tag uses plain CometLogger; force offline so it doesn't try the internet), `srun pixi run
  python main.py fit ... --trainer.devices=3 --trainer.num_nodes=2`.
- **Param-count control:** `ModelSummary` callback prints total params at startup → verify ≈**12M**
  in the slurm log (positive control that we're running the paper's larger model).

## SUBMITTED 2026-07-07 → job 36526156 (PENDING), 6×L4 (2 nodes)
Env build was non-trivial (documented so it's reproducible):
1. Paper `pixi.lock` was stale for the republished `cuda-version-12.9-3.conda` metapackage
   (hash mismatch). Fixed by `pixi lock` (backup `pixi.lock.orig`). Verified the relock left the
   model stack UNCHANGED: torch 2.7.0/cu126, flash-attn 2.7.4 wheel, python 3.12.11, numpy,
   lightning 2.5.0, numba, awkward, uproot all identical; only peripheral deps + scipy 1.16.0→1.16.2
   (Hungarian matcher output unaffected) drifted.
2. pixi 0.70.1 passes a `pixi-conda-environment` config-setting that hepattn's build backend
   (scikit-build-core) rejects → editable build failed → pixi's all-or-nothing PyPI phase installed
   NOTHING. Workaround: commented out `hepattn = {path=".",editable=true}` in pyproject
   `[tool.pixi.pypi-dependencies]` (backup `pyproject.toml.orig`) so pixi installs torch/flash-attn/
   etc., then built hepattn with **plain pip**: `pixi run python -m pip install -e .
   --ignore-requires-python` (build isolation ON; `--ignore-requires-python` bypasses the paper's
   strict `requires-python == "3.12"` which plain pip reads as 3.12.0). hepattn's compiled part is a
   pure pybind11 `_core` module (no CUDA/torch). Confirmed hepattn survives `pixi install` reconcile.
3. Submit script uses `srun pixi run --frozen` (no network / no lock race across the 6 tasks on
   offline compute nodes). pixi = /home/m.mazza/.pixi/bin/pixi.
- Config confirmed: our data paths, batch 170 (global ~1020), bf16-mixed, flash-varlen, dim 256 —
  the paper recipe (NOT fp32/torch; that's task 3). ModelSummary will print param count at start
  → expect **~12M** (positive control we're running the paper's larger model).
- Output folder: `hepattn-clic-paper/.../logs/clic_v6_<timestamp>` (paper config `name: clic_v6`).

## 🔴 2026-07-14 — job 36526156 NEVER RAN. Two launch bugs found + fixed.
Discovered a week later: 36526156 crashed **seconds after launch** on 2026-07-07T18:18 and then sat
as a **zombie holding 2 L4 nodes for 6d20h** (python died; `srun` wedged instead of exiting, so
Slurm still reported `RUNNING`). **Zero epochs were trained.** Cancelled 2026-07-14.
Lesson: never assume `RUNNING` == progressing — check for ckpts/log growth, not job state.

**Bug 1 — CometLogger offline dir (killed 36526156).**
```
comet_ml.exceptions.InvalidOfflineDirectory: .../logs/clic_v6_20260707-T182139
Reason: [Errno 2] No such file or directory
```
The paper tag uses the **stock** `CometLogger` with `save_dir=<timestamped run dir>`. In offline
mode Comet writes its archive there but **does not create the dir**, and nothing else does either —
the only `mkdir` is in `SaveConfig.on_train_start`, which never runs because the logger is built
first. (Our fork dodges this: its `MyCometLogger` takes an explicit `offline_directory`.)
→ **Fix:** `Path(log_dir_timestamp).mkdir(parents=True, exist_ok=True)` in
`utils/cli.py::before_instantiate_classes`. CLI layer only — model code stays pristine.

**Bug 2 — Triton compile-cache race across ranks (found by the smoke test).**
```
torch._inductor.exc.InductorError: OSError: [Errno 26] Text file busy:
  /home/m.mazza/.triton/cache/<hash>/tmp.pid_.../__triton_launcher.so -> .../__triton_launcher.so
```
The `Compile` callback runs the model through torch.compile → inductor → triton. All 6 ranks share
`$HOME/.triton/cache` on NFS and race on the atomic rename that installs a compiled kernel; a rank
dies and the job then hangs on NCCL. (The fork uses the same `Compile` callback and has simply been
getting lucky — this is a latent race there too.)
→ **Fix:** `run_task.sh`, a per-task launcher srun invokes once per rank, giving each rank a private
`TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` on node-local `/var/tmp` before exec'ing pixi.

**Also ported:** the CSVLogger fix (task 2) into the paper clone's `cli.py`, so this run yields
train+val loss curves too.

**Positive control CONFIRMED (smoke test 37135405):** ModelSummary prints **12.1 M trainable
params** — vs **10.1 M** on our HEAD. This is the paper's larger model, exactly as task 4 predicted.

## ✅ 2026-07-15 — smoke test PASSED, full training SUBMITTED (job 37233919)
The re-queued smoke test (**job 37135830**, `clic-paper-SMOKE`) sat in cluster backfill overnight
and finally **COMPLETED cleanly at 05:47** (exit `0:0`, 4m25s). It confirmed all four gates:
1. **No `InvalidOfflineDirectory`** → bug-1 mkdir fix works (the crash that zombied 36526156).
2. **12,065,291 (12.1 M) trainable params** in ModelSummary → positive control, paper's model.
3. **Checkpoints written** — `logs/clic_paper_smoke_20260715-T054320/ckpts/epoch=00{0,1}-*.ckpt`
   (val_loss 29.35 → 28.70, decreasing).
4. **`csv_metrics/metrics.csv` populated** → the CSVLogger fix works in the paper clone too.
Both launch bugs are dead; no NCCL hang, no Triton race (run_task.sh per-rank cache works).

**Full run submitted 2026-07-15 T16:10 → job 37233919** (`clic-paper-l4-2nodes`, 2 nodes × 3 L4,
7-day limit, hpg-turin, PENDING). Launch path is identical to the smoke test —
`submit_training_hpg_l4_2nodes.sh` → `srun ./run_task.sh` → `pixi run --frozen` — minus the
epoch/batch caps. Added **`--name clic_paper`** to the python cmd so the output folder is
`hepattn-clic-paper/.../logs/clic_paper_<timestamp>` (the config default `clic_v6` would collide by
name with the fork's drifted-HEAD runs). Recipe unchanged: base.yaml 200 epochs, batch 170 (×6 =
global 1020 ≈ paper 1024), bf16-mixed, flash-varlen, dim 256, Lion @8e-5.
**Watch:** early steps for OOM/NCCL, then that ckpts + csv_metrics actually grow (don't trust
`RUNNING` alone — that was the 36526156 zombie lesson). Eval + IQR overlay when it finishes.

## ✅ 2026-07-16 — TRAINING COMPLETE (job 37233919), eval submitted (job 37328641)
Full run **COMPLETED** exit `0:0`, ran 19h06m (start 07-15 T17:32 → end 07-16 T12:38), 6×L4.
- **Rank-0 output folder:** `hepattn-clic-paper/.../logs/clic_paper_20260715-T173417` (200 ckpts).
  ⚠️ DDP wrote 4 timestamped folders T17341{1,4,7,8}; only **T173417 is rank-0** with real ckpts +
  config.yaml + csv_metrics; the other three are empty rank stubs — ignore them.
- **Best ckpt: `epoch=194-val_loss=4.01237`** (last epochs 198/199 = 4.0152/4.0138, flat plateau).
- **Training healthy — no overfit:** csv_metrics final epoch 199 train/loss **3.956** ≈ val/loss
  **4.014**. 8000 rows logged (CSVLogger fix works).
- ⚠️ **val_loss NOT comparable across code versions:** paper-tag best 4.012 vs our HEAD 3.788 does
  NOT mean the paper model is "worse" — the loss composition/normalization differs between the
  refactored HEAD and the paper tag. The ONLY meaningful comparison is the **jet-E IQR trend**,
  which needs the eval below. Do not read anything into the raw 4.01-vs-3.79 gap.
- **Eval submitted 2026-07-16 → job 37328641** (PENDING, 1×L4, 4h). New tooling created in the paper
  clone (didn't exist): `configs/eval.yaml` (copy of fork's: 32-true, attn_type torch, is_inference
  true, PflowPredictionWriter) + `submit_eval_paper.sh` (pixi run --frozen, NOT apptainer). Produces
  `logs/clic_paper_20260715-T173417/ckpts/epoch=194-val_loss=4.01237__test.{h5,root}`.
- **NEXT (decisive):** once the root lands, overlay its jet-E IQR vs the HEAD runs + Pandora with
  `../plot_new_evals_iqr.py` (add the paper root to RUNS). If IQR **falls** with E → confirmed: the
  post-paper refactor is the cause. If it still **rises** → model code is not the cause.

## After training
Eval with the SAME pipeline and overlay IQR vs the HEAD runs + paper Fig. 4. Interpretation below.

## Interpretation
- IQR reverts to **falling** with energy → confirmed: the discrepancy is the post-paper refactor.
  Next: bisect which change (incidence width vs norm/decoder) by toggling on `clic-paper`.
- IQR still **rises** → the model code is not the cause; look upstream (data/targets/recipe),
  though those were largely cleared in Phase 1.

## Open questions / watch-outs
- The paper-tag code may need an older env than the current `pixi.sif` (100 commits of dep drift).
  Check whether the paper-tag repo builds/runs in our container; may need a matching env.
- `flash-varlen` attention on the paper tag — same GPU/precision caveats as HEAD.
- Decide compute: this likely wants the same 6×L4 (2 nodes) slot — may compete with the fp32 job
  (36518920) and the running v7 jobs for nodes.

## Checklist
- [x] clone on `clic-paper` tag created (separate `hepattn-clic-paper`, not a worktree)
- [x] data paths patched; config/env confirmed to run (env build documented above)
- [x] param count verified ≈12M (12,065,291 — smoke tests 37135405 + 37135830)
- [x] training submitted → 200 epochs (**job 37233919**, submitted 2026-07-15 T16:10, COMPLETED)
- [x] eval'd → IQR overlay vs HEAD runs (`plot_paper_iqr.py` → `jet_iqr_paper_vs_head.png`)
- [x] verdict: **trend reverts — YES.** Next: bisect which refactor change is responsible.

## ✅ Results (2026-07-20) — the trend REVERTS on the paper tag

Eval (job 37328641, COMPLETED 2026-07-16, 10m46s) produced
`logs/clic_paper_20260715-T173417/ckpts/epoch=194-val_loss=4.01237__test.root` (19,722 events in
the common intersection, same as all fork evals; the paper tag writes old flat-TTree branches
`mpflow_pt` vs the fork's `mpflow.pt`, but `performance/reader.py` already handles both).

Overlay: **`plot_paper_iqr.py`** → **`jet_iqr_paper_vs_head.png`** (paper-tag vs 4 HEAD runs +
Pandora, same pipeline/binning as `../plot_new_evals_iqr.py`).

Jet-E response IQR per truth-E bin (bin centers in GeV):

| run | E10 | E30 | E50 | E70 | E90 | E110 | E130 | E150 | E170 | E190 | trend |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **paper-tag 12.1M** | .084 | .075 | .076 | .077 | .072 | .076 | .069 | .070 | .068 | **.065** | **FALLS** ✅ |
| v6 bf16 baseline | .075 | .067 | .070 | .077 | .077 | .085 | .077 | .083 | .099 | .096 | rises |
| v6 fp32 | .078 | .069 | .074 | .078 | .077 | .087 | .083 | .085 | .107 | .105 | rises |
| v7 700k 2-node | .082 | .072 | .076 | .086 | .087 | .097 | .090 | .095 | .099 | .101 | rises |
| v7 700k 1-node | .078 | .071 | .076 | .084 | .089 | .096 | .091 | .098 | .102 | .104 | rises |
| Pandora | .091 | .076 | .068 | .064 | .061 | .063 | .059 | .060 | .056 | .054 | falls |

- The paper-tag IQR **monotonically improves with E** (0.084 → 0.065), the same qualitative shape
  as the paper's Fig. 4 Glow curve and the same direction as Pandora — while every HEAD run climbs
  to ~0.10 at high E. At E190 the paper tag is ~35% better than the best HEAD run.
- The paper tag doesn't fully reach Pandora at high E (0.065 vs 0.054) — consistent with the paper,
  where Glow's high-E IQR is comparable-ish to its baselines, and possibly limited by our smaller
  global batch (1020 vs 1024) / L4 hardware. The *trend* is the decisive observable, and it reverts.
- Side observation: the paper-tag **median** response sits ~+0.03 (a slightly larger positive bias
  than the HEAD runs, which are nearer 0–0.01 at mid-E). Median bias was never the discrepancy
  (both are small); noted for completeness.

**Conclusion:** tasks 1 (capacity), 2 (training health), 3 (precision) all ruled out; task 5 shows
the paper's code at tag `fb90390` reproduces the paper's falling trend on OUR data, OUR cluster,
OUR eval pipeline. **The regression lives in the ~100 commits of post-paper refactor** (leading
suspects from task 4: incidence-head width [256,256]→[512,512] Dense int-expansion #212, and the
norm/decoder-attention rewrite).

**Next step — bisect:** toggle individual refactor changes on top of `clic-paper` (or revert them
on HEAD) and re-train. Start with the incidence-head width, then the decoder/norm refactor.

## ✅✅ 2026-07-20 (later) — PAPER Fig. 4 REPRODUCED EXACTLY; convention discovery (PROXY!)

Prompted by: our paper-tag median sat ABOVE Pandora (+0.03) while in paper Fig. 4 GLOW's median
is below Pandora (~0 vs +0.02), and paper GLOW's IQR is below Pandora everywhere. Careful read of
the paper (arXiv:2508.20092, 6 pp) + audit of the `clic-paper` tag revealed:

**The paper plots GLOW's PROXY output, not the regression output.** The tag's own
`notebooks/performance.ipynb` (the paper-plot notebook) loads GLOW with
`network_type: "mpflow_proxy"`, `ind_threshold: 0.65` → `reader.py return_proxy` returns
`proxy_{pt,eta,phi}` — the incidence-weighted sums — NOT the regression-refined `mpflow_*`
branches this study has plotted everywhere. Both branch sets are in every eval root, so this is
re-plottable with no re-eval: **`plot_paper_iqr_proxy.py`** → **`jet_iqr_proxy_vs_regression.png`**.

Result (jet-E response, same binning):

| curve | median | IQR E10→E190 | vs paper Fig. 4 |
|---|---|---|---|
| **paper-tag PROXY** | −0.006…+0.010 (below Pandora) | **0.072 → 0.043** | **≈ EXACT match** (paper ~0.072→0.042, median within ±0.01) |
| paper-tag regression | +0.008…+0.030 (above Pandora) | 0.084 → 0.065 | what we plotted before — NOT what the paper shows |
| v6 HEAD PROXY | −0.006…−0.028 (negative bias at high E) | 0.073 → 0.057@E70 → **0.084@E170** | still rises at high E → refactor damage visible in proxy too |
| v6 HEAD regression | ~0 | 0.075 → 0.096 | the study's original rising curve |
| Pandora | +0.02 peak | 0.091 → 0.054 | ≈ exact match to paper's Pandora (pipeline control) |

**Conclusions:**
1. **We DO reproduce the paper, exactly**, with the paper's code (tag fb90390) + paper's plotting
   convention (proxy). Full setup audit passed: dataset (1,004,891 train / 20k test events),
   recipe (200 ep, global batch 1020≈1024, bf16-mixed, flash-varlen, 12.1M), eval per tag README
   (`test_clic_common_infer`, `is_inference true`, `32-true`, torch attn), jets genkt R=0.7,
   ≤2 leading jets pT>10, ΔR<0.1, `ind_threshold` 0.65, same bins.
2. **On the paper tag the regression head makes kinematics WORSE than its proxy input**
   (+0.03 median bias, IQR +0.01–0.02). The paper text implies the regression refines the proxy,
   but the paper's own plots use the proxy. (Note: paper's plotted ckpt was
   `CLIC_Pflow_FullDiceFocFix_20250613` epoch=159 val_loss=3.517 — predates the tag by 2 months —
   but since our tag retrain reproduces Fig. 4 exactly, this is moot.)
3. **The task-5 verdict STANDS, in both conventions:** HEAD's IQR turns UP at high E
   (proxy: 0.057→0.084; regression: 0.075→0.096) while paper-tag falls monotonically
   (0.072→0.043). The post-paper refactor is still the cause of the inverted trend.
4. **Study-wide action item:** all comparisons against paper figures must use
   `network_type: "mpflow_proxy"`; earlier apples-to-oranges (our regression vs paper proxy)
   exaggerated the absolute IQR gap and produced the "median above Pandora" confusion.

---

**Follow-up:** the capacity ablation on this tag (paper-tag code at v7 width) is **task 7** —
see `../07_paper_tag_small/NOTES.md`. Result: the falling IQR SURVIVES the 14.7x shrink,
closing the capacity hypothesis from both sides.
