# Task 2 — Loss curves of the plotted 3×L4 run

**Status:** ✅ DONE (2026-07-14) — curves plotted for all 7 runs; verdict below.

## Objective
Plot train + validation loss vs epoch for the **plotted** model
(`logs/clic_v6_20260605-T113014`, 3×L4, the run behind the disputed IQR figure) to check
convergence — plateau, overfitting (train↓ while val↑), or LR-schedule pathology that a
single "final val_loss" number would hide.

## Why train-loss is missing but val-loss survives (definitive, checked 2026-07-07)
Two independent mechanisms:
- **val/loss survives** because the `Checkpoint` callback (`monitor: val/loss`) bakes the
  monitored value into every checkpoint **filename** (`epoch=N-val_loss=X.ckpt`) and into the
  ckpt file — filesystem-persisted, logger-independent. Recoverable for all runs.
- **train-loss goes ONLY to the logger** (`MyCometLogger`). On HPG compute nodes there is no
  internet and no `COMET_API_KEY` (submit scripts set `COMET_EXPERIMENT_KEY` but never `export`
  it and never set an API key), so `MyCometLogger` auto-falls back to **offline mode**
  (`loggers.py`: `online=False` when no key). The offline experiment IS created (the progress
  bar shows `v_num=<key>`), but **no offline archive lands on disk** — the run folders contain
  only ckpts/config/metadata/times, no `.zip`, no `metrics.csv`. Likely cause: offline comet's
  working dir goes under `TMPDIR=/var/tmp/` (set in the submit scripts), which is container-
  ephemeral and lost at job end. Net: **Comet is effectively off on HPG, and train-loss is not
  persisted anywhere.** So val-loss = free from ckpt names; train-loss = currently unrecoverable
  for past runs (v6 + the running v7 jobs).

## ✅ FIX LANDED 2026-07-07 — CSVLogger added in cli.py
Implemented in `src/hepattn/utils/cli.py` `after_instantiate_classes` (fit branch): appends a
`lightning.pytorch.loggers.CSVLogger(save_dir=default_root_dir, name="csv_metrics", version="")`
to `self.trainer.loggers`. Done POST-instantiation so it doesn't disturb the single-logger
`name`/`offline_directory` parser wiring (link_arguments line 62 / injection line 86) — no other
experiment breaks. Validated standalone: writes `<run_dir>/csv_metrics/metrics.csv` with columns
`epoch,step,train/loss,val/loss,...` (wrapper.py logs `self.log(f"{stage}/loss")` for train+val).
Train loss is per-step, val per-epoch — both carry an `epoch` column. Since fork jobs run live
code from /blue (container binds it), the **already-queued fp32 job 36518920 will pick this up**
when it starts. NOTE: the paper-tag clone has its own cli.py + plain CometLogger — apply the same
fix there separately if train-loss is wanted for that run (val-loss from ckpt names covers it).

## Fix for FUTURE runs (answering "can this be fixed?")
The clean fix is a **`CSVLogger`** (writes `metrics.csv` with train+val loss to disk, no
internet). BUT the repo's custom `utils/cli.py` hardcodes a **single** logger — it does
`link_arguments("name", "trainer.logger.init_args.name")` (line 62) and force-injects
`offline_directory` into `trainer.logger` (line 86). So `trainer.logger` cannot be made a list
or swapped to `CSVLogger` (no `offline_directory` arg) without a small `cli.py` change that:
(a) guards the `name` link / offline_directory injection to loggers that accept them, or
(b) iterates when `logger` is a list. This is a shared-infra change → make + test locally
(jsonargparse parse) before relying on it. Until then, val-loss from ckpt names remains the
fallback for every run. **The fp32 run (task 3) currently keeps the single MyCometLogger** to
avoid risking the run — its train-loss will also be missing unless we land the cli.py fix first.

## What is available (checked 2026-07-07)
- The run dir has **NO local metrics files** (no CSV / tfevents / comet-offline) — only
  `ckpts/`, `config.yaml`, `metadata.yaml`, `times/`.
- Logger is `hepattn.utils.loggers.MyCometLogger` → metrics went to **Comet online**,
  project **`hepattn-clic`**.
- **Validation loss IS recoverable locally for free:** every checkpoint is named
  `epoch=<N>-val_loss=<X>.ckpt` for epoch 0–199. Parse the 200 filenames → full val-loss
  curve, no Comet needed. (Sanity: epoch=100 → 4.06470, epoch=199 → 3.79233, best epoch=195 → 3.788.)

## Method
1. **Val-loss curve (do first, zero dependencies):** glob `ckpts/epoch=*-val_loss=*.ckpt`,
   regex out (epoch, val_loss), sort, plot. Mark the best (epoch 195). Output PNG here.
2. **Train-loss curve (needs Comet):** pull from Comet project `hepattn-clic` — via the
   Comet web UI export, or `comet-ml` Python API with an API key + the experiment key for
   this run. Overlay train vs val on one axis. If Comet is inaccessible, val-only is a
   usable deliverable; note the gap.
3. Optionally do the same for the other v6 runs for comparison (all have named ckpts).

## Checklist
- [x] val-loss-vs-epoch plotted from ckpt names
- [x] train-loss: recovered for the fp32 run via the CSVLogger fix; documented as permanently
      unavailable for the 6 runs that predate the fix (Comet offline archive never persisted)
- [x] train vs val overlay (fp32 run — the only run with train-loss)
- [x] convergence verdict

## Tooling
**`plot_loss_curves.py`** (here). Reads `<run>/csv_metrics/metrics.csv` when present (train every
50 steps + val per epoch), else falls back to parsing `ckpts/epoch=<N>-val_loss=<X>.ckpt` names
(val only). Covers all 7 runs; add new runs to the `RUNS` dict at the top. Outputs:
- `loss_curves_per_run.png` — per-run panel, train (step trace + epoch mean) and val, log-y.
- `loss_curves_val_overlay.png` — val-loss overlay, full range + converged-tail zoom.

Run: `pixi run -e clic python studies/glow_jet_iqr/02_loss_curves/plot_loss_curves.py`

## Results (2026-07-14)

| Run | source | best val | @ epoch |
|---|---|---|---|
| v6 3×L4 **(PLOTTED)** | ckpt | **3.78800** | 195 |
| v6 6×L4 (2 nodes) | ckpt | 3.92173 | 199 |
| v6 1×B200 | ckpt | 3.83626 | 199 |
| v6 4×B200 | ckpt | 3.97507 | 190 |
| **v6 fp32 6×L4** | **csv** | **3.74189** | 197 |
| v7 700k 2-node | ckpt | 4.33644 | 192 |
| v7 700k 1-node | ckpt | 4.25845 | 197 |

**Verdict: training is healthy — the IQR discrepancy is NOT a training pathology.**
- **No overfitting anywhere.** In the fp32 run (the only one with train-loss) the train and val
  curves lie essentially on top of each other for all 200 epochs — no divergence, no gap opening.
- **No plateau and no LR-schedule pathology.** Every run descends smoothly and monotonically.
- **All runs are still (slowly) improving at epoch 200** — best val lands at epoch 190–199 in
  every single run, never earlier. So the models are marginally *under*-trained rather than
  over-trained. Gains are small (~0.01–0.02/10 epochs at the end), so this is very unlikely to
  explain a 2× IQR trend inversion, but it does mean "200 epochs" is not a converged asymptote.
- **fp32 gives the best val-loss of all runs (3.742 < 3.788 bf16).** Real but small. Whether it
  changes the *IQR trend* is task 3 — pending its eval.

This closes the original question ("did the plotted run actually converge, or is a train/val
divergence hiding behind a good final val_loss?"): it converged cleanly, no divergence.
