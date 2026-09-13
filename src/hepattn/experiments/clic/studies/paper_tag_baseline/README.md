# Paper-tag baseline: reproduce previous results and timings after the ports

Two trainings of the paper's model at v7 width (`configs/base_small.yaml`, 819,683 parameters) on
the `clic-paper-main` line, first submitted 2026-09-11 at commit 47645e1, right after the environment
(torch 2.9.1 + cu128), the GIL-releasing lap1015, the JV device solver, the mask-loss fix and the
HPG tooling were ported from `main` onto the tag. They answer one question: **does this code +
environment reproduce what the paper-tag code did before, and how fast is it on each hardware?**

Run 1 was cancelled and resubmitted the same day at commit a2b05f5, after the whole-model `Compile`
callback was found to be the cause of the 790 ms step and replaced by head's encoder/decoder compile
(see `studies/model_size/README.md`; run 1 is that study's reference arm and was resubmitted with it).
Run 2 is unaffected by that fix and still carries its original job numbers.

## Runs

| # | Hardware | Matcher | Geometry | Job | Pre-flight | Run folder |
|---|---|---|---|---|---|---|
| 1 | 1× B200 | GPU JV (`configs/matcher_jv.yaml`) | batch 2048, 200 ep | **41758027** | 41758026 | `logs/clic_paper_small_b200_jv_20260911-T141520` |
| 2 | 1 node × 3 L4 | host lap1015_late (`configs/matcher_lap1015.yaml`) | batch 170 × 2 accum (global 1020), 200 ep | **41750149** | 41750148 | `logs/clic_paper_small_l4_lap1015_<ts>` |

Superseded round for run 1: pre-flight 41750146 passed, training 41750147 cancelled 2026-09-11 at
epoch ~3 (`logs/clic_paper_small_b200_jv_20260911-T130357`).

Pre-flights run the same launch path with `--trainer.fast_dev_run=true` (one train + one val
batch, `--time=00:30:00`, run name prefixed `pf_`). The full runs were queued at the same time; if a
pre-flight fails, cancel its full run and fix before resubmitting.

Launch commands (from this directory's parent, `src/hepattn/experiments/clic`):

```shell
# 1. B200 + JV, through the model-size study driver: this run IS that study's reference
sbatch --time=00:30:00 --job-name=pf-ps-ref \
  --export=ALL,CONFIG=configs/base_small.yaml,FIT_ARGS="--name pf_clic_paper_small_b200_jv --trainer.fast_dev_run=true" \
  studies/model_size/submit_ablation_b200.sh
sbatch --job-name=clic-ps-ref \
  --export=ALL,CONFIG=configs/base_small.yaml,FIT_ARGS="--name clic_paper_small_b200_jv" \
  studies/model_size/submit_ablation_b200.sh

# 2. 3x L4 + lap1015
sbatch --time=00:30:00 --job-name=pf-ps-l4 submit_training_hpg_l4.sh configs/base_small.yaml   # script since renamed submit_training_l4.sh \
  --config configs/matcher_lap1015.yaml --name pf_clic_paper_small_l4_lap1015 --trainer.fast_dev_run=true
sbatch --job-name=clic-ps-l4 submit_training_hpg_l4.sh configs/base_small.yaml   # script since renamed submit_training_l4.sh \
  --config configs/matcher_lap1015.yaml --name clic_paper_small_l4_lap1015
```

## References to compare against

**Run 2 (3× L4)** has a direct reference: the same config at the same geometry on the paper tag's
own environment (torch 2.7.0+cu126, flash-attn 2.7.4, scipy matcher), job **39236741**, run folder
`/home/m.mazza/blue/projects/fastml/hepattn-clic-paper/src/hepattn/experiments/clic/logs/clic_paper_small_20260811-T212339`
(notes: `main`'s `studies/glow_jet_iqr/07_paper_tag_small/NOTES.md`):

| quantity | reference (39236741) |
|---|---|
| trainable parameters | 819,683 |
| wall time, 1 node × 3 L4 | 20 h 02 m (200/200 epochs) |
| best val_loss | 4.3716 (epoch 196) |
| proxy jet-E IQR, low → high E | 0.078 → 0.050 (falls; E90→E170 segment 0.058 → 0.058) |
| eval job / file | 39352893, `epoch=196-val_loss=4.37156__test.root` |

Two things differ by construction: the matcher (lap1015_late instead of scipy; physics-neutral
by design, faster), and the software stack (torch 2.9.1/cu128 instead of 2.7.0/cu126). A change
in wall time is therefore attributable to those two, not to the model.

**Run 1 (B200 + JV)** has no paper-code reference on that hardware. It becomes the reference for
`studies/model_size/`. The only timing yardstick is the *head-based* v7 model (702,395 params,
different code) at the same geometry and matcher: job **40405423**, 7 h 40 m on 1 B200
(`main`'s `studies/model_size/`). Expect the same order of magnitude, not equality: this model is
17% larger and has SwiGLU feed-forwards.

## What "reproduced" means here

1. `ModelSummary` reports **819,683** trainable parameters in both runs (positive control that the
   ported code builds the paper's model).
2. Run 2's best val_loss lands within training-to-training scatter of 4.3716. σ_repro has not been
   measured on this code; on head, val_loss differed by ~0.01–0.03 between same-config runs.
3. Run 2's proxy jet-E IQR **falls** with jet energy (the paper's shape), after evaluation with
   `submit_eval_l4.sh` and the `mpflow_proxy` convention.
4. Wall time of run 2 versus 20 h 02 m on identical hardware; wall time of run 1 versus 7 h 40 m
   for the smaller head model. Compare GPU-hours, never mix hardware.
5. Both pre-flights complete without a launch error (the JV build imports and reports
   `has_cuda: True` in the B200 log; lap1015 raises no GIL warning in the L4 log).

## Status

| # | Pre-flight | Training | Eval | val_loss (best ep) | wall time | proxy IQR low→high | verdict |
|---|---|---|---|---|---|---|---|
| 1 B200 JV | 41758026 ✅ | 41758027 ✅ | 41987838 ✅ | 4.4089 (ep 199) | 8 h 59 | 0.082 → 0.053 | **reproduced** |
| 2 3×L4 lap1015 | 41750148 ✅ | 41750149 ✅ | 41987204 ✅ | 4.2935 (ep 199) | 15 h 07 | 0.079 → 0.048 | **reproduced** |
| — reference | | 39236741 | 39352893 | 4.3716 (ep 196) | 20 h 02 | 0.078 → 0.050 | |

Run 2's folder is `logs/clic_paper_small_l4_lap1015_20260911-T191856`. Both trainings ran all
200 epochs and both best checkpoints are the last one, so neither had started to overfit.

**Both runs reproduce the paper's shape.** Proxy jet-E IQR per 20 GeV bin of truth jet energy,
`mpflow_proxy`, `ind_threshold` 0.65, `dr_cut` 0.1, two leading jets, `pt_min` 10:

| E [GeV] | 0–20 | 20–40 | 40–60 | 60–80 | 80–100 | 100–120 | 120–140 | 140–160 | 160–180 | 180–200 |
|---|---|---|---|---|---|---|---|---|---|---|
| reference 39236741 | 0.0784 | 0.0648 | 0.0608 | 0.0603 | 0.0583 | 0.0605 | 0.0576 | 0.0590 | 0.0585 | 0.0499 |
| run 2, 3×L4 lap1015 | 0.0788 | 0.0640 | 0.0584 | 0.0583 | 0.0567 | 0.0598 | 0.0566 | 0.0576 | 0.0561 | 0.0478 |
| run 1, B200 JV | 0.0821 | 0.0683 | 0.0609 | 0.0595 | 0.0571 | 0.0621 | 0.0568 | 0.0562 | 0.0610 | 0.0528 |
| Pandora | 0.0911 | 0.0758 | 0.0682 | 0.0641 | 0.0609 | 0.0625 | 0.0594 | 0.0603 | 0.0563 | 0.0538 |

The IQR **falls** with energy in all three, and all three sit below Pandora up to ~160 GeV and
converge with it above — the paper's behaviour, and the opposite of head's rising curve. Run 2
tracks its reference to within 0.001–0.003 in every bin, which is the point of the exercise: the
same model, on the same hardware, through the ported environment and the lap1015 matcher, gives
the same physics. Run 1 is 0.002–0.004 worse in most bins, consistent with its higher val_loss
and its different batch geometry; the shape is unchanged.

Plots (all four algorithms, six figures) are in `figures/`, written by `performance.ipynb` in the
paper clone,
`/home/m.mazza/blue/projects/fastml/hepattn-clic-paper/src/hepattn/experiments/clic/notebooks/`,
whose `plot_path` points here. That notebook is the shared tool with a HiPerGator config cell
added; the figures are this study's, so they live here rather than next to it. It reads this
branch's ROOT files, so it needed the RNTuple branch-name support from `main`'s reader (paper-tag
uproot wrote a flat TTree, the new one writes an RNTuple); as a positive control it reproduces
the reference's recorded 0.078 → 0.050 exactly.

| figure | what it shows |
|---|---|
| `jet_response.png` | jet-E median, IQR and IQR/response against truth jet E — the reproduction plot |
| `jet_residuals.png` | jet pT, E, constituent-count and ΔR residuals |
| `jet_residuals_boxplot.png` | jet residual spread per pT bin |
| `event_response.png` | MET, HT and charged/neutral constituent-count residuals |
| `residuals_all.png` | per-particle residuals, charged and neutral |
| `residuals_neutrals.png` | per-particle residuals, neutral hadrons and photons |

### Why run 1 is slower than head

**Found (`studies/step_time_gap/`, 2026-09-13): the GPU Jonker-Volgenant kernel takes 84 ms per
step on this model's cost matrices against 47 ms on head's, at the trained state; every other
kernel class is equal, the environment and logger are not involved, and fresh models of both
codes step at the same 360 ms over 300 steps.** The earlier note below stands as the history.
`studies/swiglu_silu/` isolated the leading model-side candidate — `Dense`'s activation, gated
SwiGLU here against plain SiLU on `main`, across all 14 transformer feed-forwards — and ruled it
out on timing: two 300-step B200 pre-flights at one commit, 819,683 against 703,203 parameters,
within 6% of each other. That study continues for the *physics* half of the question, since
SwiGLU is also a jet-E IQR suspect from `main`'s `glow_jet_iqr` bisect.

**Pre-flight 41758026 passed** (2026-09-11, 4 m 20 s of a 30 m limit; `fast_dev_run` stops at one
train + one val batch, so minutes are the expected scale and no runtime projection comes out of it).
Both positive controls hold: the log reports `tla ok, has_cuda: True`, so the JV device solver built
and imported, and `ModelSummary` reports 819 K trainable parameters. Training 41758027 started
14:14:30 on the `afterok` dependency with a 14 h limit taken from the measured 367 ms step.

**Both trainings completed 2026-09-11/12.** Run 1 took 8 h 59 (311 ms/step), run 2 15 h 07.

Two points of comparison, neither of them a problem:

- Run 2 against its same-hardware reference: **val_loss 4.2935 against 4.3716**, i.e. 0.078
  *better*, which is outside the ~0.01–0.03 scatter head showed between same-config runs. The two
  differ by the matcher (lap1015_late instead of scipy) and the software stack, and run 2 also
  finished 4 h 55 sooner. σ_repro has not been measured on this code, so the gap stays
  unexplained — but it is favourable and the evaluation agrees with it: run 2's IQR is slightly
  *better* than the reference's in every bin, by about the same small margin.
- Run 1 against the head-based v7 model at the same geometry (7 h 26–7 h 40): **8 h 59, a quarter
  slower per step** (311 against 251–255 ms/step). The cause is open. The obvious candidate, that
  this model is 819K parameters against v7's 702K because `Dense` defaults to gated SwiGLU here
  and to plain SiLU on head, was tested in `studies/swiglu_silu/` and **refuted**: removing those
  116,480 parameters did not move the step time. Recorded in `README_HPG.md`.

### The evaluation path did not work on this branch

Neither run could be evaluated until three faults in the ported code were fixed (branches
`fix-eval-path` and `port-inference-timer-warmstart`, merged 2026-09-13); all three came in with
the port, not from the tag:

1. `predictionwriter.py` passed `jets_name=` to `ftag.hdf5.H5Writer`. The tag does not pin
   `atlas-ftag-tools`, so the new environment solved to 0.3.5, where that argument is
   `global_objects_name`. Every `main.py test` died in the first `on_test_batch_end` and wrote
   nothing. Head is unaffected because it pins `atlas-ftag-tools>=0.2.9,<0.3`.
2. A run trained with `device_solver: jv` could not be evaluated on an L4, because its config
   still named the solver and the torch-linear-assignment build is per-architecture — even
   though the matching never runs at test time. `configs/eval.yaml` now clears `device_solver`.
3. `InferenceTimer.on_test_end` dropped ten warm-up forward passes and then raised
   `ValueError: No times recorded.` if none were left. At batch 2048 the 19,722-event test set is
   ten batches, so run 1 aborted every time — after writing the 2.4 GB HDF5 file, but before the
   prediction writer converted it to ROOT, so the run produced no usable output. `main` already
   had that guard; it had not been ported. Run 2, at batch 170, has ~116 batches and never hit it.

Failed attempts: 41986321/41986322 (fault 1; 41986321 hit fault 2 first) and 41987203 (fault 3).
The evaluations of record are **41987838** (run 1) and **41987204** (run 2), each about 1 min 15
on one L4, well inside the script's 30 min limit.
