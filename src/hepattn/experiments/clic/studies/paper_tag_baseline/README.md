# Paper-tag baseline: reproduce previous results and timings after the ports

Two trainings of the paper's model at v7 width (`configs/base_small.yaml`, 819,683 parameters) on
the `clic-paper-main` line, submitted 2026-09-11 at commit 47645e1, right after the environment
(torch 2.9.1 + cu128), the GIL-releasing lap1015, the JV device solver, the mask-loss fix and the
HPG tooling were ported from `main` onto the tag. They answer one question: **does this code +
environment reproduce what the paper-tag code did before, and how fast is it on each hardware?**

## Runs

| # | Hardware | Matcher | Geometry | Job | Pre-flight | Run folder |
|---|---|---|---|---|---|---|
| 1 | 1× B200 | GPU JV (`configs/matcher_jv.yaml`) | batch 2048, 200 ep | **41750147** | 41750146 | `logs/clic_paper_small_b200_jv_<ts>` |
| 2 | 1 node × 3 L4 | host lap1015_late (`configs/matcher_lap1015.yaml`) | batch 170 × 2 accum (global 1020), 200 ep | **41750149** | 41750148 | `logs/clic_paper_small_l4_lap1015_<ts>` |

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
| 1 B200 JV | 41750146 ⏳ | 41750147 ⏳ | | | | | |
| 2 3×L4 lap1015 | 41750148 ⏳ | 41750149 ⏳ | | | | | |
