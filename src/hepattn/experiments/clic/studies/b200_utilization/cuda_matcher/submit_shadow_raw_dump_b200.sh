#!/bin/bash
# Three steps of the converged v6 model, dumping the shadow comparison's PER-PAIR arrays.
#
# THE QUESTION. Job 40640611 measured, over epochs 150-199, that the device and host mask-IoU
# distributions are bitwise identical at every stored quantile (q0.01, q0.5, q0.99 equal at 100%
# of steps) and that the per-step mean agrees to ~1e-9 -- while 66% of individual swapped pairs
# move by more than 0.01. With independent signs the per-step mean would sit at ~2.4e-2, so this
# is seven orders of magnitude too exact to be cancellation by chance. The natural reading is
# that both solvers return the SAME multiset of IoU values attached to different particles, but
# that is a claim about the JOINT distribution and the .jsonl stores only five quantiles per
# step, so it cannot be checked from what was recorded. This run records the per-pair arrays
# instead. It is a correctness check on a plotted claim, not a new measurement.
#
# WHAT IT RUNS. `fit --ckpt_path <epoch 198>` resumes at epoch 199 with max_epochs 200 from the
# base config, so exactly one epoch is scheduled; --trainer.limit_train_batches=3 cuts that to
# three optimiser steps and --trainer.limit_val_batches=0 skips validation. The overlay config
# drops the Checkpoint and PflowPredictionWriter callbacks, so this writes nothing into the
# original run's directory -- it gets its own logs/clic_v6_cudamatch_shadow_raw_<timestamp>/.
# Most of the wall time is torch.compile warmup; the steps themselves are seconds.
#
# OUTPUT. matcher_shadow_raw_step{0,1,2}.npz, each holding, for every disagreeing (problem,
# target) pair: the problem and target index, the query each solver chose, the mask IoU each
# choice yields, and the two cost entries. ~300 pairs a step, so kilobytes. Three steps rather
# than one so a coincidence cannot pass as a structural result.

#SBATCH --job-name=clic-shadow-raw-dump-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

SIF=/blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif
PIXI_ENV=/blue/avery/m.mazza/projects/fastml/hepattn/.pixi/envs/clic

# The -default tree, ABI-bound to the pixi env main.py runs under. See submit_shadow_matcher_b200.sh.
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-default
export APPTAINERENV_PYTHONPATH=$VENDOR
# JV kernel block size: nothing exported. Since 2026-09-08 the production tree's SMPCores()
# carries a compute-capability major-10 case returning 32, so a B200 gets the measured
# geometry by construction and an L4 keeps 128 (NOTES.md, 2026-09-08). TLA_BLOCK_SIZE
# survives in the binary as an override for sweeps only; do not set it here.
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

CACHE="/var/tmp/clic_shadow_raw_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

CKPT="logs/clic_v6_cudamatch_shadow_20260830-T121009/ckpts/epoch=198-val_loss=3.44082.ckpt"

echo "started: $(date -Is)"

# The explicit `-e` is load-bearing: a bare `pixi run` inherits PIXI_ENVIRONMENT_NAME from the
# submitting shell. This used to say `-e default` because the clic env lacked flash-attn; clic
# now carries the same torch and flash-attn, and is the only env installed.
srun apptainer run --nv --bind /blue/,/cmsuf/ "$SIF" \
  pixi run -e clic python main.py fit \
    --config configs/clic_v6_cudamatch.yaml \
    --config configs/shadow_raw_dump.yaml \
    --trainer.devices=1 \
    --trainer.limit_train_batches=3 \
    --trainer.limit_val_batches=0 \
    --ckpt_path "$CKPT"

echo "finished: $(date -Is)"
echo "Done!"
