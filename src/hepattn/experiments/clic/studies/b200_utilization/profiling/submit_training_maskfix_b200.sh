#!/bin/bash

# Full 200-epoch CLIC training on 1x B200 with the mask-fixed loss.
#
# Purpose: test the *extrapolated* claim that the fixes cut a B200 training from 68 h 18 m to
# ~37 h. That figure came from short-run throughput (1497 samples/s median), not from a real
# training, and B200 node-to-node variance spans 1.7x -- so the honest prediction is
# **29-50 h, centred on ~37 h**. This run measures it.
#
# Geometry copied exactly from the June 2026 B200 production run (job 33954039: 1 node,
# gpu:b200:1, 1 task x 16 CPU, 60 GB, batch 2048, 200 epochs, bf16-mixed), which took
# 2-20:16:35. batch_size comes from the config (2048), matching June -- do NOT override it,
# that is the whole point of the comparison.
#
# NOTE the resulting model is NOT directly comparable to the 3x L4 mask-fixed run
# (job 38469247): global batch is 2048 here vs 768 there, and the learning rate is not
# batch-scaled. This job answers a *timing* question. For physics, use the L4 run.
#
# Submit with:  sbatch submit_training_maskfix_b200.sh

#SBATCH --job-name=clic-train-maskfix-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
echo "started: $(date -Is)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

CONFIG_PATH="configs/clic_v6_maskfix.yaml"
echo "Using config: $CONFIG_PATH  (batch_size 2048 from the config, as in June)"

PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --trainer.devices=1"

srun apptainer run --nv --bind /blue/,/cmsuf/ \
  /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run -e clic $PYTORCH_CMD

echo "finished: $(date -Is)"
echo "Done!"
