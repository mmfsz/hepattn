#!/bin/bash

# Phase 0 (GATE): what fraction of a B200 training step does the Hungarian matcher own?
#
# This is the measurement the whole study hangs on. Every existing attribution of matcher cost
# in ../profiling/NOTES.md predates the mask-loss fix, which made the step ~1.7x faster on this
# hardware -- so the matcher's *share* of it is simply unknown, and a device solver cannot buy
# back more than that share.
#
# Decision rule, fixed before the measurement (README.md §5):
#   >= 20%   go   -- proceed to the real-cost replay and the paired A/B
#   10-20%   reconsider -- small ceiling, but the CPU-core saving may still justify it
#   <  10%   kill -- record it in NOTES.md as the answer to profiling candidate 6
#
# Protocol: production settings (compiled encoder/decoder, no profiler, batch 2048, 16 CPUs),
# with explicit torch.cuda.synchronize()-bracketed timers inside the matcher. Deliberately not
# the Phase-2/3 trace protocol -- see the header of configs/profile_phase0.yaml for why.
#
# Arm: pass CONFIG=configs/clic_v6_cudamatch.yaml to time the device solver instead of the host
# path. The host arm is the one that decides the gate; the device arm is a free sanity check
# that the auction runs at all on GPU, and reports its fallback count.
#
# Submit with:  sbatch submit_phase0_matcher_share_b200.sh
#           or: sbatch --export=ALL,CONFIG=configs/clic_v6_cudamatch.yaml submit_phase0_matcher_share_b200.sh

#SBATCH --job-name=clic-phase0-matcher
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
CONFIG=${CONFIG:-configs/clic_v6_maskfix.yaml}
OUT_DIR=$REPO/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher/phase0_logs

echo "Hostname: $(hostname)"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
echo "arm config: $CONFIG"
nvidia-smi

cd $REPO/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

PYTORCH_CMD="python main.py fit \
  --config $CONFIG \
  --config configs/profile_phase0.yaml \
  --trainer.devices=1"

srun apptainer run --nv --bind /blue/,/cmsuf/ \
  $REPO/pixi.sif pixi run $PYTORCH_CMD

# Stamp the results with the job id so repeat runs and the two arms cannot clobber each other.
for ext in npz json; do
  if [ -f "$OUT_DIR/phase0_matcher_timing.$ext" ]; then
    mv "$OUT_DIR/phase0_matcher_timing.$ext" "$OUT_DIR/phase0_matcher_timing_${SLURM_JOB_ID}.$ext"
    echo "wrote $OUT_DIR/phase0_matcher_timing_${SLURM_JOB_ID}.$ext"
  fi
done

echo "Done!"
