#!/bin/bash

# Phase 3: op-level trace of the MASK-FIXED code on 1x L4 at batch 256.
#
# This is the arm that matters for CLIC. The production config is 3x L4 at 256/GPU, and the
# central lesson of this study is that batch-2048 B200 conclusions did NOT transfer to it:
# the matcher fixes looked like -31% on the B200 protocol and were worth ~7% in production,
# and the mask fix was 1.70x on B200 but 1.18x on L4. Profiling only the B200 config would
# repeat that mistake.
#
# ONE GPU, not three, on purpose. batch_size is per GPU, so a single L4 at 256 does exactly
# the per-GPU work of a production rank -- the kernel composition is the production one. What
# is NOT captured is the DDP gradient all-reduce between ranks. If NCCL turns out to matter,
# that needs a separate multi-rank trace (and PyTorchProfiler writes one file per rank).
#
# Protocol otherwise identical to submit_profile_phase3_b200.sh: same profile_phase2.yaml
# overlay (12 steps, eager mode, no validation), so the two traces are directly comparable
# in composition. Same eager-mode/profiler-overhead caveat: composition transfers, absolute
# idle fraction does not.
#
# Submit with:  sbatch submit_profile_phase3_l4.sh

#SBATCH --job-name=clic-phase3-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

PYTORCH_CMD="python main.py fit \
  --config configs/clic_v6_maskfix.yaml \
  --config configs/profile_phase2.yaml \
  --trainer.devices=1 \
  --data.batch_size=256 \
  --trainer.profiler.init_args.filename=phase3_maskfix_l4"

srun apptainer run --nv --bind /blue/,/cmsuf/ \
  /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run $PYTORCH_CMD

echo "Done!"
