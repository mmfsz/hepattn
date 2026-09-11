#!/bin/bash

# Phase 3: op-level trace of the MASK-FIXED code on 1x B200 at batch 2048.
#
# Why re-trace at all: the Phase-2 traces describe a code path that no longer exists. The four
# Triton mask-loss kernels were 67.8% of GPU-busy time; the fix made the whole step ~1.7x
# faster on this hardware, so whatever dominates now is simply unknown. Every "where the time
# goes" number in NOTES.md predates the fix.
#
# Protocol identical to Phase 2 (configs/profile_phase2.yaml): PyTorchProfiler, 12 steps, no
# validation, and the Compile callback removed so the trace is clean eager mode. Lightning
# applies its default schedule (wait 1, warmup 1, active 3) -> 3 profiled steps.
#
# Companion: submit_profile_phase3_l4.sh, same protocol at batch 256 on an L4.
#
# Analyse with analyze_trace.py (interval-union of kernel/memcpy events vs ProfilerStep spans).
# CAVEAT carried over from Phase 2: this is eager mode with profiler overhead, so the absolute
# GPU-idle fraction is NOT the production one -- only the *composition* transfers. The script
# also emits duplicate thread-level ProfilerStep spans (6 spans = 3 real steps).
#
# Submit with:  sbatch submit_profile_phase3_b200.sh

#SBATCH --job-name=clic-phase3-b200
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

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

# filename is overridden so the two arms cannot clobber each other's trace in profile_logs/.
PYTORCH_CMD="python main.py fit \
  --config configs/clic_v6_maskfix.yaml \
  --config configs/profile_phase2.yaml \
  --trainer.devices=1 \
  --trainer.profiler.init_args.filename=phase3_maskfix_b200"

srun apptainer run --nv --bind /blue/,/cmsuf/ \
  /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run -e clic $PYTORCH_CMD

echo "Done!"
