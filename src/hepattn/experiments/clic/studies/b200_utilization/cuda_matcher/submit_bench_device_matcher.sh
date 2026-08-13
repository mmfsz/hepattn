#!/bin/bash

#SBATCH --job-name=clic-bench-device-matcher
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

# Phase 1 of the CUDA matching study: is the GPU auction solver exact, and is it faster than
# the 16-thread host solver once the device->host copy is counted against the host?
# Measurement only -- no production code path changes, since device_solver defaults to off.
# See studies/b200_utilization/cuda_matcher/README.md.
#
# --cpus-per-task=16 matches the production config on purpose: the host arm must be given the
# thread count it was measured to saturate at, or the comparison flatters the GPU.

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher

export TMPDIR=/var/tmp/

BENCH_CMD="python bench_device_matcher.py --n-jobs 16 --sample 512"
CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run $BENCH_CMD"
echo "Running command: $CMD"
$CMD
echo "Done!"
