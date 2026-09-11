#!/bin/bash

# Head-to-head rerun, arm B: 3x L4 on one node with all four matcher/loss-pipeline fixes.
# SLURM geometry copied from the June 2026 production run (job 33954040: hpg-turin,
# gpu:l4:3, 1 node, 3 tasks x 16 CPU, 150 GB) so the throughput is directly comparable
# to its 1.81 it/s. Protocol as on the B200 arm (SimpleProfiler, no validation), with
# configs/profile_l4.yaml supplying the per-GPU batch of 256 and 800 steps.
# See studies/b200_utilization/profiling/NOTES.md.

#SBATCH --job-name=clic-rerun-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

# Load CUDA matching the container build
module load cuda/12.8.1

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi:"
nvidia-smi
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"

# Move to workdir
cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

echo "Running 3x L4 throughput rerun (batch 256/GPU, 800 steps, SimpleProfiler)..."

# Python command that will be run.
# devices=3 on 1 node = 3 L4 GPUs, matching the SBATCH allocation above.
PYTORCH_CMD="python main.py fit --config configs/base.yaml --config configs/profile.yaml --config configs/profile_l4.yaml --trainer.devices=3"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# srun in front for multi-GPU DDP
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
