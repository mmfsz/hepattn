#!/bin/bash

#SBATCH --job-name=clic-profile-lap
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
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

# Move to workdir
cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Phase 1 rerun with lap1015_late matcher solver (single-variable change vs job 38122563)
echo "Running lap1015 profiling run..."

# Python command that will be run
PYTORCH_CMD="python main.py fit --config configs/base.yaml --config configs/profile.yaml --config configs/profile_lap1015.yaml --trainer.devices=1"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
