#!/bin/bash
# Train on 1 B200 on HPG.
# Submit FROM this directory:
#   sbatch submit_training_hpg_1gpu.sh <config.yaml> [extra main.py args...]
# Extra arguments pass straight through to main.py, e.g. --name my_run or --trainer.max_epochs=2.

#SBATCH --job-name=clic-train-1gpu
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

CONFIG_PATH="${1:?Usage: sbatch submit_training_hpg_1gpu.sh <config.yaml> [extra args]}"
shift
echo "Using config: $CONFIG_PATH"

# Load CUDA matching the container build
module load cuda/12.8.1

# Compute nodes have no internet / COMET_API_KEY: run the CometLogger offline.
export COMET_MODE=offline

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn-paper rev-parse HEAD)"
echo "started: $(date -Is)"
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# configs/hpg.yaml layers the HPG data paths over the model config, which keeps the
# authors' paths. batch_size 1024 on 1 B200 = the paper's global batch (the throughput studies on main used 2048).
# Override any of these through the extra arguments.
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH --config configs/hpg.yaml \
    --trainer.devices=1 --trainer.num_nodes=1 --data.batch_size=1024 $*"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run -e clic $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image.
# srun in front for multi-GPU DDP; run_task.sh gives each rank a private compile cache.
APPTAINER_CMD="srun ./run_task.sh apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn-paper/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "finished: $(date -Is)"
echo "Done!"
