#!/bin/bash
# Train on 2 nodes x 3 L4 (6 GPUs) on HPG. Submit FROM this directory:
#   mkdir -p slurm_logs
#   sbatch submit_training_hpg_l4_2nodes.sh configs/base.yaml [extra main.py args...]
# Extra arguments pass straight through, e.g. --name my_run or --trainer.max_epochs=2.

#SBATCH --job-name=clic-train-l4-2nodes
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=2
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --output=slurm_logs/slurm-%j.%x.out

CONFIG_PATH="${1:?Usage: sbatch submit_training_hpg_l4_2nodes.sh <config.yaml> [extra args]}"
shift

# Compute nodes have no internet / COMET_API_KEY: run the CometLogger offline.
export COMET_MODE=offline
export PATH="$HOME/.pixi/bin:$PATH"

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git rev-parse HEAD)"
echo "started: $(date -Is)"
nvidia-smi

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working dir: ${PWD}"

# devices=3 x num_nodes=2 = 6 L4, matching the SBATCH allocation.
# batch_size 170/GPU x 6 = global 1020, the paper's global batch (512 x 2 A100); a 24 GB L4
# cannot hold 512/GPU. Override with --data.batch_size=N in the extra args if needed.
PYTORCH_CMD="main.py fit --config $CONFIG_PATH --config configs/hpg.yaml \
    --trainer.devices=3 --trainer.num_nodes=2 --data.batch_size=170 $*"

# srun launches one task per GPU; run_task.sh gives each rank a private compile cache and
# execs the command inside the pixi env.
CMD="srun ./run_task.sh $PYTORCH_CMD"
echo "Running: $CMD"
$CMD
echo "finished: $(date -Is)"
