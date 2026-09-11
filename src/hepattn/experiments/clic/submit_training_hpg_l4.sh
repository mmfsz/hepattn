#!/bin/bash
# Train on 1 node x 3 L4 on HPG. Submit FROM this directory:
#   mkdir -p slurm_logs
#   sbatch submit_training_hpg_l4.sh configs/base.yaml [extra main.py args...]
# Extra arguments pass straight through, e.g. --name my_run or --trainer.max_epochs=2.

#SBATCH --job-name=clic-train-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --output=slurm_logs/slurm-%j.%x.out

CONFIG_PATH="${1:?Usage: sbatch submit_training_hpg_l4.sh <config.yaml> [extra args]}"
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

# devices=3 x num_nodes=1 = 3 L4. batch_size 170/GPU x 3 ranks x 2 accumulation steps =
# global 1020, the paper's global batch, with the same per-rank batch as the 6-GPU script
# (the loss is normalised per rank, so this averages the same way as 6 ranks of 170).
PYTORCH_CMD="main.py fit --config $CONFIG_PATH --config configs/hpg.yaml \
    --trainer.devices=3 --trainer.num_nodes=1 --data.batch_size=170 \
    --trainer.accumulate_grad_batches=2 $*"

CMD="srun ./run_task.sh $PYTORCH_CMD"
echo "Running: $CMD"
$CMD
echo "finished: $(date -Is)"
