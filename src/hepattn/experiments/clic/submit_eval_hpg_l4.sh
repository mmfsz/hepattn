#!/bin/bash
# Evaluate a run's checkpoint on 1 L4: writes <ckpt>__test.h5 and, via PflowPredictionWriter,
# <ckpt>__test.root next to the checkpoint. Submit FROM this directory:
#   mkdir -p slurm_logs
#   sbatch --job-name=clic-eval-<tag> \
#          --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<ckpt_file> \
#          submit_eval_hpg_l4.sh

#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/slurm-%j.%x.out

: "${RUN_DIR:?export RUN_DIR=logs/<run_folder>}"
: "${CKPT_NAME:?export CKPT_NAME=<ckpt_file>}"

export COMET_MODE=offline
export PATH="$HOME/.pixi/bin:$PATH"
export TMPDIR=/var/tmp
export TRITON_CACHE_DIR="/var/tmp/triton_cache_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/var/tmp/inductor_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "RUN_DIR: ${RUN_DIR}"
echo "CKPT_NAME: ${CKPT_NAME}"
nvidia-smi

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working dir: ${PWD}"

CKPT="${RUN_DIR}/ckpts/${CKPT_NAME}"

# configs/hpg.yaml re-applies the HPG data paths in case the run's config was written elsewhere;
# configs/eval.yaml applies the README's evaluation rules (fp32, torch attention, inference data).
PYTORCH_CMD="main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config configs/hpg.yaml \
  --config configs/eval.yaml \
  --trainer.devices=1 \
  --trainer.num_nodes=1 \
  --ckpt_path $CKPT"

CMD="pixi run --frozen python $PYTORCH_CMD"
echo "Running: $CMD"
$CMD
echo "Done!"
