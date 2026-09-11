#!/bin/bash
# Fast smoke test of the evaluation path (one batch) on 1 L4. Edit CKPT/RUN_DIR below.

#SBATCH --job-name=clic-eval-test
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1
export COMET_MODE=offline

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/
echo "Working dir: ${PWD}"

export TMPDIR=/var/tmp/

RUN_DIR="logs/clic_paper_<timestamp>"
CKPT="${RUN_DIR}/ckpts/<epoch=...ckpt>"

PYTORCH_CMD="python main.py test \
  --config ${RUN_DIR}/config.yaml \
  --config configs/hpg.yaml \
  --config configs/eval.yaml \
  --trainer.devices=1 \
  --trainer.fast_dev_run=1 \
  --ckpt_path $CKPT"

echo "Running: $PYTORCH_CMD"
apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn-paper/pixi.sif \
  pixi run -e clic $PYTORCH_CMD
echo "Done!"
