#!/bin/bash
# Environment A/B for the paper-vs-head step-time gap: the SAME paper-tag code, run inside the
# pixi environment of either worktree. 300 steps, 1x B200, batch 2048, GPU JV matcher -- the
# standard pre-flight geometry, so the two arms are compared against each other only.
#
#   ENV_REPO=/blue/avery/m.mazza/projects/fastml/hepattn-paper  (control: the paper's own env,
#            lightning 2.5.0.post0, comet_ml 3.58.6)
#   ENV_REPO=/blue/avery/m.mazza/projects/fastml/hepattn        (head's env, lightning 2.5.2,
#            comet_ml 3.50.0)
#
# Submit, from src/hepattn/experiments/clic of the paper worktree:
#   sbatch --job-name=pf-gap-paperenv --export=ALL,ENV_REPO=<paper>,RUN_NAME=pf_gap_paper_code_paper_env \
#     studies/step_time_gap/submit_env_ab_b200.sh
# Optional: EXTRA_ARGS="..." appends main.py arguments; CODE_REPO=<worktree> runs that checkout's
# hepattn instead of the paper's (with CONFIG=<its config>, e.g. the head model from main).
#
# --time: a 300-step pre-flight, 6 min measured (jobs 41992197/8); 30 min is the pre-flight limit.

#SBATCH --job-name=pf-gap
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1
export COMET_MODE=offline

CODE_REPO="${CODE_REPO:-/blue/avery/m.mazza/projects/fastml/hepattn-paper}"   # checkout whose hepattn runs
CONFIG="${CONFIG:-configs/base_small.yaml}"                                     # relative to its experiments/clic
ENV_REPO="${ENV_REPO:?set ENV_REPO to the worktree whose pixi environment to use}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-300}"
EXTRA_ARGS="${EXTRA_ARGS:-}"          # e.g. EXTRA_ARGS=--trainer.logger=false

unset PIXI_ENVIRONMENT_NAME PIXI_IN_SHELL PIXI_PROMPT CONDA_PREFIX || true

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "code: $CODE_REPO at $(git -C $CODE_REPO rev-parse HEAD)"
echo "env:  $ENV_REPO/.pixi/envs/clic (main at $(git -C $ENV_REPO rev-parse HEAD))"
nvidia-smi

cd $CODE_REPO/src/hepattn/experiments/clic/
STUDY=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/studies/step_time_gap
export TMPDIR=/var/tmp/

# One JV build for every arm (torch 2.9.1+cu128 in both environments): the paper worktree's.
export APPTAINERENV_PYTHONPATH=/blue/avery/m.mazza/projects/fastml/hepattn-paper/vendor/torch-linear-assignment
# The paper's model configs carry only the model; the B200 geometry and the JV matcher come from
# overlays that exist only in the paper checkout. main's head config already contains both.
OVERLAYS=""
for o in configs/hpg.yaml configs/matcher_jv.yaml; do [ -f "$o" ] && OVERLAYS="$OVERLAYS --config $o"; done
export APPTAINERENV_LD_LIBRARY_PATH=$ENV_REPO/.pixi/envs/clic/lib
export APPTAINERENV_PAPER_SRC=$CODE_REPO/src
export APPTAINERENV_DROP_SRC=$ENV_REPO/src
[ "$ENV_REPO" != "$CODE_REPO" ] && export APPTAINERENV_SHIM_COMET=1

PIXI="pixi run --manifest-path $ENV_REPO/pyproject.toml -e clic"
RUN="srun apptainer run --nv --bind /blue/,/cmsuf/ $ENV_REPO/pixi.sif $PIXI"

$RUN python -c "from torch_linear_assignment import _backend as b; assert b.has_cuda(), 'CPU-only build'; print('torch-linear-assignment ok, has_cuda: True')"

CACHE="/var/tmp/clic_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

echo "started: $(date -Is)"
$RUN python $STUDY/launch_paper_code.py fit \
  --config $CONFIG $OVERLAYS \
  --trainer.devices=1 --trainer.num_nodes=1 --data.batch_size=2048 \
  --name "$RUN_NAME" --trainer.max_steps="$MAX_STEPS" $EXTRA_ARGS
echo "finished: $(date -Is)"
