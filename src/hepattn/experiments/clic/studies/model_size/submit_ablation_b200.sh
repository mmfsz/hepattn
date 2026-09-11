#!/bin/bash
# One model-size arm on the paper tag: 1x B200, batch 2048, GPU matcher (device_solver: jv),
# 200 epochs. Same geometry for every arm AND for the reference, so an arm is compared against
# the reference run of this study only (logs/clic_paper_small_b200_jv_*), never against a run
# from another hardware or from the head-based study on main.
#
# Submit, from this directory:
#   sbatch --job-name=clic-ps-C3 --export=ALL,CONFIG=studies/model_size/configs/clic_paper_small_C3_a2a3.yaml \
#     studies/model_size/submit_ablation_b200.sh
#   reference:  --export=ALL,CONFIG=configs/base_small.yaml,FIT_ARGS="--name clic_paper_small_b200_jv"
# Pre-flight every new geometry first (one train + one val batch, minutes):
#   ... --time=00:30:00 --export=ALL,CONFIG=<cfg>,FIT_ARGS=--trainer.fast_dev_run=true ...
# A parameter count instantiates every layer but never runs a forward pass, so it cannot see
# shape errors inside an attention kernel; only a pre-flight can.

#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=09:59:00          # the 702k head-v7 model took 7 h 40 m at this geometry (job 40405423);
                                  # per-epoch checkpoints with save_last make a timeout resumable
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1
export COMET_MODE=offline

# Pin the pixi environment explicitly: sbatch --export=ALL forwards the submitting shell's
# variables, and a submitting `pixi shell` would otherwise choose the environment.
unset PIXI_ENVIRONMENT_NAME PIXI_IN_SHELL PIXI_PROMPT CONDA_PREFIX || true

REPO=/blue/avery/m.mazza/projects/fastml/hepattn-paper

: "${CONFIG:?set CONFIG=<model config>.yaml via --export}"
FIT_ARGS="${FIT_ARGS:-}"

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
echo "config: ${CONFIG}"
nvidia-smi

cd $REPO/src/hepattn/experiments/clic/
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}"; exit 1; }
export TMPDIR=/var/tmp/

# The GPU solver's out-of-environment build; ABI-bound to the clic environment that runs main.py.
export APPTAINERENV_PYTHONPATH=$REPO/vendor/torch-linear-assignment
export APPTAINERENV_LD_LIBRARY_PATH=$REPO/.pixi/envs/clic/lib

# Fail here, before burning the allocation, if the build is wrong.
srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic \
  python -c "from torch_linear_assignment import _backend as b; print('tla ok, has_cuda:', b.has_cuda())"

# Own Inductor/Triton cache: no kernels inherited from another run.
CACHE="/var/tmp/clic_abl_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

echo "started: $(date -Is)"

# shellcheck disable=SC2086
srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif \
  pixi run -e clic python main.py fit \
    --config "${CONFIG}" \
    --config configs/hpg.yaml \
    --config configs/matcher_jv.yaml \
    --trainer.devices=1 --trainer.num_nodes=1 \
    --data.batch_size=2048 \
    ${FIT_ARGS}

echo "finished: $(date -Is)"
