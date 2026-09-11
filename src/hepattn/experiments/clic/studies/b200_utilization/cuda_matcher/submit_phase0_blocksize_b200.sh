#!/bin/bash

# Does the block-size win survive inside a real training step?
#
# The offline sweep (job 40507155, bench_jv_blocksize.py) measured TLA_BLOCK_SIZE=32 at 153.31 ms
# against the default 128's 170.64 ms -- a free 10% with bit-identical assignments. That number
# was taken on an IDLE GPU, replaying a dumped tensor. Inside training the solver runs straight
# after the model's kernels with L2 full of activations, and the whole benefit is SM coverage
# (80 of 148 SMs at block 128, all 148 at block 32), which is exactly the kind of thing a
# contended GPU can erase. This study has already been burned once by promoting an offline
# solver number: JV measured 10.5x offline and 2.3-3.1x inside training.
#
# So this is the gate before the candidate binary is allowed anywhere near production. Both
# cells run in ONE allocation on ONE node, because the effect being measured (~10% of a 39%
# term, so ~2.5% of a step) is far smaller than the node-to-node spread this study has
# documented.
#
# Uses configs/profile_phase0.yaml, so MatcherTimer attributes the step with
# torch.cuda.synchronize()-bracketed timers. Read the `device` bucket: it is the whole device
# solver path, and it is the only bucket that should move between the two cells.
#
# THE BINARY. vendor/torch-linear-assignment-candidate is the only tree with all three of:
# the DEFAULT ABI (torch 2.9.1, what main.py runs under), both sm_89 and sm_100 cubins, and the
# TLA_BLOCK_SIZE patch. Production `-default` has none of the patch, so it cannot run this test.
# See build_tla_candidate.sh.
#
# Submit with:  sbatch submit_phase0_blocksize_b200.sh

#SBATCH --job-name=clic-phase0-blocksize-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
CLIC=$REPO/src/hepattn/experiments/clic
PIXI_ENV=$REPO/.pixi/envs/clic
# Defaults to the PRODUCTION tree. Until 2026-08-28 that tree had no TLA_BLOCK_SIZE patch and
# this had to point at -candidate; the candidate was merged into production that day, so the
# default now exercises the path training actually takes. Override VENDOR to re-test a candidate.
VENDOR=${VENDOR:-/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-default}
OUT_DIR=$CLIC/studies/b200_utilization/cuda_matcher/phase0_logs

echo "Hostname: $(hostname)"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
echo "vendor: $VENDOR"
nvidia-smi

cd "$CLIC"
export TMPDIR=/var/tmp/
# Deliberately no `module load cuda/12.8.1`: it sets CUDA_HOME=/apps/..., which is not
# bind-mounted into the container. Not needed at run time in any case.
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

run_cell () {
  local label="$1" bs="$2"
  # Own Inductor/Triton cache per cell, so the second cell cannot inherit the first's compiled
  # kernels and look faster for a reason unrelated to the block size.
  local cache="/var/tmp/clic_p0bs_${SLURM_JOB_ID}_${label}"
  mkdir -p "${cache}/inductor" "${cache}/triton"
  export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
  export APPTAINERENV_TRITON_CACHE_DIR="${cache}/triton"

  if [ -n "$bs" ]; then
    export APPTAINERENV_TLA_BLOCK_SIZE="$bs"
  else
    unset APPTAINERENV_TLA_BLOCK_SIZE || true
  fi

  echo
  echo "================ ARM: ${label} ================"
  echo "TLA_BLOCK_SIZE=${bs:-<unset, SMPCores fallback = 128>}"
  echo "started: $(date -Is)"

  apptainer run --nv --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic \
    python main.py fit \
      --config configs/clic_v6_cudamatch.yaml \
      --config configs/profile_phase0.yaml \
      --trainer.devices=1

  # MatcherTimer writes phase0_matcher_timing.{json,npz}; stamp each cell so the two survive.
  local newest
  for ext in json npz; do
    newest=$(find "$OUT_DIR" -name "phase0_matcher_timing.$ext" -newermt "-30 minutes" -print0 \
             | xargs -0 -r ls -t | head -1 || true)
    if [ -n "$newest" ]; then
      cp "$newest" "$OUT_DIR/phase0_blocksize_${label}_${SLURM_JOB_ID}.$ext"
      echo "kept $OUT_DIR/phase0_blocksize_${label}_${SLURM_JOB_ID}.$ext"
    else
      echo "WARNING: no timing .$ext found for cell ${label}" >&2
    fi
  done
  echo "finished: $(date -Is)"
  echo "================ END ARM: ${label} ================"
}

# ORDER alternates which cell runs first. Submit one of each: if the win were an artefact of
# within-job drift -- a node warming up, a neighbour's job starting -- it would follow the
# position in the job rather than the block size, and reversing the order would reverse it.
ORDER=${ORDER:-default_first}
echo "ORDER: $ORDER"
if [ "$ORDER" = "bs32_first" ]; then
  run_cell bs32 32
  run_cell bs128_default ""
else
  run_cell bs128_default ""
  run_cell bs32 32
fi

echo "Done!"
