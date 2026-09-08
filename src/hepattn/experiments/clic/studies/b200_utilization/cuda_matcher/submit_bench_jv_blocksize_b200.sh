#!/bin/bash

# Occupancy sweep for the JV device solver: block size and problem-count scaling.
#
# Replays the already-dumped cost tensor, so there is no training stage and the job is minutes.
# See bench_jv_blocksize.py for what the two axes mean and why axis B is the load-bearing one.
#
# WHICH VENDOR TREE, AND WHY IT IS NOT ONE OF THE PRODUCTION ONES.
#
# Axis A needs the TLA_BLOCK_SIZE runtime override, which exists only in a patched build:
#   vendor/torch-linear-assignment-blocksize   <- built by build_tla_blocksize.sh, used here
#   vendor/torch-linear-assignment            <- PRODUCTION, clic env (torch 2.10), other benches
#   vendor/torch-linear-assignment-default    <- PRODUCTION, default env (torch 2.9.1), training
# The patched tree is a copy so this experiment cannot damage either production build. The copy
# is of the *clic* tree because this benchmark runs under `pixi run -e clic`; a torch extension
# is ABI-bound to the torch it was compiled against, and the wrong one dies on
#   undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
# LD_LIBRARY_PATH must point at the env's lib or the import dies on CXXABI_1.3.15 instead.
# Both have to be passed through the container with the APPTAINERENV_ prefix.
#
# Falling back to the production tree is deliberately NOT done: an unpatched build ignores
# TLA_BLOCK_SIZE, which would silently turn axis A into five identical rows. The benchmark
# warns if it sees that, but the right place to catch it is here.
#
# Submit with:  sbatch submit_bench_jv_blocksize_b200.sh
#   DTYPE=float64 sbatch ...   to run the diagnostic arithmetic instead of the production one

#SBATCH --job-name=clic-bench-jv-blocksize
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
STUDY=$REPO/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-blocksize
PIXI_ENV=$REPO/.pixi/envs/clic
DTYPE=${DTYPE:-float32}

if [ ! -d "$VENDOR" ]; then
  echo "FAILED: patched extension not found at $VENDOR; run build_tla_blocksize.sh first" >&2
  exit 1
fi

COSTS=${COSTS:-$(ls -t "$STUDY"/phase1_logs/*/clic_b200_matcher_costs.pt 2>/dev/null | head -1)}
if [ -z "$COSTS" ] || [ ! -f "$COSTS" ]; then
  echo "FAILED: no cost dump found; run submit_real_cost_replay_b200.sh first" >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
echo "vendor: $VENDOR"
echo "costs: $COSTS ($(du -h "$COSTS" | cut -f1))"
echo "dtype: $DTYPE"
nvidia-smi

export TMPDIR=/var/tmp/
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

cd "$STUDY"
srun apptainer run --nv --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic \
  python bench_jv_blocksize.py --costs "$COSTS" --dtype "$DTYPE"

echo "Done!"
