#!/bin/bash

# Phase 1, re-run for a Jonker-Volgenant device solver.
#
# The auction failed Phase 1 on real cost matrices (job 40228586: 89.01% exact, 21.6%
# non-convergence, 350x slower than the host path). The failure is structural -- the auction is
# pseudo-polynomial, so its cost depends on the values in the matrix, not only its shape.
# torch-linear-assignment implements Crouse (2016), the same algorithm scipy uses, batched on
# the GPU and strongly polynomial. See NOTES.md, 2026-08-25.
#
# This replays the *already dumped* cost tensor, so there is no training stage and the job is
# minutes rather than the 28 the dump-and-replay took. Point COSTS at a different dump to
# re-run on one.
#
# torch_linear_assignment is built but deliberately NOT installed into the shared pixi env
# (same discipline as the -march arms). It is reached via PYTHONPATH, and needs LD_LIBRARY_PATH
# pointed at the env's lib or the import fails with a CXXABI error. Inside apptainer both have
# to be passed through with the APPTAINERENV_ prefix.
#
# Runs both arithmetic arms back to back in one allocation. float32 is the production case and
# what job 40252873 measured (99.76% exact, worst excess 4.77e-07); float64 is the diagnostic
# that decides whether that 0.24% is fp32 rounding -- in which case it goes to 100% and Phase 1
# is a pass -- or something systematic, in which case Phase 2 does not proceed. Both arms in one
# job so the two are scored on the same node, the way the host timings never were.
#
# Submit with:  sbatch submit_bench_jv_b200.sh

#SBATCH --job-name=clic-bench-jv
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
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment
PIXI_ENV=$REPO/.pixi/envs/clic

COSTS=${COSTS:-$(ls -t $STUDY/phase1_logs/*/clic_b200_matcher_costs.pt 2>/dev/null | head -1)}
if [ -z "$COSTS" ] || [ ! -f "$COSTS" ]; then
  echo "FAILED: no cost dump found; run submit_real_cost_replay_b200.sh first" >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
echo "costs: $COSTS ($(du -h "$COSTS" | cut -f1))"
nvidia-smi

export TMPDIR=/var/tmp/
# Carried into the container; without the second one the extension fails to load.
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

cd $STUDY
for DTYPE in float32 float64; do
  echo
  echo "=== $DTYPE ==="
  srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic \
    python bench_jv_solver.py --costs "$COSTS" --n-jobs 16 --sample 2048 --dtype "$DTYPE"
done

echo "Done!"
