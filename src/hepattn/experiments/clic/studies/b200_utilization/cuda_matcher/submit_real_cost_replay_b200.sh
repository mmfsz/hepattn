#!/bin/bash

# Phase 1, step 2 of the CUDA matching study: is the auction exact on *real* cost matrices?
#
# The exactness result banked so far is on synthetic uniform costs. That is the easy case: real
# mask-BCE costs are far more degenerate, and degeneracy is precisely what provokes the
# auction's two untested weaknesses -- thrashing on ties, and a round count that scales with how
# competitive the problem is. README.md §7 risks 2 and 7. This is the run that settles them.
#
# Two stages in one allocation, because the queue is the expensive part:
#   1. a short training run whose only job is to write one step's real cost tensor to disk
#      (hepattn.callbacks.MatcherCostDump, driven by configs/dump_matcher_costs.yaml),
#   2. bench_device_matcher.py --costs on that file, scoring the auction against scipy's
#      float64 optimum and timing it against the 16-thread host path.
#
# The .pt survives the job, so every later re-run of the replay is free and needs no GPU slot
# beyond this one.
#
# Submit with:  sbatch submit_real_cost_replay_b200.sh

#SBATCH --job-name=clic-real-cost-replay
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
STUDY=$REPO/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher
OUT_DIR=$STUDY/phase1_logs
CONFIG=${CONFIG:-configs/clic_v6_maskfix.yaml}

echo "Hostname: $(hostname)"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
echo "arm config: $CONFIG"
nvidia-smi

export TMPDIR=/var/tmp/
RUN="srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic"

echo "=== STAGE 1: dump one step's real cost matrices ==="
cd $REPO/src/hepattn/experiments/clic/
$RUN python main.py fit \
  --config $CONFIG \
  --config configs/dump_matcher_costs.yaml \
  --trainer.devices=1

COSTS=$(ls -t $OUT_DIR/*/clic_b200_matcher_costs.pt 2>/dev/null | head -1)
if [ -z "$COSTS" ]; then
  echo "FAILED: no cost dump was written under $OUT_DIR" >&2
  exit 1
fi
echo "dumped: $COSTS ($(du -h "$COSTS" | cut -f1))"

echo "=== STAGE 2: replay it against scipy ==="
cd $STUDY
# --sample 2048: the fallback and exactness rates are the whole point here, and unlike the
# synthetic sweep there is only one case to spend the scipy reference solves on.
$RUN python bench_device_matcher.py --costs "$COSTS" --n-jobs 16 --sample 2048

echo "Done!"
