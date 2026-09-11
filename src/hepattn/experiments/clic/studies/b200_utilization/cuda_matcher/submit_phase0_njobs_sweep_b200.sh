#!/bin/bash

# What is the matcher's host thread pool actually worth? All arms in ONE allocation.
#
# The CLIC config asks for n_jobs=16, a number chosen while the installed lap1015 held the GIL --
# so the pool never actually ran in parallel and 16 was never really tested. Now that the
# extension releases the GIL (see NOTES.md, 2026-08-24) the number matters, and the first attempt
# to measure it one-job-per-value was worthless: each landed on a different shared node and the
# solve time swung 2.5x non-monotonically while the GPU-side `other` bucket stayed flat at
# 358-371 ms. Same confound the paired A/B runner exists to avoid.
#
# So: one allocation, one node, every n_jobs in sequence. Neighbours on the node can still change
# under us, but node identity and our own CPU share no longer differ between arms.
#
# Submit with:  sbatch submit_phase0_njobs_sweep_b200.sh
#           or: sbatch --export=ALL,JOBS_LIST="1 2 4 8 16" submit_phase0_njobs_sweep_b200.sh

#SBATCH --job-name=clic-phase0-njobs-sweep
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
CONFIG=${CONFIG:-configs/clic_v6_maskfix.yaml}
JOBS_LIST=${JOBS_LIST:-"1 2 4 8 16"}

echo "Hostname: $(hostname)"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
grep -m1 "model name" /proc/cpuinfo
echo "sweeping n_jobs over: $JOBS_LIST"
nvidia-smi -L

cd $REPO/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/
RUN="srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic"

$RUN python -c "import lap1015; print('lap1015.releases_gil:', lap1015.releases_gil)"

for N in $JOBS_LIST; do
  echo ""
  echo "================ ARM: n_jobs=$N ================"
  $RUN python main.py fit \
    --config $CONFIG \
    --config configs/profile_phase0.yaml \
    --trainer.devices=1 \
    --model.model.init_args.matcher.init_args.n_jobs=$N
done

echo ""
echo "================ SUMMARY ================"
echo "Grep the ARM markers above; each is followed by its MatcherTimer table."
echo "Done!"
