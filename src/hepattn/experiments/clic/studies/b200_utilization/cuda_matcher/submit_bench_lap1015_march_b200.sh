#!/bin/bash

# Where must the lap1015 extension be built, and what does portability cost?
#
# CMakeLists.txt compiles with `-march=native`, which targets the CPU that runs the build. On
# HiPerGator that is a trap: login nodes are AMD EPYC 7702 (Zen 2, no AVX-512), B200 nodes are
# Intel Emerald Rapids (AVX-512). Rebuilding the extension on the login node -- which is the
# only place `pixi reinstall` can be run interactively -- and then training on a B200 node took
# the Phase-0 host solve from 712.8 ms to 1760.7 ms, halving throughput. See NOTES.md.
#
# This job settles it on the node that actually trains, by building the same source three ways
# and timing each: the currently installed .so, a fresh `-march=native` build (native meaning
# *this* node), and a fixed `-march=x86-64-v3` AVX2 baseline that every node type supports.
#
# It does not touch the shared pixi environment -- each arm is compiled into a scratch directory
# and loaded by file path -- so it is safe to run while other work is using the env.
#
# Submit with:  sbatch submit_bench_lap1015_march_b200.sh

#SBATCH --job-name=clic-lap1015-march
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

module load cuda/12.8.1

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
STUDY=$REPO/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher

echo "Hostname: $(hostname)"
echo "git commit: $(git -C $REPO rev-parse HEAD)"
grep -m1 "model name" /proc/cpuinfo
echo "avx512f present: $(grep -m1 -c avx512f /proc/cpuinfo || true)"

# --cpus-per-task=16 on purpose: the production CLIC config asks for 16 solver threads, so the
# thread sweep has to be run under the same allocation it will face in training.
cd $STUDY
export TMPDIR=/var/tmp/

srun apptainer run --nv --bind /blue/,/cmsuf/ $REPO/pixi.sif pixi run -e clic \
  python bench_lap1015_march.py --reps 3

echo "Done!"
