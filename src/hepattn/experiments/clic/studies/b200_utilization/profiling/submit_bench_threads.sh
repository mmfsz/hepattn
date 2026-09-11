#!/bin/bash

#SBATCH --job-name=clic-bench-threads
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

# CPU-only: reproduces the matcher's thread scaling on the same hardware the training
# runs on, to explain fix experiment 5 (32 threads slower than 16).
echo "Hostname: $(hostname)"
lscpu | grep -E "^CPU\(s\)|Socket|Core\(s\) per socket|NUMA node\(s\)|NUMA node[0-9]"
echo

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/b200_utilization/profiling

BENCH_CMD="python bench_matcher_threads.py"
srun apptainer run --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run -e clic $BENCH_CMD
echo "Done!"
