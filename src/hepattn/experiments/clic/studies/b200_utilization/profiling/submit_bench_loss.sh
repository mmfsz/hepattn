#!/bin/bash

#SBATCH --job-name=clic-bench-loss
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

# Microbenchmark of the four loss kernels that are 68% of GPU-busy time in the post-fix
# trace, testing H1 (dynamic=True compilation) and H2 (data-dependent boolean-mask
# indexing forcing a device->host sync). Measurement only: no production code is changed.
# See studies/b200_utilization/profiling/NOTES.md.

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/b200_utilization/profiling

export TMPDIR=/var/tmp/
# Surface dynamo recompilation / graph-break decisions into the slurm log; this is itself
# evidence for H1 (a dynamic compile that keeps re-specialising is not free).
export TORCH_LOGS="recompiles,graph_breaks"

BENCH_CMD="python bench_loss_kernels.py --json-out profile_logs/bench_loss_${SLURM_JOB_ID}.json"
CMD="srun apptainer run --nv --bind /blue/,/cmsuf/ /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif pixi run -e clic $BENCH_CMD"
echo "Running command: $CMD"
$CMD
echo "Done!"
