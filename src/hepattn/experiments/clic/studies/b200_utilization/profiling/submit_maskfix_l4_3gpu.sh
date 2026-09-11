#!/bin/bash

# L4 arm of the post-mask-fix re-baseline: 3x L4 on one node, mask-fixed loss.
#
# Exists so both hardware arms are measured with the SAME short-run recipe. The 200-epoch
# training job 38469247 already gives a very well-averaged L4 number (2.27 it/s = 1743
# samples/s), but it is one node and a different protocol; comparing it directly against
# short B200 runs would reintroduce the apples-to-oranges problem this re-baseline exists
# to remove.
#
# SLURM geometry copied from the June 2026 production run (job 33954040: hpg-turin,
# gpu:l4:3, 1 node, 3 tasks x 16 CPU, 150 GB), so it also stays comparable to its 1.81 it/s.
# Protocol: profiler OFF (configs/profile_noprof.yaml), with configs/profile_l4.yaml
# supplying batch 256/GPU and 800 steps -- L4 steps are ~8x cheaper than B200 ones, so 300
# steps would be mostly torch.compile warmup. Overlay order matters: profile_l4 comes last
# so its max_steps=800 wins over profile_noprof's 300.
#
# Read throughput from the tqdm timestamps, steady state = steps 50 -> 800; rank 0's is the
# one printed. Convert with: samples/s = it/s x 768 (global batch).
#
# Submit 3 of these to sample node-to-node variance:
#   for i in 1 2 3; do sbatch submit_maskfix_l4_3gpu.sh; done
#
# See studies/b200_utilization/profiling/NOTES.md.

#SBATCH --job-name=clic-maskfix-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CPU count: $(awk '/^processor/{print $3}' /proc/cpuinfo | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

# No per-rank compile-cache overrides here, deliberately: every 1-node 3x L4 run in this
# study (33954040, 38461126, 38469247) ran fine on the default caches. The per-rank
# TRITON_CACHE_DIR workaround in studies/glow_jet_iqr/REPRODUCE.md was for the paper-tag
# clone's multi-NODE runs. Cross-job cache warmth only affects compile warmup, which the
# steady-state window (steps 50 -> 800) excludes anyway.

echo "started: $(date -Is)"

srun apptainer run --nv --bind /blue/,/cmsuf/ \
  /blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif \
  pixi run -e clic python main.py fit \
    --config configs/clic_v6_maskfix.yaml \
    --config configs/profile_noprof.yaml \
    --config configs/profile_l4.yaml \
    --trainer.devices=3

echo "finished: $(date -Is)"
echo "Done!"
