#!/bin/bash
# Phase 4 (physics equivalence) — the DEVICE arm, full length.
#
# Phase 3 passed, so §5 Phase 4 applies at full length rather than as a smoke check.
#
# ONLY the device arm is submitted here, because the host arm already exists:
#
#   logs/clic_v6_maskfix_20260803-T132601   (SLURM 38598204, 2026-08-03)
#     batch_size 2048, num_gpus 1, B200 c1003a-s15, max_epochs 200, 200 checkpoints,
#     torch 2.9.1+cu128, matcher = lap1015_late / parallel_solver / n_jobs 16,
#     already evaluated (epoch=199 .ckpt plus .root/.h5 test outputs).
#
# That is the same geometry, the same host matcher path and the same torch build as the
# Phase-3 host arm, so it is a valid baseline and saves ~31-41 h of allocation. Do NOT
# substitute clic_v6_maskfix_20260731-T152547 — that is the 3x L4 batch-256 run, and the
# profiling notes record that it is not physics-comparable to a batch-2048 B200 run.
#
# NOTE the deliberate omission of configs/profile_noprof.yaml. The Phase-3 A/B layered it on
# for throughput, and it sets max_steps=300 and limit_val_batches=0 -- its own header says
# "Do NOT use for real training runs". Phase 4 needs the full 200-epoch schedule and needs
# validation on, so it takes the base config only.
#
# Expect ~13 h of steady state (97,200 steps at the 2.066 it/s Phase 3 measured) plus compile
# and per-epoch validation. Walltime is set to the study's usual 96 h for headroom.
#
# After it finishes, compare against the baseline above:
#   ../profiling/plot_maskfix_curves.py     loss curves, both arms
#   ../profiling/plot_maskfix_jet_iqr.py    jet-E IQR, |z| < 2 as in the mask-fix validation
# and evaluate the checkpoint with ../profiling/submit_eval_b200_maskfix.sh as the template.

#SBATCH --job-name=clic-phase4-devmatch-train-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

echo "Hostname: $(hostname)"
echo "CPU count: $(awk '/^processor/{print $3}' /proc/cpuinfo | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
echo "host baseline for comparison: logs/clic_v6_maskfix_20260803-T132601 (SLURM 38598204)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

SIF=/blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif
PIXI_ENV=/blue/avery/m.mazza/projects/fastml/hepattn/.pixi/envs/default

# The -default tree, NOT the bench tree. The extension is ABI-bound to the pixi env it was
# built against: main.py runs under `default` (torch 2.9.1) while the offline benches run under
# `clic` (torch 2.10), and loading the wrong tree dies on
#   undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
# The second export is equally load-bearing: without it the import dies on CXXABI_1.3.15.
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-default
export APPTAINERENV_PYTHONPATH=$VENDOR

# JV kernel block size: nothing exported. Since 2026-09-08 the production tree's SMPCores()
# carries a compute-capability major-10 case returning 32, so a B200 gets the measured
# geometry by construction and an L4 keeps 128 (NOTES.md, 2026-09-08). TLA_BLOCK_SIZE
# survives in the binary as an override for sweeps only; do not set it here.
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

# Own Inductor/Triton cache, as in the A/B: no inherited kernels from another run.
CACHE="/var/tmp/clic_phase4_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

echo "started: $(date -Is)"

srun apptainer run --nv --bind /blue/,/cmsuf/ "$SIF" \
  pixi run python main.py fit \
    --config configs/clic_v6_cudamatch.yaml \
    --trainer.devices=1

echo "finished: $(date -Is)"
echo "Done!"
