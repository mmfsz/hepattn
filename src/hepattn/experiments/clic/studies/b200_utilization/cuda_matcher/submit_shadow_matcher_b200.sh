#!/bin/bash
# Inline shadow run — quantify how different the GPU and CPU matchers actually are.
#
# The colleague's question: "technical metrics like the matching quality of the algorithm or
# intersection-over-union of GPU matcher vs CPU matcher, so we can start quantifying how
# 'different' they are." The study cannot answer it yet, because its evidence has a hole in the
# middle. bench_device_matcher.py scores one dumped step's costs against scipy's float64 optimum
# — one scalar, one step, silent about which pairing was chosen. Phase 4 compares two full
# trainings on val loss and jet-E IQR — the thing that matters, measured where a solver
# difference is indistinguishable from SGD noise.
#
# This run measures the substitution itself, on identical inputs, three layers deep:
#   1. optimality gap   — cost(device permutation) − cost(host optimum), float64, per problem.
#                         Solver correctness only: costs are detached and carry no gradient
#   2. assignment IoU   — |A ∩ B| / |A ∪ B| over the (query, target) pairs, plus how degenerate
#                         the swaps are (the cost difference between the entries swapped). This
#                         is the layer that bears on training: the permutation is the only
#                         channel by which the matcher reaches the loss
#   3. mask IoU         — for the disagreeing pairs, the IoU of the assigned prediction against
#                         its truth particle under each solver's choice, with an equal-sized
#                         sample of agreeing pairs as the scale
#
# Inline rather than a replay of a dumped cost tensor, because the one question a replay cannot
# answer is whether agreement *drifts* as the model sharpens and its costs become less
# degenerate. Every 100 steps, all 10,240 problems, ~970 samples across the schedule.
#
# The production matcher call is untouched — the shadow solve runs beside it on the same inputs
# and its result is discarded, no torch RNG is touched, and the matcher's own step and
# device_fallbacks counters are restored — so this is simultaneously a valid device arm for
# physics on the same 200-epoch schedule as Phase 4, and a second seed of it if wanted.
#
# Config: configs/clic_v6_cudamatch.yaml + configs/shadow_matcher.yaml (the callback only;
# max_epochs, validation and checkpointing all come through from the base unchanged).
#
# Output: logs/clic_v6_cudamatch_shadow_<timestamp>/matcher_shadow.jsonl, one line per shadow
# step, appended as it goes, so a killed job still leaves everything measured up to that point.
# The same numbers also go to Comet under shadow/*.
#
# Expect ~13 h of steady state (97,200 steps at the 2.066 it/s Phase 3 measured) plus compile
# and per-epoch validation, plus well under 1% for the shadow itself. Walltime is the study's
# usual 96 h for headroom.

#SBATCH --job-name=clic-shadow-matcher-b200
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

# Own Inductor/Triton cache: no inherited kernels from another run.
CACHE="/var/tmp/clic_shadow_${SLURM_JOB_ID}"
mkdir -p "${CACHE}/inductor" "${CACHE}/triton"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${CACHE}/inductor"
export APPTAINERENV_TRITON_CACHE_DIR="${CACHE}/triton"

echo "started: $(date -Is)"

# `-e default` is NOT decorative. #SBATCH --export=ALL exports the submitting shell, and a
# shell that is inside `pixi shell -e clic` carries PIXI_ENVIRONMENT_NAME=clic, PIXI_IN_SHELL=1
# and a clic-first PATH. Apptainer passes those through, so a bare `pixi run` resolves to the
# *clic* env (torch 2.10), whose flash-attn is not the one this model needs: the encoder dies in
# sanity check with `TypeError: 'NoneType' object is not callable` at attention.py:292, ~2 min
# in and with nothing about pixi in the traceback. Job 40639789 died this way, as did the four
# model-size arms 40540644-47 before it. Pinning the environment makes the script immune to
# whatever shell submits it.
srun apptainer run --nv --bind /blue/,/cmsuf/ "$SIF" \
  pixi run -e default python main.py fit \
    --config configs/clic_v6_cudamatch.yaml \
    --config configs/shadow_matcher.yaml \
    --trainer.devices=1

echo "finished: $(date -Is)"
echo "Done!"
