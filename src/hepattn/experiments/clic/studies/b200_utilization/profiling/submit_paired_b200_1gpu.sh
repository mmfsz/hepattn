#!/bin/bash

# B200 re-baseline, PAIRED design: runs the legacy-loss and mask-fixed arms back-to-back
# inside ONE allocation, so both see the same node.
#
# Why paired: run-to-run variance on hpg-b200 is ~27% (2.29 / 2.66 / 2.90 s/step, same
# commit, three nodes -- see NOTES.md "Head-to-head results + variance controls"). That is
# larger than most effects this study has tried to measure, so unpaired n=1 comparisons
# cannot support an attribution. Holding the node fixed cancels it in the difference.
#
# This is only possible because the mask fix is a CONFIG change: the v2 losses are opt-in
# (loss.py keeps the legacy functions), so the two arms differ by exactly two config keys
# (verified with --print_config: mask_bce/mask_dice -> mask_bce_v2/mask_dice_v2, plus the
# run name). No code checkout, no rebuild.
#
# Protocol: configs/profile_noprof.yaml -- profiler OFF, 300 steps, no validation.
# Read throughput from the tqdm timestamps, steady state = steps 50 -> 300.
#
# Submit 3 of these (different nodes) and alternate ORDER so compile-cache warmth and any
# within-job drift cannot masquerade as an arm effect:
#   sbatch --export=ALL,ORDER=legacy_first  submit_paired_b200_1gpu.sh
#   sbatch --export=ALL,ORDER=maskfix_first submit_paired_b200_1gpu.sh
#   sbatch --export=ALL,ORDER=legacy_first  submit_paired_b200_1gpu.sh
#
# See studies/b200_utilization/profiling/NOTES.md.

#SBATCH --job-name=clic-paired-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

ORDER="${ORDER:-legacy_first}"

echo "Hostname: $(hostname)"
echo "CPU count: $(awk '/^processor/{print $3}' /proc/cpuinfo | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ORDER: ${ORDER}"
echo "git commit: $(git -C /blue/avery/m.mazza/projects/fastml/hepattn rev-parse HEAD)"
nvidia-smi

cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

SIF=/blue/avery/m.mazza/projects/fastml/hepattn/pixi.sif

run_arm () {
  local arm="$1" cfg="$2"
  # Each arm gets its OWN Inductor/Triton cache. Without this the second arm inherits the
  # first arm's compiled kernels and looks faster for a reason that has nothing to do with
  # the loss -- the same class of leaked-state confound that produced the bogus 32-thread
  # numbers in job 38458388.
  local cache="/var/tmp/clic_${SLURM_JOB_ID}_${arm}"
  mkdir -p "${cache}/inductor" "${cache}/triton"
  # APPTAINERENV_* is the guaranteed way to get a variable through to the container.
  export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
  export APPTAINERENV_TRITON_CACHE_DIR="${cache}/triton"

  echo ""
  echo "================ ARM: ${arm} ================"
  echo "config: ${cfg}"
  echo "inductor cache: ${cache}/inductor"
  echo "started: $(date -Is)"

  srun apptainer run --nv --bind /blue/,/cmsuf/ "$SIF" \
    pixi run python main.py fit \
      --config "${cfg}" \
      --config configs/profile_noprof.yaml \
      --trainer.devices=1

  echo "finished ${arm}: $(date -Is)"
  echo "================ END ARM: ${arm} ================"
}

if [ "$ORDER" = "maskfix_first" ]; then
  run_arm maskfix configs/clic_v6_maskfix.yaml
  run_arm legacy  configs/base.yaml
else
  run_arm legacy  configs/base.yaml
  run_arm maskfix configs/clic_v6_maskfix.yaml
fi

echo "Done!"
