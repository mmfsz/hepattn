#!/bin/bash

# Phase 3 of the CUDA matching study, PAIRED design: runs the host-matching and GPU-matching
# arms back-to-back inside ONE allocation, so both see the same node.
#
# Why paired: run-to-run variance on hpg-b200 is ~27% and throughput spans 1122-1911
# samples/s across nodes (see ../profiling/NOTES.md). That is larger than the effect this is
# trying to measure, so an unpaired n=1 comparison cannot support an attribution. Holding the
# node fixed cancels it in the difference.
#
# This is only possible because the device solver is a CONFIG change: device_solver defaults
# to None, so the two arms differ by exactly the matcher block (verified with --print_config).
# No code checkout, no rebuild.
#
# Protocol: configs/profile_noprof.yaml -- profiler OFF, 300 steps, no validation.
# Parse with ../profiling/parse_throughput.py --batch 2048, which splits on the ARM: markers.
#
# Submit 3 of these (different nodes) and alternate ORDER so compile-cache warmth and any
# within-job drift cannot masquerade as an arm effect:
#   sbatch --export=ALL,ORDER=host_first   submit_paired_device_matcher_b200.sh
#   sbatch --export=ALL,ORDER=device_first submit_paired_device_matcher_b200.sh
#   sbatch --export=ALL,ORDER=host_first   submit_paired_device_matcher_b200.sh
#
# See studies/b200_utilization/cuda_matcher/README.md.

#SBATCH --job-name=clic-paired-devmatch-b200
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1        # must match trainer.devices
#SBATCH --cpus-per-task=16         # what the HOST arm needs; the device arm should not care
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

module load cuda/12.8.1

ORDER="${ORDER:-host_first}"

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
  # the matcher -- the same leaked-state confound that produced the bogus 32-thread numbers.
  local cache="/var/tmp/clic_${SLURM_JOB_ID}_${arm}"
  mkdir -p "${cache}/inductor" "${cache}/triton"
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

if [ "$ORDER" = "device_first" ]; then
  run_arm device configs/clic_v6_cudamatch.yaml
  run_arm host   configs/clic_v6_maskfix.yaml
else
  run_arm host   configs/clic_v6_maskfix.yaml
  run_arm device configs/clic_v6_cudamatch.yaml
fi

echo "Done!"
