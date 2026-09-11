#!/bin/bash

# Phase 3, L4 ARM: the same paired host-vs-device matcher A/B as
# submit_paired_device_matcher_b200.sh, but on the hardware CLIC actually trains on --
# 3x L4 at batch 256/GPU. Both arms run back-to-back inside ONE allocation.
#
# WHY THIS RUN EXISTS
# -------------------
# README.md §6's ship criterion is three conditions joined by AND. Two are met: exact
# assignment cost on real matrices (0 fallbacks), and >= 10% B200 throughput (+134% to +212%,
# jobs 40291275-40291277). The third -- "no L4 regression beyond noise" -- had never been
# measured. Every statement about the L4 in this study so far is an EXPECTATION, and the whole
# opt-in design (device_solver defaults to None) is justified by it. An untested assumption
# holding up a ship decision is an open item, not a formality.
#
# WHAT WE EXPECT, AND WHY A LOSS IS THE GOOD OUTCOME
# --------------------------------------------------
# The B200 win is not a solver win, it is a HOST-STALL win: at batch 2048 the B200 is 69.5%
# GPU-idle, so moving the ~713 ms host solve onto an idle GPU costs nothing and removes it from
# the critical path. The L4 at batch 256 is 85.3% GPU-BUSY (profiling NOTES.md, Phase 3) -- it
# has no stall to reclaim, and the JV kernels have to contend with the model for the same SMs.
# So the device arm is expected to LOSE here, and that is the result that CONFIRMS the design.
# Do not tune the device arm to make it win; report what it measures.
#
# THE BINARY, AND WHY IT IS A NEW ONE
# -----------------------------------
# Until 2026-08-28 every torch-linear-assignment build in vendor/ was compiled
# TORCH_CUDA_ARCH_LIST=10.0, i.e. sm_100 cubins only, so on an L4 (sm_89) the device arm died
# at the first solve with "no kernel image is available for execution on the device". The
# first L4 run (job 40525026) used a throwaway sm_89+sm_100 tree built for the purpose. Since
# the block-size deployment that same day, the PRODUCTION tree -- vendor/torch-linear-assignment-default,
# built by build_tla_candidate.sh -- carries "8.9;10.0" itself, so this script now loads the
# same .so that training loads. That removes a confound: the L4 arm and any B200 repeat run
# the SAME binary, so "different build" cannot explain a difference between hardware. The
# TLA_BLOCK_SIZE override compiled into it is inert unless the variable is set, and this
# script does not set it: the L4 has never been measured at 32 and keeps SMPCores()'s 128.
# The tree is ABI-bound to the torch it was compiled against; main.py trains under pixi env
# `default` (torch 2.9.1), which is what -default was built for.
# with `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`.
#
# GEOMETRY -- copied from ../profiling/submit_maskfix_l4_3gpu.sh, not from the B200 script
# ----------------------------------------------------------------------------------------
# hpg-turin, gpu:l4:3, 1 node, 3 tasks x 16 CPU, 150 GB: the June 2026 production geometry
# (job 33954040) and the one clic_v6_maskfix_20260731-T152547 trained at. The overlays are
# stacked profile_noprof.yaml THEN profile_l4.yaml, and the order matters: profile_l4 supplies
# batch_size 256 and max_steps 800, and its max_steps must win over profile_noprof's 300.
# 800 rather than 300 because an L4 step is ~8x cheaper than a B200 one, so 300 steps would be
# mostly torch.compile warmup -- the same reason the maskfix L4 rerun used 800.
#
# WHY PAIRED, AGAIN
# -----------------
# The B200 arms scattered 32% across nodes (host) against 0.8% (device); node-to-node variance
# is larger than the effect being measured, so an unpaired comparison measures the node. Both
# arms in one allocation cancels that in the difference. This is only possible because the two
# arms differ by exactly one config block -- no checkout, no rebuild.
#
# Submit 3 of these (different nodes) and alternate ORDER so compile-cache warmth and any
# within-job drift cannot masquerade as an arm effect:
#   sbatch --export=ALL,ORDER=host_first   submit_paired_device_matcher_l4.sh
#   sbatch --export=ALL,ORDER=device_first submit_paired_device_matcher_l4.sh
#   sbatch --export=ALL,ORDER=host_first   submit_paired_device_matcher_l4.sh
#
# --name IS LOAD-BEARING, and this is why. utils/cli.py builds the run directory as
# "<name>_<YYYYmmdd-THHMMSS>", and clic_v6_cudamatch.yaml inherits `name: clic_v6_maskfix`
# from the config it was branched from -- the two arms are deliberately identical everywhere
# but the matcher block, name included. So two of these jobs that land in the SAME SECOND on
# different nodes resolve to the SAME directory on /blue, and the loser dies in setup with
#   RuntimeError: SaveConfigCallback expected .../config.yaml to NOT exist
# taking one arm of a paired allocation with it (seen on 40516656 vs 40516657). Namespacing
# the run by SLURM_JOB_ID makes that impossible. Both arms of a job share the name -- they are
# separated by their own timestamps, minutes apart -- so the arms still differ by exactly the
# matcher block, which is what the pairing rests on.
#
# Parse with ../profiling/parse_throughput.py --batch 768 (3 x 256 global batch, NOT the B200's
# 2048), which splits each log on its ARM: markers.
#
# See studies/b200_utilization/cuda_matcher/README.md §5 Phase 3 and §6.

#SBATCH --job-name=clic-paired-devmatch-l4
#SBATCH -p hpg-turin
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l4:3
#SBATCH --ntasks-per-node=3        # must match trainer.devices
#SBATCH --cpus-per-task=16         # production L4 request; what the HOST arm's 16-thread solve needs
#SBATCH --mem=150G
#SBATCH --time=03:00:00
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
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-default
PIXI_ENV=/blue/avery/m.mazza/projects/fastml/hepattn/.pixi/envs/clic

# Carried into the container. Harmless for the host arm, which never imports the extension;
# without the second one the import dies on CXXABI_1.3.15.
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

run_arm () {
  local arm="$1" cfg="$2"
  # Each arm gets its OWN Inductor/Triton cache. Without this the second arm inherits the
  # first arm's compiled kernels and looks faster for a reason that has nothing to do with
  # the matcher. Per-ARM, not per-rank: the 3 ranks share a node and a cache here exactly as
  # every 1-node 3x L4 run in this study has (33954040, 38461126, 38469247); the per-rank
  # workaround in studies/glow_jet_iqr/REPRODUCE.md was for multi-NODE runs.
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
    pixi run -e clic python main.py fit \
      --config "${cfg}" \
      --config configs/profile_noprof.yaml \
      --config configs/profile_l4.yaml \
      --trainer.devices=3 \
      --name="clic_v6_devmatch_l4_j${SLURM_JOB_ID}"

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
