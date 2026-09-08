#!/bin/bash

# Does the extension pick block size 32 on a B200 without being told to?
#
# `build_tla_smpcores10.sh` adds the missing compute-capability major-10 case to `SMPCores()`,
# so the +4.4% block-size geometry comes from the device instead of from a `TLA_BLOCK_SIZE=32`
# that every submit script has to remember. `strings` on the binary can only confirm the
# override is compiled in, not what the default resolves to on real hardware -- so this job
# times the solve with the variable unset against the two geometries it could have picked.
#
# Replays the already-dumped cost tensor: no training stage, minutes not hours.
#
# WHICH VENDOR TREE, AND WHICH ENV. The candidate tree is built against the `default` env
# (torch 2.9.1), because that is what main.py trains under and this fix is destined for the
# production path. So this runs `pixi run -e default`, NOT the `-e clic` the other offline
# benches use: a torch extension is ABI-bound to the torch it was compiled against and the
# wrong one dies on `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`.
# LD_LIBRARY_PATH must point at that env's lib or the import dies on CXXABI_1.3.15 instead.
# Both have to be passed through the container with the APPTAINERENV_ prefix.
#
# TLA_BLOCK_SIZE is deliberately NOT exported here. The unset default is the thing under test.
#
# Submit with:  sbatch submit_verify_smpcores10_b200.sh

#SBATCH --job-name=clic-verify-smpcores10
#SBATCH -p hpg-b200
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mmazza@fsu.edu
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
STUDY=$REPO/src/hepattn/experiments/clic/studies/b200_utilization/cuda_matcher
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-smpcores10
PIXI_ENV=$REPO/.pixi/envs/default

if [ ! -d "$VENDOR" ]; then
  echo "FAILED: candidate extension not found at $VENDOR; run build_tla_smpcores10.sh first" >&2
  exit 1
fi

COSTS=${COSTS:-$(ls -t "$STUDY"/phase1_logs/*/clic_b200_matcher_costs.pt 2>/dev/null | head -1)}
if [ -z "$COSTS" ] || [ ! -f "$COSTS" ]; then
  echo "FAILED: no cost dump found; run submit_real_cost_replay_b200.sh first" >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "git commit: $(git -C "$REPO" rev-parse HEAD)"
echo "vendor: $VENDOR"
echo "costs: $COSTS ($(du -h "$COSTS" | cut -f1))"
nvidia-smi

export TMPDIR=/var/tmp/
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib

cd "$STUDY"
srun apptainer run --nv --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e default \
  python verify_smpcores10.py --costs "$COSTS"

echo "Done!"
