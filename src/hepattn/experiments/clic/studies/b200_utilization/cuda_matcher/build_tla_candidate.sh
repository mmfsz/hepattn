#!/bin/bash
# Build the candidate replacement for the production extension: DEFAULT ABI + both GPU
# architectures + the TLA_BLOCK_SIZE patch, in one tree.
#
# Why a candidate tree and not a rebuild in place: `vendor/torch-linear-assignment-default` is
# what every training run loads. Overwriting it before the 10% block-size win is confirmed
# inside a real training step would put an unvalidated binary on the production path.
#
# It combines what the two experiment trees each have half of:
#   -blocksize  the TLA_BLOCK_SIZE patch, but built against the `clic` ABI (torch 2.10) and
#               sm_100 only. main.py trains under `default` (torch 2.9.1), so it cannot load.
#   -multiarch  sm_89 + sm_100, correct ABI, but no patch.
#
# Only the .cu is taken from -blocksize. That tree's setup.py also carries a compat_include
# shim for a supposedly truncated header in the `clic` env -- that header turned out to be
# intact (verified against torch's own RECORD manifest), and `default` never had the problem,
# so the shim is deliberately NOT carried over.
#
# Traps, all previously paid for:
#   FORCE_CUDA=1          setup.py gates on torch.cuda.is_available(), false on a login node;
#                         without it the build silently yields a CPU-only extension
#   TORCH_CUDA_ARCH_LIST  no GPU here to autodetect; 8.9 = L4, 10.0 = B200
#   CUDA_HOME=$PIXI_ENV   do NOT `module load cuda/12.8.1` -- it sets CUDA_HOME=/apps/..., which
#                         is not bind-mounted into the container, and nvcc is then not found
set -euo pipefail
REPO=/blue/avery/m.mazza/projects/fastml/hepattn
PIXI_ENV=$REPO/.pixi/envs/default
V=/blue/avery/m.mazza/projects/fastml/vendor
SRC=$V/torch-linear-assignment-default
PATCHED=$V/torch-linear-assignment-blocksize
VENDOR=$V/torch-linear-assignment-candidate

export TMPDIR=/var/tmp/
rm -rf "$VENDOR"
cp -a "$SRC" "$VENDOR"
cp "$PATCHED/src/torch_linear_assignment_cuda_kernel.cu" "$VENDOR/src/"
rm -rf "$VENDOR/build"; find "$VENDOR" -name '*.so' -delete

export APPTAINERENV_FORCE_CUDA=1
export APPTAINERENV_TORCH_CUDA_ARCH_LIST="8.9;10.0"
export APPTAINERENV_CUDA_HOME=$PIXI_ENV
export APPTAINERENV_MAX_JOBS=4

cd "$REPO"
apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e default \
  bash -c "cd $VENDOR && python setup.py build_ext --inplace"

SO=$(find "$VENDOR/torch_linear_assignment" -name '_backend*.so' | head -1)
ELVES=$("$PIXI_ENV/bin/cuobjdump" --list-elf "$SO")
for arch in sm_89 sm_100; do
  case "$ELVES" in *"$arch".cubin*) ;; *) echo "ERROR: no $arch cubin" >&2; exit 1 ;; esac
done
# grep -c, not grep -q: under `set -o pipefail`, grep -q exits on the first match and hands
# strings a SIGPIPE, so the pipeline reports failure exactly when the string IS present.
NPATCH=$(strings -a "$SO" | grep -c TLA_BLOCK_SIZE || true)
[ "${NPATCH:-0}" -ge 1 ] || { echo "ERROR: patch missing from binary" >&2; exit 1; }

export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib
apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e default python -c "
import torch_linear_assignment._backend as b
from torch_linear_assignment import batch_linear_assignment
assert b.has_cuda(), 'CPU-only build -- FORCE_CUDA did not take'
print('has_cuda: True')"
echo "OK: $VENDOR  (sm_89 + sm_100, TLA_BLOCK_SIZE patch, default ABI)"
