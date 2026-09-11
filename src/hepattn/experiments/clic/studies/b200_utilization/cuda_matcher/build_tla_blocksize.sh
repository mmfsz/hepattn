#!/bin/bash
#
# Build the block-size-sweep copy of torch-linear-assignment for the B200.
#
# WHAT THIS BUILDS
# ----------------
# /blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-blocksize is a *copy* of
# the `clic`-env vendored tree, patched so the CUDA kernel launch block size can be overridden
# at run time with the TLA_BLOCK_SIZE environment variable (multiple of 32 in [32, 1024];
# unset/invalid falls back to the upstream SMPCores() default, unchanged). Reading the variable
# on every call means one process can sweep several block sizes without a rebuild.
#
# WHY A COPY AND NOT AN IN-PLACE PATCH
# ------------------------------------
# Two built trees are PRODUCTION and must not be touched:
#   vendor/torch-linear-assignment          -- built against pixi env `clic`    (torch 2.10),
#                                              used by the offline benchmarks
#   vendor/torch-linear-assignment-default  -- built against pixi env `default` (torch 2.9.1),
#                                              used by training runs
# The extension is ABI-bound to the torch it was compiled against; loading the wrong one dies
# with `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv`. Keeping three separate trees
# reached via PYTHONPATH (never pip-installed into the shared env, same discipline as the
# -march arms) is what keeps those two from colliding.
#
# TRAPS -- all of these have bitten this package before
# -----------------------------------------------------
# 1. FORCE_CUDA=1 IS MANDATORY. setup.py gates the CUDAExtension on
#    `torch.cuda.is_available()`, which is False on a login/compile node with no GPU. Without
#    the flag the build SILENTLY SUCCEEDS and produces a CPU-only extension -- no error, just a
#    backend whose has_cuda() is False. The verification step below is the only thing that
#    catches this, so do not skip it.
# 2. TORCH_CUDA_ARCH_LIST=10.0 -- the B200 is sm_100. There is no GPU on the build host, so
#    torch cannot auto-detect the arch and would otherwise emit its default (much older) list.
# 3. DO NOT `module load cuda/...`. The module exports CUDA_HOME=/apps/compilers/cuda/12.8.1,
#    apptainer carries that variable into the container, but /apps is NOT bind-mounted -- so the
#    build fails with `nvcc not found at '/apps/compilers/cuda/12.8.1/bin/nvcc'`. The `clic`
#    pixi env ships its own nvcc 12.8.93 (cuda-nvcc), which is what torch picks up on its own.
#    CUDA_HOME is set explicitly below so the toolchain is pinned rather than auto-detected.
# 4. LD_LIBRARY_PATH must point at $PIXI_ENV/lib for the *import* check, or it dies on
#    `CXXABI_1.3.15` (the container's system libstdc++ is older than the env's).
# 5. Inside apptainer, environment variables have to be passed with the APPTAINERENV_ prefix.
# 6. THE `clic` ENV SHIPS A TRUNCATED TORCH HEADER. Not previously documented -- found while
#    writing this script. site-packages/torch/include/ATen/ops/ctc_loss_ops.h in the clic env's
#    torch 2.10.0+cu128 is 0 BYTES. It is the only zero-byte file among the 9247 headers in that
#    include tree, and it is hardlinked to an equally truncated copy in the uv archive cache
#    (/blue/avery/m.mazza/.cache/pixi/uv-cache/archive-v0/Q5jRu3eya6J3nfoxwW4NA), so the damage
#    is in the cached wheel extraction, not just the env. Every torch C++/CUDA extension built
#    against this env therefore dies with:
#       ATen/ops/ctc_loss.h:29: error: 'at::_ops::ctc_loss_IntList' has not been declared
#    (ATen/Functions.h pulls in ctc_loss.h, which needs the declarations from ctc_loss_ops.h.)
#    Workaround, contained entirely in the copied tree: $VENDOR/compat_include/ holds a good
#    copy of the header and setup.py passes that directory as the extension's include_dirs.
#    torch's CUDAExtension appends its own include paths *after* the caller's, so compat_include
#    shadows the empty file and nothing else. The header is byte-identical across torch 2.7.0,
#    2.8.0 and 2.9.1, so the 2.9.1 copy from the `default` env is a faithful stand-in; it is
#    declarations only and this extension never calls at::ctc_loss.
#    The proper fix is to repair the clic env / uv cache, which is deliberately NOT done here:
#    that env is shared with production training and benchmark runs.
#
# The compile is a single .cpp plus a single .cu; on a 1-core node it takes roughly 15 minutes,
# almost all of it the .cu. Run it on a node with a few cores if you can.
#
# Usage:  bash build_tla_blocksize.sh
# Re-runnable from scratch: it re-copies the tree from the pristine `clic` vendor tree,
# re-applies the patch if the copy does not already carry it, and rebuilds.

set -euo pipefail

REPO=/blue/avery/m.mazza/projects/fastml/hepattn
PIXI_ENV=$REPO/.pixi/envs/clic
VENDOR_SRC=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment
VENDOR=/blue/avery/m.mazza/projects/fastml/vendor/torch-linear-assignment-blocksize

export TMPDIR=/var/tmp/

# --- 1. copy the pristine clic tree, then strip anything prebuilt -----------------------------
# Removing build/ and the shipped .so guarantees the artefact that comes out is the one this
# script produced and not a leftover from the production tree.
if [ ! -d "$VENDOR" ]; then
  cp -a "$VENDOR_SRC" "$VENDOR"
fi
rm -rf "$VENDOR/build"
find "$VENDOR" -name '*.so' -delete

# --- 2. patch the block size to a runtime override --------------------------------------------
# Idempotent: only applied if the marker is absent.
KERNEL=$VENDOR/src/torch_linear_assignment_cuda_kernel.cu
if ! grep -q TLABlockSize "$KERNEL"; then
  echo "ERROR: $KERNEL is not patched (no TLABlockSize)." >&2
  echo "       Re-apply the TLA_BLOCK_SIZE patch to solve_cuda_batch before building." >&2
  exit 1
fi

# --- 2b. work around the truncated torch header (trap 6) --------------------------------------
# Rebuild compat_include/ from a known-good copy every time, so a fresh `cp -a` of the pristine
# tree (which has neither the directory nor the setup.py hook) still builds.
GOOD_HEADER=$REPO/.pixi/envs/clic/lib/python3.12/site-packages/torch/include/ATen/ops/ctc_loss_ops.h
BAD_HEADER=$PIXI_ENV/lib/python3.12/site-packages/torch/include/ATen/ops/ctc_loss_ops.h
if [ ! -s "$BAD_HEADER" ]; then
  echo "NOTE: $BAD_HEADER is empty -- applying the compat_include shim (trap 6)."
  if [ ! -s "$GOOD_HEADER" ]; then
    echo "ERROR: no good copy of ctc_loss_ops.h at $GOOD_HEADER" >&2
    exit 1
  fi
  mkdir -p "$VENDOR/compat_include/ATen/ops"
  cp "$GOOD_HEADER" "$VENDOR/compat_include/ATen/ops/ctc_loss_ops.h"
  chmod 644 "$VENDOR/compat_include/ATen/ops/ctc_loss_ops.h"
fi
if ! grep -q compat_include "$VENDOR/setup.py"; then
  echo "ERROR: $VENDOR/setup.py does not pass compat_include as include_dirs." >&2
  echo "       Without it the build fails on the truncated ctc_loss_ops.h (trap 6)." >&2
  exit 1
fi

# --- 3. build ---------------------------------------------------------------------------------
# NOTE: no `module load cuda` -- see trap 3.
export APPTAINERENV_FORCE_CUDA=1              # trap 1
export APPTAINERENV_TORCH_CUDA_ARCH_LIST=10.0 # trap 2
export APPTAINERENV_CUDA_HOME=$PIXI_ENV       # trap 3
export APPTAINERENV_MAX_JOBS=4

cd "$REPO"
apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic \
  bash -c "cd $VENDOR && python setup.py build_ext --inplace"

# --- 4. verify the extension is a CUDA build, not a silent CPU fallback ------------------------
export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib  # trap 4

apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic python -c "
import torch_linear_assignment._backend as backend
from torch_linear_assignment import batch_linear_assignment
print('backend:', backend.__file__)
print('has_cuda:', backend.has_cuda())
print('batch_linear_assignment:', batch_linear_assignment)
assert backend.has_cuda(), 'CPU-ONLY BUILD -- rebuild with FORCE_CUDA=1 (trap 1)'
print('OK')
"

echo
echo "Built: $VENDOR"
echo "Use it with:"
echo "  export PYTHONPATH=$VENDOR"
echo "  export LD_LIBRARY_PATH=$PIXI_ENV/lib"
echo "  export TLA_BLOCK_SIZE=256   # multiple of 32 in [32,1024]; unset = upstream default"
