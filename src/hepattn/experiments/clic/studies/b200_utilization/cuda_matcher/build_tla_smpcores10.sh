#!/bin/bash
# Build the extension with a compute-capability major-10 case in SMPCores(), so a B200 picks
# block size 32 by construction instead of needing TLA_BLOCK_SIZE=32 in the environment.
#
# WHY. The block-size win (2026-08-28, +4.4% end-to-end over three allocations) is currently
# switched on by an environment variable set in individual submit scripts. `device_solver: jv`
# comes from the config, `TLA_BLOCK_SIZE=32` comes from the environment, and the two are
# decoupled -- so any new submit script gets the device matcher at block 128 and silently loses
# the 4.4%. No error, no warning, just the old number. That is a fail-open default, and the
# fix recorded in NOTES.md is to make the *device* decide, which is what SMPCores() is for.
#
# The upstream switch covers compute capability majors 2-9. The B200 is major 10, so it fell
# through to `return 128; // Unknown device` -- the shipped block size was a fallback for
# hardware the library has never heard of, not a property of the machine.
#
# This tree keeps the TLA_BLOCK_SIZE override as well: it is what the sweep needs, and an
# explicit environment value still wins over the new default, so every existing submit script
# and every recorded measurement keeps behaving exactly as it did.
#
# Same build traps as build_tla_candidate.sh, all previously paid for:
#   FORCE_CUDA=1          setup.py gates on torch.cuda.is_available(), false on a login node;
#                         without it the build silently yields a CPU-only extension
#   TORCH_CUDA_ARCH_LIST  no GPU here to autodetect; 8.9 = L4, 10.0 = B200
#   CUDA_HOME=$PIXI_ENV   do NOT `module load cuda/12.8.1` -- it sets CUDA_HOME=/apps/..., which
#                         is not bind-mounted into the container, and nvcc is then not found
#
# Builds into a candidate tree, NOT over production. Verify on a GPU first:
#   sbatch submit_verify_smpcores10_b200.sh
set -euo pipefail
REPO=/blue/avery/m.mazza/projects/fastml/hepattn
PIXI_ENV=$REPO/.pixi/envs/clic
V=/blue/avery/m.mazza/projects/fastml/vendor
SRC=$V/torch-linear-assignment-default
VENDOR=$V/torch-linear-assignment-smpcores10

export TMPDIR=/var/tmp/
rm -rf "$VENDOR"
cp -a "$SRC" "$VENDOR"
rm -rf "$VENDOR/build"; find "$VENDOR" -name '*.so' -delete

# The source edit, applied here rather than kept as a patched copy of the tree, so that this
# script is the single reproducible statement of what changed. Both replacements assert their
# target text, so a drifting upstream source fails loudly instead of building something else.
python3 - "$VENDOR/src/torch_linear_assignment_cuda_kernel.cu" <<'PY'
import sys, pathlib

path = pathlib.Path(sys.argv[1])
src = path.read_text()

old_switch = """  case 9: // Hopper
    if (devProp.minor == 0) return 128;
    break;
  }
  return 128; // Unknown device"""
new_switch = """  case 9: // Hopper
    if (devProp.minor == 0) return 128;
    break;
  case 10: // Blackwell -- B100 / B200 / GB200
    // 32, and deliberately NOT the 128 FP32 cores a Blackwell SM actually has. This function
    // has exactly one caller, TLABlockSize() below, which uses the result as the CUDA launch
    // block size; 32 is what was measured to be fastest on a B200. The kernel runs one thread
    // per problem, so a smaller block spreads the same 10,240 threads over more SMs: at 128
    // the launch is 80 blocks for 148 SMs and 46% of the machine gets no work, at 32 it is
    // 320 blocks. Measured 170.64 -> 153.31 ms on the real cost tensor (job 40507155) and
    // +4.4% end to end across three allocations (jobs 40533246 / 40534109 / 40538140), with
    // bit-identical assignments at every block size from 32 to 512.
    //
    // Without this case the B200 fell through to `return 128; // Unknown device` and the 4.4%
    // had to be bought back with TLA_BLOCK_SIZE=32 in the environment of every submit script
    // -- which fails open, because a script that omits it silently gets the slow geometry.
    return 32;
  }
  return 128; // Unknown device"""
assert src.count(old_switch) == 1, "SMPCores() tail not found exactly once -- upstream source changed"
src = src.replace(old_switch, new_switch)

old_note = """// The sweep is worth doing because the upstream default is not actually tuned here. Upstream
// uses SMPCores(device_index), but on this hardware that returns its `128 // Unknown device`
// fallback: the B200 is compute capability major 10 and the switch above only covers majors
// 2-9. So the shipped block size is an arbitrary constant, not a property of the device."""
new_note = """// The sweep is worth doing because the upstream default is not tuned for this hardware.
// Upstream covers compute capability majors 2-9; the B200 is major 10 and used to fall through
// to `128 // Unknown device`, an arbitrary constant rather than a property of the device. The
// major-10 case added to SMPCores() above now returns the measured 32, so an unset
// TLA_BLOCK_SIZE already gives the fast geometry on a B200 and this override exists for
// sweeping and for hardware that has not been measured."""
assert src.count(old_note) == 1, "TLABlockSize() rationale comment not found exactly once"
src = src.replace(old_note, new_note)

path.write_text(src)
print("patched SMPCores(): case 10 -> 32")
PY

export APPTAINERENV_FORCE_CUDA=1
export APPTAINERENV_TORCH_CUDA_ARCH_LIST="8.9;10.0"
export APPTAINERENV_CUDA_HOME=$PIXI_ENV
export APPTAINERENV_MAX_JOBS=4

cd "$REPO"
apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic \
  bash -c "cd $VENDOR && python setup.py build_ext --inplace"

SO=$(find "$VENDOR/torch_linear_assignment" -name '_backend*.so' | head -1)
ELVES=$("$PIXI_ENV/bin/cuobjdump" --list-elf "$SO")
for arch in sm_89 sm_100; do
  case "$ELVES" in *"$arch".cubin*) ;; *) echo "ERROR: no $arch cubin" >&2; exit 1 ;; esac
done
# grep -c, not grep -q: under `set -o pipefail`, grep -q exits on the first match and hands
# strings a SIGPIPE, so the pipeline reports failure exactly when the string IS present.
NPATCH=$(strings -a "$SO" | grep -c TLA_BLOCK_SIZE || true)
[ "${NPATCH:-0}" -ge 1 ] || { echo "ERROR: TLA_BLOCK_SIZE override missing from binary" >&2; exit 1; }

export APPTAINERENV_PYTHONPATH=$VENDOR
export APPTAINERENV_LD_LIBRARY_PATH=$PIXI_ENV/lib
apptainer run --bind /blue/,/cmsuf/ "$REPO/pixi.sif" pixi run -e clic python -c "
import torch_linear_assignment._backend as b
assert b.has_cuda(), 'CPU-only build -- FORCE_CUDA did not take'
print('has_cuda: True')"

echo "OK: $VENDOR  (sm_89 + sm_100, TLA_BLOCK_SIZE override, SMPCores major-10 = 32)"
echo "Next: sbatch submit_verify_smpcores10_b200.sh   -- the default must now match block 32 on a GPU"
