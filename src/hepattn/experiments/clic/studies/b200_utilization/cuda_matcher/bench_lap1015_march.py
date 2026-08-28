"""How much does `-march` cost the host LAP solver, and where must the extension be built?

`CMakeLists.txt` compiles `lap1015` with `-march=native`, which targets whatever CPU runs the
build. On this cluster that is a trap: the login nodes are AMD EPYC 7702 (Zen 2, no AVX-512)
and the B200 nodes are Intel Emerald Rapids (AVX-512). An extension built on the login node
and run on a B200 node is tuned for the wrong machine -- and a rebuild done that way turned the
Phase-0 host solve from 712.8 ms into 1760.7 ms, which is what prompted this script.

Three arms, all measured in the same job on the node that actually trains:

- ``installed``   -- the `.so` currently in the environment, whatever it was built on.
- ``native``      -- rebuilt here, so `-march=native` means *this* node.
- ``x86-64-v3``   -- rebuilt against a fixed AVX2 baseline that every node in the cluster
                     supports. The portable option: one binary that is correct everywhere and
                     reproducible for anyone cloning the repo.

Each arm is built into its own throwaway package directory and exercised in a subprocess, so
the shared pixi environment is never modified. The gap between `native` and `x86-64-v3` is the
price of portability; the gap between `installed` and `native` is the price of building in the
wrong place.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
LAP_SRC = REPO / "src" / "lap1015" / "src"

# Mirrors CMakeLists.txt, minus -march which is the variable under test.
BASE_FLAGS = [
    "-Wall",
    "-O2",
    "-Wfatal-errors",
    "-fstrict-aliasing",
    "-std=c++17",
    "-fopenmp",
    "-flto",
    "-m64",
    "-mfpmath=sse",
]


def pybind11_include_dirs() -> list[Path]:
    """Where to find the pybind11 headers at run time.

    `pybind11` is a build-time requirement only, so it is absent from the installed environment.
    Torch vendors a copy and torch is a hard dependency of this project, which makes it the one
    include path that is always there.

    Raises:
        RuntimeError: If no pybind11 headers can be located.
    """
    try:
        import pybind11  # noqa: PLC0415

        return [Path(pybind11.get_include())]
    except ImportError:
        pass

    import torch  # noqa: PLC0415

    torch_include = Path(torch.__file__).parent / "include"
    if (torch_include / "pybind11" / "pybind11.h").exists():
        return [torch_include]

    raise RuntimeError("no pybind11 headers found: neither the pybind11 package nor torch's vendored copy")


def build_extension(march: str, out_dir: Path) -> Path:
    """Compile `main.cpp` for one `-march` into a standalone importable `lap1015` package."""
    pkg = out_dir / "lap1015"
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy(LAP_SRC / "lap1015" / "__init__.py", pkg / "__init__.py")

    includes = [f"-I{d}" for d in pybind11_include_dirs()]

    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        *BASE_FLAGS,
        f"-march={march}",
        *includes,
        f"-I{LAP_SRC}",
        f"-I{sysconfig.get_paths()['include']}",
        str(LAP_SRC / "main.cpp"),
        "-o",
        str(pkg / "_core.so"),
    ]
    print(f"  building -march={march} ...", flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


# Run in a subprocess so each arm imports its own extension into a clean interpreter.
BENCH_SNIPPET = """
import importlib.util, json, sys, time
from functools import partial
import numpy as np, torch
import hepattn.models.matcher as M

so = {path!r}
if so:
    # Load the freshly built extension by file path. `sys.path` is not enough: the editable
    # install registers a meta-path finder for `lap1015` that wins over any path entry, so an
    # `import lap1015` here would silently measure the installed build instead of this arm's.
    # The name must be `_core`: CPython derives the init symbol it looks for from the module
    # name, and the extension only exports `PyInit__core`.
    spec = importlib.util.spec_from_file_location("_core", so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    lap_late = partial(core.linear_sum_assignment, omp=False, eps=True)
    releases_gil = bool(getattr(core, "releases_gil", False))
    M.SOLVERS["lap1015_late"] = lambda cost: lap_late(cost)
else:
    import lap1015
    releases_gil = bool(lap1015.releases_gil)

B, Q, T = {batch}, 150, 50
g = torch.Generator().manual_seed(0)
costs = torch.rand(B, Q, T, generator=g)
# Minimum of two targets: single-row problems trip a rare lap1015 edge case that falls back to
# scipy, which would price a solver bug into a timing comparison.
lengths = torch.randint(2, T + 1, (B,), generator=g)
valid = torch.arange(T)[None, :] < lengths[:, None]

out = {{"releases_gil": releases_gil}}
for solver in ("lap1015_late", "scipy"):
    for n in (1, 4, 8, 16):
        m = M.Matcher(default_solver=solver, adaptive_solver=False,
                      parallel_solver=(n > 1), n_jobs=n)
        m(costs, valid, None)                      # warm the thread pool
        ts = [ ]
        for _ in range({reps}):
            t = time.perf_counter(); m(costs, valid, None); ts.append(time.perf_counter() - t)
        out[f"{{solver}}:{{n}}"] = min(ts)
print("RESULT " + json.dumps(out))
"""


def run_arm(label: str, path: str, batch: int, reps: int) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(BENCH_SNIPPET).format(path=path, batch=batch, reps=reps)],
        capture_output=True,
        text=True,
        check=False,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if line is None:
        print(f"  {label}: FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", flush=True)
        return {}
    return json.loads(line[len("RESULT ") :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=2048 * 5, help="problems per solve; the CLIC B200 geometry")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--marches", nargs="*", default=["native", "x86-64-v3"])
    args = parser.parse_args()

    cpu = next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines() if "model name" in ln)
    print(f"CPU: {cpu}")
    print(f"geometry: {args.batch} problems of 150 preds x <=50 targets\n")

    arms = {"installed": ""}
    tmp = Path(tempfile.mkdtemp(prefix="lap1015_march_"))
    try:
        for march in args.marches:
            arms[march] = str(build_extension(march, tmp / march) / "lap1015" / "_core.so")

        results = {}
        for label, path in arms.items():
            print(f"timing {label} ...", flush=True)
            results[label] = run_arm(label, path, args.batch, args.reps)

        header = f"\n{'arm':12s} {'gil':>5s} " + " ".join(f"{f'{s}:{n}':>16s}" for s in ("lap1015_late", "scipy") for n in (1, 4, 8, 16))
        print(header)
        print("-" * len(header))
        for label, r in results.items():
            if not r:
                print(f"{label:12s} {'--':>5s}  (failed)")
                continue
            cells = " ".join(f"{r.get(f'{s}:{n}', float('nan')):16.3f}" for s in ("lap1015_late", "scipy") for n in (1, 4, 8, 16))
            print(f"{label:12s} {r['releases_gil']!s:>5s} {cells}")

        print(
            "\nRead it as: `installed` vs `native` is the cost of having built the extension on the\n"
            "wrong CPU; `native` vs `x86-64-v3` is the cost of a portable binary that is safe to\n"
            "ship to every node type. Compare both against the scipy column, which needs no build."
        )
        # Stamped with the host: the whole point of this benchmark is that the answer depends
        # on which machine it ran on, so an unstamped filename would be actively misleading.
        out = Path.cwd() / f"lap1015_march_{socket.gethostname().split('.')[0]}.json"
        out.write_text(json.dumps({"cpu": cpu, "host": socket.gethostname(), "results": results}, indent=2))
        print(f"\nwrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
