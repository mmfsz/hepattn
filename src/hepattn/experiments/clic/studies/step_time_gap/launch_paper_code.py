"""Run the paper-tag CLIC entry point inside ANOTHER worktree's pixi environment.

Both worktrees install ``hepattn`` editable through scikit-build-core, which registers a
``ScikitBuildRedirectingFinder`` on ``sys.meta_path`` that maps ``hepattn`` to the
environment's own checkout regardless of ``sys.path``. This launcher strips only the
``hepattn`` entries from that finder (its ``lap1015._core`` wheel mapping stays), puts the
paper checkout's ``src`` first on ``sys.path``, prints what actually got imported, and then
runs ``main.py`` with the remaining arguments exactly as ``python main.py`` would.

Lightning 2.5.2 (head's environment) rewrote ``CometLogger`` over ``comet_ml.start`` and
renamed its arguments, so the paper's base.yaml (``project_name``, plus the ``experiment_name``
and ``save_dir`` links in ``hepattn.utils.cli``) does not parse there. When the installed
logger lacks ``project_name``, the launcher swaps ``lightning.pytorch.loggers.CometLogger``
for a shim that accepts the paper's names and forwards them, offline.

Usage (from src/hepattn/experiments/clic of the paper worktree):
    PAPER_SRC=<paper>/src DROP_SRC=<env repo>/src python studies/step_time_gap/launch_paper_code.py fit ...
"""

import os
import runpy
import sys

paper_src = os.environ["PAPER_SRC"]  # the checkout to run: <repo>/src
drop_src = os.environ.get("DROP_SRC", "")

for finder in sys.meta_path:
    if type(finder).__name__ == "ScikitBuildRedirectingFinder":
        for attr in ("known_source_files", "submodule_search_locations"):
            mapping = getattr(finder, attr)
            for key in [k for k in mapping if k.split(".")[0] == "hepattn"]:
                mapping.pop(key)
        finder.pkgs = [p for p in finder.pkgs if p.split(".")[0] != "hepattn"]

sys.path = [p for p in sys.path if p != drop_src]
sys.path.insert(0, paper_src)

import inspect  # noqa: E402

import comet_ml  # noqa: E402
import lightning  # noqa: E402
import lightning.pytorch.loggers as pl_loggers  # noqa: E402
import torch  # noqa: E402

import hepattn  # noqa: E402

print(f"hepattn from: {hepattn.__file__}")
print(f"torch {torch.__version__}, lightning {lightning.__version__}, comet_ml {comet_ml.__version__}")
print(f"python prefix: {sys.prefix}")

# Only when the paper config meets a foreign lightning (SHIM_COMET=1, set by the submit script
# when the environment is not the checkout's own): main's config names its own MyCometLogger.
if os.environ.get("SHIM_COMET") == "1" and "project_name" not in inspect.signature(pl_loggers.CometLogger.__init__).parameters:

    class PaperNamesCometLogger(pl_loggers.CometLogger):
        def __init__(self, project_name: str = "hepattn-clic", experiment_name: str | None = None, save_dir: str | None = None, **kwargs):
            super().__init__(project=project_name, name=experiment_name, offline_directory=save_dir, online=False, **kwargs)

    # jsonargparse round-trips the class by module + name; resolve it back to the shim.
    PaperNamesCometLogger.__module__ = "lightning.pytorch.loggers"
    PaperNamesCometLogger.__name__ = PaperNamesCometLogger.__qualname__ = "CometLogger"
    pl_loggers.CometLogger = PaperNamesCometLogger
    print("CometLogger shimmed for this lightning version")

sys.argv = ["main.py", *sys.argv[1:]]
runpy.run_path(os.path.join(os.path.dirname(paper_src), "src/hepattn/experiments/clic/main.py"), run_name="__main__")
