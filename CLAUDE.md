# hepattn — paper-tag line (`clic-paper-main`)

End-to-end ML for particle physics reconstruction: a single encoder–decoder
transformer (MaskFormer-style) applied to many reconstruction tasks — hit
filtering, tracking, vertexing, particle flow — across different detectors. See
[README.md](README.md) for the paper list and task overview.

This checkout is the branch `clic-paper-main`, based on the tag `clic-paper`
(fb90390, the code of the GLOW paper, arXiv:2508.20092). The group moved its
development here on 2026-09-11 because the model at upstream head performs worse
than the paper's (its jet-energy resolution degrades with jet energy instead of
improving; the responsible post-paper change was never bisected). Everything
built on the head-based `main` is being ported here feature by feature.

## Repository layout

- `src/hepattn/models/` — core building blocks: `encoder.py`, `decoder.py`,
  `maskformer.py`, `attention.py`, `task.py`, `matcher.py`, `loss.py`, etc.
- `src/hepattn/experiments/` — one subdirectory per experiment/detector
  (`clic`, `cld`, `itk`, `pixel`, `tide`, `trackml`). Each holds its own configs,
  data readers, and run instructions.
- `src/hepattn/callbacks/`, `src/hepattn/flex/`, `src/hepattn/utils/` —
  training callbacks, FlexAttention helpers, and shared utilities.
- `tests/` — pytest suite (see markers below).

## Environment & common commands

The project uses **pixi**. At the tag `pyproject.toml` defines only the `default`
environment. The HPG submit scripts expect the `clic` environment (default plus the
CLIC analysis packages) inside the Apptainer image `pixi.sif`, the way `main` runs;
that environment is being ported from `main` and is not installed in this worktree
yet, so nothing launches from here until it is.

- Run a command: `pixi run -e clic <cmd>`; enter the env: `pixi shell -e clic`
- Re-solve the lock only inside the container (`apptainer exec pixi.sif ... pixi lock`):
  the host pixi writes lock format v7, upstream and the container use v6.
- Run all tests: `pytest`
- CI-equivalent tests (no GPU, no external data): `pytest -m 'not gpu and not requiresdata'`
- Lint & format before committing: `ruff check --fix .` then `ruff format .`

## Terminology

- **constituent** — input entity fed to the encoder/decoder (e.g. a detector hit).
- **object** — reconstructed output from the decoder (e.g. a particle/track).
- **input** / **input_object** — generic term for any module input.
- **output** — generic term for any module output.

## Documentation & instruction files

- [README.md](README.md) — setup, environments, running tests, contributing.
- Per-experiment READMEs under `src/hepattn/experiments/*/` — how to run each experiment.
- [CLIC on HiPerGator](src/hepattn/experiments/clic/README_HPG.md) — submit scripts,
  data location, evaluation on the cluster.
- The CLIC particle-flow explainer lives on `main` only, at
  `/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/EXPLAINER.md`.
  It was written against head code; the architecture, task heads, matching and
  evaluation pipeline it describes are the same here, the code details may differ.

## Development workflow: what goes upstream and what stays here

`clic-paper-main` holds two kinds of work, and the rule that keeps them apart is:

1. **Anything under `src/hepattn/experiments/*/studies/` never goes upstream.** A
   study directory owns everything that exists only for it: its notes, scripts,
   figures, and its configs (`studies/<study>/configs/`). The shared `configs/`
   directory holds only the models and run modes that are not study-specific.
2. **Everything outside `studies/` is written to upstream quality**, with no
   study references, job numbers or personal paths, and is committed on a
   feature branch first (cut from the tag `clic-paper`, or from Lindsey's
   `clic-paper-main` once it exists, or from an open PR branch), then merged
   into `clic-paper-main`. Study commits go straight onto `clic-paper-main`.
   Never commit shared-code changes directly on `clic-paper-main`; that is what
   forces cherry-picking later.
3. **Recorded run directories under `logs/` are evidence.** Never rewrite them,
   even for a rename that would keep them loadable; bookmark the old code with a
   branch instead.

PRs go to Lindsey's fork (`lgray` remote), against his `clic-paper-main`, never
against `main`: a tag-based branch diffed against head shows the undoing of a
hundred upstream commits.

To see what is currently pushable:

```shell
git diff --stat clic-paper clic-paper-main -- . ':!**/studies/**'
```

If that shows only things you would send upstream, `clic-paper-main` is in a clean state.

## Previous results (head-based work on `main`)

The studies done on head code are not ported here; they stay on `main`, checked out
in the worktree next door. Read them there:

- `/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/`
  - `glow_jet_iqr/` — why head's jet-energy IQR rises with energy while the paper's
    falls; task 5 and 7 are the paper-tag reproductions (full and 0.82M-parameter
    small model) that this branch is built on. Compare with the paper using the
    `mpflow_proxy` output only.
  - `b200_utilization/` — throughput profiling, the mask-loss normalisation bug, the
    CUDA (Jonker-Volgenant) matcher study and its shadow-matcher validation.
  - `model_size/` — shrinking the head-v7 model for FPGA deployment; its factorial
    designs and the reproducibility-spread measurement are on head code and must be
    redone here.
- The paper-tag reproduction runs themselves (checkpoints, csv metrics, eval ROOT
  files) are in the paper clone:
  `/home/m.mazza/blue/projects/fastml/hepattn-clic-paper/src/hepattn/experiments/clic/logs/`.

Both worktrees share one repository, so `git log main` and `git show main:<path>`
work from here.
