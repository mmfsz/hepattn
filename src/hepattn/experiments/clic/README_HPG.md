# Running CLIC on HiPerGator (HPG)

HPG-specific instructions for training and evaluating the CLIC particle-flow model on
UF's HiPerGator cluster. For the model overview, evaluation flags, and data-format
details see [`README.md`](./README.md).

## Prerequisites

- **GPU partitions:**
  - `hpg-b200` — B200 192 GB (up to 4/node)
  - `hpg-turin` — L4 24 GB (up to 3/node)
- **Container + env:** the submit scripts run inside the pixi Apptainer image at the
  repo root (`pixi.sif`) via `apptainer run --nv ... pixi run -e clic ...`, so the `clic`
  pixi environment must be installed in the checkout you submit from. You do not need to
  enter the container by hand to submit.
- **CUDA module:** `module load cuda/12.8.1` (matches the container build; already in
  the submit scripts).
- **Data:** the CLIC ROOT files must be readable from the compute nodes.
  [`configs/hpg.yaml`](./configs/hpg.yaml) points at the group's copy on `/blue`:

  ```
  /blue/avery/m.mazza/projects/fastml/hepattn/data/clic/
  ├── train_clic_fix.root          # 12 GB
  ├── val_clic_fix.root            # 309 MB
  └── test_clic_common_infer.root  # 250 MB
  ```

  The scripts layer that overlay after your config, so `base.yaml` keeps the authors'
  paths and no model config needs editing to run here. If the data moves, change the
  overlay only.

## Submitting a training

All commands run from the experiment directory, and `slurm_logs/` must exist (sbatch
refuses to start if the `--output` directory is missing):

```shell
cd /blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic
mkdir -p slurm_logs
```

| Script | Partition | Nodes × GPU | Default batch | Global batch |
|---|---|---|---|---|
| `submit_training_hpg.sh` | `hpg-b200` | 1 × 4 B200 | 256/GPU | 1024 |
| `submit_training_hpg_1gpu.sh` | `hpg-b200` | 1 × 1 B200 | 1024 | 1024 |
| `submit_training_hpg_l4.sh` | `hpg-turin` | 1 × 3 L4 | 170/GPU, 2 accumulation steps | 1020 |
| `submit_training_hpg_l4_2nodes.sh` | `hpg-turin` | 2 × 3 L4 | 170/GPU | 1020 |

Every script takes the config as its first argument; anything after it passes straight
through to `main.py fit`, so no script needs editing to change a run:

```shell
sbatch submit_training_hpg_l4_2nodes.sh configs/base.yaml --name clic_paper
sbatch submit_training_hpg.sh configs/base_small.yaml --data.batch_size=512
```

The defaults reproduce the paper's global batch of 1024 on each hardware. The learning
rate is not batch-scaled, so change the global batch only when you mean to.

### Smoke test first

Before committing a multi-day allocation, run the same launch path for two tiny epochs
and check the SLURM log for the parameter count in `ModelSummary`, checkpoints under
`logs/<run>/ckpts/`, and `logs/<run>/csv_metrics/metrics.csv`:

```shell
sbatch --time=00:40:00 --job-name=clic-smoke submit_training_hpg_l4_2nodes.sh configs/base.yaml \
    --name clic_smoke --trainer.max_epochs=2 --trainer.limit_train_batches=20 --trainer.limit_val_batches=5
```

### Key rule: devices must match the allocation

`--ntasks-per-node` (SBATCH) must equal `--trainer.devices`, and `--nodes` must equal
`--trainer.num_nodes`. Each script sets both consistently; if you change the GPU count,
change both. Global batch = `num_nodes × devices × data.batch_size × accumulate_grad_batches`.

### What the launcher does

`srun` starts one task per GPU through [`run_task.sh`](./run_task.sh), which gives each
rank a private Triton/Inductor compile cache on node-local disk before starting the
container. Without it, the ranks race on the shared `$HOME/.triton` cache when the
`Compile` callback compiles the model, one dies with `Text file busy`, and the job hangs
on NCCL.

`COMET_MODE=offline` is exported because the compute nodes have no internet and no
`COMET_API_KEY`. The Comet archive lands in the run folder; the CLI also attaches a
`CSVLogger`, so train and val losses are always in `logs/<run>/csv_metrics/metrics.csv`
regardless of Comet. `hepattn.utils.loggers.MyCometLogger` is available for configs that
want the offline switch to happen automatically when no API key is set.

### Optional: solve the matching on the GPU

A single B200 is not saturated by the CLIC model, so the training step there is
host-bound and the Hungarian matching (device-to-host copy plus a threaded solve)
dominates it. The `Matcher` has an opt-in GPU solver for exactly that case: build it
once with `pixi run -e clic bash setup/build_torch_linear_assignment.sh`, export the
`PYTHONPATH` and `LD_LIBRARY_PATH` it prints in the submit script, and set
`device_solver: jv` on the matcher in the config. See the
[top-level README](../../../../README.md#optional-solving-the-matching-on-the-gpu).
Leave it off on the L4 nodes, which the model already keeps busy.

## Outputs & monitoring

- **Run folder:** `logs/<name>_<YYYYMMDD>-T<HHMMSS>/` with `ckpts/`, the resolved
  `config.yaml`, `csv_metrics/`, and the Comet offline archive. `<name>` is the
  config's `name:` field unless overridden with `--name`.
- **SLURM stdout/stderr:** `slurm_logs/slurm-<jobid>.<jobname>.out`.

```shell
squeue -u $USER                                        # queued/running jobs
tail -f slurm_logs/slurm-<jobid>.*.out                 # live training log
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it finishes
scancel <jobid>                                        # cancel
```

## Evaluation and validation

All on one L4. Each layers [`configs/eval.yaml`](./configs/eval.yaml) over the run's
config, which applies the evaluation rules from [`README.md`](./README.md): fp32,
`torch` attention, inference-mode data, and no `Compile` callback.

| Script | Purpose |
|---|---|
| `submit_eval_run.sh` | Parameterised eval: `RUN_DIR` and `CKPT_NAME` from the environment. Writes `<ckpt>__test.root` next to the checkpoint. |
| `submit_eval_hpg_l4.sh` | Same, with the run and checkpoint edited into the script. |
| `submit_eval_test_hpg_l4.sh` | One-batch smoke test of the eval path. |
| `submit_validate_run.sh` | Validation-only pass: re-score a checkpoint on the val set under the current code, into a fresh `logs/_val_<jobid>/`. Optional `CONFIG` scores it under another run's objective. |

```shell
sbatch --job-name=clic-eval-paper \
       --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<epoch=...ckpt> \
       submit_eval_run.sh
```

Produce the performance plots with
[`notebooks/performance.ipynb`](./notebooks/performance.ipynb). To compare with the
paper's figures use the `mpflow_proxy` output, which is what the paper plots.
