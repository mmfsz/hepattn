# Running CLIC on HiPerGator (HPG)

HPG-specific instructions for training and evaluating the CLIC particle-flow model on
UF's HiPerGator cluster. For the model overview, evaluation flags, and data-format
details see [`README.md`](./README.md).

## Prerequisites

- **GPU partition:** `hpg-turin`, L4 24 GB, up to 3 per node. The recipes below reproduce
  the paper's global batch of 1024 on L4s.
- **Environment:** the submit scripts run `pixi run --frozen python`, so the pixi
  environment must already be installed in the checkout you submit from
  (`pixi install --frozen` from the repo root). `--frozen` means the lock file is used
  as-is: compute nodes have no internet.
- **Data:** the CLIC ROOT files must be readable from the compute nodes.
  [`configs/hpg.yaml`](./configs/hpg.yaml) points at the group's copy on `/blue`:

  ```
  /blue/avery/m.mazza/projects/fastml/hepattn/data/clic/
  ├── train_clic_fix.root          # 12 GB
  ├── val_clic_fix.root            # 309 MB
  └── test_clic_common_infer.root  # 250 MB
  ```

  The scripts append that overlay after your config, so `base.yaml` keeps the authors'
  paths and nothing in a model config needs editing to run here. If the data moves,
  change the overlay only.

## Submitting a training

Everything is submitted **from the experiment directory**, and the scripts `cd` back to
it on the compute node (`$SLURM_SUBMIT_DIR`):

```shell
cd /path/to/hepattn/src/hepattn/experiments/clic
mkdir -p slurm_logs        # sbatch refuses to start if the --output directory is missing
```

| Script | Nodes × GPU | Global batch | Config source |
|---|---|---|---|
| `submit_training_hpg_l4_2nodes.sh` | 2 × 3 L4 | 170 × 6 = 1020 | argument (`$1`) |
| `submit_training_hpg_l4.sh` | 1 × 3 L4 | 170 × 3 × 2 accumulation = 1020 | argument (`$1`) |

The config is the first argument; anything after it passes straight through to
`main.py fit`, so no script needs editing to change a run:

```shell
sbatch submit_training_hpg_l4_2nodes.sh configs/base.yaml --name clic_paper
sbatch submit_training_hpg_l4.sh configs/base.yaml --name clic_paper_1node
```

Add `--mail-type=END,FAIL --mail-user=<you>` to the `sbatch` line for e-mail
notifications.

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
The learning rate is not batch-scaled, so keep the global batch at the paper's 1020 unless
you mean to change it.

### What the launcher does

`srun` starts one task per GPU through [`run_task.sh`](./run_task.sh), which gives each
rank a private Triton/Inductor compile cache on node-local disk before running Python in
the pixi environment. Without it, the ranks race on the shared `$HOME/.triton` cache when
the `Compile` callback compiles the model, one dies with `Text file busy`, and the job
hangs on NCCL.

`COMET_MODE=offline` is exported because the compute nodes have no internet and no
`COMET_API_KEY`. The Comet archive lands in the run folder; the CLI also attaches a
`CSVLogger`, so train and val losses are always in
`logs/<run>/csv_metrics/metrics.csv` regardless of Comet.

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

## Evaluation

[`submit_eval_hpg_l4.sh`](./submit_eval_hpg_l4.sh) runs `main.py test` on one L4 from a
run's checkpoint and writes `<ckpt>__test.root` next to it. It layers
[`configs/eval.yaml`](./configs/eval.yaml) over the run's config, which applies the
evaluation rules from [`README.md`](./README.md): fp32, `torch` attention, inference-mode
data, and no `Compile` callback.

```shell
sbatch --job-name=clic-eval-paper \
       --export=ALL,RUN_DIR=logs/<run_folder>,CKPT_NAME=<epoch=...ckpt> \
       submit_eval_hpg_l4.sh
```

Produce the performance plots with
[`notebooks/performance.ipynb`](./notebooks/performance.ipynb). To compare with the
paper's figures use the `mpflow_proxy` output, which is what the paper plots.
