# Reproducing our CLIC / GLOW results

Two result sets come out of this study, trained from **two different code
versions**:

| | **A — current model** ("newer" results) | **B — paper reproduction** |
|---|---|---|
| Code | `lgray/hepattn` @ **`1df05cc`** | `hepattn` @ tag **`clic-paper`** = **`fb90390`** |
| Params | 10,126,115 | 12,065,291 |
| Config to use | [`config_v6_baseline.yaml`](config_v6_baseline.yaml) | [`config_paper_tag.yaml`](config_paper_tag.yaml) |
| Reference run | `logs/clic_v6_20260605-T113014`, best `epoch=195-val_loss=3.78800` | `logs/clic_paper_20260715-T173417`, best `epoch=194-val_loss=4.01237` |
| Jet-E IQR vs E | **rises** (0.075 → 0.096) | **falls** (0.072 → 0.043, = paper Fig. 4) |

The two configs above are the *resolved* configs Lightning wrote for those runs — they
are the exact, complete settings used, not templates. Everything below just says which
code they must be run against.

## Data

Same dataset in both cases (1,004,891 train / 20,000 test events), from the GLOW authors
(originally `/share/gpu1/syw24/dmitrii_clic/`). On HiPerGator:

```
/cmsuf/data/store/user/mmazza/hepattn_clic_data/
├── train_clic_fix.root            # 12 GB
├── val_clic_fix.root              # 309 MB
└── test_clic_common_infer.root    # 250 MB   (test/inference file)
```

Off-cluster you need these files from the authors; then edit
`data.{train,valid,test}_path` in the config.

---

## A — Current model (10.1M)

Clone Lindsey's fork and **pin the commit** — `main` keeps moving, and the CLIC model was
refactored after the paper:

```shell
git clone git@github.com:lgray/hepattn.git
cd hepattn && git checkout 1df05cc     # "Implement k-Max-DeepLab (#251)", 2026-02-27
```

At the time of writing this *is* `lgray/main`'s tip, and its committed code is
byte-identical to `samvanstroud/hepattn` upstream. Our own fork
(`mmfsz/hepattn` @ `6ee4c60`) is the same commit plus HPG SLURM scripts, docs, and a
CSVLogger fix — **no model, loss, or matcher changes**, so either clone reproduces the
numbers.

Train:

```shell
cd src/hepattn/experiments/clic
python main.py fit --config <path-to>/config_v6_baseline.yaml \
    --trainer.devices=3 --trainer.num_nodes=1
```

Note that `batch_size` is **per GPU**, so global batch =
`num_nodes × devices × batch_size` (ours: 3 × 256 = 768). Keep the global batch fixed if
you train on a different GPU count — the learning rate is not batch-scaled.

Evaluate the best checkpoint (writes `<ckpt>__test.h5` and `__test.root` beside it):

```shell
python main.py test \
    --config logs/<run-folder>/config.yaml \
    --config configs/eval.yaml \
    --trainer.devices=1 --trainer.num_nodes=1 \
    --ckpt_path logs/<run-folder>/ckpts/<best>.ckpt
```

Passing both configs is the standard eval workflow (see the experiment
[`README.md`](../../README.md)): the run's `config.yaml` defines the model, and
`configs/eval.yaml` layers the inference settings on top —

- `data.is_inference=true` — matches `test_clic_common_infer.root`, which already carries
  correct truth particles; without it the loader applies the training-time relabelling of
  trackless particles and you score against the wrong truth.
- `precision=32-true`, `matmul_precision=highest`, encoder `attn_type=torch` — inference
  in full fp32 on the plain attention path.
- callbacks — **replaces** the training list, adding `PflowPredictionWriter` (which writes
  the `.h5`/`.root`; without it `test` prints metrics and saves nothing) and dropping
  `Compile`.

`batch_size` at test time affects only speed and memory, not the predictions — set it to
whatever fits the eval GPU.

---

## B — Paper reproduction (12.1M)

This is the code the GLOW paper (arXiv:2508.20092) was actually written against —
**100 commits before** the version in A.

```shell
git clone git@github.com:samvanstroud/hepattn.git hepattn-clic-paper
cd hepattn-clic-paper && git checkout clic-paper      # = fb90390, "update clic (#103)"
```

Four small local patches were needed to run it on HPG. If you hit the same issues, apply
them with [`05_reproduce_paper_tag/paper_tag_hpg.patch`](05_reproduce_paper_tag/paper_tag_hpg.patch):

1. `configs/base.yaml` — data paths → our copies; `batch_size 512 → 170`
   (170 × 6 L4 = global 1020 ≈ the paper's 512 × 2 A100 = 1024).
2. `utils/cli.py` — `mkdir` the run dir before the logger is built (offline Comet
   otherwise crashes at launch).
3. `pyproject.toml` — pixi cannot build `hepattn`, so build the environment first and
   install `hepattn` into it by hand:
   ```shell
   pixi install                                        # dependencies only
   pixi run pip install -e . --ignore-requires-python   # hepattn itself
   pixi run --frozen python main.py fit ...             # --frozen from here on
   ```
   The patch just comments out the `hepattn = { path = ".", editable = true }` line under
   `[tool.pixi.pypi-dependencies]`. Why each piece:
   - **Commenting it out** — pixi ≥0.70 passes a build config-setting that `hepattn`'s
     build backend (`scikit-build-core`) rejects. While `hepattn` is in the manifest that
     failure aborts the whole `pixi install`, leaving you with no environment; removing it
     lets the dependencies install, and `pip` then does the same editable build without
     the offending flag.
   - **`--ignore-requires-python`** — the tag pins `requires-python = "== 3.12"`, which
     means exactly 3.12.0, and the environment ships 3.12.11. Nothing is actually
     incompatible; the flag skips the check.
   - **`--frozen`** — pixi doesn't know about the hand-installed `hepattn` and would
     reconcile it away on a later `pixi run`. `--frozen` uses the env and lock as they are
     on disk (and needs no network, which compute nodes lack anyway).
4. Multi-rank only: give each rank its own Triton/Inductor cache
   (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` under `/var/tmp/...$SLURM_PROCID`) —
   otherwise ranks race on shared `$HOME/.triton` and die with `Text file busy`. See
   `run_task.sh` in the clone.

Then train and evaluate exactly as in A, using `config_paper_tag.yaml`
(200 epochs, `bf16-mixed`, `flash-varlen`, 3 devices × 2 nodes). `configs/eval.yaml` in
that clone is the same override set as A's. `ModelSummary` should print
**12,065,291 parameters** — a good early check that you are on the right code.

---

## Comparing against the paper's figures

**Use the proxy output, not the regression output.** The paper's Fig. 3/4 GLOW curves are
`network_type: "mpflow_proxy"` with `ind_threshold: 0.65` (as in the tag's own
`notebooks/performance.ipynb`) — the incidence-weighted proxy kinematics, *not* the
regression-refined `mpflow_*` branches. Both branch sets are in every eval `.root`, so
this is a plotting choice, not a re-run. Plotting the wrong one shifts the median by
~0.03 and the IQR by 0.01–0.02.

Jet definition used throughout: gen-kt R = 0.7, at most 2 jets with pT > 10 GeV,
truth-match dR < 0.1.

