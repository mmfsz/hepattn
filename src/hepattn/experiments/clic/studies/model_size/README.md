# Model-size study on the paper tag

Shrinking the CLIC particle-flow model toward an FPGA-sized network, **starting from the paper's
code** (tag `clic-paper`) instead of upstream head. The head-based version of this study lives on
`main`: `/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/studies/model_size/`
(`STUDY.md` there holds the plan, the judging protocol and rounds 1–3). Its arm *definitions* are
reused here; its *numbers* are not comparable to anything in this directory, because the code
differs (SwiGLU vs SiLU feed-forwards, no q/k/v norms, the paper's incidence head, …) and so does
the training-to-training scatter.

## Reference and geometry

Every run here is **1× B200, batch 2048, GPU matcher (`device_solver: jv`), 200 epochs**, launched
by `submit_ablation_b200.sh`. The reference is the paper's model at v7 width,
`configs/base_small.yaml` (819,683 parameters), trained at that same geometry with
`--name clic_paper_small_b200_jv`. Arms are compared against that run only.

The same config was trained on 3× L4 (batch 170 × 2 accumulation steps, scipy matcher, job
39236741 in the paper clone, 20 h 02 m) during the jet-IQR investigation; that run is the
positive control for *this code*, not the reference for the arms.

## The arms

Definitions as on `main` (round 3, the 2³ factorial over A2/A3/A4 at fixed decoder depth), derived
from `configs/base_small.yaml` by `make_arms.py`:

| Arm | Change | Params | Fraction |
|---|---|---:|---:|
| reference | `configs/base_small.yaml` | 819,683 | 1.000 |
| C5 a2a4 | A2 + A4 | 616,219 | 0.752 |
| C4 a3a4 | A3 + A4 | 443,435 | 0.541 |
| C3 a2a3 | A2 + A3 | 369,089 | 0.450 |
| C1 a2a3a4 | A2 + A3 + A4 | 352,331 | 0.430 |

- **A2** `dense_kwargs.hidden_dim_scale: 2 → 1` in every encoder and decoder block (on the paper's
  SwiGLU `Dense` this halves the gated hidden width).
- **A3** `dim 64 → 48`, the dim-derived head literals scaled by 0.75 (class head `[48, 96, 24]`,
  regression input 102 and hidden `[96, 96, 96, 48, 24]`; the incidence head follows `dim`),
  `num_heads 8 → 6` so `head_dim` stays 8 (flash-attn needs `head_dim % 8 == 0`).
- **A4** encoder `num_layers 6 → 5`.

Parameter counts are from instantiating each resolved config (`count_params`, CPU); verify against
the trained checkpoints' `ModelSummary` line.

On head, C1 failed to train (its last decoder layer) while C2 = C1 + S1 trained, and the factorial
C3/C4/C5 was queued to locate the failure. Whether the paper's code shows the same behaviour is
the first question this directory answers.

## Running

```shell
cd src/hepattn/experiments/clic && mkdir -p slurm_logs
# pre-flight (minutes) then the full run, per arm
sbatch --time=00:30:00 --job-name=pf-C3 \
  --export=ALL,CONFIG=studies/model_size/configs/clic_paper_small_C3_a2a3.yaml,FIT_ARGS=--trainer.fast_dev_run=true \
  studies/model_size/submit_ablation_b200.sh
sbatch --job-name=clic-ps-C3 \
  --export=ALL,CONFIG=studies/model_size/configs/clic_paper_small_C3_a2a3.yaml \
  studies/model_size/submit_ablation_b200.sh
```

Regenerate the arm configs after changing `configs/base_small.yaml`:
`python studies/model_size/make_arms.py configs/base_small.yaml studies/model_size/configs clic_paper_small`.

## Judging

Same protocol as `main`'s `STUDY.md` §4: jet-energy median and IQR versus jet energy, evaluated with
the host solver, compared with the paper's figures in the **`mpflow_proxy`** convention only, and a
delta only counts when it clears the training-to-training scatter. **σ_repro has not been measured
on this code**; the head numbers (0.0029 `mpflow`, 0.0007 proxy, n = 4 seeds) do not transfer.
Evaluate with `submit_eval_l4.sh` (fp32, torch attention, inference data). Plotting tools will be
brought over from `main`'s study once the first results exist.

## Run register

| Arm | Job | Run folder | Pre-flight | Status |
|---|---|---|---|---|
| reference (B200, jv) | 41750147 | `clic_paper_small_b200_jv_*` | 41750146 | queued 2026-09-11 (submitted from `studies/paper_tag_baseline/`) |
| C5 a2a4 | 41750166 | | 41750165 | queued 2026-09-11 |
| C4 a3a4 | 41750168 | | 41750167 | queued 2026-09-11 |
| C3 a2a3 | 41750170 | | 41750169 | queued 2026-09-11 |
| C1 a2a3a4 | 41750172 | | 41750171 | queued 2026-09-11 |

The arm preflights were submitted without a distinguishing `--name`, so each arm has TWO run
folders: the earlier one (12:15-12:24) is the preflight (`fast_dev_run`, no checkpoints), the
later one (13:03 onwards) is the full run. Name preflights `pf_<run>` from now on.

All ten submitted 2026-09-11 against commit 47645e1 of `clic-paper-main`; each full run is chained
`afterok` behind its pre-flight (30 min), and the full runs were trimmed to 9 h 59 m on the B200 from the head-v7 measurement. That was
wrong for this code: the first epochs run at 7 min each, i.e. 24 h for 200 epochs, so these
runs will hit their limit near epoch 80 and must be resumed from `last.ckpt`. Diagnostic
runs 41755411 (jv) / 41755412 (host) with `MatcherTimer` attribute the step time. If a pre-flight fails,
cancel the matching full run (`scancel <job>`), fix, resubmit both.
