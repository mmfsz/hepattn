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
| reference (B200, jv) | 41750147 | `clic_paper_small_b200_jv_20260911-T130357` | 41750146 passed | CANCELLED 2026-09-11 at epoch ~3 |
| C5 a2a4 | 41750166 | `clic_paper_small_C5_a2a4_20260911-T130709` | 41750165 passed | CANCELLED |
| C4 a3a4 | 41750168 | `clic_paper_small_C4_a3a4_20260911-T132105` | 41750167 passed | CANCELLED |
| C3 a2a3 | 41750170 | `clic_paper_small_C3_a2a3_20260911-T13*` | 41750169 passed | CANCELLED |
| C1 a2a3a4 | 41750172 | `clic_paper_small_C1_a2a3a4_20260911-T130715` | 41750171 passed | CANCELLED |

**Round 1 cancelled 2026-09-11.** The runs stepped at 790 ms (7 min/epoch, 24 h projected) against
head-v7's 360 ms at the same geometry, with the matcher at 0.5% of the step (`MatcherTimer`, job
41755411) and the GPU saturated. The paper code on 3x L4 (job 39236741) was *faster* per GPU than
head-v7, so the loss is specific to the B200 + torch 2.9 stack; the leading suspect is the tag's
Compile callback compiling the whole model as one graph and losing autocast (the attention bf16
cast exists for exactly that failure). Diagnostics 41755907 (stacked matching + head's compile),
41756034 (no compile), 41756035 (PyTorch profiler), 41755775 (host solver), 41755985 (simple
profiler). **Cause found and fixed the same day.** The paper's `Compile` callback compiled the whole model as
one dynamic graph; head's compiles only the encoder and decoder, after the sanity check. With
head's callback (merged as dd72cd1) the same run steps at **367 ms** (job 41755907, 5,582
samples/s), head-v7's pace; without any compilation it ran at 1.74 it/s over the first 300 steps
(job 41756034), so the whole-model graph was actively slower than eager. The matcher stays at
0.4% of the step.

**Round 1, resubmitted 2026-09-11 on ae43410** (14 h requested from the 10.4 h projection, each
full run chained `afterok` behind a named pre-flight):

| Arm | Pre-flight | Full run | params | ms/step | wall time | state |
|---|---|---|---|---|---|---|
| reference (B200, jv) | 41758026 | 41758027 | 819,683 | 311 | 8 h 59 | COMPLETED |
| C5 a2a4 | 41758028 | 41758029 | 616 K | 300 | 8 h 42 | COMPLETED |
| C4 a3a4 | 41758030 | 41758031 | 443 K | 292 | 8 h 11 | COMPLETED |
| C3 a2a3 | 41758032 | 41758033 | n/a¹ | 278 | 8 h 01 | COMPLETED |
| C1 a2a3a4 | 41758034 | 41758035 | 352 K | 278 | 8 h 15 | COMPLETED |

¹ 41758033's log lost its header to the same `srun: error: unpack_header: protocol_version 515
not supported` that truncated 41758027's stdout at epoch 57. Both jobs themselves were fine —
`metrics.csv` runs to epoch 199 and the last checkpoint is there. **Trust `metrics.csv` over the
progress bars for this batch.**

**All five completed 2026-09-11/12, none near its limit.** 14 h was requested from the
preflight's 10.4 h projection; the measurement is 8 h 01 to 8 h 59, so the driver now asks for
12 h (1.3x the reference arm). The step time orders exactly with parameter count, and every arm
sits 11-24% above its head-v7 counterpart at the same geometry (235-255 ms/step). That offset is
not the arms, and it is **not** `Dense`'s gated SwiGLU default either: `studies/swiglu_silu/`
tested exactly that and found no step-time difference. It is unexplained, and it applies to every
arm equally, so arm-against-arm comparisons within this study are unaffected.


The arm preflights were submitted without a distinguishing `--name`, so each arm has TWO run
folders: the earlier one (12:15-12:24) is the preflight (`fast_dev_run`, no checkpoints), the
later one (13:03 onwards) is the full run. Name preflights `pf_<run>` from now on.

That paragraph described the **cancelled** round, whose 9 h 59 m limit came from the head-v7
measurement and would have truncated runs stepping at 790 ms near epoch 80. The compile fix
removed the 790 ms step, and the resubmitted round finished inside 9 h with no resume needed.
Diagnostic runs 41755411 (jv) / 41755412 (host) with `MatcherTimer` attribute the step time. If a
pre-flight fails, cancel the matching full run (`scancel <job>`), fix, resubmit both.
