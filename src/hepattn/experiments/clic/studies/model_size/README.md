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

| Arm | Change | Params | Fraction | Round |
|---|---|---:|---:|---|
| reference | `configs/base_small.yaml` | 819,683 | 1.000 | 1 |
| A4 enc5 | A4 | 777,627 | 0.949 | 2 |
| A6 dec3 | A6 | 719,971 | 0.878 | 2 |
| A2 mlp1x | A2 | 645,859 | 0.788 | 2 |
| S1 nobidir | S1 | 653,539 | 0.797 | 2 |
| C5 a2a4 | A2 + A4 | 616,219 | 0.752 | 1 |
| A3 dim48 | A3 | 467,201 | 0.570 | 2 |
| C4 a3a4 | A3 + A4 | 443,435 | 0.541 | 1 |
| C3 a2a3 | A2 + A3 | 369,089 | 0.450 | 1 |
| C1 a2a3a4 | A2 + A3 + A4 | 352,331 | 0.430 | 1 |

- **A2** `dense_kwargs.hidden_dim_scale: 2 → 1` in every encoder and decoder block (on the paper's
  SwiGLU `Dense` this halves the gated hidden width).
- **A3** `dim 64 → 48`, the dim-derived head literals scaled by 0.75 (class head `[48, 96, 24]`,
  regression input 102 and hidden `[96, 96, 96, 48, 24]`; the incidence head follows `dim`),
  `num_heads 8 → 6` so `head_dim` stays 8 (flash-attn needs `head_dim % 8 == 0`).
- **A4** encoder `num_layers 6 → 5`.
- **S1** `bidirectional_ca: false` in `decoder_layer_config` — drops the reverse
  cross-attention `kv_ca` from every decoder layer. Like A6 it is outside the {A2, A3, A4}
  factorial. It is the only arm that removes an *attention site*, so it is the one that cuts
  data × data MACs without touching `dim`.
- **A6** decoder `num_decoder_layers: 4 → 3`. Not part of the {A2, A3, A4} factorial — it varies
  depth, which the factorial holds fixed. Its val_loss is **not comparable** to the reference's:
  the loss sums over the intermediate decoder layers and A6 has one fewer, so a lower number is an
  artefact of fewer terms. Only the final-layer metrics and the jet physics compare.

Round 1 is the factorial's pairs and triple, trained 2026-09-11. Round 2 is the three singles plus
A6, queued 2026-09-14: each single is exactly one change off the reference, which turns round 1's
pair measurements into per-change main effects without assuming additivity — and round 1 already
showed additivity fails for the triple.

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
Evaluate with `submit_eval_l4.sh` (fp32, torch attention, inference data).

**The plotting tools are now here**, ported from `main`'s study 2026-09-14 with paths and the arm
table repointed at this branch:

| Script | Reads | Where it runs |
|---|---|---|
| `arm_sets.py` | — | the single copy of the arm table; `ARM_SET` picks a set, `all` is the only one so far |
| `plot_size_ablation_jet_iqr.py` | `__test.root` | Tier 1, the verdict. Batch — jet clustering wants 16 cores |
| `plot_size_ablation_performance.py` | `__test.root` | the breadth check, every other notebook figure. Batch |
| `plot_size_ablation_training_curves.py` | `metrics.csv` | Tier 0, "did each arm train?". Seconds on a login node |
| `plot_decoder_layers.py` | `metrics.csv` | the per-layer profile. Seconds on a login node |

The two `.root` readers go through `submit_size_plots.sh`; the two `metrics.csv` readers need no
evaluation and no batch job.

**One deliberate departure from head's scripts: no verdict column.** Head's jet-IQR script printed
REAL / not-detectable against thresholds of 0.0073 (global IQR), 0.0164 (high-E) and 0.0170 (low-E).
Those are properties of head's code, measured from four seed trainings of head's reference, and they
do not transfer here. Every delta is reported with its bootstrap σ_stat and nothing else — and
σ_stat is the *smaller* half of the error, the test-sample term only. A delta inside it is certainly
not real; one outside it is merely not excluded. The two-convention sign check is the strongest
statement available until a seed set is trained on this code, because it asks for consistency rather
than significance. The bar an arm must clear is 2·√2·σ_repro, the √2 because both sides are
independently trained.

**C1 is on the canvas here**, unlike in head's arm table, where it is excluded from every ablation
set because it failed to train twice. See the results section below.

**Checkpoint selection.** Every arm is read at its own lowest-val_loss checkpoint, head's rule
throughout. For C4 and C3 that is **not** the last epoch — 189 (4.70576) and 197 (4.83713) against
199's 4.70814 and 4.83990. The differences are ~0.003 and change nothing, but the arm table records
the selected stem and the eval jobs were run on it.

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
41755411 -- a figure since found to be the kernel's *launch* time, see `studies/step_time_gap/`)
and the GPU saturated. The paper code on 3x L4 (job 39236741) was *faster* per GPU than
head-v7, so the loss is specific to the B200 + torch 2.9 stack; the leading suspect is the tag's
Compile callback compiling the whole model as one graph and losing autocast (the attention bf16
cast exists for exactly that failure). Diagnostics 41755907 (stacked matching + head's compile),
41756034 (no compile), 41756035 (PyTorch profiler), 41755775 (host solver), 41755985 (simple
profiler). **Cause found and fixed the same day.** The paper's `Compile` callback compiled the whole model as
one dynamic graph; head's compiles only the encoder and decoder, after the sanity check. With
head's callback (merged as dd72cd1) the same run steps at **367 ms** (job 41755907, 5,582
samples/s), head-v7's pace; without any compilation it ran at 1.74 it/s over the first 300 steps
(job 41756034), so the whole-model graph was actively slower than eager. (`MatcherTimer` read
the matcher at 0.4% of the step here; that was the asynchronous launch, not the solve --
`studies/step_time_gap/`.)

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
tested exactly that and found no step-time difference. `studies/step_time_gap/` found it: the GPU
JV kernel's time depends on the trained model's cost matrices (84 ms/step for the paper model
against 47 for head's), so it applies to every arm, and arm-against-arm comparisons within this
study are unaffected.


The arm preflights were submitted without a distinguishing `--name`, so each arm has TWO run
folders: the earlier one (12:15-12:24) is the preflight (`fast_dev_run`, no checkpoints), the
later one (13:03 onwards) is the full run. Name preflights `pf_<run>` from now on.

That paragraph described the **cancelled** round, whose 9 h 59 m limit came from the head-v7
measurement and would have truncated runs stepping at 790 ms near epoch 80. The compile fix
removed the 790 ms step, and the resubmitted round finished inside 9 h with no resume needed.
Diagnostic runs 41755411 (jv) / 41755412 (host) with `MatcherTimer` attribute the step time. If a
pre-flight fails, cancel the matching full run (`scancel <job>`), fix, resubmit both.

## Evaluation register

All five arms evaluated 2026-09-14 with `submit_eval_l4.sh` (1× L4, fp32, torch attention,
inference data, **host** solver — so the eval path is identical across arms and contributes nothing
to any difference seen). Each arm at its own lowest-val_loss checkpoint.

| Arm | Job | Checkpoint | Elapsed |
|---|---|---|---|
| reference (B200, jv) | 41987838 | `epoch=199-val_loss=4.40887` | 00:01:13 |
| C5 a2a4 | 42119454 | `epoch=199-val_loss=4.62587` | 00:01:15 |
| C4 a3a4 | 42119469 | `epoch=189-val_loss=4.70576` | 00:01:18 |
| C3 a2a3 | 42119470 | `epoch=197-val_loss=4.83713` | 00:01:13 |
| C1 a2a3a4 | 42119459 | `epoch=199-val_loss=4.96997` | 00:01:13 |

All COMPLETED. The reference's evaluation predates this batch — it was run on 2026-09-13 in the
course of `studies/step_time_gap/`, at the same checkpoint and through the same script, so it is
the same evaluation the arms get. **Measured eval time is ~1 m 15 s; the script requests 30 min.**

C4 and C3 were first submitted at epoch 199 (jobs 42119457/58) and cancelled before they ran: the
selection rule is each arm's lowest val_loss, which for those two is epoch 189 and 197. The
resubmitted jobs are the ones in the table.

Figures: `sbatch --export=ALL,ARM_SET=all studies/model_size/submit_size_plots.sh
studies/model_size/plot_size_ablation_jet_iqr.py studies/model_size/plot_size_ablation_performance.py`
(job 42119839).

## Results

### C1 trained. The first question is answered, and the answer is "no".

On head, C1 (A2+A3+A4) failed to train twice over: val `final_classification_object_ce` 1.92 and
2.13 against the reference's 0.67, a jet-E IQR near 0.4 against a threshold of 0.008, and a third
of the reference's matched jets. On the paper tag it completed 200 epochs at val_loss 4.96997 and
is simply the worst arm of five, by a small and ordered margin.

`plot_decoder_layers.py`, at each arm's evaluated checkpoint:

```
val classification_object_ce   layer_0  layer_1  layer_2  layer_3    final
reference 820k                  0.8055   0.2372   0.2467   0.2362   0.4670
C5 a2a4 616k                    0.8191   0.2510   0.2629   0.2470   0.4823
C4 a3a4 443k                    0.8072   0.2497   0.2579   0.2411   0.5136
C3 a2a3 369k                    0.8013   0.2468   0.2563   0.2434   0.5405
C1 a2a3a4 352k                  0.8069   0.2616   0.2696   0.2557   0.5428
```

Head's last-layer observation survives in graded form, and it is worth keeping: every arm
*including the reference* degrades from `layer_3` to `final` (0.2362 → 0.4670 for the reference),
which is a property of the architecture and not of any arm. The arms then fan out about four times
more at `final` (0.467 → 0.543, a spread of 0.076) than at `layer_3` (0.236 → 0.256, 0.020). So the
last decoder layer is where shrinking costs most on this code too — it just does not break there.

### Every metric orders monotonically with parameter count

`plot_size_ablation_training_curves.py`, mean over epochs 190–199, each arm minus the reference:

| metric | reference | C5 616k | C4 443k | C3 369k | C1 352k |
|---|---:|---:|---:|---:|---:|
| mask exact match | 0.4942 | −0.0153 | −0.0363 | −0.0601 | −0.0625 |
| mask purity | 0.7785 | −0.0091 | −0.0283 | −0.0471 | −0.0532 |
| object CE | 0.4667 | +0.0173 | +0.0502 | +0.0730 | +0.0753 |
| class acc (macro) | 0.8956 | −0.0045 | −0.0081 | −0.0095 | −0.0140 |
| \|E residual\| | 0.0177 | +0.0007 | +0.0021 | +0.0037 | +0.0045 |
| val total loss | 4.41349 | +0.220 | +0.297 | +0.426 | +0.560 |

No arm is an outlier and no ordering inverts. That is a different picture from head's round 3,
where the three pairs trained healthily and only the triple broke.

⚠️ **None of this is the verdict.** These are type-A (loss) numbers: the matcher picks which truth
particle each query is scored against, so an arm that has learned to choose differently can look
unchanged in val_loss and still be worse — head's A3 was +0.069 in val_loss, the worst arm there,
and +0.0006 in global jet-E IQR, i.e. nothing. The ranking comes from the jet-E IQR figure, and
even that is an ordering rather than a verdict until σ_repro is measured on this code.

### Tier 1: the jet-E IQR (job 42119839)

32,188 matched jets for the reference; 29,330–32,265 across arms. Global jet-E IQR, both
conventions, with bootstrap σ_stat:

| Arm | params | ratio | IQR `mpflow` | ΔIQR | IQR `proxy` | ΔIQR | σ_stat |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 819,683 | 1.000× | 0.0827 | — | 0.0601 | — | 0.0006 |
| C5 a2a4 | 616,219 | 0.752× | 0.0840 | +0.0012 | 0.0632 | +0.0031 | 0.0006 |
| C4 a3a4 | 443,435 | 0.541× | 0.0926 | +0.0098 | 0.0674 | +0.0073 | 0.0007 |
| C3 a2a3 | 369,089 | 0.450× | 0.0980 | +0.0153 | 0.0719 | +0.0118 | 0.0007 |
| C1 a2a3a4 | 352,331 | 0.430× | 0.1006 | +0.0179 | 0.0738 | +0.0137 | 0.0008 |
| Pandora | — | — | 0.0628 | −0.0199 | 0.0628 | +0.0027 | 0.0004 |

**Pandora crosses the arms in one convention and not the other.** It is the classical
reconstruction and has no convention of its own — the same 0.0628 appears in both columns — so the
comparison flips: in `mpflow` Pandora beats every arm including the reference (0.0628 against
0.0827), while in `mpflow_proxy` the reference beats it (0.0601 against 0.0628) and C5 is level
with it. That is the concrete reason this study compares against the paper in the **`mpflow_proxy`
convention only**: the regression head's kinematics are worse than the incidence-head proxy by more
than the entire size ablation spans. Read the grey line as the bar to beat on the proxy row; on the
`mpflow` row it is telling you about the regression head, not about model size.

**The resolution ordering is monotone and both conventions agree on every arm** — the sign check
passes for global IQR and for high-E IQR across all four arms. C1 does not blow the y-range; it is
the worst arm at +0.0179, roughly 15× head's C1.

**The damage is resolution, not energy scale.** The global median response moves in *opposite*
directions between conventions for C4, C3 and C1 (e.g. C1: `mpflow` +0.0057, proxy −0.0029), which
is the sign check's definition of noise. No arm develops an energy-scale bias.

### A3 is the change that costs; A4 is nearly free

The three pairs plus the triple over-determine a main-effects fit (reference = 0). Solving the
three pair equations for the individual contributions to ΔIQR (`mpflow`):

| change | what it does | contribution to ΔIQR |
|---|---|---:|
| A3 | `dim 64 → 48`, heads 8 → 6 | **+0.0120** |
| A2 | `hidden_dim_scale 2 → 1` | +0.0034 |
| A4 | encoder `num_layers 6 → 5` | −0.0022 (free, marginally helpful) |

Every arm containing A3 costs ≥ +0.0098; C5, the one pair without it, costs +0.0012 for a 25%
parameter cut. **If a cheap shrink is wanted, C5 is it** — a quarter of the parameters for an IQR
change three-to-five times σ_stat and far below anything head would have called detectable.

The fit also says the triple is **super-additive**: main effects predict C1 at +0.0132, measured
+0.0179, an interaction of +0.0048. Shrinking width and depth together costs more than the sum of
the parts. That is the paper-tag echo of head's C1 failure — there the interaction was
catastrophic, here it is a 36% overshoot.

⚠️ **Read this as an ordering, not as verdicts.** σ_stat (~0.0007) is the test-sample term only.
The bar an arm must clear is 2·√2·σ_repro, and σ_repro is unmeasured on this code; on head it was
4× larger than σ_stat in `mpflow` and it is the term that decides whether +0.0012 (C5) is
distinguishable from zero. The main-effects fit inherits that same missing error bar, and it rests
on four trainings with no replicates — the singles A2/A3/A4 were never trained on this branch.

### The same cut costs ~4x more here than on head

Head's factorial (job 41394227 on `main`) ran the same three pairs. Global jet-E IQR in
`mpflow_proxy`, arm minus its own reference:

| arm | head Δ | this branch Δ |
|---|---:|---:|
| C5 a2a4 | −0.0001 | +0.0031 |
| C4 a3a4 | +0.0021 | +0.0073 |
| C3 a2a3 | +0.0030 | +0.0118 |

Not an artefact of the comparison: 32,120 matched jets on head against 32,188 here, per-bin counts
within 2%, both host solver, both 200 epochs. And not a deeper cut — **A3 removes 43.0% of the
parameters here and 42.78% on head**, the same shrink, for +0.0005 there (measured, single arm)
against ~+0.0080 here (inferred from the pairs; the single is now queued to measure it).

This is what makes the arms cross Pandora. The reference's margin over Pandora is widest at low
energy (+0.009 at E10) and gone by E170 (−0.005) — particle flow's advantage *is* the low-energy
region. A penalty of +0.010 there consumes it, so C3 and C1 fall behind Pandora below 75 GeV while
head's arms, penalised only +0.002, never did.

**Hypothesis: head's model was not capacity-limited.** Head's reference carries the rising-IQR
pathology this branch exists to escape — its `mpflow` IQR runs 0.078 → 0.100 across energy where
the paper's runs 0.082 → 0.078. If something other than width is the binding constraint there,
removing width is nearly free. A supporting sign: head's measured σ_repro (proxy) is 0.0007, so its
detection bar is 2·√2·σ = 0.0020 — head's A3 single sits *below* its own bar and C3 barely above.
Head's whole size ablation lived at its noise floor.

If that holds, the head-based study understated what shrinking costs, and its conclusions should not
be carried into the FPGA target. The singles A2/A3/A4 queued here test it directly: they make the
per-change comparison measurement-against-measurement instead of fit-against-measurement.

### Tier 2: the breadth check (job 42120447)

Fourteen figures, seven per convention, in `figures/size_ablation_<convention>_*.png`. The
criterion is qualitative and comparative — curves must not develop a new turn-over, and per-class
splits must degrade proportionally rather than one class collapsing. **It passes.**

`eff_fr_purity` is the figure that matters most, because it is the only one that splits by particle
class and so the only one that can see a failure the jet-E IQR integrates away — the photon and
neutral-hadron classes are the ones inferred from calorimetry alone, with no track to anchor them:

- **Efficiency is flat at ~1.0** for every arm, both classes, across the whole p_T range. No arm
  develops a turn-over.
- **Neutral-hadron fake rate degrades gradually** with size — C1 and C3 sit near 0.31 at high p_T
  against the reference's ~0.25 — and the photon curves stay clustered near 0.01–0.10 for every
  arm. Both classes move together; neither collapses. C4 is marginally *below* the reference at
  high p_T, which is the arms interleaving, not a result.
- **Class-match purity** interleaves in the 0.90–0.96 band with no arm separating out.

So the cost of shrinking on this code is resolution, spread proportionally across classes — not a
class collapse and not an efficiency failure. That is the opposite of head's C1, which kept a third
of the reference's matched jets.

## Round 2 register — the singles and the depth arm

Submitted 2026-09-14, same geometry as round 1 (1× B200, batch 2048, `device_solver: jv`, 200
epochs). 12 h requested, 1.3× the reference arm's measured 8 h 59; every arm here is smaller than
the reference, so none should approach it. Each full run chained `afterok` behind its own
pre-flight, and pre-flights carry `--name pf_<run>` so they no longer collide with the full run's
folder.

| Arm | Params | Pre-flight | Full run |
|---|---:|---|---|
| A2 mlp1x | 645,859 | 42122352 | 42122353 |
| A3 dim48 | 467,201 | 42122354 | 42122355 |
| A4 enc5 | 777,627 | 42122356 | 42122357 |
| A6 dec3 | 719,971 | 42122358 | 42122359 |
| S1 nobidir | 653,539 | 42149891 | 42149892 |

S1 was added to the round 2026-09-14, after the other four: head trained it and this branch had
not, so the same "never carried over" gap that A6 had. It is the only arm that removes an
attention site rather than narrowing one.

Why these four: the three singles make the per-change comparison against head
measurement-against-measurement rather than fit-against-measurement, which is what the "4× more
sensitive" claim above needs. A6 has never been trained on this branch at all — head trained it in
its round 3 and it was not carried over.
