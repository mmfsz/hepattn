# Mask-loss broadcast bug — correctness analysis (2026-07-31)

Follow-up to the "Loss-kernel efficiency benchmark" section of [`NOTES.md`](NOTES.md)
(job 38463563), which found that four mask-loss functions in
`src/hepattn/models/loss.py` do an accidental `[B, N_valid, C]` broadcast. That
section established the *performance* facts; this document answers the
*correctness* question:

> Is this only a wastefully large intermediate that produces the right number,
> or does it change the loss value and gradients the model actually trains on —
> and if so, did it affect any physics result?

**Everything here was measured on real CLIC data** (first 12,175 events of
`data/clic/train_clic_fix.root`, the production training file) on CPU in fp32,
with the actual uncompiled functions from `src/hepattn/models/loss.py` at HEAD of
branch `matcher-perf`. Script: [`analyze_loss_bug.py`](analyze_loss_bug.py); raw
numbers: `profile_logs/loss_bug_analysis_results.json`. Nothing in production
code was changed.

## Verdict, in one paragraph

**It is a real correctness bug, not just a compute defect: the loss the model
trains on differs from the documented/intended loss by −11.5% (BCE) to +190%
(dice, late training), and the gradient points in a measurably different
direction (cosine similarity 0.88–0.93, i.e. ~20–28 degrees off, systematic not
noise).** Every training run of every experiment that uses `ObjectHitMaskTask`
since commit b3985a3 (2025-07-01), including the `clic-paper` tag, optimised
this distorted objective. **However, the distortion is almost completely
independent of batch size** (mean discrepancy moves by ≤0.4 percentage points
across a 64× batch range), so all runs — at any batch size — optimised
essentially the *same* wrong objective, and cross-configuration comparisons
within this codebase remain internally consistent. **In particular the bug is
genuinely excluded as the cause of the `glow_jet_iqr` rising-IQR regression**
(section "glow_jet_iqr verdict" below — both arms of the decisive experiment
trained on the identical buggy loss). Whether the *absolute* physics results
(e.g. jet-E IQR vs Pandora) would improve, worsen, or stay the same with the
corrected loss is **unknown — no run with the corrected loss exists** — but the
one available natural experiment (paper-original vs tag-retrain, below) suggests
jet-level metrics are insensitive to changes of the mask-loss reduction
semantics at least this large.

## 1. The three loss semantics in play

For a batch of `B` events, `N` object slots, `C = max_nodes = 160` constituent
slots, with `pad[b,c]` the constituent-validity mask, `V[b] = sum_c pad[b,c]`,
`ovm[b,n]` the object-validity mask, `Nv = sum ovm`, and `e(n)` the event that
flattened valid object `n` belongs to:

- **V2 — intended** (what the docstrings and the `/valid_counts` normalisation
  describe; never yet trained on):
  each object is masked by *its own event's* pad mask and normalised by its own
  event's count. For BCE: `L = mean_n [ sum_c l[n,c] * pad[e(n),c] / V[e(n)] ]`.
- **V1 — buggy** (b3985a3 → HEAD, *and* the `clic-paper` tag fb90390):
  `pred_logits[object_valid_mask]` collapses `[B,N,C] → [Nv,C]`; the pad mask
  `[B,1,C]` then broadcasts against the flattened object axis giving
  `[B,Nv,C]` — every object crossed with every event's pad mask.
- **V0 — pre-bug** (before b3985a3; the code the paper's original June-2025
  plotted run trained on): no pad mask at all —
  `L = mean over [Nv, 160]` of the elementwise loss; padded logits are
  `finfo.min` so they contribute ~0 to BCE but still dilute the denominator
  (uniform per-object weight 1/160 instead of 1/V).

### Exact closed form of the buggy loss (PROVEN)

For **`mask_bce_loss`** the buggy value reduces exactly to

```
L_buggy = (1/Nv) * sum_n sum_c l[n,c] * w[c],   w[c] = (1/B) * sum_b pad[b,c] / V[b]
```

i.e. each object's per-constituent loss is weighted by a **batch-averaged,
constituent-index-dependent weight** `w[c]` instead of its own event's
`pad[e(n),c]/V[e(n)]`. The elementwise gradient ratio buggy/intended is exactly
`w[c] * V[e(n)]`. (This confirms the algebra proposed when this task was
commissioned.) For **`mask_dice_loss`** the buggy value reduces exactly to

```
L_buggy = mean_b mean_n dice( object n truncated to the first V[b] constituents )
```

(truncation because CLIC pads are contiguous — valid-first, `do_padding` in
`pflow_data.py`; verified for all 12,175 events). Each object's dice is averaged
over *B truncated copies* of itself; truncating below `V[e(n)]` deletes true
intersection mass, pushing dice toward 1. `mask_focal_loss` has the same
structure as BCE (confirmed numerically at toy scale: buggy 0.5296 vs intended
0.5861). `mask_kl_div_loss` shares the rank-collapsing pattern via
`masked_fill` but is **used by no config** in the repo.

Both closed forms were verified to match the actual production functions
**exactly** (values to `rtol=1e-5` and gradients to `atol=1e-6` via autograd),
at toy scale and on real 64-event CLIC batches, against per-event Python-loop
references for the intended semantics. All large-scale numbers below use the
validated closed forms (the naive buggy tensor at B=2048 would be ~130 TB).

Two structural facts that shape the physics interpretation:

1. **No padding leakage.** Padded constituents carry `logit = finfo.min`
   (`task.py:562`) and target 0, so their BCE/dice contribution is exactly 0
   regardless of which pad mask hits them. The damage is purely (a) a
   positional reweighting `w[c]` of *valid* constituents and (b) a wrong
   per-object normaliser.
2. **`w[c]` is monotonically decreasing in `c`** (a constituent index is valid
   in fewer events the higher it is). Since the CLIC constituent axis is
   ordered tracks-first-then-topoclusters, high-index constituents — mostly
   topocluster hits of busy events — are systematically down-weighted.

## 2. Real-data padding distribution (this is why the bug is not a no-op)

From 12,175 real training events (`max_nodes = 160`):

| V = valid constituents/event | value |
|---|---|
| mean ± std | **67.0 ± 26.8** (CV = 0.40) |
| min / max | 1 / 159 |
| 1% / 25% / 50% / 75% / 99% quantiles | 11 / 49 / 62 / 81 / 145 |
| tracks per event (mean) | 21.0 |
| valid objects per event | 49.6 ± 20.0 |

V varies by an order of magnitude across events, so the intended per-event
normalisation `1/V[e(n)]` and the buggy batch-average differ strongly.
Concretely, the **total BCE weight given to one object** (intended: exactly 1.0
for every object) becomes `sum_{c<V_e} w[c]`:

| event size V_e | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 159 |
|---|---|---|---|---|---|---|---|---|
| buggy total weight | 0.35 | 0.66 | 0.86 | 0.95 | 0.98 | 0.99 | 1.00 | 1.00 |

Mean over events 0.83, minimum 0.02 (a V=1 event). **Objects in sparse events
lose up to ~3–50× of their intended loss weight; the intended amplification of
sparse events (divide by a small V) is removed entirely.** This measurement is
data-only (no model, no surrogate) and is exact.

## 3. Loss-value and gradient discrepancy on real data

Surrogate logits on real targets/masks: `logit = s*(2t−1) + noise`, padded
positions set to `finfo.min` exactly as `ObjectHitMaskTask.forward` does;
`sample_weight = t + 1.0*(1−t)` (base.yaml `null_weight: 1.0`); three sharpness
regimes bracketing training progress ("early" s=0, "mid" s=2, "late" s=6).
Results at B=2048 (production B200 batch size; means ± std over 6 disjoint real
batches; "combined" is the production `5*mask_bce + 1*mask_dice`):

| quantity | early training | late training |
|---|---|---|
| BCE: (buggy−intended)/intended | **−11.5% ± 0.1%** | **−11.5% ± 0.1%** |
| dice: (buggy−intended)/intended | −0.5% ± 0.02% | **+189% ± 1%** (buggy ≈ 2.9× intended) |
| combined loss rel. diff | −9.9% | +143% |
| combined **gradient cosine** (buggy vs intended) | **0.896 ± 0.001** | **0.932 ± 0.001** |
| combined gradient norm ratio | 0.926 | 0.819 |

The BCE distortion is constant in training progress; the dice distortion
*grows* as predictions sharpen (a converged object's own-event dice → 0, but
its B−1 truncated copies keep O(1) values), so late in training the dice term
the optimiser sees is ~3× the intended one and the mask task's loss floor is
inflated. A gradient cosine of 0.90 is a systematic ~26-degree rotation of the
training signal (per-batch std < 0.01 — this is not noise), applied at every
one of the 5 supervised decoder outputs on every step of every epoch.

**Conclusion: the model demonstrably did not train on the documented
objective.** In that sense the physics-relevant training signal was affected.
What is *not* demonstrated is that final physics metrics moved — see §5.

## 4. Batch-size dependence — the headline negative result

The commissioning hypothesis was that `w[c]` depends on batch composition, so
different batch sizes might optimise measurably different objectives. Measured
(same protocol, disjoint real batches; 16/8/6 repeats at B=32/256/2048):

| B (per-rank) | BCE rel. diff | combined grad cosine (early) | w[c] rel. RMS across batches | w[c] deviation from population |
|---|---|---|---|---|
| 32 | −11.16% ± 0.6% | 0.901 ± 0.016 | 14.0% | 2.5% |
| 256 | −11.30% ± 0.3% | 0.900 ± 0.007 | 4.6% | 1.0% |
| 2048 | −11.50% ± 0.06% | 0.896 ± 0.001 | 1.6% | 0.2% |

- The **mean** discrepancy is nearly batch-size independent: `w[c]` converges to
  a population function of the data, and even at B=32 the batch estimate sits
  within 2.5% of it. The residual drift of the mean (−11.16% → −11.50% over a
  64× range, a few σ, from the O(1/B) self-term of an object's own event in the
  average) is ~0.4 percentage points — negligible against the 11.5% offset.
- Only the per-batch **variance** of the objective shrinks with B (an extra
  gradient-noise source at small batch, 14% relative RMS on the weights at
  B=32, on top of ordinary minibatch noise).

**So: runs at different batch sizes did NOT optimise measurably different
objectives.** All post-b3985a3 runs optimised the same distorted loss, with
batch-size-dependent noise only. Cross-configuration comparisons inside this
project (L4 vs B200, batch 128 vs 2048) are not confounded by this bug's batch
dependence.

## 5. glow_jet_iqr verdict: genuinely excluded

Re-examined, as commissioned, the earlier claim that this bug cannot explain the
jet-E-IQR regression (`../glow_jet_iqr/`). The claim survives — and is now
supported by configuration evidence and measurement rather than only by the
commit timeline:

1. **Both arms of the decisive experiment trained on the identical buggy
   loss.** The `clic-paper` tag clone (`hepattn-clic-paper`, fb90390) contains
   the same boolean-index-then-broadcast code in all four functions, the same
   `ObjectHitMaskTask.loss` passing both masks (its `task.py:362`), the same
   `mask_bce: 5.0 + mask_dice: 1.0` config, the same valid-first `do_padding`,
   the same `max_nodes: 160`, and the same dataset files. The tag-retrain
   (falling IQR) and every HEAD run (rising IQR) optimised the same distorted
   objective.
2. **Batch composition cannot rescue the hypothesis.** Checked in the actual
   logged `config.yaml` of each run: HEAD runs used per-rank batch 128 (fp32),
   256 (3xL4 plotted run), 512 (v7), 2048 (both B200 runs) — **all rising**;
   the paper-tag retrain used 170 — inside that range — and **falls**. And §4
   measured the bug's batch dependence at ≤0.4 pp of loss and <0.01 of gradient
   cosine over 32→2048; there is nothing there that could flip a physics trend
   between batch 170 and 256.
3. **A stronger natural experiment exists and points the same way.** The
   paper's own plotted checkpoint (`CLIC_Pflow_FullDiceFocFix_20250613`,
   trained 2025-06-13) predates the bug commit (2025-07-01) and therefore
   trained on the V0 loss — no pad mask at all, a *larger* semantic difference
   from V1 than the V1→V2 bug fix is (V0 BCE = 0.542 vs V1 = 0.949 vs
   V2 = 1.069 on the same real batch). Yet the tag-retrain (V1) reproduced the
   paper's Fig. 4 proxy IQR **exactly** (0.072→0.043 vs paper ~0.072→0.042, per
   `../glow_jet_iqr/05_reproduce_paper_tag/NOTES.md`). Jet-level metrics were
   insensitive to a mask-loss semantics change of this size. (The V0 dating is
   INFERRED from commit/run dates, not from the paper group's actual code.)

The regression therefore remains attributed to the post-paper model refactor,
as the glow study concluded. One caveat cuts the other way: because *every* run
in that study is V1-trained, the study says nothing about what the *corrected*
(V2) loss would do to the IQR — that would require a retrain.

## 6. Matching / cost path: clean

Confirmed by inspection at HEAD: the `*_cost` functions
(`mask_dice_cost`, `mask_bce_cost`, `mask_focal_cost`, `mask_iou_cost`,
`mask_kl_div_cost`) never boolean-index; they keep the `[B,N,C]` rank, so
`input_pad_mask.unsqueeze(1)` broadcasts correctly as `[B,1,C]`. `input_pad_mask`
reaches loss functions from exactly one call site in the entire model code,
`ObjectHitMaskTask.loss` (`task.py:620-621`); the cost call site
(`task.py:597`) uses the cost functions. Hungarian matching inputs are
therefore computed as intended (and the matcher itself was separately
equivalence-verified during fix experiments 1–4). The bug is confined to the
loss/gradient path.

## 7. Scope across experiments

`ObjectHitMaskTask.loss` always passes both masks, so every config that
instantiates it trains on the buggy mask losses (weights as in config):

| experiment | affected configs | mask losses used |
|---|---|---|
| clic | base, clic_v6_fp32, clic_v7 (+ paper tag) | bce 5, dice 1 |
| trackml | tracking (dice 2, focal 100); tracking-lite/strip/queryPE-lite (bce 100, focal 50) | bce, dice, focal |
| itk | tracking | bce 100 |
| tide | base, pixel, regression, tagging/tide | bce 5–10 |
| cld | base (bce 0.25, dice 0.75/1.0), tracking (bce 100, focal 50), regression (bce 10) | bce, dice, focal |
| colliderml | base | dice 2, focal 100 |
| atlas_muon | config/muon_tracking | bce 1 |
| pixel | — (no ObjectHitMaskTask) | not affected |

Magnitudes were only measured for CLIC; each experiment's distortion depends on
its own V-spread (a hit-padding distribution as broad as CLIC's implies a
comparable effect).

## 8. PROVEN / INFERRED / confounds

**PROVEN (measured in this analysis):**

- The buggy functions compute the closed forms of §1 (values `rtol=1e-5`,
  gradients `atol=1e-6`, direct code vs closed form, toy + real batches), and
  differ from per-event-loop references implementing the docstring semantics.
- On real CLIC batches: BCE −11.5%, dice −0.5% (early) to +189% (late),
  combined-gradient cosine 0.90–0.93, norm ratio 0.82–0.93 (§3).
- V-distribution 67 ± 27, range 1–159; per-object total weight 0.35–1.00 vs
  intended 1.0 (§2, data-only, exact).
- Batch-size dependence of the mean objective ≤0.4 pp over B=32→2048; only the
  variance changes (§4).
- Both HEAD and the `clic-paper` tag contain the identical buggy code and call
  it identically; the glow runs' actual per-rank batch sizes are 128–2048
  (rising) vs 170 (falling) (§5, from the tag clone and logged configs).
- Cost/matching path clean by code inspection; single loss call site (§6).
- Padded constituents contribute exactly 0 to BCE/dice (finfo.min logits,
  zero targets) — no padding leakage, in fp32 on real batches.

**INFERRED (not directly measured):**

- That the paper's original plotted run trained on V0 (dated from commits and
  the run name only).
- That final physics metrics are insensitive to the V1→V2 correction. Supported
  by the V0-vs-V1 natural experiment, but **no V2-trained run exists**; the §3
  numbers prove the objective differs, not that end-metrics would move or not.
- That other experiments (trackml, itk, …) see CLIC-magnitude distortions
  (structure identical; their V-distributions were not measured).

**Confounds:**

1. **Surrogate logits.** Predictions were `s*(2t−1)+noise` on real
   targets/masks, not a trained checkpoint's outputs; the three regimes bracket
   training progress, and the BCE numbers are regime-independent, but the exact
   late-training dice ratio (2.9×) depends on the assumed sharpness.
2. **fp32 on CPU**, not production bf16-mixed autocast; and the uncompiled
   functions, not the `torch.compile(dynamic=True)` wrappers (job 38463563
   showed compiled == eager values, so this is weak).
3. **Loss applied to dataset-ordered targets, not Hungarian-matched pairs.**
   Matching permutes which (object, target) pairs are compared but not the
   reduction structure the bug lives in; the surrogate's target-correlation
   stands in for post-matching alignment.
4. Real events are the first 12,288 of the training file (file order assumed
   unbiased); B=2048 uses 6 resamples from a 12k-event pool, so batches share
   events across repeats (not across a single batch).

## 9. Relation to the recommended fix

The "mulmask" rewrite recorded in `NOTES.md` (job 38463563) implements V2 and
was used here as the vectorised intended reference (it matches the per-event
loop to float precision on real data). The validation checklist there stands,
with one addition from this analysis: **a V2 retrain changes the effective
objective by ~−10% (BCE) and up to ~3× (late-training dice), re-amplifies
sparse events, and restores per-event normalisation — so loss curves and
possibly converged metrics will shift; do not compare V2-trained runs against
V1-trained baselines without re-baselining.**
