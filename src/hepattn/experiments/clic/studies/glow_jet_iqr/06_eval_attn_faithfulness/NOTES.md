# Task 6 — Eval attention/precision faithfulness (train-matched eval)

**Status:** ⬜ TODO — added 2026-07-07 (hypothesis raised by user).

## Objective
Test whether the **encoder-only** train/eval attention+precision mismatch alters the jet IQR —
i.e., is the rising-IQR discrepancy partly an **eval-faithfulness artifact** rather than a
property of the trained model?

| | Encoder attn | Decoder attn | Precision |
|---|---|---|---|
| **Train** (base.yaml) | flash-varlen | torch | bf16-mixed |
| **Eval** (eval.yaml) | **torch** | torch | **32-true** + matmul highest |

The decoder is `torch` in BOTH (its `decoder_layer_config` never sets `attn_type` → default
`"torch"`, decoder.py:75/384), so the mismatch is **encoder only**: flash-varlen→torch and
bf16→fp32. Plausible mechanism: `encoder.py:236` special-cases
`if self.attn_type == "flash-varlen" and kv_mask is not None:` — the flash-varlen and torch paths
handle the ragged `kv_mask` / 8 register tokens via **different code**, so they are not guaranteed
bit-identical.

## Prior reasoning (why this is probably NOT the main cause, but untested directly)
- fp32/torch eval is MORE faithful than bf16, not less (gold-standard eval).
- The jet-E **median is correct** (flat ~0); only the high-E IQR tail is off — a broken/mismatched
  attention would usually perturb the median too.
- The paper's README prescribes this SAME eval switch (encoder→torch, 32-true, matmul highest), so
  it's likely a non-differentiator vs the paper (which still got a falling IQR).
- The discrepancy is a systematic ~2× energy-trend inversion, not a small numerical delta.
- BUT: we have never DIRECTLY re-evaluated a checkpoint under training-matched attention. Do it.

## Method (decisive test)
1. Make `configs/eval_trainmatch.yaml` = a copy of `eval.yaml` but matching TRAINING inference:
   - encoder `attn_type: flash-varlen` (was torch)
   - `precision: bf16-mixed` (was 32-true); drop/relax `matmul_precision` (moot for bf16 flash)
   - keep `is_inference: true`, PflowPredictionWriter, single device.
2. Eval the **plotted v6 checkpoint** `logs/clic_v6_20260605-T113014/ckpts/epoch=195-val_loss=3.78800.ckpt`
   with this config → write to a **distinct** output name (e.g. `__test_trainmatch.root`, NEVER
   overwrite the canonical `__test.root`). Note: `submit_eval_run.sh` hardcodes `--config
   configs/eval.yaml`; make a variant or pass `--config configs/eval_trainmatch.yaml`.
3. Overlay the train-matched IQR vs the standard torch/fp32 eval (reuse `00_cross_run/compare_runs_iqr.py`
   / `01_.../plot_v7_iqr.py` with both roots for the same run).

## Watch-outs
- flash-varlen needs bf16 + GPU; the L4 bf16-cast fix in `attention.py` is in this path.
- Confirm flash-varlen eval actually runs (varlen kernel + register tokens) — if it errors, that
  itself is informative about the train/eval path divergence.

## Interpretation
- IQR identical (torch/fp32 vs flash/bf16) → the encoder attention/precision eval switch is **ruled
  out**; the discrepancy is a true model property (→ task 5 remains the explanation).
- IQR differs (esp. reverts toward falling) → the eval switch **is** a real factor; revisit the eval
  recipe and the flash-vs-torch kv_mask handling.

## Overlap
Partly folds into **task 5**: if the encoder/mask handling changed between the `clic-paper` tag and
HEAD, the paper-tag reproduction will expose it independently.

## Checklist
- [ ] `eval_trainmatch.yaml` written (flash-varlen + bf16)
- [ ] plotted v6 ckpt eval'd → `__test_trainmatch.root` (canonical root untouched)
- [ ] IQR overlay: train-matched vs standard eval
- [ ] verdict (eval switch matters? Y/N)

## Results
_(fill in)_
