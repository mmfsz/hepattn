# Linformer attention in hepattn

A guide to the `linformer` attention backend on this branch: what the method is,
how it is implemented here, and what it does and does not preserve.

Assumed background: transformers as a user, not as an implementer. No prior
Linformer knowledge, and no linear algebra beyond matrix multiplication.

---

## 1. The problem: attention is quadratic

One attention layer takes `n` input tokens (for CLIC: detector constituents —
tracks and topoclusters) each carrying a vector of `d` numbers, and produces `n`
updated vectors. It does so by letting every token look at every other token:

```
scores = Q Kᵀ        shape (n, n)      "how much does token i care about token j"
weights = softmax(scores) row-wise
out = weights · V    shape (n, d)
```

The `(n, n)` score matrix is the cost driver: both the arithmetic and the memory
grow like `n²`. For `n` in the thousands (language models, full detector hit
collections) that is the wall Linformer was written to get around.

## 2. The idea: compress the keys and values, not the tokens

**Observation (the Linformer paper, arXiv:2006.04768).** The `(n, n)` attention
matrix is very nearly *low rank*. That is easy to believe here: before the
softmax, `scores = Q Kᵀ` is a product of an `(n, d_head)` and a `(d_head, n)`
matrix, so its rank cannot exceed `d_head` — 16 in our model. Softmax formally
breaks that bound, but the paper shows (via the Johnson–Lindenstrauss lemma,
which says random projections preserve distances between vectors) that the result
stays close to a rank-`k` matrix, with `k` depending on the head dimension and
the accuracy you want — **not** on `n`.

If the answer lives in a `k`-dimensional space anyway, you should not pay to
build the `n × n` object in the first place. So: **compress the key and value
sequences down to `k` "virtual tokens" before attending.**

Concretely, introduce a learned matrix `E` of shape `(n, k)` and replace

```
K  (n, d)   →   K' = Eᵀ K   (k, d)
V  (n, d)   →   V' = Fᵀ V   (k, d)      (F is a second, independent projection)
```

Then

```
scores = Q K'ᵀ       shape (n, k)      ← linear in n, not quadratic
out = softmax(scores) · V'   shape (n, d)
```

Each query still produces one output vector of size `d`; it just attends to `k`
summary slots instead of `n` real tokens.

**What `E` is doing, in words.** Virtual token `m` is a fixed weighted sum of all
`n` real tokens: `K'_m = Σ_j E_{jm} K_j`. It is a learned re-binning of the
sequence — like rebinning a histogram, except the bin weights are free parameters
trained by gradient descent, and bins may overlap and carry negative weights.

**The one thing to keep in mind:** the weights `E_{jm}` depend on the *slot index*
`j`, not on the *content* of token `j`. Attention is content-addressed; this
projection is position-addressed. Slot 7 is always mixed the same way, whatever
happens to be sitting in slot 7.

## 3. What you give up

Three consequences follow directly from "the projection mixes positions", and all
three show up in the code:

1. **No per-position attention masks.** A mask says "query `i` may not look at
   token `j`". After projection there is no token `j` any more — every virtual
   token contains a bit of every real token. There is no correct way to apply the
   mask, so this implementation refuses it. For MaskFormer this is not free: the
   decoder's *masked attention* (each object query attending only to the
   constituents it currently claims) has to be switched off. Whether that is
   fixable is the open question in section 8.
2. **Padding has to be handled before the projection.** A padded slot is not a
   token you can simply ignore later; if it is non-zero it contaminates every
   virtual token. The implementation zeroes padded keys and values first.
3. **A fixed maximum sequence length.** `E` has one row per slot, so the model
   must be built with `seq_len` ≥ any sequence it will ever see. Shorter
   sequences use the first `kv_len` rows of `E`.

---

## 4. The implementation

### 4.1 `src/hepattn/models/linformer.py` — the attention itself

`LinformerAttention` is a self-contained attention module: unlike the other
backends in this repo (which are bare kernels wrapped by `Attention`), it owns
its own input and output projections.

Parameters created in `__init__`:

| Parameter | Shape | Role |
| --- | --- | --- |
| `to_q`, `to_k`, `to_v` | `(dim, heads·dim_head)` | the usual Q/K/V linear maps, no bias |
| `proj_k` | `(seq_len, k)` | `E` — sequence compression for the keys |
| `proj_v` | `(seq_len, k)` | `F` — sequence compression for the values |
| `to_out` | `(heads·dim_head, dim)` | output projection |

`proj_k` / `proj_v` are initialised uniform in `±1/√k`. Two options from the
original paper are carried over but unused in our configs: `share_kv` (use one
projection for both K and V) and `one_kv_head` (a single K/V head shared by all
query heads).

The forward pass, with CLIC encoder shapes (batch `B`, `n = m = 168` tokens,
`dim = 256`, `heads = 16`, `dim_head = 16`, `k = 64`):

| Step | Code | Shape |
| --- | --- | --- |
| inputs | `q`, `kv` | `(B, 168, 256)` |
| project to Q, K, V | `to_q`, `to_k`, `to_v` | `(B, 168, 256)` |
| **zero padded slots** | `keys * valid`, `values * valid` | `(B, 168, 256)` |
| slice the projection | `proj_k[:kv_len]` | `(168, 64)` |
| **compress the sequence** | `einsum("bnd,nk->bkd", keys, proj_k)` | `(B, 64, 256)` |
| split heads | reshape + transpose | q `(B, 16, 168, 16)`, k/v `(B, 16, 64, 16)` |
| scores | `einsum("bhnd,bhkd->bhnk", …) / √d_head` | `(B, 16, 168, 64)` |
| softmax over the 64 virtual tokens, dropout | | `(B, 16, 168, 64)` |
| weighted values | `einsum("bhnk,bhkd->bhnd", attn, values)` | `(B, 16, 168, 16)` |
| merge heads, output projection | `to_out` | `(B, 168, 256)` |

The `einsum` strings read directly: `"bnd,nk->bkd"` means "for each batch `b` and
feature `d`, sum over the sequence index `n` with weights `E[n, k]`" — i.e. the
re-binning of section 2, applied to every feature channel independently.

Two guards: `kv_len ≤ seq_len` is asserted, and an `attn_mask` never reaches this
module (it is refused one level up).

### 4.2 `src/hepattn/models/attention.py` — how it plugs in

`Attention(attn_type="linformer", linformer_seq_len=…, linformer_proj_dim=…)`.
Because `LinformerAttention` owns its projections, the wrapper deliberately does
*less* than usual for this backend:

- no `in_proj_weight` / `out_proj` are created;
- no Q/K/V norms and no value-residual mix are created, because the backend never
  exposes the projected values for them to act on. (Creating them anyway would
  mean parameters that never receive a gradient, which DDP rejects.)
- `forward` short-circuits: `return self.attn(q, kv, kv_mask=kv_mask)`, before the
  usual `_prepare_qkv` path.

Backend membership lists encode section 3: `linformer` is in `VARLEN_ATTN_TYPES`
(it handles key padding) and *not* in `ATTN_MASK_ATTN_TYPES` or
`ATTN_BIAS_ATTN_TYPES` (an `attn_mask` or `attn_bias` trips an assertion).

`set_backend` — used at evaluation time to swap kernels — refuses to switch
between `linformer` and any other backend in either direction: the two have
different parameters, so a swapped model would have nothing to load its weights
into. Calling `set_backend("linformer")` again is a no-op that keeps the trained
projections.

### 4.3 `src/hepattn/experiments/clic/configs/linformer.yaml` — the CLIC run

`base.yaml` with `attn_type: linformer` in the encoder and the decoder, and a
2e-4 peak learning rate. The sizes:

- `linformer_seq_len: 168` — the longest sequence anywhere in the model. The
  encoder sees 160 constituents (CLIC pads every event to `max_nodes = 160`) plus
  8 register tokens. The decoder sees 160 constituents and 150 object queries,
  both shorter, so they use a slice of the same projection.
- `linformer_proj_dim: 64` — this is `k`, the compression knob, and the thing to
  study. 168 → 64 is a factor 2.6 on the key/value axis.
- `mask_attention: false` in the decoder, forced by section 3.1.

Do **not** evaluate this model with `configs/eval.yaml`'s `attn_type: torch`
override: a Linformer checkpoint has Linformer parameters and must be evaluated
with the Linformer backend.

---

## 5. What it buys at CLIC scale

Counting the dominant matrix multiplications for one self-attention layer with
`n = m` tokens and embedding size `d`:

- standard: `scores` + `weights·V` ≈ `2 n² d`
- Linformer: two sequence projections `2 m k d` + `2 n k d` ≈ `4 n k d`

So Linformer is cheaper only when `k < n/2`. At `n = 168`, `k = 64` that is
`4·64 = 256` versus `2·168 = 336` — roughly a 25 % reduction in attention FLOPs,
before any overhead. The attention matrix itself shrinks from `168 × 168` to
`168 × 64` (a factor 2.6 in activation memory), which is the more useful number
on memory-bound hardware. Parameter count *increases* slightly: `2 · 168 · 64 ≈
21.5 k` extra per attention layer, about +8 % on that layer's parameters.

These are arithmetic estimates, not measurements: at `n = 168` attention is not
the dominant cost of this model, so a wall-clock win is not guaranteed and has
not been measured on this branch. Linformer's real promise here is not speed at
today's sequence length but a smaller, more regular attention block — the reason
it is interesting alongside the model-size work.

## 6. Gotchas worth knowing before reading results

- **Empty virtual tokens still get attention weight.** If a virtual token's real
  contributors are all padding, its key becomes the zero vector, its score is
  exactly 0, and `softmax` still gives it weight `e⁰/Z`. It contributes a zero
  vector to the output, so it cannot inject garbage — the test checks exactly
  this — but it does dilute the attention weights, and the dilution varies with
  event occupancy. This is a property of the method as published, not a bug.
- **Slot index is not physically meaningful in CLIC.** The constituent list is
  tracks first, then topoclusters, then padding, and the number of tracks varies
  event by event. So slot 7 is a track in one event and a topocluster in another,
  while `proj_k` treats it the same way in both. The projection has to learn
  something occupancy-averaged. (In the decoder, by contrast, query slots are
  consistent across events.)
- **Masked attention is off**, so the decoder differs from the paper model by
  more than just the attention kernel. Any comparison of physics performance is a
  comparison of two changes at once.

---

## 7. Edits made when porting

Ported from `lgray/compression-work` (76f2078), which itself is a rewriting of
[lucidrains/linformer](https://github.com/lucidrains/linformer). The following
was changed during the port; everything here is a behaviour change, not style.

1. **The attention-mask handling was a no-op, on the wrong axis.** The original
   did

   ```python
   dots[..., :kv_len].masked_fill((attn_mask == 0)[:, None, ...], float("-inf"))
   dots[..., kv_len:] = float("-inf")  # mask out anything past the current seq_len too.
   ```

   *The intent was to make masked attention work*, not to disable it: the same
   commit (`1c927c0`) added `linformer` to `ATTN_MASK_ATTN_TYPES` — the list that
   declares a backend mask-capable — threaded `attn_mask` down into the module, and
   its config kept `mask_attention: true`. The two lines are two separate
   intentions: the first applies the per-query mask, the second (per its comment)
   handles padding.

   Both land on the wrong axis. The last axis of `dots` is the *projected* axis of
   length `k`: column `m` is the query's score against a learned blend of **all**
   `kv_len` tokens, so column index and key index have no correspondence and there
   is nothing meaningful to mask there. On top of that, `masked_fill` is not
   in-place (that is `masked_fill_`), so the first line built a masked copy and
   threw it away — the mask never had any effect at all. Only the second line did
   something: with the branch's defaults (`k = 256`, `kv_len = 160`) it set virtual
   tokens 160–255 to `-inf` whenever a mask was passed, disabling 96 of the 256.

   Two accidents kept this invisible. The discarded `masked_fill` raises nothing,
   and `k` was never moved off its default of 256, which is *larger* than any CLIC
   sequence — so `dots[..., :160]` broadcast against a `(B, 1, 150, 160)` mask by
   coincidence. At a `k` that actually compresses (64, as used here) the same line
   raises a shape error immediately.

   Now: masks are refused outright (`linformer` removed from
   `ATTN_MASK_ATTN_TYPES`) and the CLIC config turns decoder mask attention off, so
   nothing is silently ignored. Whether they can be made to coexist properly is
   section 8.

2. **Key/value padding was not handled at all.** `kv_mask` never reached the
   module, so padded constituents were projected into every virtual token and
   attended to like real ones. Now padded keys and values are zeroed before the
   projection, and a test replaces padded slots with large garbage values and
   asserts the output is unchanged.

3. **Unused parameters were created.** The wrapper still built the Q/K/V norms and
   the value-residual mix for the Linformer path, even though that path never uses
   them. Parameters that never receive a gradient make DDP raise (the CLIC runs
   are multi-GPU), besides being dead weight. They are no longer created.

4. **`set_backend` re-created the module on every call.** Each call built a fresh
   `LinformerAttention` with freshly initialised projections, so any later
   `set_backend` — the evaluation path does exactly this — would have thrown away
   the trained projections. Now the module is created once, a repeated
   `set_backend("linformer")` keeps the weights, and switching to or from another
   backend raises instead of producing a silently broken model.

5. **Signature mismatch with this branch.** The original took `(q, k, v)` and
   asserted `k.shape[1] == v.shape[1]`, which raises on self-attention where
   `k is None`. The paper-tag `Attention` passes `(q, kv)`, and the module now
   follows that convention.

6. **The config compressed nothing.** The branch left `linformer_seq_len` and
   `linformer_proj_dim` at their defaults of 256/256. For CLIC's 168 tokens that
   is a projection to *more* virtual tokens than there are real ones — no
   compression, just extra parameters. Set to 168/64 here, with the reasoning in
   the config header. Its `mask_attention: true` was turned off (point 1).

7. **Repo fit.** Typed signatures, numpydoc docstrings, no `lambda`/`map`
   pipelines, and no dependency on the `linformer` package — the module is ~120
   lines of self-contained code. The other backends are untouched: their outputs
   and state dicts are unchanged by this port.

## 8. Next step: can Linformer and masked attention coexist?

This is the open question the port leaves behind, and it should be answered before
any physics conclusion is drawn from a Linformer run.

**Why it matters.** Turning `mask_attention` off is not a small concession.
Masked attention — each object query attending only to the constituents it
currently claims — is a defining piece of MaskFormer, and it is the mechanism that
sharpens the mask predictions layer by layer. As things stand, a Linformer run
differs from the baseline by *two* changes at once (compressed attention **and**
no masked attention), so its results cannot be attributed to either.

**Step 0 — find out whether the question is worth answering.** Run the baseline
model, unchanged, with `mask_attention: false` and standard `torch` attention.
That single ablation separates the two changes and costs one training run. If the
baseline barely moves without masked attention, the rest of this section is
unnecessary: keep `mask_attention: false` and study compression on its own. If it
degrades noticeably, the coexistence question is the blocker and the avenues below
are worth the effort.

**Avenues, cheapest first.**

1. **Use Linformer only where there is no mask.** The mask exists only in the
   decoder's cross-attention (`q_ca`, and its transpose in `kv_ca`). The encoder's
   self-attention (168 tokens, 6 layers) and the decoder's query self-attention
   (150 queries) are unmasked and could use Linformer today, with the masked
   cross-attention left as ordinary attention. This keeps the model faithful and
   still compresses where the sequences are longest. It is a config/plumbing change,
   not research, and it is the obvious thing to try first.
2. **Project the mask along with the keys.** The mask `M` has shape
   `(B, n_q, kv_len)` — the same sequence axis the keys are compressed along. A
   soft per-virtual-token weight `M' = M · E` (shape `(B, n_q, k)`, normalised, and
   with a non-negative projection since `E` may be negative) would say "how much of
   virtual token `m` is made of constituents this query is allowed to see", and
   could be applied as a multiplicative weight or additive bias on the scores. This
   is an approximation, not a mask, and it needs a check that it does not simply
   reproduce the unmasked model — but it is the only avenue that keeps the
   compression and the mask in the same layer.
3. **Check the literature before inventing anything.** Linformer as published is
   bidirectional-encoder only; the paper does not handle causal or positional
   masking, and that is a known limitation of the method rather than an oversight
   in this code. Landmark-based linear attentions (Nyströmformer and its causal
   variants, Luna, and related work) choose *data-dependent* summary tokens instead
   of a fixed learned projection, which is exactly the property a mask needs. If one
   of them supports masking cleanly, adopting it beats patching Linformer.

**What would settle it.** Any candidate has to clear the same bar as the baseline:
jet-energy IQR that falls with energy, compared against the measured
reproducibility spread — not validation loss. Anything that only moves `val_loss`
proves nothing here.

## 9. Where things are

| File | What |
| --- | --- |
| `src/hepattn/models/linformer.py` | `LinformerAttention` |
| `src/hepattn/models/attention.py` | backend selection, mask/padding policy, `set_backend` |
| `src/hepattn/experiments/clic/configs/linformer.yaml` | the CLIC run config |
| `tests/models/test_linformer.py` | shapes, padding-invariance, refusals, weight persistence |

The code itself lives outside this directory: the port is shared code, merged into
`clic-paper-main` on 2026-09-11 (`a9b1435`, merging `ca90076` and `3033053`). This
study directory holds only the notes and, later, the runs.

## 10. Status

Checked against `clic-paper-main` at `5c3ba33`, after the thirty commits that
landed on top of the merge:

- `tests/models/test_linformer.py` — 6 passed.
- `configs/linformer.yaml` differs from the current `base.yaml` by the intended
  changes only (name, 2e-4 peak LR, `attn_type`, the two projection sizes,
  `mask_attention: false`); none of the later config work drifted past it.
- Building the model from `configs/linformer.yaml` and running one forward and
  backward pass on CPU: 12.418 M parameters, all four decoder layers plus the
  final one, and **no parameter left without a gradient** — the condition section
  4.2 and point 3 of section 7 exist to keep, and the one DDP would raise on.
- The baseline `configs/base.yaml` is 12.065 M parameters, so compression costs
  +0.353 M (+2.9 %) here; section 5 explains why the projections add parameters
  while removing arithmetic. That count is also the only comparison available on
  CPU: `base.yaml`'s `flash-varlen` backend has no CPU kernel, Linformer runs
  anywhere.

No training run exists yet: nothing under `logs/`, and no submit script refers to
this config. Before one is launched, section 8's step 0 — the baseline with
`mask_attention: false` and ordinary attention — is what makes the result
attributable.

Reference: Wang et al., *Linformer: Self-Attention with Linear Complexity*,
[arXiv:2006.04768](https://arxiv.org/abs/2006.04768).
