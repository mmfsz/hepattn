# Slides

Typst deck for the Linformer study. Six slides: three on how it works, three on what it costs,
the costing ones on the **820 k paper-tag model**
(`base_small`, `dim = 64`) so the numbers line up with the model-size deck's slide 7.

| file | what it is |
|---|---|
| `linformer.typ` | the deck |
| `template.typ` | the reusable deck template, copied verbatim from `../../model_size/slides/`. Keep them in sync by copying, not by editing one of them |
| `logos/` | FSU lockups used by the title slide and the footer bar |

## Building

```bash
typst compile --root .. linformer.typ      # -> linformer.pdf
typst watch   --root .. linformer.typ      # live preview while editing
```

## The slides

The first three are pedagogical and carry one worked example all the way through: 6 constituents,
3 summaries, 4 queries. The numbers are chosen so the third slide shows the mechanism *and* its
failure mode in the same table — q1..q3 claim constituents that line up with the summaries and keep
a sharp mask, while q4's claims are spread across all three, so its row comes out nearly flat and
cancels in the softmax. That is the `Var(w)` check of the study README 11.5, made visible.

1. **Masked attention** — the mask is one yes/no per (query, constituent); every column of the
   scores *is* a constituent, so "not constituent 5" has somewhere to point.
2. **Linformer** — the 6 constituents become 3 learned summaries, each a blend of all of them. No
   column is constituent 4 any more, which is why the mask cannot come along.
3. **Step 2** — blend the mask by the same matrix, `w = M|E| / (valid |E|)`, and add `log w` to the
   scores. Sharp for q1..q3, flat for q4.

4. **The cost model** — which multiplications need a DSP, in symbols. `data × data` is both
   operands from the event; `data × weight` is one operand a trained constant and is counted
   separately. Linformer's `K' = Eᵀ K` is data × weight, so it **moves** work out of the DSP budget.
   Every compressed site costs `k / n_kv` of what it did, independent of `D`.
5. **k = 64** — the value `configs/linformer.yaml` uses.
6. **k = 32** — the same table, one scan point down.

Columns are the study README's own steps, so the two documents agree on what each name means:

| column | what it is |
|---|---|
| baseline | ordinary attention everywhere, masked attention on — the model-size deck's slide 7 unchanged |
| step 0 | masked attention off, no Linformer. Cheaper for one reason only: four of the mask head's five calls exist to drive it |
| step 1 | Linformer at the two sites that carry no mask, ① and ③, masked attention still on. The arm that is running |
| step 2 | step 1 plus the projected mask at ② |

Rows are the four attention sites plus the mask head. Site ④ (`kv_ca`) never moves: it is out of
reach for both the mask and the padding-axis reasons in section 8.1.

| | baseline | step 0 | step 1 | step 2 |
|---|---|---|---|---|
| `k = 64` | 65.5 M | 59.3 M (−9 %) | 45.4 M (−31 %) | 38.1 M (−42 %) |
| `k = 32` | 65.5 M | 59.3 M (−9 %) | 38.8 M (−41 %) | 29.0 M (−56 %) |

Two further rows under each table carry the other buckets, so both are on one page: `data × weight
added here`, and `accumulate-only — needs no multiplier`, which is the projected mask. `M` is
boolean, so `M · |E|` selects and sums columns of `|E|`: additions, no DSPs.

**Step 3 (per-query `K'`) is deliberately not a column.** It is identical to step 2 on this metric
— one `K'` per query still leaves each query attending to `k` summaries — so giving it a column
only invites the reading that it is equally good. It lives in the slide note instead, with the
number that actually rules it out: 786 M of data × weight at `k = 64`, **6.3×** the model's entire
125.8 M budget (393 M and 3.1× at `k = 32`).

## Where the numbers come from

Every figure on slides 5 and 6 is [`../mac_budget.py`](../mac_budget.py) — run it rather than
re-deriving. Step 0 reproduces the model-size deck's 65.5 M, which is the check that this deck uses
that convention and not a parallel one. The bucket rule is section 5.1 of the study README,
including the trap that reverses the sign of the answer if `Eᵀ K` is counted as data × data.

Parameter counts for the same four steps are in section 8 of the study README; they are the
**reference** config at `dim = 256`, not the 820 k model these MACs describe.
