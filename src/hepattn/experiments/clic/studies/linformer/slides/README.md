# Slides

Typst deck for the Linformer study. One slide: the four attention sites of the model-size deck's
data × data budget, re-costed under Linformer.

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

`--root ..` matches the model-size deck and leaves room for `../figures/` if the study ever grows
figures; nothing outside `slides/` is embedded yet.

## The one slide

Columns are three options for the same model, not three models:

| column | what it is |
|---|---|
| `now` | ordinary attention everywhere — reproduces the model-size deck's data × data slide unchanged |
| Linformer, shared `K'` | the ceiling: Linformer at every site, *if* masked attention could use one. The cells at ② and ④ carry a † because nothing implements that today — only ① and ③ are green |
| Linformer, per-query `K'` | the mask applied before the projection — the option [../README.md](../README.md) section 8 rules out |

Colour carries the status: green = available now, grey with † = hypothetical, red = ruled out.
The accent row, **58.6 M (−10 %)**, is the only total reachable on current code: Linformer at the two
unmasked sites (① encoder self-attention, ③ query self-attention), ordinary masked attention at ②
and ④. That is avenue 1 of section 8.

Shapes are the **820 k paper-tag model** (`base_small`, `dim = 64`) so the numbers line up with the
model-size deck — *not* the 12.4 M reference that `configs/linformer.yaml` trains. The percentages
are shape ratios and carry over; the absolute MACs do not.

Every number is an arithmetic estimate, not a measurement — the same caveat as section 5 of
[../README.md](../README.md). No MAC counter exists yet.
