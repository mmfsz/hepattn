# Slides

Typst deck for the paper-tag model-size study. Copied from the head-based deck on `main`
(`src/hepattn/experiments/clic/studies/model_size/slides/`) and re-based on this branch's model
and results — see the comment block at the top of `model_size.typ` for exactly which slides were
replaced, which were rewritten, and which were deliberately left alone.

| file | what it is |
|---|---|
| `model_size.typ` | the deck |
| `template.typ` | the reusable deck template, copied verbatim from head's. Keep the two in sync by copying, not by editing one of them |
| `logos/` | FSU lockups used by the title slide and the footer bar |

## Building

```bash
typst compile --root .. model_size.typ      # -> model_size.pdf
typst watch   --root .. model_size.typ      # live preview while editing
```

**`--root ..` is load-bearing.** The deck embeds the study's own figures from `../figures/`, and
Typst refuses any path that escapes the project root — which defaults to the input file's
directory. Pointing the root one level up at `studies/model_size/` puts both `slides/` and
`figures/` inside it. Without the flag the build fails with *"would escape the project root"*.

Needs `typst` (0.15 here) and network access on the first build, to fetch `@preview/touying:0.7.4`
into the local package cache.

## Figures

The deck embeds PNGs from `../figures/`, produced by the plot scripts one level up (see
[../README.md](../README.md)). Only one arm set exists on this branch so far, `all`, so every
figure carries the `size_ablation_` prefix.

| figure | produced by | on which slide |
|---|---|---|
| `size_ablation_<convention>_jet_median_iqr.png` | `plot_size_ablation_jet_iqr.py` | 9 — the jet-energy response |
| `size_ablation_jet_iqr.png` | same | not in the deck; the 2×2 working figure, both conventions at once |
| `size_ablation_<convention>_*.png` | `plot_size_ablation_performance.py` | not in the deck; the breadth check |
| `size_ablation_decoder_layers.png` | `plot_decoder_layers.py` | not in the deck |

**The deck does not use `plot_jet_response`'s output.** That helper stacks three panels vertically
on this branch — median, IQR, and IQR/median — where head's put median and IQR side by side with
no ratio. Rather than restyle a helper `performance.ipynb` shares, the per-convention slide figure
is drawn by `plot_size_ablation_jet_iqr.py`, which already has the numbers, the bootstrap errors
and the Pandora backdrop.

## What is missing relative to head's deck

Head's deck had a measured σ_repro slide and a round-3 figure. Neither exists here:

- **σ_repro is unmeasured on this code.** Slide 12 says so rather than quoting head's 0.0007,
  which does not transfer across the activation, norm and incidence-head differences. Every Δ in
  the deck therefore carries σ_stat only, and is an ordering rather than a verdict.
- **The singles A2/A3/A4 and the depth arm A6 were queued 2026-09-14** and are marked *training*
  on slide 8. When they land, slide 8's table and the head comparison in `../README.md` both need
  updating.
