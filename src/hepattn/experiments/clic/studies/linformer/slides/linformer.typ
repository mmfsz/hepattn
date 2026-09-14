// =========================================================================
//  Linformer attention on the PAPER TAG (`clic-paper`) — what compression can
//  and cannot remove from the data x data budget.
//
//  Three slides:
//    1  the cost model, in symbols — which multiplications land in the DSP
//       budget and which do not
//    2  the same table at k = 64, the value configs/linformer.yaml uses
//    3  the same table at k = 32
//
//  THE METRIC. data x data = both operands come from the event, so a real
//  multiplier is needed: Q K^T, scores . V, the mask head bilinear. Anything
//  multiplied by a LEARNED matrix -- including Linformer's compression E^T K
//  and avenue 2's mask projection M . E -- is data x weight and is counted
//  separately (the model-size deck: 125.8 M of those against 65.5 M here).
//  Putting E^T K in the wrong bucket reverses the sign of the answer; see
//  "The trap" in ../README.md section 5.1.
//
//  Sources for every number quoted here:
//    ../mac_budget.py                       -- every figure on slides 2 and 3; run it
//    ../README.md section 5.1               -- the metric and the bucket rule
//    ../README.md sections 8 and 8.1        -- the steps, and why kv_ca is out of reach
//    ../../model_size/slides/model_size.typ -- slide 7, the baseline column
//
//  Shapes are the 820 k paper-tag model (`base_small`, dim 64), NOT the 12.4 M
//  reference that `configs/linformer.yaml` trains.
//
//  Compile:  typst compile --root .. linformer.typ
// =========================================================================

#import "template.typ": *

#show: deck.with(
  title: "Linformer and the data × data budget",
  subtitle: "paper tag (clic-paper), 820 k reference",
  event: "hepattn CLIC / MaskFormer particle flow",
  date: "September 14, 2026",
)


// ==================== 1. THE COST MODEL ====================

#cslide("The cost model: which multiplies need a DSP")[
  #flow(80pt, 185pt, 1760pt, gap: 0.9cm)[
    #bold[data × data] — both operands come from the event, so a real multiplier is needed. That is
    the scarce resource. #bold[data × weight], one operand a trained constant, is counted separately.
    Linformer's compression $K' = E^T K$ multiplies activations by a #emph[learned] matrix, so it is
    data × weight: it #bold[moves] work out of the DSP budget rather than adding to it.
  ]

  #at(80pt, 330pt, box(width: 1760pt)[
    #set text(size: 24pt)
    #table(
      columns: (auto, 1fr, auto, auto, auto, auto, auto, auto),
      stroke: none,
      align: (center, left, center, center, center, center, center, center),
      inset: (x: 11pt, y: 9pt),
      [], [#bold[Site]], [#bold[calls]],
        [#bold[baseline]],
        [#bold[Linformer] #text(size: 20pt, fill: muted)[the limit]],
        [#bold[step 1]], [#bold[step 2]],
        [#bold[per-query] #text(size: 20pt, fill: warn)[ruled out]],
      table.hline(stroke: 0.8pt),
      [①], [encoder self-attention], [6], [$2 S^2 D$], [$2 S k D$],
        [#text(fill: good)[$2 S k D$]], [#text(fill: good)[$2 S k D$]], [#text(fill: muted)[$2 S k D$]],
      table.hline(stroke: 0.4pt + luma(200)),
      [②], [cross-attention `q_ca`], [4], [$2 Q N D$], [$2 Q k D$],
        [$2 Q N D$], [#text(fill: good)[$2 Q k D$]], [#text(fill: muted)[$2 Q k D$]],
      table.hline(stroke: 0.4pt + luma(200)),
      [③], [query self-attention `q_sa`], [4], [$2 Q^2 D$], [$2 Q k D$],
        [#text(fill: good)[$2 Q k D$]], [#text(fill: good)[$2 Q k D$]], [#text(fill: muted)[$2 Q k D$]],
      table.hline(stroke: 0.4pt + luma(200)),
      [④], [reverse cross-attention `kv_ca`], [4], [$2 N Q D$],
        [#text(fill: warn)[out of reach]], [$2 N Q D$], [$2 N Q D$], [$2 N Q D$],
      table.hline(stroke: 0.4pt + luma(200)),
      [], [mask head bilinear], [5], [$Q N D$], [—], [$Q N D$], [$Q N D$], [$Q N D$],
      table.hline(stroke: 0.8pt),
      [], [#text(fill: muted)[data × weight added — no DSPs]], [], [#text(fill: muted)[—]],
        [], [#text(fill: muted)[$2 n_"kv" k D$]],
        [#text(fill: muted)[$+ Q N k$ #text(size: 18pt)[adds only]]],
        [#text(fill: warn)[$2 Q N k D$]],
      table.hline(stroke: 0.8pt),
    )
  ])

  #at(80pt, 880pt, box(width: 1760pt, text(size: 23pt, fill: muted)[
    $S = 168$ padded nodes, $Q = 150$ queries, $N = 160$ nodes, $D$ width, $k$ projected length.
    
    ④ is out
    of reach for two independent reasons (../README.md section 8.1): it carries the transposed mask,
    and its padding sits on the query axis — the one axis Linformer's zero-before-projection cannot
    touch.
  ]))
]


// ==================== 2 & 3. THE NUMBERS ====================
//
//  Columns are ALTERNATIVES, not a progression: step 0 turns masked attention off and adds no
//  Linformer; step 1 keeps masked attention and puts Linformer where there is no mask. Every
//  figure comes from ../mac_budget.py -- run it rather than editing numbers here.

#let brow(marker, name, calls, v0, v1, v2, v3, best: ()) = (
  [#marker], [#name], [#calls], [#v0],
  if 1 in best { [#text(fill: good)[#v1]] } else { [#v1] },
  if 2 in best { [#text(fill: good)[#v2]] } else { [#v2] },
  if 3 in best { [#text(fill: good)[#v3]] } else { [#v3] },
)

#let budget-table(rows, totals, pcts, dxw, adds) = box(width: 1760pt)[
  #set text(size: 25pt)
  #table(
    columns: (auto, 1fr, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, center, right, right, right, right),
    inset: (x: 12pt, y: 8pt),
    [], [#bold[Site]], [#bold[calls]],
      [#bold[baseline]],
      [#bold[step 0] #text(size: 20pt, fill: muted)[mask off]],
      [#bold[step 1] #text(size: 20pt, fill: muted)[unmasked]],
      [#bold[step 2] #text(size: 20pt, fill: muted)[proj. mask]],
    table.hline(stroke: 0.8pt),
    ..rows,
    table.hline(stroke: 0.8pt),
    [], [#bold[all data × data]], [],
      [#bold[#totals.at(0)]],
      [#bold[#totals.at(1)] #text(size: 20pt, fill: muted)[#pcts.at(1)]],
      [#bold[#totals.at(2)] #text(size: 20pt, fill: muted)[#pcts.at(2)]],
      [#text(fill: good)[#bold[#totals.at(3)]] #text(size: 20pt, fill: muted)[#pcts.at(3)]],
    table.hline(stroke: 0.4pt + luma(200)),
    [], [#text(fill: muted)[data × weight added here]], [],
      ..dxw.map(v => [#text(size: 22pt, fill: muted)[#v]]),
    table.hline(stroke: 0.4pt + luma(200)),
    [], [#text(fill: muted)[accumulate-only — needs no multiplier]], [],
      ..adds.map(v => [#text(size: 22pt, fill: muted)[#v]]),
    table.hline(stroke: 0.8pt),
  )
]

#let budget-note(perq, times) = box(width: 1760pt, text(size: 21pt, fill: muted)[
  Step 0 is cheaper for one reason: four of the mask head's five calls exist only to drive masked
  attention, so with it off they are never evaluated. No approximation, no new mechanism.
  #linebreak()
  The accumulate-only row is the projected mask $M dot |E|$. $M$ is boolean, so it selects and
  sums columns of $|E|$ — additions, not multiplications, and no DSPs.
  #linebreak()
  #text(fill: warn)[Per-query projection is ruled out on cost:] it makes $E$ depend on the query,
  so compression becomes $2 Q N k D$ per call — #perq here, #bold[#times] the model's entire
  125.8 M data × weight budget. Its data × data would match step 2; nothing else would.
])


#cslide("Where the multiplies go — k = 64")[
  #at(80pt, 185pt, budget-table(
    (
      ..brow([①], [encoder self-attention — node ↔ node], [6], [21.7 M], [21.7 M], [8.3 M], [8.3 M], best: (2, 3)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([②], [cross-attention `q_ca` — queries ← nodes], [4], [12.3 M], [12.3 M], [12.3 M], [4.9 M], best: (3,)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([③], [query self-attention `q_sa` — query ↔ query], [4], [11.5 M], [11.5 M], [4.9 M], [4.9 M], best: (2, 3)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([④], [reverse cross-attention `kv_ca` — #text(fill: warn)[out of reach]], [4],
        [12.3 M], [12.3 M], [12.3 M], [12.3 M]),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([], [mask head bilinear], [5], [7.7 M], [1.5 M], [7.7 M], [7.7 M], best: (1,)),
    ),
    ("65.5 M", "59.3 M", "45.4 M", "38.1 M"),
    ("", "(−9%)", "(−31%)", "(−42%)"),
    ("—", "—", "13.2 M", "18.4 M"),
    ("—", "—", "—", "6.1 M"),
  ))
  #at(80pt, 585pt, budget-note[786 M][6.3×])
]

#cslide("Where the multiplies go — k = 32")[
  #at(80pt, 185pt, budget-table(
    (
      ..brow([①], [encoder self-attention — node ↔ node], [6], [21.7 M], [21.7 M], [4.1 M], [4.1 M], best: (2, 3)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([②], [cross-attention `q_ca` — queries ← nodes], [4], [12.3 M], [12.3 M], [12.3 M], [2.5 M], best: (3,)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([③], [query self-attention `q_sa` — query ↔ query], [4], [11.5 M], [11.5 M], [2.5 M], [2.5 M], best: (2, 3)),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([④], [reverse cross-attention `kv_ca` — #text(fill: warn)[out of reach]], [4],
        [12.3 M], [12.3 M], [12.3 M], [12.3 M]),
      table.hline(stroke: 0.4pt + luma(200)),
      ..brow([], [mask head bilinear], [5], [7.7 M], [1.5 M], [7.7 M], [7.7 M], best: (1,)),
    ),
    ("65.5 M", "59.3 M", "38.8 M", "29.0 M"),
    ("", "(−9%)", "(−41%)", "(−56%)"),
    ("—", "—", "6.6 M", "9.2 M"),
    ("—", "—", "—", "3.1 M"),
  ))
  #at(80pt, 585pt, budget-note[393 M][3.1×])
]
