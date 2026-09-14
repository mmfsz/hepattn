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


// ==================== 0. HOW IT WORKS ====================
//
//  Three pedagogical slides before the costing. One worked example runs through all three:
//  6 constituents, 3 summaries, 4 queries. The numbers are chosen so that the last slide shows
//  both the mechanism AND its failure mode -- queries q1..q3 claim constituents that line up with
//  the summaries and keep a sharp mask; q4's claims are spread across all three, so its projected
//  mask comes out nearly flat and cancels in the softmax. That is the Var(w) check of README 11.5,
//  made visible rather than asserted.

#let yes = box(width: 25pt, height: 25pt, fill: good, radius: 3pt)
#let no = text(size: 26pt, fill: luma(185))[·]

#let mat(colhdr, rowhdr, rows, cellw: 66pt, cellh: 50pt, size: 25pt, hdrw: 58pt) = {
  let cells = ([],)
  for h in colhdr { cells.push(text(size: size, fill: muted)[#h]) }
  for i in range(rows.len()) {
    cells.push(text(size: size, fill: muted)[#rowhdr.at(i)])
    for c in rows.at(i) { cells.push(c) }
  }
  table(
    columns: (hdrw,) + (cellw,) * colhdr.len(),
    rows: (cellh,) * (rows.len() + 1),
    stroke: 0.6pt + luma(205),
    align: center + horizon,
    inset: 3pt,
    ..cells,
  )
}

#let n(v, fill: black, w: "regular") = text(size: 24pt, fill: fill, weight: w)[#v]

#let CONST = ([c1], [c2], [c3], [c4], [c5], [c6])
#let QUERY = ([q1], [q2], [q3], [q4])
#let SUMM = ([A], [B], [C])

// the mask: which constituents each query currently claims
#let MASK = (
  (yes, yes, no, no, no, no),
  (no, no, yes, yes, no, no),
  (no, no, no, no, yes, yes),
  (yes, no, no, yes, no, yes),
)


#cslide("Masked attention: the mask names constituents")[
  #flow(90pt, 180pt, 1740pt, gap: 0.7cm)[
    Each object query attends only to the constituents it currently claims. The mask is one
    yes/no per #bold[(query, constituent)] pair, rebuilt at every decoder layer.
  ]

  #at(110pt, 330pt, mat(CONST, QUERY, MASK))
  #at(110pt, 300pt, text(size: 23pt, fill: muted)[$M$ — 4 queries × 6 constituents])

  #at(700pt, 330pt, box(width: 1130pt)[
    #set text(size: 27pt)
    #set list(spacing: 0.75cm)
    - Scores $Q K^T$ have the #bold[same shape] as $M$: one number per (query, constituent).
    - A masked entry is set to $-infinity$ before the softmax, so its weight is exactly 0.
    - #text(fill: accent)[Column 4 is constituent 4.] To exclude it, you zero that column.
  ])

  #at(110pt, 625pt, box(width: 1740pt, fill: panel-fill, inset: 22pt)[
    #text(size: 28pt)[#bold[The property that matters:] every column of the scores #emph[is] one
    constituent, so "query 2 may not see constituent 5" has somewhere to point.]
  ])
]


#cslide("Linformer: the constituents stop existing")[
  #flow(90pt, 180pt, 1740pt, gap: 0.7cm)[
    Linformer replaces the 6 constituents by #bold[3 learned summaries]. Summary $A$ is a fixed
    weighted blend of #emph[all] of them: $K'_A = 0.8 K_1 + 0.7 K_2 + 0.1 K_3 + ...$
  ]

  #at(110pt, 340pt, mat(SUMM, CONST, (
    (n[0.8], n[0.1], n[0.1]),
    (n[0.7], n[0.2], n[0.1]),
    (n[0.1], n[0.8], n[0.1]),
    (n(fill: warn)[0.2], n(fill: warn)[0.7], n(fill: warn)[0.1]),
    (n[0.1], n[0.1], n[0.8]),
    (n[0.1], n[0.2], n[0.7]),
  ), cellw: 78pt))
  #at(110pt, 310pt, text(size: 23pt, fill: muted)[$E$ — 6 constituents × 3 summaries])

  #at(590pt, 340pt, box(width: 1240pt)[
    #set text(size: 27pt)
    #set list(spacing: 0.7cm)
    - Scores are now 4 × #bold[3]: each query attends to 3 summaries, not 6 constituents.
      That is the saving.
    - #text(fill: warn)[But which column is constituent 4?] It is 0.2 of $A$, 0.7 of $B$ and
      0.1 of $C$ — a bit of every column, and no column of its own.
    - The weights depend on the #bold[slot], not on what is in it. The blend is the same in
      every event.
  ])

  #at(110pt, 745pt, box(width: 1740pt, fill: panel-fill, inset: 22pt)[
    #text(size: 28pt)[#bold[Why the mask cannot come along:] it says "not constituent 4", and
    after the projection there is no constituent 4 left to exclude. Zeroing a column of $M$ now
    removes part of everything.]
  ])
]


#cslide("Step 2: project the mask the same way")[
  #flow(90pt, 178pt, 1740pt, gap: 0.65cm)[
    If the keys are blended by $E$, blend the mask by $E$ too: $w = (M |E|) / ("valid" |E|)$ —
    #bold[the fraction of each summary a query is allowed to see.] Add $log w$ to the scores: 1
    changes nothing, 0 is the hard mask.
  ]

  #at(110pt, 360pt, mat(SUMM, QUERY, (
    (n(w: "bold")[0.75], n[0.14], n[0.11]),
    (n[0.15], n(w: "bold")[0.71], n[0.11]),
    (n[0.10], n[0.14], n(w: "bold")[0.79]),
    (n(fill: warn)[0.55], n(fill: warn)[0.48], n(fill: warn)[0.47]),
  ), cellw: 92pt))
  #at(110pt, 330pt, text(size: 23pt, fill: muted)[$w$ — 4 queries × 3 summaries])

  #at(640pt, 360pt, box(width: 1190pt)[
    #set text(size: 27pt)
    #set list(spacing: 0.7cm)
    - #text(fill: good)[q1 claims c1, c2 — which are mostly summary $A$.] It keeps a sharp mask:
      0.75 against 0.14 and 0.11.
    - #text(fill: warn)[q4 claims c1, c4, c6 — one from each summary.] Its row comes out nearly
      flat: 0.55, 0.48, 0.47.
  ])

  #at(110pt, 620pt, box(width: 1740pt, fill: panel-fill, inset: 22pt)[
    #set text(size: 27pt)
    #bold[The risk, and the test.] A row that is flat across summaries adds the #emph[same]
    number to every score, and a constant shift #bold[cancels in the softmax] — so for q4 the
    mask does nothing at all. #linebreak()
    Before spending a training run, measure how much $w$ varies across summaries. If it is flat
    everywhere, step 2 is an expensive no-op and the answer is data-dependent summaries instead.
  ])
]


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
