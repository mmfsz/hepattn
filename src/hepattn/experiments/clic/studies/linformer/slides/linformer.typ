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


#let sq(c) = box(width: 34pt, height: 26pt, fill: c, radius: 2pt)
#let blue = rgb("#CFE0F5")
#let grey = rgb("#E4E4E4")

#cslide("Linformer: the constituents stop existing")[
  #flow(90pt, 175pt, 1740pt, gap: 0.6cm)[
    Linformer replaces the 6 constituents by #bold[3 learned summaries]. Summary $A$ is a fixed
    linear combination of #emph[all] of them, with the same coefficients in every event.
  ]

  #at(110pt, 300pt, mat((text(fill: accent)[A], [B], [C]), CONST, (
    (n[0.8], n[0.1], n[0.1]),
    (n[0.7], n[0.2], n[0.1]),
    (n[0.1], n[0.8], n[0.1]),
    (n(fill: warn)[0.2], n(fill: warn)[0.7], n(fill: warn)[0.1]),
    (n[0.1], n[0.1], n[0.8]),
    (n[0.1], n[0.2], n[0.7]),
  ), cellw: 74pt, cellh: 46pt))
  #at(110pt, 272pt, text(size: 22pt, fill: muted)[$E$ — 6 constituents × 3 summaries])
  #at(110pt, 655pt, box(width: 520pt)[
    #text(size: 22pt, fill: muted)[Read #text(fill: accent)[column $A$] downwards — those six
    numbers #emph[are] its coefficients:]
    #v(6pt)
    #text(size: 23pt)[
      $ K'_A = &0.8 K_1 + 0.7 K_2 + 0.1 K_3 \
               &+ 0.2 K_4 + 0.1 K_5 + 0.1 K_6 $
    ]
  ])

  #at(610pt, 290pt, box(width: 1240pt)[
    #set text(size: 26pt)
    #show math.equation.where(block: true): set align(left)
    #text(size: 22pt, fill: muted)[ordinary attention]
    #v(2pt)
    $ underbrace(S, 4 times 6) = underbrace(Q, 4 times d) thin underbrace(K^T, d times 6) $
    #v(14pt)
    #text(size: 22pt, fill: muted)[Linformer — two steps]
    #v(2pt)
    $ underbrace(K', 3 times d) = underbrace(E^T, 3 times 6) thin underbrace(K, 6 times d)
      quad quad underbrace(S', 4 times 3) = underbrace(Q, 4 times d) thin underbrace(K'^T, d times 3) $
  ])

  #at(640pt, 620pt, box[
    #text(size: 22pt, fill: muted)[$S$ — one column per constituent]
    #v(4pt)
    #mat(CONST, QUERY, ((sq(blue),) * 6,) * 4, cellw: 52pt, cellh: 38pt, size: 21pt, hdrw: 46pt)
  ])
  #at(1040pt, 690pt, text(size: 46pt, fill: muted)[→])
  #at(1130pt, 620pt, box[
    #text(size: 22pt, fill: muted)[$S'$ — one column per #text(fill: warn)[summary]]
    #v(4pt)
    #mat(SUMM, QUERY, ((sq(grey),) * 3,) * 4, cellw: 52pt, cellh: 38pt, size: 21pt, hdrw: 46pt)
  ])

  #at(110pt, 895pt, box(width: 1740pt, fill: panel-fill, inset: 20pt)[
    #text(size: 27pt)[#bold[Why the mask cannot come along:] $M$ is 4 × 6 — it names constituents.
    $S'$ is 4 × #bold[3]. #text(fill: warn)[Constituent 4 is 0.2 of $A$, 0.7 of $B$, 0.1 of $C$] —
    a bit of every column and no column of its own, so "not constituent 4" has nothing to point at.]
  ])
]


#let op(x, y, sym) = at(x, y, text(size: 40pt, fill: muted)[#sym])

#cslide("Step 2: project the mask the same way")[
  #flow(90pt, 172pt, 1740pt, gap: 0.5cm)[
    $E$ combines the keys. Combine the mask the same way, then divide by what an
    #bold[unmasked] query would get — so $w$ is #bold[the fraction of each summary a query may see.]
  ]

  // M  x  |E|  =  M|E|  ÷ column sums  =  w
  #at(90pt, 330pt, box[
    #text(size: 21pt, fill: muted)[$M$ — the mask]
    #v(3pt)
    #mat(CONST, QUERY, MASK, cellw: 50pt, cellh: 38pt, size: 21pt, hdrw: 44pt)
  ])
  #op(500pt, 415pt, [×])

  #at(560pt, 292pt, box[
    #text(size: 21pt, fill: muted)[$|E|$]
    #v(3pt)
    #mat(SUMM, CONST + (text(size: 18pt, fill: accent)[valid$|E|$],), (
      (n[0.8], n[0.1], n[0.1]), (n[0.7], n[0.2], n[0.1]), (n[0.1], n[0.8], n[0.1]),
      (n[0.2], n[0.7], n[0.1]), (n[0.1], n[0.1], n[0.8]), (n[0.1], n[0.2], n[0.7]),
      (n(fill: accent)[2.0], n(fill: accent)[2.1], n(fill: accent)[1.9]),
    ), cellw: 58pt, cellh: 38pt, size: 21pt, hdrw: 72pt)
  ])
  #op(840pt, 415pt, [=])

  #at(900pt, 330pt, box[
    #text(size: 21pt, fill: muted)[$M|E|$]
    #v(3pt)
    #mat(SUMM, QUERY, (
      (n[1.5], n[0.3], n[0.2]), (n[0.3], n[1.5], n[0.2]),
      (n[0.2], n[0.3], n[1.5]), (n[1.1], n[1.0], n[0.9]),
    ), cellw: 64pt, cellh: 38pt, size: 21pt, hdrw: 44pt)
  ])
  #at(1160pt, 392pt, box(width: 170pt, align(center, text(size: 21pt, fill: accent)[
    normalise #linebreak() by valid$|E|$ #linebreak() #text(size: 34pt)[→]
  ])))

  #at(1340pt, 330pt, box[
    #text(size: 21pt, fill: muted)[$w$ — what the query may see]
    #v(3pt)
    #mat(SUMM, QUERY, (
      (n(w: "bold", fill: good)[0.75], n(fill: good)[0.14], n(fill: good)[0.11]),
      (n[0.15], n(w: "bold")[0.71], n[0.11]),
      (n[0.10], n[0.14], n(w: "bold")[0.79]),
      (n(fill: warn)[0.55], n(fill: warn)[0.48], n(fill: warn)[0.47]),
    ), cellw: 66pt, cellh: 38pt, size: 21pt, hdrw: 44pt)
  ])
  #at(1655pt, 432pt, text(size: 22pt, fill: good)[sharp])
  #at(1655pt, 546pt, text(size: 22pt, fill: warn)[flat])

  #at(90pt, 760pt, box(width: 1740pt, text(size: 26pt)[
    Then add $log w$ to the scores — $1$ changes nothing, $0$ is the hard mask.
  ]))

  #at(90pt, 845pt, box(width: 1740pt, fill: panel-fill, inset: 20pt)[
    #text(size: 26pt)[
      #text(fill: good)[q1 claims c1, c2 — both mostly summary $A$], so its mask survives.
      #text(fill: warn)[q4 claims c1, c4, c6 — one from each summary], so its row is flat.
      #linebreak()
      #bold[A flat row adds the same number to every score, and a constant cancels in the softmax]
      — so q4 gets no mask at all. Measure how much $w$ varies before spending a training run.
    ]
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
