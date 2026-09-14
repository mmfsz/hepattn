// =========================================================================
//  Linformer attention on the PAPER TAG (`clic-paper`) — what compression can
//  and cannot remove from the data x data budget.
//
//  One slide. It re-costs the four attention sites of the model-size deck's
//  "data x data MACs" slide
//  (../../model_size/slides/model_size.typ, slide 7) under three options:
//
//    now          the model as trained — ordinary attention everywhere
//    shared       Linformer at every site, if masked attention could use it
//    per-query    Linformer at every site, with the mask applied before the
//                 projection — the option section 8 of ../README.md rules out
//
//  Sources for every number quoted here:
//    ../../model_size/slides/model_size.typ -- the `now` column and the shapes,
//                                              unchanged; this deck only adds columns
//    ../README.md sections 3, 5 and 8       -- the cost model and the ruled-out option
//    ../../../../../../docs/FPGA_ATTENTION_STUDY.md -- the four attention sites
//
//  Shapes are the 820 k paper-tag model (`base_small`, dim 64), NOT the 12.4 M
//  reference that `configs/linformer.yaml` trains. The percentages are shape
//  ratios and carry over; the absolute MACs do not.
//
//  Every number is an arithmetic estimate, not a measurement -- ../README.md
//  section 5 makes the same caveat.
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


#cslide("Where Linformer can cut data × data")[
  #flow(80pt, 185pt, 1760pt, gap: 0.9cm)[
    Linformer compresses the key/value sequence to $k$ virtual tokens before attending, so a site
    costs $2 dot "kv-len" dot k dot D$ to build $K'$ and $V'$ and $2 dot "q-len" dot k dot D$ to
    attend — instead of $2 dot "q-len" dot "kv-len" dot D$. #bold[It needs an unmasked site]: the
    two cross-attentions carry a per-query mask, and applying it before the projection forces one
    $K'$ per query.
  ]

  #at(80pt, 320pt, box(width: 1760pt)[
    #set text(size: 25pt)
    #table(
      columns: (auto, 1fr, auto, auto, auto, auto, auto, auto),
      stroke: none,
      align: (center, left, center, center, right, right, right, right),
      inset: (x: 10pt, y: 8pt),
      [], [#bold[Site]], [#bold[calls]], [#bold[Masked?]],
        [#bold[Step 0] \ #text(size: 20pt, fill: muted)[baseline]],
        [#text(fill: good)[#bold[Step 1]] \ #text(size: 20pt, fill: muted)[unmasked sites]],
        [#bold[Step 2] \ #text(size: 20pt, fill: muted)[with mask $M dot E$]],
        [#text(fill: warn)[#bold[Step 3]] \ #text(size: 20pt, fill: muted)[per-query $K'$]],
      table.hline(stroke: 0.8pt),
      [①], [encoder self-attention — node ↔ node #text(size: 20pt, fill: muted)[$168 times 168$]],
        [6], [no], [21.7 M], [#text(fill: good)[16.5 M]], [16.5 M], [16.5 M],
      table.hline(stroke: 0.4pt + luma(200)),
      [②], [cross-attention `q_ca` — queries ← nodes #text(size: 20pt, fill: muted)[$150 times 160$]],
        [4], [#text(fill: warn)[yes]], [12.3 M], [12.3 M], [10.2 M], [#text(fill: warn)[791 M]],
      table.hline(stroke: 0.4pt + luma(200)),
      [③], [query self-attention `q_sa` — query ↔ query #text(size: 20pt, fill: muted)[$150 times 150$]],
        [4], [no], [11.5 M], [#text(fill: good)[9.8 M]], [9.8 M], [9.8 M],
      table.hline(stroke: 0.4pt + luma(200)),
      [④], [reverse cross-attention `kv_ca` — nodes ← queries #text(size: 20pt, fill: muted)[$160 times 150$]],
        [4], [#text(fill: warn)[yes]], [12.3 M], [12.3 M], [10.2 M], [#text(fill: warn)[791 M]],
      table.hline(stroke: 0.4pt + luma(200)),
      [], [mask head bilinear], [5], [—], [7.7 M], [7.7 M], [7.7 M], [7.7 M],
      table.hline(stroke: 0.4pt + luma(200)),
      [], [#emph[mask projection] $M dot E$ #text(size: 20pt, fill: muted)[— what Step 2 adds]],
        [8], [—], [—], [—], [#text(fill: warn)[12.3 M]], [—],
      table.hline(stroke: 0.8pt),
      [], [#bold[all data × data]], [], [],
        [#bold[65.5 M]], [#text(fill: good)[#bold[58.6 M]] #text(size: 20pt, fill: muted)[(−10%)]],
        [#bold[66.6 M] #text(size: 20pt, fill: muted)[(+2%)]],
        [#text(fill: warn)[#bold[1.62 G]] #text(size: 20pt, fill: muted)[(25×)]],
      table.hline(stroke: 0.8pt),
      [], [#text(size: 22pt, fill: muted)[#emph[ceiling] — Step 2 if the mask projection were free]],
        [], [], [], [], [#text(size: 22pt, fill: muted)[54.3 M (−17%)]], [],
    )
  ])

  #at(80pt, 745pt, box(width: 1760pt, text(size: 22pt, fill: muted)[
    $S = 168$ padded nodes, $Q = 150$ queries, $N = 160$ nodes, $D = 64$ width, $k = 64$ projected
    length — the shapes of the model-size deck's data × data slide, which #bold[Step 0] reproduces.
    Arithmetic estimates, not measurements. Parameter counts for the same four steps are in
    ../README.md section 8 (reference config, $"dim" = 256$): #bold[+0, +215 k, +387 k, +387 k].

    #bold[Step 1] is the only one buildable today — Linformer needs an unmasked site, and ① and ③
    are the unmasked ones. #bold[Step 2] adds a projected mask so the masked sites can compress too,
    but $M dot E$ costs $Q N k$ per call, rebuilt every decoder layer: 12.3 M, more than the 4.3 M the
    compression buys there. #bold[Step 3] applies the mask before the projection, which forces
    $K'_i = E^T "diag"(M_i) K$ — one $K'$ per query, $Q N k D$ instead of $N k D$. No extra
    parameters in any step beyond the projections, only MACs.
  ]))

]
