// =========================================================================
//  Shrinking the CLIC particle-flow model, on the PAPER TAG (`clic-paper`).
//
//  Copied from the head-based deck on `main`
//  (src/hepattn/experiments/clic/studies/model_size/slides/model_size.typ) and re-based on this
//  branch's model and results. What changed, and what deliberately did not:
//
//    slide 5  parameters   REPLACED -- 819,683, not head's 702,395. The attention projections are
//                          identical (same shapes); the MLPs are 1.50x head's because the paper's
//                          `Dense` is gated (SwiGLU); the mask and incidence heads swap sizes.
//    slide 7  data x data  UNCHANGED, and that is the correct answer: both models share dim,
//                          layer counts, heads, node and query counts, so every shape-derived
//                          number is identical. The constant-weight total is NOT quoted -- head's
//                          126 M is its own model under its own convention.
//    slide 8  experiments  REPLACED -- this branch's arms and parameter counts. S1 and C2 dropped
//                          (never trained here); C1 kept, because here it trained.
//    slide 9  the plots    REPLACED -- median and IQR side by side, no ratio panel, from
//                          `plot_size_ablation_jet_iqr.py`. The performance helper's
//                          `plot_jet_response` stacks three panels vertically on this branch.
//    slide 10 resolution   NEW -- the jet-E IQR table, Pandora included.
//    slide 12 sigma_repro  REWRITTEN -- head measured it, this branch has not, and head's number
//                          does not transfer.
//
//  Sources for every number quoted here:
//    ../README.md                     -- the arms, the jobs, the results, the head comparison
//    logs/<run>/metadata.yaml         -- the trained runs' own parameter counts
//    slurm_logs/slurm-42119839.clic-ms-plots.out -- the jet-E IQR numbers on slides 9 and 10
//    ../../../../../../docs/FPGA_ATTENTION_STUDY.md -- the cost model and the L1 budget
//
//  Compile:  typst compile --root .. model_size.typ
// =========================================================================

#import "template.typ": *

#show: deck.with(
  title: "Shrinking the particle-flow model",
  subtitle: "paper tag (clic-paper), 820 k reference",
  event: "hepattn CLIC / MaskFormer particle flow",
  date: "September 14, 2026",
)


// ==================== OVERVIEW ====================

#cslide("Overview")[
  #flow(80pt, 195pt, 1760pt, gap: 1.0cm)[
    These studies aim at reducing the models' FPGA footprint by reducing the number of parameters and
    resources required by the model. We start from the #bold[820 k paper-tag model] (`base_small`,
    the tag `clic-paper`), not the `clic_v7` head model the earlier version of this deck used —
    head's model carries the jet-IQR regression this branch exists to escape.
  ]

  #at(80pt, 350pt, box(width: 1760pt)[
    #set list(spacing: 0.85cm)
    1. #bold[Where the cost is]: 819,683 parameters, of which 65.5 M multiply–accumulates (MACs)
      per event are data × data, and the resources they map onto. 
    2. #bold[Targeted changes and their performance]
  ])

]


#cslide("The model")[
  #flow(80pt, 190pt, 1760pt, gap: 0.8cm)[
    A #bold[MaskFormer] encoder–decoder. Up to 160 input nodes (reconstructed tracks and
    topological calorimeter clusters, one 27-dimensional vector each) and #bold[150 candidate output
    particles], each with a class, a set of claimed nodes and reconstructed kinematics.
  ]

  #cell(80pt, 340pt, 350pt, 150pt, [
    nodes \
    #text(size: 23pt, fill: muted)[$<= 160 times 27$]
  ], size: 27pt)
  #arw(445pt, 415pt, 490pt, 415pt)
  #cell(505pt, 340pt, 340pt, 150pt, [
    encoder \
    #text(size: 23pt, fill: muted)[6 layers, self-attention]
  ], size: 27pt)
  #arw(860pt, 415pt, 905pt, 415pt)
  #cell(920pt, 340pt, 420pt, 150pt, [
    decoder \
    #text(size: 23pt, fill: muted)[4 layers, 150 object queries]
  ], size: 27pt)
  #arw(1355pt, 415pt, 1400pt, 415pt)
  #cell(1415pt, 340pt, 425pt, 150pt, [
    four task heads \
    #text(size: 23pt, fill: muted)[class · mask · incidence · kinematics]
  ], size: 27pt, fill: panel-fill)

  #at(80pt, 550pt, box(width: 1760pt)[
    #set text(size: 29pt)
    #table(
      columns: (auto, auto, 1fr, auto, auto),
      stroke: none,
      align: (left, right, left, left, right),
      inset: (x: 16pt, y: 11pt),
      [width `dim`], [64], [], [object queries `Q`], [150],
      [heads × head dim], [8 × 8], [], [max nodes `N`], [160 (+8 registers)],
      [encoder layers], [6], [], [median nodes per real event], [63],
      [decoder layers], [4], [], [attention calls per forward pass], [18],
      [MLP hidden], [2 × `dim` #text(size: 21pt, fill: muted)[, gated]], [], [trainable parameters], [819,683],
    )
  ])

  #at(80pt, 855pt, callout(1760pt, size: 26pt)[
    #bold[On an FPGA every shape is fixed at compile time], so both blocks always
    run fully padded: a 63-node event costs exactly what a full one does.

  ])
]




// One decoder layer, step by step. Order verified against decoder.py:452-476 --
// q_ca, q_dense, q_sa, kv_ca, kv_dense, with the mask head run before all of them.

#cslide("Inside one decoder layer")[
  #flow(80pt, 182pt, 1760pt, gap: 1cm)[
    The decoder carries #bold[two sets of vectors] and improves both: #bold[150 object queries],
    one per candidate particle, and the #bold[160 node embeddings] the encoder produced. 
    
    Each layer has four steps:
  ]

  #cell(80pt, 350pt, 390pt, 200pt, [
    #bold[mask head] \
    #v(2pt)
    #text(size: 21pt)[$150 times 160$ scores] \
    #text(size: 21pt, fill: muted)[which hits belong \ to each particle?]
  ], size: 26pt, fill: panel-fill)
  #arw(478pt, 450pt, 529pt, 450pt)

  #cell(537pt, 350pt, 390pt, 200pt, [
    #bold[② cross-attention] \
    #v(2pt)
    #text(size: 21pt)[queries ← nodes] \
    #text(size: 21pt, fill: muted)[each particle reads \ the hits it claims]
  ], size: 26pt)
  #arw(935pt, 450pt, 986pt, 450pt)

  #cell(993pt, 350pt, 390pt, 200pt, [
    #bold[③ query self-attention] \
    #v(2pt)
    #text(size: 21pt)[queries ↔ queries] \
    #text(size: 21pt, fill: muted)[so two never claim \ the same particle]
  ], size: 26pt)
  #arw(1391pt, 450pt, 1442pt, 450pt)

  #cell(1450pt, 350pt, 390pt, 200pt, [
    #bold[④ reverse cross-attention] \
    #v(2pt)
    #text(size: 21pt)[nodes ← queries] \
    #text(size: 21pt, fill: muted)[the hits are updated \ from the particles]
  ], size: 25pt)

  #at(80pt, 575pt, box(width: 1760pt, text(size: 24pt, fill: muted)[
    Steps ② and ④ are each followed by a small MLP. 
  ]))

  #at(80pt, 650pt, box(width: 1760pt, text(size: 27pt)[
    #bold[This repeats in all four decoder layers.] The queries and nodes that come out are what
    the four task heads read (next slide).
  ]))

  #at(80pt, 760pt, callout(1760pt, size: 26pt)[
    #bold[Only ④ writes back to the nodes.] Without it the node embeddings stay exactly as the
    encoder left them for the whole decoder, which is what standard Mask2Former does, and what
    the S1 arm tests. 
    
    #bold[③ is the only place the queries see each other]: it is the mechanism
    that keeps two queries from reconstructing the same particle.
  ])
]



// What each task head computes and what it reads. Verified against task.py: classification
// (#95), mask (#479, einsum query x node), incidence (#1294, softmax over the query axis),
// regression (#1379, proxy from raw inputs + scale correction).

#cslide("What the four task heads compute")[
  #flow(80pt, 182pt, 1760pt, gap: 0.7cm)[
    After the last decoder layer, four heads turn the queries and nodes into the actual
    output.
  ]

  #at(80pt, 290pt, box(width: 1760pt)[
    #set text(size: 26pt)
    #table(
      columns: (auto, 1fr, auto),
      stroke: none,
      align: (left + top, left + top, left + top),
      inset: (x: 16pt, y: 14pt),
      [#bold[head]], [#bold[what it computes]], [#bold[what it reads]],
      table.hline(stroke: 0.8pt),
      [#bold[class]], [six class probabilities for each of the 150 queries],
        [the query alone],
      table.hline(stroke: 0.4pt + luma(200)),
      [#bold[mask]], [a $150 times 160$ score, one per (particle, hit) pair \
        #text(size: 23pt, fill: muted)[through a sigmoid: does this hit belong to this particle?]],
        [query × node],
      table.hline(stroke: 0.4pt + luma(200)),
      [#bold[incidence]], [the same kind of score, but normalised down the #emph[query] axis \
        #text(size: 23pt, fill: muted)[what fraction of this hit belongs to each particle — so one
        cluster can be shared]],
        [query × node],
      table.hline(stroke: 0.4pt + luma(200)),
      [#bold[kinematics]], [$E$, $p_T$, $eta$, $sin phi$, $cos phi$ for each particle],
        [query, node and \ the measured hits],
    )
  ])

  #at(80pt, 700pt, callout(1760pt, size: 25pt)[
    #bold[Only the kinematics head looks at the detector again.] It first builds a guess out of the
    measured hits the incidence assigns it — for a charged particle, the kinematics of its
    most-weighted track; for a neutral one, an energy-weighted sum over its clusters — and the
    network then only #emph[scales] that guess.

    #v(0.4em)
    #bold[That guess on its own is `mpflow_proxy`; the scaled version is `mpflow`.] 
  ])
]


// §2 of STUDY.md. Both columns computed twice -- analytically from the config and by summing
// the instantiated model's parameters -- and they agree exactly, matching metadata.yaml.

#cslide("Where the 819,683 parameters live")[
  #at(80pt, 185pt, box(width: 900pt)[
    #set text(size: 28pt)
    #table(
      columns: (1fr, auto, auto),
      stroke: none,
      align: (left, right, right),
      inset: (x: 14pt, y: 10pt),
      [#bold[component]], [#bold[params]], [#bold[%]],
      table.hline(stroke: 0.8pt),
      [input embedding `27→54→64`], [6,544], [0.8%],
      [Fourier position encoding], [#text(fill: muted)[0 — frozen]], [0.0%],
      [encoder register tokens], [512], [0.1%],
      [#bold[encoder, 6 layers]], [#bold[251,816]], [#bold[30.7%]],
      [decoder query codebook], [9,600], [1.2%],
      [#bold[decoder, 4 layers]], [#bold[398,848]], [#bold[48.7%]],
      [task head — classification], [16,806], [2.1%],
      [task head — object–hit mask], [49,792], [6.1%],
      [task head — incidence], [24,960], [3.0%],
      [task head — regression], [60,805], [7.4%],
      table.hline(stroke: 0.8pt),
      [#bold[total]], [#bold[819,683]], [100%],
    )
  ])

  #at(1020pt, 185pt, box(width: 820pt)[
    #set text(size: 28pt)
    #table(
      columns: (1fr, auto, auto),
      stroke: none,
      align: (left, right, right),
      inset: (x: 14pt, y: 10pt),
      [#bold[cut a different way]], [#bold[params]], [#bold[%]],
      table.hline(stroke: 0.8pt),
      [attention projections Q/K/V/O], [299,520], [#bold[36.5%]],
      [transformer MLPs #text(size: 22pt, fill: muted)[— SwiGLU, 3 matrices]], [348,544], [#bold[42.5%]],
      [all four task heads], [152,363], [18.6%],
      [input embedding, learned tokens], [19,256], [2.3%],
    )
  ])

  #at(1020pt, 480pt, callout(820pt, size: 25pt)[
    #bold[79.0% is projections and MLPs, and both scale as $D^2$.] Halving `dim` quarters
    four fifths of the model. This is the single parameter lever that matters. \
    #text(size: 22pt)[The attention projections are #emph[identical] to head's 299,520 — same
    `dim`, layers and heads. The MLPs are #bold[1.50×] head's 232,064: the paper's `Dense` is
    gated (SwiGLU), three weight matrices where head's SiLU has two.]
  ])

  #at(80pt, 720pt, callout(1760pt, size: 26pt)[
    Note: #bold[this model has no parameter sharing of any kind]  \
    (1) The #bold[task-head hidden widths are hard-coded
    absolute numbers], not tied to `dim`, e.g. the regression head is `134→128→128→128→64→32→5`.
    When v6→v7 cut `dim` from 256 to 64 they barely moved, so they #emph[take over] as the model shrinks. \
    (2) #bold[No weights are tied anywhere],
    each of the four decoder layers carries a full private copy. \
    (3) #bold[The mask and incidence heads swap places against head]: 49,792 / 24,960 here
    against head's 16,576 / 49,664. Same two jobs, different split — this is the paper's
    incidence head, one of the three code differences from head.
  ])
]


// §3 of STUDY.md. The point of this slide is the second and third rows, not the first.

#cslide("Not every multiply costs the same")[
  #flow(80pt, 185pt, 1760pt, gap: 0.7cm)[
    Multiplies split into two kinds, which on an FPGA the two cost 20–40×
    differently:
  ]

  #cell(80pt, 255pt, 850pt, 290pt, [
    #bold[constant × data] \
    #v(5pt)
    #text(size: 23pt)[projections, MLPs, task heads] \
    #v(8pt)
    #text(size: 23pt, fill: muted)[the weight is known before the event arrives, \
    so the multiply becomes a fixed adder graph] \
    #v(8pt)
    #text(size: 25pt)[$approx$ 1.5–3 LUT each] \
    #bold[125.8 M per event]
  ], size: 28pt)

  #cell(990pt, 255pt, 850pt, 290pt, [
    #bold[data × data] \
    #v(5pt)
    #text(size: 23pt)[$Q K^T$, $"scores" dot V$, the mask head bilinear] \
    #v(8pt)
    #text(size: 23pt, fill: muted)[both numbers come from the event, \
    so a real multiplier is needed] \
    #v(8pt)
    #text(size: 25pt)[$approx$ half a DSP, or $approx$ 60 LUT each] \
    #bold[65.5 M per event]
  ], size: 28pt)

  #at(80pt, 600pt, callout(1760pt, size: 26pt)[
    The constant-weight column scales as D² while the data × data column scales only as D, so its share of all MACs per event therefore rises as the model gets smaller.
    
    Data × data also scales with occupancy squared. These numbers are at CLIC occupancy, but might change with HL-LHC values.

    The data x data usage is reducible only by a different attention formulation, or by cutting
    queries or nodes.
  ])

  #flow(80pt, 845pt, 1760pt, gap: 0.7cm)[
    #bold[Softmax exponentials and sigmoids] are a separate cost: 3.75 M per event, 96% of it
    attention softmax. Each one is a lookup table in on-chip memory, built once per concurrent
    lane, for 822 Mbit of BRAM. The count scales with the number of heads and with
    occupancy squared, and not with `dim`.
  ]


]


// The four attention sites, from FPGA_ATTENTION_STUDY.md §4.1 — verified with forward hooks on
// every Attention module. Site (4) is the one S1 deletes, which is why it gets its own slide.

#cslide("data x data MACs")[
  #flow(80pt, 185pt, 1760pt, gap: 0.9cm)[
    Attention runs in four place, once in the encoder and three times inside every
    decoder layer: #bold[6 + 3 × 4 = 18 calls per forward pass]. 
    
    Each decoder layer first re-runs
    the mask head. The same mask is then used in step ② and ④.
  ]

  #at(80pt, 345pt, box(width: 1760pt)[
    #set text(size: 27pt)
    #table(
      columns: (auto, 1fr, auto, auto, auto, auto, auto),
      stroke: none,
      align: (center, left, center, center, center, center, right),
      inset: (x: 13pt, y: 10pt),
      [], [#bold[Site]], [#bold[calls]], [#bold[Shape]], [#bold[Masked?]],
        [#bold[MACs per call]],
        [#bold[data × data MACs] \ #text(size: 22pt, fill: muted)[all calls, per event (% of 65.5 M)]],
      table.hline(stroke: 0.8pt),
      [①], [encoder self-attention — node ↔ node], [6], [$168 times 168$], [no],
        [$2 S^2 D$], [21.7 M #text(fill: muted)[(33%)]],
      table.hline(stroke: 0.4pt + luma(200)),
      [②], [cross-attention `q_ca` — queries ← nodes], [4], [$150 times 160$], [yes],
        [$2 Q N D$], [12.3 M #text(fill: muted)[(19%)]],
      table.hline(stroke: 0.4pt + luma(200)),
      [③], [query self-attention `q_sa` — query ↔ query], [4], [$150 times 150$], [no],
        [$2 Q^2 D$], [11.5 M #text(fill: muted)[(18%)]],
      table.hline(stroke: 0.4pt + luma(200)),
      [④], [reverse cross-attention `kv_ca` — nodes ← queries], [4], [$160 times 150$],
        [yes], [$2 N Q D$], [12.3 M #text(fill: muted)[(19%)]],
      table.hline(stroke: 0.8pt),
      [], [mask head bilinear],
        [5 \ #text(size: 21pt, fill: muted)[4 in ② #emph[and] ④, 1 in output]],
        [$150 times 160$], [—], [$Q N D$], [7.7 M #text(fill: muted)[(12%)]],
      table.hline(stroke: 0.8pt),
      [], [all data × data (what needs DSPs)], [], [], [], [], [65.5 M],
    )
  ])

  #at(80pt, 745pt, box(width: 1760pt, text(size: 24pt, fill: muted)[
    $S = 168$ padded nodes, $Q = 150$ queries, $N = 160$ nodes, $D = 64$ width. 
    
    Attention pays
    twice — once for $Q K^T$ and once for $"scores" dot V$ — the bilinear only once. 

    #bold[Every number in this table is unchanged from head's deck, and that is correct rather
    than stale]: the two models share `dim` $= 64$, 6 encoder and 4 decoder layers, 8 heads, 150
    queries and 160 nodes, and `bidirectional_ca` is on in both. Data × data MACs depend on those
    shapes alone, so they are identical.

    #text(fill: warn, weight: "bold")[The constant-weight MACs are NOT carried over and are not
    quoted here.] They scale with the weight matrices, and this model's feed-forwards are
    #bold[1.50×] head's — the paper's `Dense` is gated (SwiGLU), three matrices where head's SiLU
    has two. Head's 126 M is its model's number under its own counting convention; recomputing it
    for this one is open.
  ]))


]



// Every experiment in one table: the six single changes with what each one removes (the old
// "which change reduces which resource" slide) and the five stacks built out of them. The
// stacks' resource columns are deliberately blank -- no MAC counter exists yet (P0.5), and the
// single-change values do not simply compound. Parameter counts are each run's own metadata.yaml.

#cslide("The experiments")[
  #flow(80pt, 176pt, 1760pt, gap: 0.6cm)[
    Every row is one edit to the 820 k paper-tag `base_small` config, or a stack of them. Each
    column on the right is what that edit removes. #bold[No single change removes all three.]
  ]

  #at(80pt, 262pt, box(width: 1760pt)[
    #set text(size: 25pt)
    #table(
      columns: (auto, 1fr, auto, auto, auto, auto),
      stroke: none,
      align: (left + horizon, left + horizon, right + horizon, right + horizon,
              right + horizon, right + horizon),
      inset: (x: 14pt, y: 9pt),
      [#bold[arm]], [#bold[the change]], [#bold[params]], [#bold[vs ref]],
        [#bold[data × data]], [#bold[lookup tables]],
      table.hline(stroke: 0.8pt),
      [#bold[reference]], [`base_small` as built], [819,683], [1.000×],
        [#text(fill: muted)[—]], [#text(fill: muted)[—]],
      table.hline(stroke: 0.4pt + luma(200)),
      [#bold[A2] mlp1x], [transformer MLP hidden $2 D -> D$ #h(6pt) #text(size: 21pt, fill: muted)[training]],
        [645,859], [0.788×],
        [#text(fill: muted)[none]], [#text(fill: muted)[none]],
      [#bold[A3] dim48], [`dim` 64 → 48 #text(size: 22pt, fill: muted)[— forces heads 8 → 6] #h(6pt) #text(size: 21pt, fill: muted)[training]],
        [467,201], [#bold[0.570×]], [−25%], [−25%],
      [#bold[A4] enc5], [encoder 6 → 5 layers #h(6pt) #text(size: 21pt, fill: muted)[training]],
        [777,627], [0.949×], [−6%], [−6%],
      [#bold[A6] dec3], [decoder 4 → 3 layers #h(6pt) #text(size: 21pt, fill: muted)[training]],
        [719,971], [0.878×], [−16%], [−16%],
      table.hline(stroke: 0.8pt),
      table.cell(colspan: 6, inset: (x: 14pt, y: 7pt))[
        #text(size: 23pt, fill: muted)[#bold[stacks of the changes above]]],
      table.hline(stroke: 0.4pt + luma(200)),
      [#bold[C5]], [A2 + A4], [616,219], [0.752×], [−5%], [−6%],
      [#bold[C4]], [A3 + A4], [443,435], [0.541×], [−29%], [−29%],
      [#bold[C3]], [A2 + A3], [369,089], [0.450×], [−25%], [−24%],
      [#bold[C1]], [A2 + A3 + A4 #h(8pt) #text(fill: good, weight: "bold")[trained here]],
        [352,331], [#bold[0.430×]], [−29%], [−29%],
    )
  ])

  #at(80pt, 800pt, box(width: 1760pt, text(size: 23pt, fill: muted)[
    Parameters are from instantiating each resolved config, and reproduce each trained run's own
    `metadata.yaml`. The other two columns are counted from the shapes, not measured, and are
    #bold[carried over from head unchanged]: this model has the same `dim`, layer counts, heads,
    node and query counts, so every shape-derived number is identical. \
    #bold[S1 and C2 are absent] — they were never trained on this branch. #bold[C1 trained here],
    where on head it failed twice.
  ]))


]


// =========================================================================
//  Jet energy response, the plots alone. Two slides of the same four panels:
//  round 1/2's six arms, then the same figure with round 3's C3/C4/C5 and the
//  depth arm A6 added. Both conventions on each slide, proxy over mpflow.
//
//  The two canvases are NOT line-for-line comparable. The pipeline intersects
//  events across the arms it is given, so the ten-arm figure runs on a smaller
//  common event set and every line -- the reference included -- shifts a little
//  against the six-arm figure. Each canvas is read against itself.
// =========================================================================

#cslide("Jet energy response — both conventions")[
  #at(260pt, 174pt, text(size: 26pt)[#bold[`mpflow_proxy`] — kinematics rebuilt from the incidence head])
  #img-at(260pt, 206pt, 1400pt, "../figures/size_ablation_mpflow_proxy_jet_median_iqr.png")

  #at(260pt, 592pt, text(size: 26pt)[#bold[`mpflow`] — kinematics from the regression head])
  #img-at(260pt, 624pt, 1400pt, "../figures/size_ablation_mpflow_jet_median_iqr.png")
]


#cslide("Jet energy resolution — what the shrink costs")[
  #flow(80pt, 176pt, 1760pt, gap: 0.5cm)[
    Global jet-E IQR against the reference. #bold[The ordering is monotone in parameter count and
    both conventions agree on every arm.] C1 does not blow the range — it is the worst arm by a
    graded margin, not a failure.
  ]

  #at(80pt, 268pt, box(width: 1060pt)[
    #set text(size: 25pt)
    #table(
      columns: (1fr, auto, auto, auto, auto),
      stroke: none,
      align: (left, right, right, right, right),
      inset: (x: 14pt, y: 9pt),
      [#bold[arm]], [#bold[vs ref]], [#bold[IQR `mpflow`]], [#bold[Δ]], [#bold[Δ `proxy`]],
      table.hline(stroke: 0.8pt),
      [reference 820k], [1.000×], [0.0827], [—], [—],
      [C5 a2a4], [0.752×], [0.0840], [+0.0012], [+0.0031],
      [C4 a3a4], [0.541×], [0.0926], [+0.0098], [+0.0073],
      [C3 a2a3], [0.450×], [0.0980], [+0.0153], [+0.0118],
      [C1 a2a3a4], [0.430×], [0.1006], [+0.0179], [+0.0137],
      table.hline(stroke: 0.4pt + luma(200)),
      [#text(fill: muted)[Pandora] #text(size: 21pt, fill: muted)[— classical, not an arm]],
        [#text(fill: muted)[—]], [#text(fill: muted)[0.0628]],
        [#text(fill: muted)[−0.0199]], [#text(fill: muted)[+0.0027]],
    )
  ])

  #at(1170pt, 268pt, callout(670pt, size: 24pt)[
    #bold[A3 is what costs.] Solving the three pairs for per-change contributions to ΔIQR
    (`mpflow`): #bold[A3 +0.0120], A2 +0.0034, A4 #bold[−0.0022] — free. \
    #bold[C5 is the cheap shrink]: a quarter of the parameters for +0.0012. \
    The triple is #bold[super-additive]: +0.0132 predicted, +0.0179 measured.
  ])

  #at(1170pt, 620pt, callout(670pt, size: 23pt, fill: rgb("#fdf3f3"))[
    #text(fill: warn, weight: "bold")[Orderings, not verdicts.] σ#sub[stat] here is ≈ 0.0007 and is
    the test-sample term only. The bar an arm must clear is 2√2 σ#sub[repro], and σ#sub[repro] has
    #bold[not been measured on this code].
  ])

  #at(80pt, 790pt, box(width: 1760pt, text(size: 22pt, fill: muted)[
    Pandora has no convention of its own, so the same 0.0628 appears in both columns and the
    comparison flips: it beats every arm in `mpflow` and loses to the reference in `mpflow_proxy`.
    That gap is wider than the whole ablation — which is why the paper comparison is read in
    `mpflow_proxy` only.
  ]))
]


#closing("Backup")


// The seed floor, finally measured -- the missing denominator under every Part 3 verdict.

#cslide("Backup — the resolution floor is NOT measured here")[
  #flow(80pt, 190pt, 1760pt, gap: 0.7cm)[
    On head this slide carried a measurement: the same reference config trained four times at
    seeds 42/43/44/45, the band those four lines span being the resolution of every comparison.
    #bold[That measurement has not been repeated on this code, and head's number does not
    transfer] — the feed-forward activation, the q/k/v norms and the incidence head all differ,
    and the training-to-training scatter is its own quantity.
  ]

  #at(80pt, 360pt, callout(1760pt, size: 27pt)[
    What head measured, #emph[for head's model]: #bold[$sigma_"repro"$ = 0.0007] on the global
    jet-E IQR (`mpflow_proxy`, $n = 4$, so $plus.minus 41%$), giving a bar of
    #bold[$2 sqrt(2) sigma = 0.0020$] on a difference between two independently trained runs.
    #tag("HEAD ONLY", col: warn)
  ])

  #at(80pt, 560pt, callout(1760pt, size: 26pt)[
    #bold[What this means for the numbers in this deck.] Every Δ quoted carries only
    $sigma_"stat"$, the bootstrap over matched jets — $approx 0.0007$, and the #emph[smaller] half
    of the error. A Δ inside it is certainly not real; one outside it is merely not excluded.
    The C4 / C3 / C1 separations are an order of magnitude above any plausible bar, so they are
    safe. #bold[C5's +0.0012 is not] — and C5 is the arm most likely worth shipping.
  ])

  #at(80pt, 800pt, callout(1760pt, size: 25pt, fill: rgb("#f3f7fb"))[
    #bold[What would settle it:] four trainings of `base_small` at different seeds, the same
    design head ran. Until then the ablation is an #emph[ordering], not a set of verdicts.
  ])
]


// Plain-language glossary of the FPGA vocabulary, from FPGA_ATTENTION_STUDY.md §2.3 and §2.4.
// Nothing here is specific to this model; the slide exists so the budget slide that follows is
// readable without the companion document. Device counts are the VU13P datasheet (A1).

#cslide("Backup — the FPGA vocabulary")[
  #flow(80pt, 182pt, 1760pt, gap: 0.7cm)[

  ]

  #at(80pt, 230pt, text(size: 29pt, fill: accent, weight: "bold")[What the chip is made of
    #text(size: 24pt, fill: muted, weight: "regular")[— counts are for a VU13P]])
  #at(990pt, 230pt, text(size: 29pt, fill: accent, weight: "bold")[How the chip is clocked])

  #at(80pt, 276pt, box(width: 860pt)[
    #set text(size: 24pt)
    #table(
      columns: (auto, 1fr),
      stroke: none,
      align: (left, left),
      inset: (x: 10pt, y: 10pt),
      [#bold[LUT]], [A 6-input truth table — the fabric's general-purpose logic cell. Builds
        adders, muxes, control, and #emph[small] multipliers. #text(fill: muted)[~1.7 M.]],
      [#bold[DSP]], [A hardened $approx 18 times 27$-bit multiply–accumulate block. The only
        dedicated multiplier, and the scarcest resource. #text(fill: muted)[12,288.]],
      [#bold[BRAM]], [On-chip memory in #bold[36 Kbit] blocks, #bold[preloaded from the
        bitstream]. Every $exp$, $tanh$, $1 slash sqrt(x)$ is a stored lookup table, and lives
        here. #text(fill: muted)[94.5 Mbit.]],
      [#bold[URAM]], [Memory in bigger #bold[288 Kbit] blocks: more bits, but #bold[written at
        run time], never preloaded. #text(fill: muted)[360 Mbit.]],
    )
  ])

  #at(990pt, 276pt, box(width: 860pt)[
    #set text(size: 24pt)
    #table(
      columns: (auto, 1fr),
      stroke: none,
      align: (left, left),
      inset: (x: 10pt, y: 10pt),
      [#bold[latency]], [The #bold[time] from the first input bit entering to the last output bit
        leaving, for #bold[one] event. Fixed-latency firmware makes it a whole number of cycles,
        so it is quoted that way: `latency = cycles / f_clk`.
        #text(fill: muted)[3–10 µs is 1,100–3,600 cycles at 360 MHz.]],
      [#bold[II]], [#bold[Initiation interval] — cycles between successive #emph[inputs]: how often
        a new event may enter. Set by time multiplexing, since each board sees one event every
        `TMUX × 25 ns`.],
      [#bold[RF]], [#bold[Reuse factor] — how many times one physical multiplier is reused within
        one event. RF = 1 gives every MAC its own hardware: fastest, largest.],
    )
  ])

  #at(80pt, 636pt, callout(1760pt, size: 24pt)[
    #bold[multiplier] — the circuit that carries out one MAC. It is #emph[built], from $approx 1 slash 2$ a DSP or $approx 60$ LUTs. A #bold[MAC]
    is the operation. Only MACs where #bold[both operands are data] need a multiplier — a
    MAC against a compile-time constant folds into a shared adder graph (1.5–3 LUTs) and needs no
    multiplier at all.
  ])
  #at(80pt, 752pt, callout(1760pt, size: 24pt)[
    #bold[The analogy.] A car production line. #bold[Latency] is how long one car takes from raw
    steel to driving off the end. #bold[II] is how often a new car is started down the line — a
    line can take 8 hours per car and still start one every 5 minutes, because many are on it at
    once. #bold[RF] is how many cars share a single welding robot.
  ])
]


// The budget, from FPGA_ATTENTION_STUDY.md §2.6 and §5.7. The two things that have to be said
// before any cost number means anything: throughput binds rather than latency, and the two
// classes of MAC differ by 20-40x per operation.

#cslide("Backup — what we are shrinking towards")[
  #flow(80pt, 188pt, 1760pt, gap: 0.6cm)[
    #bold[The II sets how much hardware is needed.] A new event enters every II cycles (
    `II = TMUX × f_clk / 40 MHz`) so in every II-cycle window the design must start one event's
    worth of MACs. Different MACs can be
    pipelined inside a multiplier, but one multiplier can start only a new MAC every cycle and produce
    at most one result per clock cycle. So the design must contain #bold[multipliers ≥ `MAC / II`] (for the MACs that need a real multiplier).

  ]

  #at(80pt, 392pt, box(width: 1760pt)[
    #set text(size: 27pt)
    #table(
      columns: (1fr, auto, auto, auto, auto),
      stroke: none,
      align: (left, right, right, right, right),
      inset: (x: 16pt, y: 10pt),
      [], [#bold[v7 today]], [#bold[II = 54]], [#bold[II = 162]], [#bold[II = 324]],
      table.hline(stroke: 0.8pt),
      [constant-weight MAC / event], [125.8 M],
        [15.6 M #text(size: 22pt, fill: muted)[(8.1×)]],
        [46.7 M #text(size: 22pt, fill: muted)[(2.7×)]],
        [93.3 M #text(size: 22pt, fill: muted)[(1.3×)]],
      [#bold[data × data MAC / event]], [#bold[65.5 M]],
        [#bold[2.1 M] #text(size: 22pt, fill: warn, weight: "bold")[(31×)]],
        [#bold[6.3 M] #text(size: 22pt, fill: warn, weight: "bold")[(10.4×)]],
        [#bold[12.6 M] #text(size: 22pt, fill: warn, weight: "bold")[(5.2×)]],
      [softmax table storage
        #text(size: 21pt, fill: muted)[— II-independent]], [822 Mbit],
        table.cell(colspan: 3, align: right)[455 Mbit
          #text(size: 22pt, fill: muted)[(1.8×)]],
    )
  ])

  #at(80pt, 604pt, box(width: 1760pt)[
    #set text(size: 23pt, fill: muted)
    Budget = the #bold[whole] VU13P (12,288 DSP48E2, 1.728 M LUT, 94.5 Mbit BRAM + 360 Mbit URAM),
    LUT fabric split evenly between the two columns. 
    
    `II = 162` is TMUX 18 at 360 MHz — 9× the
    40 MHz bunch clock. 
    
    #text(fill: warn, weight: "bold")[Two further caveats:] the constant-weight row assumes
    multipliers are time-shared (RF = II), whereas `da4ml`'s adder graphs require #bold[RF = 1];
    and the softmax row assumes tables pack perfectly across both memory pools.
    The #bold[≈ 1–3 LUT] constant-weight cost is back-solved from the published zero-DSP trigger
    transformer.
  ])

]
