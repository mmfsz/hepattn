// =========================================================================
//  Reusable Touying slide template (16:9, 1920 x 1080 pt canvas).
//
//  Adapted from ../../../../../../../../vbsvvh/slides/vbsvvh_corrections/
//  vbsvvh_20260709_jecs.typ so that positions can be read straight off a
//  reference layout in pt and placed 1:1 with `at`.
//
//  Usage from a deck file living next to this one:
//
//      #import "template.typ": *
//      #show: deck.with(
//        title: "My talk",
//        subtitle: "an optional second line",
//        event: "Some meeting",
//        date: "August 28, 2026",
//      )
//      #cslide("First slide")[ ... ]
//      #divider("A section")
//
//  Everything below the SETTINGS block is layout machinery; the knobs worth
//  touching for a new deck are all in SETTINGS.
// =========================================================================

#import "@preview/touying:0.7.4": *
#import themes.simple: *


// ==================== SETTINGS ====================

#let body-font = "Liberation Sans"
// The 16:9 canvas is 1920 x 1080 pt, so any pt value can be placed 1:1. The base
// font is scaled to that canvas.
#let body-font-size = 68pt
// Regular text size on the content (body) slides.
#let body-text-size = 34pt

#let author-name = "Maria Mazza"
#let affiliation = "Florida State University"

// --- Logo sizes (the FSU lockups are horizontal, so sized by height) ------
#let title-logo-height = 105pt   // white FSU logo, top-left of the dark title slide
#let footer-logo-height = 46pt   // white FSU logo inside the dark footer bar
// Figures on the regular content slides (fraction of the text-column width).
#let body-figure-width = 95%

// --- Credit / citation text ---------------------------------------------
#let credit-size = 24pt
#let credit-color = rgb("#3274B5")
#let credit(body, fill: credit-color, bg: none) = {
  let content = text(size: credit-size, fill: fill)[#body]
  if bg == none {
    content
  } else {
    box(fill: bg, inset: (x: 0.35em, y: 0.2em), radius: 0.15em)[#content]
  }
}

// --- Content-slide master layout (pt from the slide's top-left corner) ---
// The body text flows from `body-text-pos` (this is set as the page margin),
// and the slide title is placed absolutely at `body-title-pos`.
#let body-title-pos = (x: 60pt, y: 55pt)    // slide title (the `cslide` argument)
#let body-title-size = 70pt
#let body-text-pos = (x: 90pt, y: 170pt)    // body text — first line start
#let body-margin-right = 60pt
#let body-margin-bottom = 60pt

// Thin dark footer bar: name/affiliation | page number | logo.
#let footer-name-size = 30pt
#let footer-page-size = 24pt
#let footer-bar-height = 58pt   // visible height of the thin dark bar
#let footer-bar-fill = rgb("#1D1D1D")
#let footer-text-fill = rgb("#CDB98D")

// --- Accent colours used by the helpers below ---------------------------
#let accent = rgb("#3274B5")     // "this is the good path" blue
#let warn = rgb("#C00000")       // "this is the failure" red
#let good = rgb("#1B7F3B")       // "this passed" green
#let panel-fill = rgb("#F6F4EC") // callout box background
#let muted = luma(110)           // de-emphasised text


// ==================== FOOTER ====================

#let my-footer = context {
  // Baseline size only sets the em unit used by the insets below; the visible
  // footer elements get explicit pt sizes.
  set text(size: 0.6 * body-font-size)
  // Cancel the `v(.5em)` the theme inserts before the footer, so the bar sits
  // flush at the page bottom and the overhanging logo is not pushed off-page.
  v(-0.5em)
  block(
    width: 100%,
    // +6pt bleeds the bar a few pt past the page edge (clipped) so it always
    // reaches the bottom — kills the thin white sliver under the bar.
    height: footer-bar-height + 6pt,
    fill: footer-bar-fill,
    inset: (x: 18pt),
    {
      // Name (left) and page number (center), vertically centered on the bar.
      place(left + horizon, text(fill: footer-text-fill, size: footer-name-size)[#author-name (FSU)])
      place(center + horizon, text(fill: footer-text-fill, size: footer-page-size)[
        #utils.slide-counter.display() / #utils.last-slide-number
      ])
      place(right + horizon, image("logos/FSU_white.png", height: footer-logo-height))
    },
  )
}


// ==================== SLIDE HELPERS ====================
// The canvas is 1920x1080 pt, so element positions can be read straight off a
// reference layout (in pt) and placed with `at`.
//
// `at(x, y, body)` places `body` with its top-left at absolute slide coordinate
// (x, y) measured from the slide's top-left corner. The content area starts at
// `body-text-pos`, so we offset by it; placing into the margins is allowed.
#let at(x, y, body) = place(top + left, dx: x - body-text-pos.x, dy: y - body-text-pos.y, body)

// Small upward nudge so a placed text box's visible cap-line lands on the given
// y (Typst places the line box top, which sits above the caps). Tuned for 34pt.
#let text-y-nudge = 8pt

// `tb(x, y, w, body)` — a left-aligned text paragraph of fixed width `w` placed
// at absolute (x, y). Width fixes the line wrapping to match the source.
#let tb(x, y, w, body) = at(x, y - text-y-nudge, box(width: w, body))

// `vstack(x, y0, dy, w, ..items)` — place each item as a `tb` of width `w` at the
// same x, stepping the y position by `dy` per item.
#let vstack(x, y0, dy, w, ..items) = {
  for (i, it) in items.pos().enumerate() {
    tb(x, y0 + i * dy, w, it)
  }
}

// `eqn(x, y, body)` — a displayed equation block placed at absolute (x, y).
#let eqn(x, y, body) = at(x, y, body)

// `ceqn(y, body)` — a displayed equation horizontally centered on the slide,
// with its top at absolute vertical position `y`.
#let ceqn(y, body) = place(top + center, dy: y - body-text-pos.y, body)

// `flow(x, y, w, body)` — anchor the TOP-LEFT corner of a text column at absolute
// (x, y) with fixed width `w`, then let the content flow and space itself:
//   - sentences (separated by a blank line in the body) wrap automatically at `w`;
//   - block equations `$ ... $` (note the spaces) are auto-centered in the column;
//   - the vertical gap between consecutive items is inserted automatically.
// `gap` tunes the uniform spacing between paragraphs and around equations.
#let flow(x, y, w, gap: 3.0cm, body) = place(
  top + left,
  dx: x - body-text-pos.x,
  dy: y - text-y-nudge - body-text-pos.y,
  block(width: w, {
    set par(spacing: gap)
    show math.equation.where(block: true): set block(above: gap, below: gap)
    body
  }),
)

// `img-at(x, y, w, path)` — image of width `w` placed at absolute (x, y).
#let img-at(x, y, w, path) = at(x, y, image(path, width: w))

// --- Text styling helpers ------------------------------------------------
// Bold that forces black (for slide bodies on the white content slides).
#let bold(b) = text(weight: "bold", fill: black, b)
// Bold that inherits the surrounding color.
#let bnum(b) = text(weight: "bold", b)
// Bold-italic for emphasized portions of block quotations.
#let bi(b) = text(weight: "bold", style: "italic", fill: black, b)
// Red speaker-note annotation placed at absolute (x, y).
#let rednote(x, y, sz, b) = at(x, y, text(weight: "bold", fill: warn, size: sz, b))

// `callout(w, body)` — the pale panel used for "the thing to remember" boxes.
#let callout(w, body, fill: panel-fill, size: 28pt) = box(
  width: w,
  fill: fill,
  inset: (x: 22pt, y: 18pt),
  radius: 6pt,
  text(size: size, body),
)

// `tag(label, col)` — a small coloured pill, for PASS / FAIL / DEAD END markers.
#let tag(label, col: accent) = box(
  fill: col,
  inset: (x: 12pt, y: 6pt),
  radius: 5pt,
  text(fill: white, weight: "bold", size: 26pt, label),
)

// Straight arrow with a filled-look head from absolute (x1,y1) to (x2,y2).
#let arw(x1, y1, x2, y2, th: 3.5pt, hd: 22pt, col: black) = {
  let dx = x2 - x1
  let dy = y2 - y1
  let ang = calc.atan2(dx / 1pt, dy / 1pt)
  place(top + left, dx: x1 - body-text-pos.x, dy: y1 - body-text-pos.y, line(end: (dx, dy), stroke: (
    paint: col,
    thickness: th,
  )))
  place(top + left, dx: x2 - body-text-pos.x, dy: y2 - body-text-pos.y, line(
    end: (hd * calc.cos(ang + 155deg), hd * calc.sin(ang + 155deg)),
    stroke: (paint: col, thickness: th),
  ))
  place(top + left, dx: x2 - body-text-pos.x, dy: y2 - body-text-pos.y, line(
    end: (hd * calc.cos(ang + 205deg), hd * calc.sin(ang + 205deg)),
    stroke: (paint: col, thickness: th),
  ))
}

// `ln(x1, y1, x2, y2)` — a plain line between two absolute points.
#let ln(x1, y1, x2, y2, th: 3pt, col: black, dash: none) = place(
  top + left,
  dx: x1 - body-text-pos.x,
  dy: y1 - body-text-pos.y,
  line(end: (x2 - x1, y2 - y1), stroke: (paint: col, thickness: th, dash: dash)),
)

// `node(x, y, r, label)` — a labelled circle centred on absolute (x, y).
#let node(x, y, r, label, fill: white, stroke-col: black, size: 28pt, text-col: black) = at(
  x - r,
  y - r,
  box(
    width: 2 * r,
    height: 2 * r,
    radius: r,
    fill: fill,
    stroke: (paint: stroke-col, thickness: 3pt),
    align(center + horizon, text(size: size, fill: text-col, weight: "bold", label)),
  ),
)

// `cell(x, y, w, h, body)` — one rectangular matrix cell centred on its content.
#let cell(x, y, w, h, body, fill: white, stroke-col: luma(140), size: 28pt) = at(
  x,
  y,
  box(
    width: w,
    height: h,
    fill: fill,
    stroke: (paint: stroke-col, thickness: 1.5pt),
    align(center + horizon, text(size: size, [#body])),
  ),
)

// `grid-matrix(x0, y0, cw, ch, rows)` — draw a numeric matrix as a grid of cells
// with the top-left cell at (x0, y0). `rows` is an array of arrays; each entry is
// either a value or a `(value, fill)` pair for a highlighted cell.
#let grid-matrix(x0, y0, cw, ch, rows, size: 28pt) = {
  for (i, row) in rows.enumerate() {
    for (j, entry) in row.enumerate() {
      let value = entry
      let f = white
      if type(entry) == array {
        value = entry.at(0)
        f = entry.at(1)
      }
      cell(x0 + j * cw, y0 + i * ch, cw, ch, value, fill: f, size: size)
    }
  }
}

// `cslide(title, body)` — standard content slide: bold title at `body-title-pos`
// then a custom body (typically a sequence of `at`/`tb`/`eqn`/`img-at` calls).
//
// The title is measured and shrunk (never below 0.72x) if it would wrap onto a
// second line, since a wrapped title overruns the body's first row of content.
// Shrinking is a safety net, not a licence: a title that needs it wants rewording.
#let cslide(title, body) = slide[
  #context {
    let avail = 1920pt - body-title-pos.x - body-margin-right
    let w = measure(text(size: body-title-size, weight: "bold", title)).width
    let sz = if w > avail { calc.max(0.72, avail / w) * body-title-size } else { body-title-size }
    at(body-title-pos.x, body-title-pos.y - 10pt, text(size: sz, weight: "bold", title))
  }
  #body
]

// `fmark(n)` / `fnote(n, body)` — a footnote implemented by hand, since Typst's
// automatic footnotes collide with this deck's flush dark footer bar.
#let fmark(n) = super(text(size: 34pt)[#n])
#let fnote(n, body, y: 952pt) = {
  at(80pt, y - 12pt, line(length: 560pt, stroke: 0.6pt + luma(80)))
  at(80pt, y, box(width: 1700pt, text(size: 24pt)[#super(text(size: 34pt)[#n]) #body]))
}

// `divider(title)` — a full-bleed dark section divider with a centered title.
#let divider(title, subtitle: none) = slide(config: utils.merge-dicts(
  config-page(fill: rgb("#1D1D1D")),
  config-store(footer: none, footer-right: none),
))[
  #place(center + horizon, box(width: 1500pt, align(center, {
    text(size: body-title-size, fill: white, title)
    if subtitle != none {
      linebreak()
      v(0.4em)
      text(size: 38pt, fill: footer-text-fill, subtitle)
    }
  })))
]

// `closing(title)` — the dark end/backup slide.
#let closing(title) = slide(config: utils.merge-dicts(
  config-page(fill: rgb("#1D1D1D")),
  config-store(footer: none, footer-right: none),
))[
  #place(center + horizon, text(size: body-title-size, fill: white, title))
]


// ==================== DECK WRAPPER ====================
// Applies the theme and emits the title slide, so a deck file is nothing but
// `#show: deck.with(...)` followed by its slides.

#let deck(title: "", subtitle: "", event: "", date: "", body) = {
  show: simple-theme.with(
    aspect-ratio: "16-9",
    footer: my-footer,
    footer-right: none,
    // Titles are placed by `cslide`, so the per-subslide preamble is disabled
    // to avoid showing a stale heading on custom slides.
    subslide-preamble: none,
    config-page(
      fill: rgb("#ffffff"),
      width: 1920pt,
      height: 1080pt,
      margin: (
        left: body-text-pos.x,
        top: body-text-pos.y,
        right: body-margin-right,
        bottom: body-margin-bottom,
      ),
    ),
    config-common(zero-margin-footer: true),
    config-methods(
      init: (self: none, body) => {
        // Base size for content slides. The title slide sets its own size
        // locally, so this does not affect its em-based elements.
        set text(font: body-font, size: body-text-size)
        show footnote.entry: set text(size: 24pt)
        show heading.where(level: 1): set text(1.4em)
        set enum(tight: false, spacing: 1.5cm)
        set list(tight: false, spacing: 1.5cm)
        show raw: set text(size: 0.92em)
        body
      },
    ),
  )

  title-slide(
    config: config-page(
      fill: rgb("#1D1D1D"),
      header: none,
      footer: none,
      margin: 0em,
    ),
  )[
    #set text(fill: white, size: body-font-size)

    #box(width: 100%, height: 100%)[
      // Top-left corner: FSU logo, inset from the slide edge.
      #place(top + left, dx: 0.9cm, dy: 0.9cm)[
        #image("logos/FSU_white.png", height: title-logo-height)
      ]

      // Center: talk title and subtitle.
      #place(center + horizon, dy: -1.0cm)[
        #stack(
          dir: ttb,
          spacing: 1.6cm,
          align(center)[#text(size: 1.05em, weight: "bold")[#title]],
          align(center)[#text(size: 0.55em, fill: footer-text-fill)[#subtitle]],
        )
      ]

      // Bottom-left: author / event / date block.
      #place(bottom + left, dx: 3.0cm, dy: -2.6cm)[
        #stack(
          dir: ttb,
          spacing: 1.0cm,
          text(size: 50pt, weight: "bold")[#author-name],
          text(size: 40pt)[#affiliation],
          text(size: 40pt)[#event],
          text(size: 40pt)[#date],
        )
      ]
    ]
  ]

  body
}
