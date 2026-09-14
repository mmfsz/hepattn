"""Generate the Linformer study's arm configs from base_small.yaml.

Four configs, each a standalone copy of the small paper model with the lines that differ
carrying an "L:" comment. An overlay config cannot express these: `decoder_layer_config` and
`attn_kwargs` are plain dicts, so a second --config file replaces them wholesale instead of
merging, silently dropping `dim`, `hybrid_norm` and `num_heads` -- the confound these arms
exist to avoid.

    ref    the reference. Nothing but the phi ordering, which a permutation-equivariant model
           is insensitive to. Needed because the topocluster phi fix changed the encoder's
           inputs, so the paper-tag reproductions are no longer a baseline for anything here.
    step0  masked attention off, no Linformer. Measures what masked attention is worth on its
           own, which is what says whether step 2 is worth building.
    step1  Linformer on every attention that carries no mask: the encoder, and the decoder's
           query self-attention. Masked cross-attention untouched, so this differs from ref by
           the attention mechanism and nothing else.
    step2  step1 plus the projected mask, once that exists. Not generated yet.

Usage: python make_configs.py <base_small.yaml> <outdir>
"""

import sys
from pathlib import Path

# The encoder sees 160 constituents (max_nodes) plus 8 register tokens; the decoder's query
# self-attention sees the 150 object queries. Each projection is sized to its own sequence.
ENCODER_SEQ_LEN = 168
DECODER_SEQ_LEN = 150
PROJ_DIM = 64

base_path, outdir = sys.argv[1], Path(sys.argv[2])
base = Path(base_path).read_text()
body = base[base.index("\nname: ") + 1 :]

ENCODER_ATTN = "          attn_kwargs:\n            num_heads: 8                  # (2) 16 -> 8\n"
DECODER_ATTN = "          attn_kwargs:\n            num_heads: 8                  # (3) 16 -> 8\n"


def sub1(s: str, old: str, new: str) -> str:
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


def with_phi_sorting(s: str) -> str:
    """Sort the constituents by phi inside the encoder (it un-sorts them on the way out)."""
    return sub1(
        s,
        "      dim: &dim 64                        # (1) 256 -> 64\n",
        "      dim: &dim 64                        # (1) 256 -> 64\n"
        "      input_sort_field: phi               # L: slot index must mean something (6.1)\n",
    )


def make_ref(s: str) -> str:
    return with_phi_sorting(s)


def make_step0(s: str) -> str:
    s = with_phi_sorting(s)
    return sub1(
        s,
        "        mask_attention: true\n",
        "        mask_attention: false           # L: the ablation this arm exists for\n",
    )


def make_step1(s: str) -> str:
    s = with_phi_sorting(s)
    # Encoder self-attention: unmasked, 168 tokens
    s = sub1(
        s,
        "          attn_type: flash-varlen\n",
        "          attn_type: linformer            # L: unmasked self-attention, so Linformer fits\n",
    )
    s = sub1(
        s,
        ENCODER_ATTN,
        ENCODER_ATTN + f"            linformer_seq_len: {ENCODER_SEQ_LEN}      # L: 160 constituents + 8 registers\n"
        f"            linformer_proj_dim: {PROJ_DIM}      # L: the compression, k\n",
    )
    # Decoder query self-attention only: q_ca and kv_ca keep their masked attention
    return sub1(
        s,
        DECODER_ATTN,
        DECODER_ATTN + "          sa_attn_kwargs:                 # L: the query self-attention carries no mask\n"
        "            attn_type: linformer\n"
        f"            linformer_seq_len: {DECODER_SEQ_LEN}      # L: the object queries\n"
        f"            linformer_proj_dim: {PROJ_DIM}\n",
    )


ARMS = {
    "clic_paper_small_phi_ref": (
        make_ref,
        "The reference for every other arm: the small paper model, phi-sorted and nothing else.\n"
        "#\n"
        "# The model is permutation equivariant over constituents -- attention, the task heads and the\n"
        "# matcher all are, and position enters as content through the Fourier encoder -- so the sort\n"
        "# leaves it statistically unchanged. It is on here anyway, so that the arms differ from this\n"
        "# one by their attention and by nothing else.\n"
        "#\n"
        "# This run is not optional. The topocluster phi fix changed what the encoder is fed, so the\n"
        "# paper-tag reproductions were trained on different inputs and cannot serve as the baseline.",
    ),
    "clic_paper_small_no_mask_attn": (
        make_step0,
        "Step 0: masked attention off, everything else the reference model.\n"
        "#\n"
        "# Masked attention -- each object query attending only to the constituents it currently\n"
        "# claims -- is a defining piece of MaskFormer, and Linformer cannot honour it. This arm\n"
        "# measures what it is worth on its own. If the reference barely moves without it, the\n"
        "# projected-mask work of step 2 is unnecessary and compression can be studied with the mask\n"
        "# simply off; if it degrades, step 2 is the blocker and worth the effort.",
    ),
    "clic_paper_small_linformer_unmasked": (
        make_step1,
        "Step 1: Linformer on every attention that carries no mask.\n"
        "#\n"
        "# The encoder's self-attention (168 tokens) and the decoder's query self-attention (150\n"
        "# queries). The decoder's cross-attentions keep ordinary masked attention, so mask_attention\n"
        "# stays on and this arm differs from the reference by the attention mechanism alone.\n"
        "#\n"
        "# linformer_proj_dim is the knob. At k = 64 the compression matrices are 2 x 168 x 64 =\n"
        "# 21,504 parameters per encoder layer, which is more than the 16,640 of the attention block\n"
        "# they join: the small model pays about +15% in parameters to save about 24% of the\n"
        "# attention multiplies. That trade is the measurement, so k is worth scanning.",
    ),
}

outdir.mkdir(parents=True, exist_ok=True)
for name, (make, blurb) in ARMS.items():
    arm = sub1(make(body), "name: clic_paper_small", f"name: {name}")
    header = (
        f"# {name} -- {blurb}\n"
        "#\n"
        "# Generated from configs/base_small.yaml by studies/linformer/make_configs.py; every line\n"
        '# that differs from it carries an "L:" comment. See studies/linformer/README.md.\n'
    )
    out = outdir / f"{name}.yaml"
    out.write_text(header + arm)
    print(f"wrote {out}")
