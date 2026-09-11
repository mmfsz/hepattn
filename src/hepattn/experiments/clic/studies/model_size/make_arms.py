"""Generate the paper-tag model-size arm configs from base_small.yaml.

Arms (same definitions as the head-based study on main, studies/model_size):
  A2  MLP hidden width halved:   dense_kwargs.hidden_dim_scale 2 -> 1 (encoder + decoder blocks)
  A3  dim 64 -> 48 (+ dim-derived head literals, num_heads 8 -> 6 so head_dim stays 8)
  A4  encoder num_layers 6 -> 5
  C1 = A2+A3+A4, C3 = A2+A3, C4 = A3+A4, C5 = A2+A4
Usage: python make_arms.py <base_small.yaml> <outdir> <reference_name>
"""
import re
import sys
from pathlib import Path

base_path, outdir, ref_name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
base = Path(base_path).read_text()
# strip the clone's leading comment block (everything before the first top-level key)
body = base[base.index("\nname: ") + 1 :]

def sub1(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)

def apply_a2(s):
    s = sub1(s, "          attn_kwargs:\n            num_heads: 8                  # (2) 16 -> 8\n",
             "          dense_kwargs:\n            hidden_dim_scale: 1           # A2: MLP hidden 2*dim -> 1*dim\n          attn_kwargs:\n            num_heads: 8                  # (2) 16 -> 8\n")
    s = sub1(s, "          attn_kwargs:\n            num_heads: 8                  # (3) 16 -> 8\n",
             "          dense_kwargs:\n            hidden_dim_scale: 1           # A2: MLP hidden 2*dim -> 1*dim\n          attn_kwargs:\n            num_heads: 8                  # (3) 16 -> 8\n")
    return s

def apply_a3(s):
    s = sub1(s, "dim: &dim 64                        # (1) 256 -> 64", "dim: &dim 48                        # A3: 64 -> 48")
    s = s.replace("num_heads: 8                  # (2) 16 -> 8", "num_heads: 6                  # A3: 8 -> 6 (head_dim stays 8; flash-attn needs head_dim % 8 == 0)")
    s = s.replace("num_heads: 8                  # (3) 16 -> 8", "num_heads: 6                  # A3: 8 -> 6 (head_dim stays 8)")
    s = sub1(s, "hidden_layers: [64, 128, 32]     # (4) [256, 128, 32] -> [64, 128, 32]", "hidden_layers: [48, 96, 24]      # A3: dim-derived, [64, 128, 32] * 0.75")
    s = sub1(s, "input_size: 134                  # (5) 518 -> 134 (= 2*dim + 6 raw node vars)", "input_size: 102                  # A3: 2*48 + 6 raw node vars")
    s = sub1(s, "hidden_layers: [128, 128, 128, 64, 32]   # was [512, 256, 128, 64, 32]", "hidden_layers: [96, 96, 96, 48, 24]      # A3: [128, 128, 128, 64, 32] * 0.75")
    return s

def apply_a4(s):
    return sub1(s, "          num_layers: 6\n          dim: *dim\n", "          num_layers: 5                 # A4: encoder 6 -> 5 layers\n          dim: *dim\n")

arms = {
    "C1_a2a3a4": ("A2 + A3 + A4", [apply_a2, apply_a3, apply_a4]),
    "C3_a2a3":   ("A2 + A3",      [apply_a2, apply_a3]),
    "C4_a3a4":   ("A3 + A4",      [apply_a3, apply_a4]),
    "C5_a2a4":   ("A2 + A4",      [apply_a2, apply_a4]),
}
outdir.mkdir(parents=True, exist_ok=True)
for arm, (desc, fns) in arms.items():
    s = body
    for f in fns:
        s = f(s)
    s = sub1(s, "name: clic_paper_small", f"name: {ref_name}_{arm}")
    header = f"""# {ref_name}_{arm} -- {desc}, on the paper tag's small model.
#
# Generated from configs/base_small.yaml by studies/model_size/make_arms.py; every line that
# differs from base_small.yaml carries an "A2:", "A3:" or "A4:" comment. The arm definitions are
# the head-based study's (main: studies/model_size), re-derived on the paper tag's code:
#   A2  dense_kwargs.hidden_dim_scale 2 -> 1 in every encoder and decoder block
#   A3  dim 64 -> 48, the dim-derived head literals scaled by 0.75, num_heads 8 -> 6
#   A4  encoder num_layers 6 -> 5
# Everything else, including the training recipe, is the reference's. Compare against the
# reference run of this study (base_small.yaml at the same geometry), never against head runs.
#
"""
    (outdir / f"{ref_name}_{arm}.yaml").write_text(header + s)
    print("wrote", arm)
