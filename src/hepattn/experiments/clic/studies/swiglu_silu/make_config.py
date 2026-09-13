"""Generate the SiLU arm config from base_small.yaml.

One variable: `Dense`'s activation in every encoder and decoder feed-forward, gated SwiGLU
(this line's default) -> plain SiLU (`main`'s default). Nothing else moves.

A config overlay cannot do this. `decoder_layer_config` is a plain dict, so a second
--config file replaces it wholesale instead of merging, silently dropping `dim`,
`hybrid_norm` and `num_heads` -- which is exactly the confound this arm must not have.

Usage: python make_config.py <base_small.yaml> <outdir> <name>
"""

import sys
from pathlib import Path

base_path, outdir, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
base = Path(base_path).read_text()
body = base[base.index("\nname: ") + 1 :]


def sub1(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


# The same two insertion points the model-size study's A2 uses: the encoder block and
# decoder_layer_config, the only two places that build transformer feed-forwards.
for marker in ("# (2) 16 -> 8", "# (3) 16 -> 8"):
    body = sub1(
        body,
        f"          attn_kwargs:\n            num_heads: 8                  {marker}\n",
        "          dense_kwargs:\n"
        "            activation: SiLU              # S: SwiGLU -> SiLU (main's default)\n"
        f"          attn_kwargs:\n            num_heads: 8                  {marker}\n",
    )
body = sub1(body, "name: clic_paper_small", f"name: {name}")

header = f"""# {name} -- SwiGLU -> SiLU in every encoder and decoder feed-forward.
#
# Generated from configs/base_small.yaml by studies/swiglu_silu/make_config.py; the only lines
# that differ carry an "S:" comment. `Dense` defaults to a gated SwiGLU on this line and to a
# plain SiLU on `main`, and neither the encoder block nor decoder_layer_config sets
# dense_kwargs, so all 14 feed-forwards (6 encoder + 8 decoder) take that default. Gating
# doubles the inner projection, so this arm is worth 14 x 8,320 = 116,480 parameters.
#
# The task heads already name torch.nn.SiLU explicitly and are untouched.
"""
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"{name}.yaml"
out.write_text(header + body)
print(f"wrote {out}")
