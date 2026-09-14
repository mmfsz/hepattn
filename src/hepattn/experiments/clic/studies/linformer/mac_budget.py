"""Data x data MAC budget for the Linformer arms, on the small model.

The FPGA cost this project is trying to reduce is the number of DSPs, and the deck's slide 7
counts what maps onto them: multiplications where **both operands come from the event**, so a
real multiplier is needed. Q K^T, scores . V, and the mask head's bilinear. A multiplication by
a learned matrix -- every weight projection, and Linformer's sequence compression E^T K -- is
data x weight and is counted separately; the deck quotes 125.8 M for those against 65.5 M here.

Getting an operation into the wrong bucket reverses the sign of the answer, so the rule is
applied explicitly below rather than left to the reader. The baseline total this prints,
65.45 M, is the 65.5 M of the deck, which is the check that the convention here is that one.

Usage: python mac_budget.py
"""

# The small paper model, configs/base_small.yaml
DIM = 64
N_ENC = 168  # 160 constituents + 8 register tokens
N_CONST = 160  # max_nodes
N_QUERY = 150  # object queries
K = 64  # linformer_proj_dim
L_ENC = 6
L_DEC = 4
MASK_HEADS = L_DEC + 1  # one per decoder layer, plus the final prediction

M = 1e6


def attn_dxd(n_q: int, n_kv: int, dim: int) -> int:
    """Data x data MACs of one ordinary attention: scores = Q K^T, then weights . V."""
    return 2 * n_q * n_kv * dim


def linformer_dxd(n_q: int, k: int, dim: int) -> int:
    """Data x data MACs of one Linformer attention.

    Only the compressed scores and the weighted values: the two sequence compressions are an
    activation times the learned matrix E, so they are data x weight and are not counted here.
    """
    return 2 * n_q * k * dim


def linformer_dxw(n_kv: int, k: int, dim: int) -> int:
    """Data x weight MACs Linformer adds: compressing the keys and the values."""
    return 2 * n_kv * k * dim


SITES = {
    "encoder SA": attn_dxd(N_ENC, N_ENC, DIM) * L_ENC,
    "decoder q_ca": attn_dxd(N_QUERY, N_CONST, DIM) * L_DEC,
    "decoder q_sa": attn_dxd(N_QUERY, N_QUERY, DIM) * L_DEC,
    "decoder kv_ca": attn_dxd(N_CONST, N_QUERY, DIM) * L_DEC,
    "mask head bilinear": N_QUERY * N_CONST * DIM * MASK_HEADS,
}

# Step 1 puts Linformer on every attention that carries no mask; avenue 2 adds q_ca.
STEP1 = dict(SITES)
STEP1["encoder SA"] = linformer_dxd(N_ENC, K, DIM) * L_ENC
STEP1["decoder q_sa"] = linformer_dxd(N_QUERY, K, DIM) * L_DEC

AVENUE2 = dict(STEP1)
AVENUE2["decoder q_ca"] = linformer_dxd(N_QUERY, K, DIM) * L_DEC

ARMS = {"baseline": SITES, "step 1": STEP1, "avenue 2": AVENUE2}

print(f"data x data MACs per event, small model (dim {DIM}, k {K})\n")
print(f"{'site':22s}" + "".join(f"{name:>12s}" for name in ARMS))
print("-" * (22 + 12 * len(ARMS)))
for site in SITES:
    print(f"{site:22s}" + "".join(f"{arm[site] / M:11.2f} " for arm in ARMS.values()))
print("-" * (22 + 12 * len(ARMS)))
totals = {name: sum(arm.values()) for name, arm in ARMS.items()}
print(f"{'TOTAL':22s}" + "".join(f"{t / M:11.2f} " for t in totals.values()))
base = totals["baseline"]
print(f"{'vs baseline':22s}" + "".join(f"{100 * (t / base - 1):+10.0f}% " for t in totals.values()))
print(f"\nthe deck's slide 7 quotes 65.5 M for the baseline; this prints {base / M:.2f} M")

# What avenue 2 adds, and why none of it lands in the budget above.
mask_proj = N_QUERY * N_CONST * K * L_DEC
comp_qca = linformer_dxw(N_CONST, K, DIM) * L_DEC
dxd_removed = SITES["decoder q_ca"] - AVENUE2["decoder q_ca"]

per_call_ordinary = attn_dxd(N_QUERY, N_CONST, DIM)
per_call_linformer = linformer_dxd(N_QUERY, K, DIM) + linformer_dxw(N_CONST, K, DIM)
per_call_saving = per_call_ordinary - per_call_linformer
per_call_mask = mask_proj / L_DEC
intermediate_masks = (MASK_HEADS - 1) * N_QUERY * N_CONST * DIM

print(f"""
What avenue 2 adds, in the other buckets:

  key/value compression at q_ca   {comp_qca / M:6.2f} M   data x weight; E is learned
  mask projection  M . |E|        {mask_proj / M:6.2f} M   data x weight, and M is boolean
                                            (task.py: sigmoid() >= threshold), so this is a
                                            masked accumulation of |E| columns: additions,
                                            not multiplications. No DSPs at all.
  q_ca data x data removed        {dxd_removed / M:6.2f} M

THE TRAP. Counting every MAC equally reverses the sign at this site. Per call, ordinary
attention is {per_call_ordinary / M:.2f} M and Linformer is {per_call_linformer / M:.2f} M, a saving of only {per_call_saving / M:.2f} M,
against {per_call_mask / M:.2f} M for the mask projection -- which reads as a net loss of
{(per_call_mask - per_call_saving) / M:.2f} M per call. Both numbers are correct. The conclusion drawn from them is
not, because neither the compression nor the mask projection needs a multiplier fed
from two activations, and the mask projection needs no multiplier at all.

SEPARATELY, and nothing to do with Linformer: {intermediate_masks / M:.2f} M of the mask head's
{SITES["mask head bilinear"] / M:.2f} M is its {MASK_HEADS - 1} intermediate predictions, which exist only to drive
masked attention. With mask_attention off, the final one is the output and the rest
can be skipped at inference: real data x data, removed with no approximation.""")
