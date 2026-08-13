"""Verify the *_v2 mask losses against explicit per-event reference loops.

Covers mask_bce, mask_dice, mask_focal and mask_kl_div. Each v2 must equal a reference that
masks and normalises each object with its OWN event's constituent mask; the legacy version
should differ (that difference is the bug). Two structural checks are also made: with uniform
padding legacy and v2 must agree exactly, and the no-mask path must be unchanged.
"""

import math

import torch
import torch.nn.functional as F

from hepattn.models.loss import (
    mask_bce_loss,
    mask_bce_loss_v2,
    mask_dice_loss,
    mask_dice_loss_v2,
    mask_focal_loss,
    mask_focal_loss_v2,
    mask_kl_div_loss,
    mask_kl_div_loss_v2,
)

torch.manual_seed(0)
B, N, C = 5, 7, 11

logits = torch.randn(B, N, C, dtype=torch.float64) * 3
targets = (torch.rand(B, N, C, dtype=torch.float64) < 0.3).double()
obj_valid = torch.rand(B, N) < 0.7
# Variable number of valid constituents per event (prefix masks), the real CLIC situation
V = torch.randint(1, C + 1, (B,))
pad = torch.arange(C)[None, :] < V[:, None]
null_weight = 1.0
sw = targets + null_weight * (1 - targets)


def ref_bce():
    vals = []
    for b in range(B):
        v = int(pad[b].sum())
        for n in range(N):
            if not obj_valid[b, n]:
                continue
            tot = 0.0
            for c in range(C):
                if not pad[b, c]:
                    continue
                tot += float(
                    F.binary_cross_entropy_with_logits(
                        logits[b, n, c], targets[b, n, c], weight=sw[b, n, c], reduction="none"
                    )
                )
            vals.append(tot / max(v, 1))
    return sum(vals) / len(vals)


def ref_dice():
    vals = []
    for b in range(B):
        for n in range(N):
            if not obj_valid[b, n]:
                continue
            probs = torch.sigmoid(logits[b, n]) * pad[b].double()
            num = 2 * float((probs * targets[b, n]).sum())
            den = float(probs.sum() + targets[b, n].sum())
            vals.append(1 - (num + 1) / (den + 1))
    return sum(vals) / len(vals)


r_bce, r_dice = ref_bce(), ref_dice()
v2_bce = float(mask_bce_loss_v2(logits, targets, obj_valid, pad, sw))
v2_dice = float(mask_dice_loss_v2(logits, targets, obj_valid, pad, sw))
lg_bce = float(mask_bce_loss(logits, targets, obj_valid, pad, sw))
lg_dice = float(mask_dice_loss(logits, targets, obj_valid, pad, sw))

print(f"BCE   reference={r_bce:.10f}  v2={v2_bce:.10f}  legacy={lg_bce:.10f}")
print(f"DICE  reference={r_dice:.10f}  v2={v2_dice:.10f}  legacy={lg_dice:.10f}")
print()
ok_b = abs(v2_bce - r_bce) < 1e-9
ok_d = abs(v2_dice - r_dice) < 1e-9
print(f"v2 BCE  matches reference: {ok_b}   (legacy differs: {abs(lg_bce - r_bce) > 1e-6})")
print(f"v2 DICE matches reference: {ok_d}   (legacy differs: {abs(lg_dice - r_dice) > 1e-6})")

# Degenerate cases: no masks at all, and all-valid masks (bug should vanish when V is constant)
pad_full = torch.ones(B, C, dtype=torch.bool)
same = abs(float(mask_bce_loss(logits, targets, obj_valid, pad_full, sw)) - float(mask_bce_loss_v2(logits, targets, obj_valid, pad_full, sw)))
print(f"\nconstant-V check (legacy vs v2 should agree): delta={same:.3e}  -> {same < 1e-9}")
none_ok = abs(float(mask_bce_loss_v2(logits, targets, None, None, sw)) - float(mask_bce_loss(logits, targets, None, None, sw))) < 1e-9
print(f"no-mask path unchanged vs legacy: {none_ok}")


GAMMA = 2.0
EPS = 1e-8


def ref_focal():
    vals = []
    for b in range(B):
        v = int(pad[b].sum())
        for n in range(N):
            if not obj_valid[b, n]:
                continue
            tot = 0.0
            for c in range(C):
                if not pad[b, c]:
                    continue
                ce = float(
                    F.binary_cross_entropy_with_logits(
                        logits[b, n, c], targets[b, n, c], weight=sw[b, n, c], reduction="none"
                    )
                )
                pr = float(torch.sigmoid(logits[b, n, c]))
                tg = float(targets[b, n, c])
                p_t = pr * tg + (1 - pr) * (1 - tg)
                tot += ce * (1 - p_t) ** GAMMA
            vals.append(tot / max(v, 1))
    return sum(vals) / len(vals)


def ref_kl():
    vals = []
    for b in range(B):
        v = int(pad[b].sum())
        for n in range(N):
            if not obj_valid[b, n]:
                continue
            lg = logits[b, n].masked_fill(~pad[b], float("-inf"))
            tg = targets[b, n] * pad[b].double()
            tg = tg / max(float(tg.sum()), EPS) + EPS
            probs = torch.softmax(lg, dim=-1)
            per_c = -tg * torch.log(probs + EPS)
            vals.append(float((per_c * pad[b].double()).sum()) / (v + EPS))
    return sum(vals) / len(vals)


r_focal, r_kl = ref_focal(), ref_kl()
v2_focal = float(mask_focal_loss_v2(logits, targets, GAMMA, obj_valid, pad, sw))
v2_kl = float(mask_kl_div_loss_v2(logits, targets, obj_valid, pad, sw))
lg_focal = float(mask_focal_loss(logits, targets, GAMMA, obj_valid, pad, sw))
lg_kl = float(mask_kl_div_loss(logits, targets, obj_valid, pad, sw))

print(f"\nFOCAL reference={r_focal:.10f}  v2={v2_focal:.10f}  legacy={lg_focal:.10f}")
print(f"KLDIV reference={r_kl:.10f}  v2={v2_kl:.10f}  legacy={lg_kl:.10f}")
ok_f = abs(v2_focal - r_focal) < 1e-9
ok_k = abs(v2_kl - r_kl) < 1e-9


def legacy_verdict(lg, ref):
    """NaN compares False against everything, so report it explicitly rather than as 'agrees'."""
    if math.isnan(lg):
        return "legacy is NaN"
    return f"legacy differs: {abs(lg - ref) > 1e-6}"


print(f"v2 FOCAL matches reference: {ok_f}   ({legacy_verdict(lg_focal, r_focal)})")
print(f"v2 KLDIV matches reference: {ok_k}   ({legacy_verdict(lg_kl, r_kl)})")

# constant-V: the bug is a no-op when every event has the same number of constituents
same_f = abs(
    float(mask_focal_loss(logits, targets, GAMMA, obj_valid, pad_full, sw))
    - float(mask_focal_loss_v2(logits, targets, GAMMA, obj_valid, pad_full, sw))
)
same_k = abs(float(mask_kl_div_loss(logits, targets, obj_valid, pad_full, sw)) - float(mask_kl_div_loss_v2(logits, targets, obj_valid, pad_full, sw)))
print(f"constant-V check FOCAL: delta={same_f:.3e} -> {same_f < 1e-9}")
print(f"constant-V check KLDIV: delta={same_k:.3e} -> {same_k < 1e-9}")

# NaN guard: an all-zero-target object slot must not poison the result (kl_div renormaliser)
t_zero = targets.clone()
t_zero[0, :] = 0.0                      # every object in event 0 has empty targets
ov = obj_valid.clone()
ov[0, :] = False                        # ...and all are invalid, so they must not contribute
nan_ok = torch.isfinite(mask_kl_div_loss_v2(logits, t_zero, ov, pad, sw)).item()
print(f"KLDIV all-zero-target invalid objects stay finite: {nan_ok}")

extra_ok = ok_f and ok_k and same_f < 1e-9 and same_k < 1e-9 and nan_ok

print("\nALL PASS" if (ok_b and ok_d and same < 1e-9 and none_ok and extra_ok) else "\nFAILURE")
