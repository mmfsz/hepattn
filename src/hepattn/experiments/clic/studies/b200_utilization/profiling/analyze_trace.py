import json, sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as f:
    trace = json.load(f)

events = trace["traceEvents"]

# GPU-side events: kernels, memcpy, memset
gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
gpu_ev = [e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"} and "dur" in e]
steps = [e for e in events if e.get("name", "").startswith("ProfilerStep") and "dur" in e]
steps.sort(key=lambda e: e["ts"])

print(f"GPU events: {len(gpu_ev)}, ProfilerSteps: {len(steps)}")

def union_busy(intervals):
    if not intervals:
        return 0.0
    intervals.sort()
    busy, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    return busy

# Overall GPU window: from first to last GPU event
iv = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_ev]
t0 = min(s for s, _ in iv); t1 = max(e for _, e in iv)
busy = union_busy(iv[:])
print(f"GPU window {(t1-t0)/1e6:.3f}s  busy {busy/1e6:.3f}s  ({100*busy/(t1-t0):.1f}%)  idle {100*(1-busy/(t1-t0)):.1f}%")

# Per ProfilerStep (approx: GPU events whose start falls in the step's CPU span)
for i, st in enumerate(steps):
    s0, s1 = st["ts"], st["ts"] + st["dur"]
    sub = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_ev if s0 <= e["ts"] < s1]
    b = union_busy(sub)
    print(f"step {i}: wall {(s1-s0)/1e6:.3f}s  gpu busy {b/1e6:.3f}s ({100*b/(s1-s0):.1f}%)")

# Busy-time breakdown by coarse kernel family
def family(name):
    if "binary_cross_entropy" in name: return "loss: mask BCE (triton fused)"
    if "sigmoid" in name and "triton" in name: return "loss: dice/other fused (triton)"
    if name.startswith("triton_"): return "loss/other triton fused"
    if "Memcpy DtoH" in name: return "Memcpy DtoH (matcher cost -> CPU)"
    if "Memcpy HtoD" in name: return "Memcpy HtoD"
    if "flash" in name.lower() or "fmha" in name: return "attention kernels"
    if "gemm" in name.lower() or "cutlass" in name.lower() or "gemv" in name.lower(): return "GEMM"
    if "elementwise" in name or "vectorized" in name: return "elementwise"
    if "layer_norm" in name: return "layernorm"
    if "reduce" in name.lower() or "Reduce" in name: return "reductions"
    return "other"

fam = defaultdict(float)
for e in gpu_ev:
    fam[family(e["name"])] += e["dur"]
tot = sum(fam.values())
print(f"\nGPU busy-time breakdown (total {tot/1e6:.3f}s, may double-count overlapping streams):")
for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
    print(f"  {k:42s} {v/1e6:7.3f}s  {100*v/tot:5.1f}%")

# Top 12 individual GPU ops by summed duration
byname = defaultdict(float)
for e in gpu_ev:
    byname[e["name"]] += e["dur"]
print("\nTop individual GPU ops:")
for k, v in sorted(byname.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v/1e6:7.3f}s  {k[:110]}")
