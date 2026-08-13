"""Why do the mask loss kernels move bytes ~400x slower than the mask cost kernel?

Context (see NOTES.md / PROFILING_EXPLAINED.md section 10). In the post-fix trace
`profile_logs/phase2_postfix_trace.pt.trace.json`, four Triton kernels are 68% of all
GPU-busy time:

  triton_red_fused__to_copy_binary_cross_entropy_with_logits_mul_sum_unsqueeze_1
      -> mask_bce_loss forward,  86.1 ms/call, grid [3_218_560, 1, 1] x 128 thr, 38% occ
  triton_red_fused_add_div_mul_sigmoid_sum_unsqueeze_1
      -> mask_dice_loss forward, 50.9 ms/call, grid [3_218_560, 1, 1] x 256 thr, 100% occ
  triton_red_fused__to_copy_add_div_expand_mul_neg_sigmoid_sigmoid_backward_sum_...
      -> mask_dice_loss backward, 43.5 ms/call, grid [251_450, 1, 1]
  triton_red_fused__to_copy_div_expand_mul_sigmoid_sub_sum_unsqueeze_view_0
      -> mask_bce_loss backward,  28.9 ms/call, grid [251_450, 1, 1]

while `triton_red_fused_sum_0` in the *same* trace does the *same* [N_valid, 160] -> [N_valid]
reduction in 0.05 ms, and the dice-cost `aten::bmm` sustains ~2.9 TB/s.

Two hypotheses:

  H1  `torch.compile(fn, dynamic=True)` (loss.py:350-370) forbids Inductor from
      specialising on the real, fixed 160-long reduction axis, so it emits a generic
      looping reduction with a pathological grid.
  H2  the boolean-mask indexing at the top of every mask loss
          pred_logits = pred_logits[object_valid_mask]
      has a data-dependent output shape, which forces `nonzero()` + a device->host
      sync, and (under dynamic=True) keeps the reduction extent symbolic.

This script separates them by measurement. Note that the `object_valid_mask=None` and
`mulmask` variants DO NOT compute the same number as production -- they exist purely to
attribute TIME, and are labelled as such in the output.

Run on one B200 via submit_bench_loss.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import time
import warnings
from collections import defaultdict
from statistics import median

import torch
import torch._dynamo
import torch.nn.functional as F

from hepattn.models.loss import mask_bce_loss, mask_dice_loss

# ---------------------------------------------------------------------------------------
# Shapes: exactly the CLIC production configuration.
#   configs/base.yaml   batch_size: 2048, num_objects: &num_particles 150, precision bf16-mixed
#   pflow_data.py:39    max_nodes: int = 160
# N_VALID is read straight out of the post-fix trace: `triton_red_fused_sum_0` (which is
# targets.sum(-1) inside mask_dice_loss, i.e. one output per surviving object) launches a
# grid of 100_580 / 100_999 / 102_312 blocks on the three profiled steps. So ~102k of the
# 2048*150 = 307_200 object slots survive `object_valid_mask` -- a valid fraction of ~0.333,
# i.e. ~50 real particles per event. The three different values across three consecutive
# steps are themselves the signature of H2's data-dependent shape.
# ---------------------------------------------------------------------------------------
BATCH = 2048
NUM_OBJECTS = 150
NUM_CONSTITUENTS = 160
N_VALID_REF = 102_312
VALID_FRAC = N_VALID_REF / (BATCH * NUM_OBJECTS)  # 0.333
HIT_VALID_FRAC = 0.75  # node_valid occupancy; affects values, not shapes/timings

FNS = {"mask_bce": mask_bce_loss, "mask_dice": mask_dice_loss}


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------
def make_inputs(seed: int = 0, n_valid_target: int = N_VALID_REF, dtype=torch.bfloat16):
    """Synthetic batch shaped like one CLIC decoder layer's mask-task output.

    Mirrors task.py:557-562 (logit = logit_scale * einsum, then padded constituents are
    overwritten with finfo.min) and task.py:606-621 (target cast with .type_as(output),
    sample_weight = target + null_weight * (1 - target) with null_weight = 1.0).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    # node_valid: [B, C] bool, contiguous prefix of real hits per event
    n_hits = torch.randint(
        int(0.5 * NUM_CONSTITUENTS), NUM_CONSTITUENTS + 1, (BATCH,), generator=g, device=dev
    )
    ar_c = torch.arange(NUM_CONSTITUENTS, device=dev)
    input_pad_mask = ar_c.unsqueeze(0) < n_hits.unsqueeze(1)

    # particle_valid: [B, N] bool, contiguous prefix, mean count tuned to hit n_valid_target
    mean_valid = n_valid_target / BATCH
    n_obj = torch.clamp(
        torch.round(torch.randn(BATCH, generator=g, device=dev) * 12 + mean_valid), 1, NUM_OBJECTS
    ).long()
    # correct the total so N_valid lands on the target exactly enough for a fair comparison
    ar_n = torch.arange(NUM_OBJECTS, device=dev)
    object_valid_mask = ar_n.unsqueeze(0) < n_obj.unsqueeze(1)

    # targets: sparse binary assignment, each valid object owns a few hits
    targets = (torch.rand((BATCH, NUM_OBJECTS, NUM_CONSTITUENTS), generator=g, device=dev) < 0.05).to(dtype)
    targets = targets * input_pad_mask.unsqueeze(1).to(dtype)

    logits = (torch.randn((BATCH, NUM_OBJECTS, NUM_CONSTITUENTS), generator=g, device=dev) * 4).to(dtype)
    logits = torch.where(input_pad_mask.unsqueeze(1), logits, torch.finfo(dtype).min)
    logits = logits.detach().requires_grad_(True)

    null_weight = 1.0  # configs/base.yaml, mask task
    sample_weight = targets + null_weight * (1 - targets)

    return {
        "pred_logits": logits,
        "targets": targets,
        "object_valid_mask": object_valid_mask,
        "input_pad_mask": input_pad_mask,
        "sample_weight": sample_weight,
        "n_valid": int(object_valid_mask.sum().item()),
    }


# ---------------------------------------------------------------------------------------
# The candidate rewrite: multiplicative mask instead of boolean indexing.
# Mathematically equivalent to the production functions (same value in exact arithmetic),
# but the tensor shapes are now static, so nothing is data dependent.
# ---------------------------------------------------------------------------------------
def mask_bce_loss_mulmask(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None):
    loss = F.binary_cross_entropy_with_logits(pred_logits, targets, weight=sample_weight, reduction="none")
    if input_pad_mask is not None:
        loss = loss * input_pad_mask.unsqueeze(1)
        valid_counts = input_pad_mask.sum(-1, keepdim=True)
        loss = loss.sum(-1) / valid_counts.clamp_min(1.0)
    else:
        loss = loss.mean(-1)
    if object_valid_mask is None:
        return loss.mean()
    w = object_valid_mask.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


def mask_dice_loss_mulmask(pred_logits, targets, object_valid_mask=None, input_pad_mask=None, sample_weight=None):  # noqa: ARG001
    probs = pred_logits.sigmoid()
    if input_pad_mask is not None:
        probs = probs * input_pad_mask.unsqueeze(1)
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if object_valid_mask is None:
        return loss.mean()
    w = object_valid_mask.to(loss.dtype)
    return (loss * w).sum() / w.sum().clamp_min(1.0)


MULMASK = {"mask_bce": mask_bce_loss_mulmask, "mask_dice": mask_dice_loss_mulmask}


# ---------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------
def build_variants(fn_name: str, skip_autotune: bool):
    """(label, callable, uses_object_valid_mask, exactness note)."""
    base = FNS[fn_name]
    mul = MULMASK[fn_name]
    v = [
        ("eager", base, True, "production semantics"),
        ("compile dynamic=True  [PRODUCTION]", torch.compile(base, dynamic=True), True, "production semantics"),
        ("compile dynamic=False", torch.compile(base, dynamic=False), True, "production semantics"),
        ("compile dynamic=None (auto)", torch.compile(base), True, "production semantics"),
        ("eager, valid_mask=None", base, False, "DIFFERENT VALUE - timing attribution only"),
        ("compile dyn=True, valid_mask=None", torch.compile(base, dynamic=True), False, "DIFFERENT VALUE - timing attribution only"),
        ("compile dyn=False, valid_mask=None", torch.compile(base, dynamic=False), False, "DIFFERENT VALUE - timing attribution only"),
        ("mulmask eager", mul, True, "math-equivalent rewrite"),
        ("mulmask compile dynamic=True", torch.compile(mul, dynamic=True), True, "math-equivalent rewrite"),
        ("mulmask compile dynamic=False", torch.compile(mul, dynamic=False), True, "math-equivalent rewrite"),
    ]
    if not skip_autotune:
        v.append(
            (
                "mulmask compile dyn=False max-autotune",
                torch.compile(mul, dynamic=False, mode="max-autotune-no-cudagraphs"),
                True,
                "math-equivalent rewrite",
            )
        )
    return v


# ---------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------
def call(fn, data, use_valid_mask: bool):
    return fn(
        data["pred_logits"],
        data["targets"],
        object_valid_mask=data["object_valid_mask"] if use_valid_mask else None,
        input_pad_mask=data["input_pad_mask"],
        sample_weight=data["sample_weight"],
    )


def timed(fn, data, use_valid_mask, iters, backward):
    """Median/min/max seconds per call, synchronised."""
    ts = []
    for _ in range(iters):
        if data["pred_logits"].grad is not None:
            data["pred_logits"].grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = call(fn, data, use_valid_mask)
        if backward:
            out.backward()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def bench_variant(label, fn, use_valid_mask, data, warmup, iters):
    """Returns dict with warmup(compile) seconds, fwd and fwd+bwd timings, and the value."""
    res = {"label": label}
    # ---- warmup. For the compiled variants this is dominated by compilation, and covers
    #      both the inference graph and the training (fwd+bwd) graph.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = call(fn, data, use_valid_mask)
    for _ in range(warmup):
        out = call(fn, data, use_valid_mask)
        out.backward()
    torch.cuda.synchronize()
    res["warmup_s"] = time.perf_counter() - t0
    data["pred_logits"].grad = None

    res["value"] = float(call(fn, data, use_valid_mask).detach())
    data["pred_logits"].grad = None
    torch.cuda.synchronize()

    with torch.no_grad():
        fwd = timed(fn, data, use_valid_mask, iters, backward=False)
    fwd_grad = timed(fn, data, use_valid_mask, iters, backward=False)  # graph-building fwd
    fb = timed(fn, data, use_valid_mask, iters, backward=True)
    data["pred_logits"].grad = None

    res["fwd_ms"] = median(fwd) * 1e3
    res["fwd_lo_ms"] = min(fwd) * 1e3
    res["fwd_hi_ms"] = max(fwd) * 1e3
    res["fwdgrad_ms"] = median(fwd_grad) * 1e3
    res["fwdbwd_ms"] = median(fb) * 1e3
    res["bwd_ms"] = res["fwdbwd_ms"] - res["fwdgrad_ms"]
    return res


# ---------------------------------------------------------------------------------------
# Bandwidth reference: what this GPU actually achieves on a trivially streaming kernel
# ---------------------------------------------------------------------------------------
def measure_peak_bandwidth(nbytes=2 << 30):
    n = nbytes // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    a.normal_()
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        b.copy_(a)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    t = median(ts)
    del a, b
    torch.cuda.empty_cache()
    return 2 * nbytes / t  # read + write


def min_traffic_bytes(fn_name, n_valid, use_valid_mask, dtype_bytes=2):
    """Bytes the fused forward reduction *must* touch, in the best case.

    bce reads pred_logits, targets, sample_weight over the surviving rows and writes one
    fp32 partial per row. dice reads pred_logits and targets. The [B, C] pad mask is
    negligible (0.3 MB).  This is a lower bound, i.e. it flatters the kernel.
    """
    rows = n_valid if use_valid_mask else BATCH * NUM_OBJECTS
    n_read = {"mask_bce": 3, "mask_dice": 2}[fn_name]
    return rows * NUM_CONSTITUENTS * dtype_bytes * n_read + rows * 4


# ---------------------------------------------------------------------------------------
# H2: does a host sync fire inside the loss function?
# ---------------------------------------------------------------------------------------
def probe_sync(data):
    print("\n" + "=" * 96)
    print("H2 PROBE A -- torch.cuda.set_sync_debug_mode")
    print("=" * 96)
    for label, fn, use_mask in [
        ("mask_bce_loss eager, object_valid_mask SET", mask_bce_loss, True),
        ("mask_bce_loss eager, object_valid_mask None", mask_bce_loss, False),
        ("mask_dice_loss eager, object_valid_mask SET", mask_dice_loss, True),
        ("mask_bce_loss_mulmask eager, mask SET", mask_bce_loss_mulmask, True),
    ]:
        torch.cuda.synchronize()
        got = []
        try:
            torch.cuda.set_sync_debug_mode("warn")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with torch.no_grad():
                    call(fn, data, use_mask)
                got = [str(x.message)[:110] for x in w]
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

        err = None
        try:
            torch.cuda.set_sync_debug_mode("error")
            with torch.no_grad():
                call(fn, data, use_mask)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:150]}"
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        print(f"  {label}")
        print(f"      warn-mode warnings : {got if got else 'none'}")
        print(f"      error-mode raised  : {err or 'nothing (no sync)'}")


def probe_profiler(data):
    print("\n" + "=" * 96)
    print("H2 PROBE B -- torch.profiler op counts for the PRODUCTION path")
    print("=" * 96)
    fn = torch.compile(mask_bce_loss, dynamic=True)
    for _ in range(3):
        with torch.no_grad():
            call(fn, data, True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(3):
            out = call(fn, data, True)
            out.backward()
        torch.cuda.synchronize()
    data["pred_logits"].grad = None

    agg = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        n = str(e.name)
        if any(k in n for k in ("nonzero", "Synchronize", "Memcpy", "aten::index", "item", "masked_select")):
            agg[n][0] += 1
            agg[n][1] += float(getattr(e, "cpu_time_total", 0.0))
    if not agg:
        print("  no nonzero / sync / memcpy / index ops recorded")
    for k, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {c:5d} x  {t / 1e3:9.2f} ms total   {k[:80]}")

    print("\n  top CUDA kernels (3 fwd+bwd of the production mask_bce path):")
    ka = defaultdict(lambda: [0, 0.0])
    for e in prof.key_averages():
        t = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if t > 0:
            ka[e.key][0] += e.count
            ka[e.key][1] += t
    for k, (c, t) in sorted(ka.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {c:5d} x  {t / 1e3:9.2f} ms gpu     {k[:80]}")


def probe_recompiles(skip_autotune, iters=8):  # noqa: ARG001
    """Realistic case: N_valid changes every step, because it is data dependent.

    A `dynamic=False` compile of a function whose internal shapes move every step is not
    free -- it either recompiles or gives up and falls back to dynamic. This is the thing
    that decides whether 'just flip dynamic=False' is a usable production change.
    """
    print("\n" + "=" * 96)
    print(f"H1/H2 PROBE C -- N_valid varies every step ({iters} distinct batches), recompilation cost")
    print("=" * 96)
    print(f"  torch._dynamo.config.cache_size_limit = {torch._dynamo.config.cache_size_limit}")
    batches = [make_inputs(seed=100 + i, n_valid_target=N_VALID_REF + (i - 4) * 900) for i in range(iters)]
    print("  N_valid per batch:", [b["n_valid"] for b in batches])

    for label, fn, use_mask in [
        ("mask_bce compile dynamic=True  [PRODUCTION]", torch.compile(mask_bce_loss, dynamic=True), True),
        ("mask_bce compile dynamic=False", torch.compile(mask_bce_loss, dynamic=False), True),
        ("mask_bce_mulmask compile dynamic=False", torch.compile(mask_bce_loss_mulmask, dynamic=False), True),
    ]:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        # warm on batch 0 so the first-call compile is not attributed to the loop
        with torch.no_grad():
            call(fn, batches[0], use_mask)
        torch.cuda.synchronize()
        per = []
        for b in batches:
            t0 = time.perf_counter()
            with torch.no_grad():
                call(fn, b, use_mask)
            torch.cuda.synchronize()
            per.append((time.perf_counter() - t0) * 1e3)
        c = torch._dynamo.utils.counters
        print(f"\n  {label}")
        print(f"    per-batch wall (ms): {[round(x, 1) for x in per]}")
        print(f"    unique_graphs={c['stats'].get('unique_graphs')}  frames={dict(c['frames'])}")
        recompiles = {k: dict(v) for k, v in c.items() if "recompile" in k or "graph_break" in k}
        print(f"    recompile/graph-break counters: {recompiles or 'none'}")


# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-autotune", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print("=" * 96)
    print("SETUP")
    print("=" * 96)
    print(f"  device            : {name}")
    print(f"  torch             : {torch.__version__}")
    print(f"  SMs / total mem   : {props.multi_processor_count} / {props.total_memory / 2**30:.0f} GiB")
    print(f"  shapes            : pred_logits [{BATCH}, {NUM_OBJECTS}, {NUM_CONSTITUENTS}] bfloat16")
    print("  autocast          : bf16 (matches trainer precision: bf16-mixed)")

    bw = measure_peak_bandwidth()
    print(f"  measured HBM copy bandwidth (2 GiB read + 2 GiB write): {bw / 1e12:.2f} TB/s")
    print("    -- B200 datasheet HBM3e peak is ~8 TB/s; a device-to-device copy typically")
    print("       reaches 75-85% of peak, so treat the number above as the achievable ceiling.")

    data = make_inputs()
    print(f"  N_valid           : {data['n_valid']} of {BATCH * NUM_OBJECTS} slots "
          f"({data['n_valid'] / (BATCH * NUM_OBJECTS):.3f}); trace reference {N_VALID_REF}")
    print(f"  pred_logits size  : {data['pred_logits'].numel() * 2 / 2**20:.1f} MiB")

    results = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for fn_name in ("mask_bce", "mask_dice"):
            print("\n" + "=" * 96)
            print(f"{fn_name}_loss  --  median of {args.iters} iters, bf16 autocast, "
                  f"[{BATCH}, {NUM_OBJECTS}, {NUM_CONSTITUENTS}]")
            print("=" * 96)
            hdr = (f"{'variant':<40}{'fwd ms':>10}{'bwd ms':>10}{'fwd+bwd':>10}"
                   f"{'fwd GB/s':>10}{'warmup s':>10}{'value':>12}")
            print(hdr)
            print("-" * len(hdr))
            rows = []
            for label, fn, use_mask, note in build_variants(fn_name, args.skip_autotune):
                if "compile" in label:
                    torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    r = bench_variant(label, fn, use_mask, data, args.warmup, args.iters)
                except Exception as e:  # noqa: BLE001
                    print(f"{label:<40}  FAILED: {type(e).__name__}: {str(e)[:60]}")
                    continue
                r["note"] = note
                r["use_valid_mask"] = use_mask
                nbytes = min_traffic_bytes(fn_name, data["n_valid"], use_mask)
                r["fwd_GBps"] = nbytes / (r["fwd_ms"] / 1e3) / 1e9
                r["min_traffic_MB"] = nbytes / 1e6
                rows.append(r)
                print(f"{label:<40}{r['fwd_ms']:>10.2f}{r['bwd_ms']:>10.2f}{r['fwdbwd_ms']:>10.2f}"
                      f"{r['fwd_GBps']:>10.1f}{r['warmup_s']:>10.1f}{r['value']:>12.5f}", flush=True)
            results[fn_name] = rows
            base = next((x for x in rows if "PRODUCTION" in x["label"]), None)
            if base:
                print("\n  speedups vs production (dynamic=True, mask set), forward:")
                for r in rows:
                    print(f"    {r['label']:<42} {base['fwd_ms'] / r['fwd_ms']:6.2f}x fwd  "
                          f"{base['fwdbwd_ms'] / r['fwdbwd_ms']:6.2f}x fwd+bwd   [{r['note']}]")
                print(f"\n  minimum forward traffic assumed: "
                      f"{min_traffic_bytes(fn_name, data['n_valid'], True) / 1e6:.1f} MB (mask set) / "
                      f"{min_traffic_bytes(fn_name, data['n_valid'], False) / 1e6:.1f} MB (no mask)")
                print(f"  production fwd effective bandwidth: {base['fwd_GBps']:.1f} GB/s "
                      f"= {100 * base['fwd_GBps'] * 1e9 / bw:.2f}% of the {bw / 1e12:.2f} TB/s copy ceiling")

    if not args.skip_probes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for probe, probe_args in ((probe_sync, (data,)), (probe_profiler, (data,)),
                                      (probe_recompiles, (args.skip_autotune,))):
                try:
                    probe(*probe_args)
                except Exception as e:  # noqa: BLE001
                    print(f"  PROBE {probe.__name__} FAILED: {type(e).__name__}: {str(e)[:300]}")
                    torch.cuda.set_sync_debug_mode("default")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"device": name, "bandwidth_Bps": bw, "n_valid": data["n_valid"], "results": results}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    with contextlib.suppress(AttributeError):
        torch.set_float32_matmul_precision("high")
    main()
