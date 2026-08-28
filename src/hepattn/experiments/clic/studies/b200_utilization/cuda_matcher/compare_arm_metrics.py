"""Paired host-vs-device metric comparison from the Phase-3 A/B runs.

Phase 3 ran both arms back-to-back in each of three allocations, from the same seed, for the
same 300 steps. That means the CSV metrics it left behind are already a paired physics
comparison -- three (host, device) pairs, each pair on one node -- at no extra allocation cost.
This is a SCREEN, not Phase 4: 300 steps is chaotic early training, and the quantity Phase 4
actually cares about (jet-E IQR on a converged model) is not in here.

Pairing is per allocation, which controls the node the same way the throughput A/B does.

Usage:
    python compare_arm_metrics.py [--logs <clic logs dir>] [--step N]
"""

import argparse
import csv
import statistics
from pathlib import Path

# job -> {arm: run directory}. Read off each run's metadata.yaml (slurm_job_id + hostname) and
# its config.yaml (matcher.device_solver), not inferred from timestamps.
PAIRS = {
    "40291275": {"host": "clic_v6_maskfix_20260826-T123529", "device": "clic_v6_maskfix_20260826-T124546"},
    "40291276": {"host": "clic_v6_maskfix_20260826-T124206", "device": "clic_v6_maskfix_20260826-T123531"},
    "40291277": {"host": "clic_v6_maskfix_20260826-T123537", "device": "clic_v6_maskfix_20260826-T124742"},
}


def load(run_dir):
    """Read one run's CSV metrics into {metric: {step: float}}."""
    out = {}
    with (Path(run_dir) / "csv_metrics" / "metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if not row.get("step"):
                continue
            step = int(row["step"])
            for k, v in row.items():
                if k in {"step", "epoch"} or not v:
                    continue
                out.setdefault(k, {})[step] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/logs")
    ap.add_argument("--step", type=int, default=None, help="step to compare (default: last common)")
    args = ap.parse_args()

    data = {job: {arm: load(Path(args.logs) / d) for arm, d in arms.items()} for job, arms in PAIRS.items()}

    # Steps present in every run.
    stepsets = [set(m.get("train/loss", {})) for j in data.values() for m in j.values()]
    steps = sorted(set.intersection(*stepsets))
    step = args.step if args.step is not None else steps[-1]
    print(f"logged steps: {steps}\ncomparing at step {step}\n")

    metrics = sorted(set.intersection(*[set(m) for j in data.values() for m in j.values()]))

    # Within-arm reproducibility first: it is the noise floor, and without it no difference
    # between the arms is interpretable.
    print("=" * 100)
    print("WITHIN-ARM SPREAD (the noise floor) -- train/loss")
    print("=" * 100)
    for s in steps:
        row = f"  step {s:<4}"
        for arm in ("host", "device"):
            vals = [data[j][arm]["train/loss"][s] for j in PAIRS]
            row += f"   {arm}: mean {statistics.mean(vals):9.5f} spread {max(vals) - min(vals):8.5f}"
        print(row)

    print()
    print("=" * 100)
    print(f"PAIRED device - host, per allocation, at step {step}")
    print("(sign consistency across all three pairs is the thing to look at, not the p-value: df=2)")
    print("=" * 100)
    print(f"{'metric':<52}{'host mean':>12}{'dev mean':>12}{'diff':>11}{'rel':>9}  {'per-pair diffs':<28}{'':>6}")

    flagged = []
    for m in metrics:
        try:
            h = [data[j]["host"][m][step] for j in PAIRS]
            d = [data[j]["device"][m][step] for j in PAIRS]
        except KeyError:
            continue
        diffs = [di - hi for di, hi in zip(d, h, strict=False)]
        hm, dm = statistics.mean(h), statistics.mean(d)
        # Within-arm spread as the reference scale; a difference smaller than it is not a signal.
        floor = max(max(h) - min(h), max(d) - min(d))
        consistent = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
        exceeds = abs(statistics.mean(diffs)) > floor
        rel = (dm - hm) / abs(hm) if hm else float("nan")
        mark = "  <== consistent sign AND exceeds within-arm spread" if (consistent and exceeds) else ""
        if mark:
            flagged.append((m, hm, dm, rel))
        pp = " ".join(f"{x:+.2e}" for x in diffs)
        print(f"{m:<52}{hm:>12.5f}{dm:>12.5f}{statistics.mean(diffs):>+11.4f}{rel:>+8.2%}  {pp:<28}{mark}")

    print()
    print("=" * 100)
    print(f"FLAGGED: {len(flagged)} of {len(metrics)} metrics differ consistently and beyond the within-arm spread")
    print("=" * 100)
    for m, hm, dm, rel in flagged:
        print(f"  {m:<50} host {hm:12.5f}   device {dm:12.5f}   {rel:+.2%}")
    if not flagged:
        print("  none -- every difference sits inside the run-to-run spread of a single arm")


if __name__ == "__main__":
    main()


def trajectory_lag(data, steps, metrics):
    """Is the device arm *behind* on the same curve, or *off* it?

    If exact matching holds, the two arms cannot compute different physics -- but a tiny fp
    difference at step 1 puts them on different trajectories, and at step 299 they sit at
    slightly different points along the same curve. That shows up as a consistent step LAG.
    So for each metric, solve for the shift d such that device(last) == host(last - d) by
    interpolating the host curve over the final logged interval. A single shared lag explaining
    every metric is strong evidence for "same curve, slightly behind"; scattered or
    opposite-signed lags would mean the arms are genuinely off each other's curve.
    """
    last, prev = steps[-1], steps[-2]
    width = last - prev
    print()
    print("=" * 100)
    print(f"TRAJECTORY-LAG TEST: device({last}) == host({last} - lag)?")
    print(f"(host curve interpolated over steps {prev}->{last}; positive lag = device behind)")
    print("=" * 100)
    print(f"{'metric':<52}{'lag (steps)':>14}   per-pair")

    lags = []
    for m in metrics:
        per = []
        for job in PAIRS:
            h_prev = data[job]["host"][m].get(prev)
            h_last = data[job]["host"][m].get(last)
            d_last = data[job]["device"][m].get(last)
            if None in {h_prev, h_last, d_last}:
                per = []
                break
            slope = h_last - h_prev
            # A flat metric carries no timing information; skip rather than divide by ~0.
            if abs(slope) < 1e-9 or abs(d_last - h_last) > abs(slope) * 4:
                per = []
                break
            per.append(-width * (d_last - h_last) / slope)
        if not per:
            continue
        mean = statistics.mean(per)
        lags.append(mean)
        print(f"{m:<52}{mean:>+14.2f}   " + " ".join(f"{x:+7.2f}" for x in per))

    if lags:
        pos = sum(1 for x in lags if x > 0)
        print()
        print(f"  {len(lags)} metrics carry timing information")
        print(
            f"  median lag {statistics.median(lags):+.2f} steps, mean {statistics.mean(lags):+.2f}, "
            f"IQR-ish spread {min(lags):+.2f} to {max(lags):+.2f}"
        )
        print(f"  sign agreement: {pos}/{len(lags)} positive (device behind)")
        print()
        print("  A tight, consistently-signed lag => same trajectory, offset in time: what exact")
        print("  matching plus chaotic early training predicts. It is NOT a physics difference,")
        print("  and it is not what Phase 4 measures. Scattered signs would have been the worry.")
