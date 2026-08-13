"""Steady-state throughput from a Lightning tqdm log.

The rate tqdm prints is a *cumulative* average over the whole run, so it is dragged down by
torch.compile warmup (the first ~50 steps cost minutes). This reads the elapsed-time stamps
instead and measures only the steady-state window, which is the number the study compares.

It also reports the spread of per-interval rates, not just a point estimate: run-to-run
variance on these nodes is ~27%, so a single number without a spread is not interpretable.

Paired B200 logs contain two arms delimited by "===== ARM: <name> =====" and are split
automatically.

Usage:
    python parse_throughput.py <log> [<log> ...] [--warmup 50] [--batch 2048]
"""

import argparse
import re
import statistics
from pathlib import Path

# tqdm bar: "Epoch 0:  62%|##| 800/1295 [07:39<04:44,  1.74it/s, ...]"
BAR = re.compile(r"(\d+)/(\d+)\s*\[(\d+):(\d+)(?::(\d+))?<")
ARM = re.compile(r"=+ ARM: (\S+) =+")


def elapsed_s(m):
    """Convert a tqdm MM:SS or HH:MM:SS stamp to seconds."""
    a, b, c = m.group(3), m.group(4), m.group(5)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def parse_segment(lines):
    """-> sorted [(step, elapsed_seconds)], keeping the FIRST sighting of each step."""
    seen = {}
    for line in lines:
        for m in BAR.finditer(line):
            step = int(m.group(1))
            seen.setdefault(step, elapsed_s(m))
    return sorted(seen.items())


def split_arms(text):
    """-> [(arm_name, lines)]; a log with no ARM markers is one unnamed segment."""
    lines = text.replace("\r", "\n").split("\n")
    marks = [(i, m.group(1)) for i, line in enumerate(lines) if (m := ARM.search(line))]
    if not marks:
        return [(None, lines)]
    out = []
    for j, (i, name) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        seg = lines[i:end]
        # the "END ARM" banner repeats the name; keep only segments that hold a progress bar
        if any(BAR.search(x) for x in seg):
            out.append((name, seg))
    return out


def report(label, pts, warmup, batch):
    pts = [(s, t) for s, t in pts if s >= warmup]
    if len(pts) < 3:
        print(f"  {label:<28} insufficient data ({len(pts)} points past step {warmup})")
        return None
    (s0, t0), (s1, t1) = pts[0], pts[-1]
    if t1 <= t0:
        print(f"  {label:<28} degenerate timing window")
        return None
    it_s = (s1 - s0) / (t1 - t0)
    # per-interval rates -> spread
    # zip() without strict= / pairwise() on purpose: pts vs pts[1:] are meant to differ in
    # length, and this keeps the script runnable by the login node's pre-3.10 system python.
    rates = [(b - a) / (tb - ta) for (a, ta), (b, tb) in zip(pts, pts[1:]) if tb > ta]  # noqa: B905, RUF007
    med = statistics.median(rates) if rates else float("nan")
    lo, hi = (min(rates), max(rates)) if rates else (float("nan"),) * 2
    sps = it_s * batch if batch else None
    extra = f"  {sps:7.0f} samples/s" if sps else ""
    print(f"  {label:<28} {it_s:6.3f} it/s  ({s1 - s0} steps over {t1 - t0}s){extra}")
    print(f"  {'':<28} per-interval median {med:.3f}, range {lo:.3f}-{hi:.3f} it/s")
    return it_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--warmup", type=int, default=50, help="first step of the steady-state window")
    ap.add_argument("--batch", type=int, default=0, help="global batch, to convert to samples/s")
    a = ap.parse_args()

    for path in a.logs:
        p = Path(path)
        print(f"\n{p.name}")
        if not p.exists():
            print("  MISSING")
            continue
        for arm, lines in split_arms(p.read_text(errors="replace")):
            report(arm or "(whole log)", parse_segment(lines), a.warmup, a.batch)


if __name__ == "__main__":
    main()
