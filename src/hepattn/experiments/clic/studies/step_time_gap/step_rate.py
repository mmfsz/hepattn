"""ms/step between two progress-bar samples of a pre-flight log (default steps 200 -> 300).

    python studies/step_time_gap/step_rate.py slurm_logs/slurm-<job>.<name>.out [--lo 200 --hi 300]

Reads the epoch-0 progress bar, which prints every refresh_rate (50) steps, and reports the
elapsed time between the two requested samples divided by the step count, i.e. the steady
state after torch.compile warm-up, the figure studies/swiglu_silu/ compares.
"""

import argparse
import re
from pathlib import Path

BAR = re.compile(r"Epoch 0:\s+\d+%\|[^|]*\|\s*(\d+)/\d+ \[(\d+):(\d+)(?::(\d+))?<")

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--lo", type=int, default=200)
ap.add_argument("--hi", type=int, default=300)
a = ap.parse_args()

t = {}
for m in BAR.finditer(Path(a.log).read_text(errors="replace")):
    step, *clock = m.groups()
    parts = [int(c) for c in clock if c is not None]
    secs = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
    t.setdefault(int(step), secs)
if a.lo not in t or a.hi not in t:
    raise SystemExit(f"{a.log}: samples at steps {sorted(t)}; need {a.lo} and {a.hi}")
print(f"{a.log}: {1000 * (t[a.hi] - t[a.lo]) / (a.hi - a.lo):.0f} ms/step over steps {a.lo}->{a.hi} (samples {sorted(t)})")
