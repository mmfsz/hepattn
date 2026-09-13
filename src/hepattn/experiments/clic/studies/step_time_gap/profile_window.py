"""Profile a short window of training steps with torch.profiler, compile left on.

Lightning's PyTorchProfiler cannot take a schedule from YAML, and profiling from step 0 records
the torch.compile warm-up instead of the steady state. This callback counts the batches it sees
(so it works on a resumed run whose global_step starts at 97,200), starts the profiler at
``start_step``, records ``active`` steps with a ``ProfilerStep`` marker per batch, then writes the
chrome trace and prints the key-average tables into the log.

    --trainer.callbacks+=profile_window.ProfileWindow --trainer.callbacks.init_args.start_step=250
"""

from pathlib import Path

import torch
from lightning import Callback


class ProfileWindow(Callback):
    def __init__(self, start_step: int = 250, active: int = 6, filename: str = "trace_window"):
        super().__init__()
        self.start_step = start_step
        self.active = active
        self.filename = filename
        self._seen = 0
        self._prof = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._seen == self.start_step:
            self._prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            self._prof.__enter__()
            print(f"ProfileWindow: profiling {self.active} steps from batch {self._seen}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._seen += 1
        if self._prof is None:
            return
        self._prof.step()
        if self._seen == self.start_step + self.active:
            self._prof.__exit__(None, None, None)
            out = Path(trainer.log_dir or ".") / f"{self.filename}.pt.trace.json"
            self._prof.export_chrome_trace(str(out))
            ka = self._prof.key_averages()
            print(f"ProfileWindow: wrote {out}")
            print("ProfileWindow: top by CUDA time")
            print(ka.table(sort_by="cuda_time_total", row_limit=40))
            print("ProfileWindow: top by self CPU time")
            print(ka.table(sort_by="self_cpu_time_total", row_limit=40))
            self._prof = None
