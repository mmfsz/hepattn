"""Plot train/val loss vs epoch for the CLIC runs in the GLOW jet-IQR investigation.

Two sources of loss history, in order of preference:

1. ``<run>/csv_metrics/metrics.csv`` — written by the CSVLogger added to utils/cli.py on
   2026-07-07. Has BOTH train/loss (every 50 steps) and val/loss (once per epoch).
2. ``<run>/ckpts/epoch=<N>-val_loss=<X>.ckpt`` — the filename fallback. val-loss only; this
   is all that survives for runs started before the CSVLogger fix, because MyCometLogger runs
   offline on HPG and its archive never persists.

Run from the clic experiment dir:  pixi run -e clic python studies/glow_jet_iqr/02_loss_curves/plot_loss_curves.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CLIC = Path(__file__).resolve().parents[3]
LOGS = CLIC / "logs"
OUT = Path(__file__).resolve().parent

# label -> run folder. Order controls plot/legend order.
RUNS = {
    "v6 3xL4 (PLOTTED)": "clic_v6_20260605-T113014",
    "v6 6xL4 (2 nodes)": "clic_v6_20260605-T143447",
    "v6 1xB200": "clic_v6_20260606-T000306",
    "v6 4xB200": "clic_v6_20260612-T102707",
    "v6 fp32 6xL4": "clic_v6_fp32_20260707-T150453",
    "v7 700k 2-node": "clic_v7_20260706-T181417",
    "v7 700k 1-node": "clic_v7_20260706-T181418",
}

CKPT_RE = re.compile(r"epoch=(\d+)-val_loss=([0-9.]+)\.ckpt$")


def load_losses(run_dir: Path):
    """Return (train_steps, train_epoch, val_epoch, source).

    train_steps: DataFrame[step, epoch, loss] or None (step-level trace)
    train_epoch: DataFrame[epoch, loss] or None (per-epoch mean)
    val_epoch:   DataFrame[epoch, loss]
    """
    csv = run_dir / "csv_metrics" / "metrics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        tr = df.loc[df["train/loss"].notna(), ["epoch", "step", "train/loss"]].rename(columns={"train/loss": "loss"})
        va = df.loc[df["val/loss"].notna(), ["epoch", "val/loss"]].rename(columns={"val/loss": "loss"})
        tr_ep = tr.groupby("epoch", as_index=False)["loss"].mean()
        return tr, tr_ep, va.reset_index(drop=True), "csv"

    # Fallback: recover val-loss from checkpoint filenames.
    rows = []
    for ckpt in (run_dir / "ckpts").glob("*.ckpt"):
        m = CKPT_RE.search(ckpt.name)
        if m:
            rows.append({"epoch": int(m.group(1)), "loss": float(m.group(2))})
    if not rows:
        return None, None, None, "none"
    va = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return None, None, va, "ckpt"


def main() -> None:
    loaded = {}
    for label, folder in RUNS.items():
        run_dir = LOGS / folder
        if not run_dir.exists():
            print(f"SKIP {label}: {run_dir} missing")
            continue
        tr, tr_ep, va, src = load_losses(run_dir)
        if va is None:
            print(f"SKIP {label}: no loss history found")
            continue
        loaded[label] = (tr, tr_ep, va, src)
        best = va.loc[va["loss"].idxmin()]
        print(f"{label:22s} [{src:4s}] epochs={len(va):3d}  best val={best['loss']:.5f} @ epoch {int(best['epoch'])}")

    if not loaded:
        raise SystemExit("no runs loaded")

    # ---- Figure 1: per-run train + val curves --------------------------------------------
    n = len(loaded)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for ax, (label, (tr, tr_ep, va, src)) in zip(axes.flat, loaded.items(), strict=False):
        if tr is not None:
            ax.plot(tr["epoch"], tr["loss"], color="tab:blue", alpha=0.18, lw=0.7, label="train (per step)")
            ax.plot(tr_ep["epoch"], tr_ep["loss"], color="tab:blue", lw=1.6, label="train (epoch mean)")
        ax.plot(va["epoch"], va["loss"], color="tab:red", lw=1.6, label="val")
        best = va.loc[va["loss"].idxmin()]
        ax.plot(best["epoch"], best["loss"], "k*", ms=11, zorder=5, label=f"best val {best['loss']:.3f}")
        note = "" if src == "csv" else "  (val only — no CSVLogger)"
        ax.set_title(f"{label}{note}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("CLIC training loss curves (log y)", fontsize=13)
    fig.tight_layout()
    f1 = OUT / "loss_curves_per_run.png"
    fig.savefig(f1, dpi=140)
    print(f"\nwrote {f1}")

    # ---- Figure 2: val-loss overlay across runs ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, (_tr, _tre, va, _src) in loaded.items():
        for ax in axes:
            ax.plot(va["epoch"], va["loss"], lw=1.4, label=label)
    axes[0].set_yscale("log")
    axes[0].set_title("val loss — all runs (log y, full range)")
    axes[1].set_title("val loss — zoom on the converged tail")
    axes[1].set_xlim(100, None)
    tails = [va["loss"].min() for _t, _te, va, _s in loaded.values()]
    axes[1].set_ylim(min(tails) * 0.98, min(tails) * 1.25)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    f2 = OUT / "loss_curves_val_overlay.png"
    fig.savefig(f2, dpi=140)
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
