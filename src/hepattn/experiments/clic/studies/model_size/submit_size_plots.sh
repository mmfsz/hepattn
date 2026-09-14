#!/bin/bash
# Jet-level physics comparisons for the paper-tag size ablation.
#
# A batch job rather than an interactive run because jet clustering wants cores: the vscode
# allocation this study is usually driven from has ONE, and `compute_jets(n_procs=16)` on one core
# turns a ten-minute job into an hour. CPU-only -- the performance module never touches a GPU, it
# reads the .root files `submit_eval_l4.sh` already wrote.
#
# Sized as head's equivalent: five arms, and both the jet clustering and the shared-event
# intersection hold every arm's jets in memory at once.
#
# Takes any number of script paths, relative to the clic experiment dir, and runs them in order:
#   sbatch studies/model_size/submit_size_plots.sh \
#     studies/model_size/plot_size_ablation_jet_iqr.py \
#     studies/model_size/plot_size_ablation_performance.py
#
# ARM_SET picks which set of trainings the scripts draw; `all` is the only set defined so far.
# It has to be exported through sbatch, and every script reads it from the environment:
#   sbatch --export=ALL,ARM_SET=all studies/model_size/submit_size_plots.sh <scripts...>
#
# `plot_size_ablation_training_curves.py` and `plot_decoder_layers.py` need NEITHER an evaluation
# nor jet clustering -- they read csv_metrics/metrics.csv only and run in seconds on a login node.
# Don't queue a 16-core job for them.

#SBATCH --job-name=clic-size-plots
#SBATCH -p hpg-default
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
# --time: no measurement on this branch yet. Head's equivalent job ran the same two scripts over
# the same number of arms well inside its 8 h limit; that is the closest thing to a measurement
# there is, so 8 h is kept and this comment is the flag that it is inherited, not measured.
# Replace it with 1.3x the first completed run's elapsed time and record that in README.md.
#SBATCH --time=08:00:00
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail
cd /blue/avery/m.mazza/projects/fastml/hepattn-paper/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

export ARM_SET="${ARM_SET:-all}"
echo "Hostname: $(hostname)   cores: $(nproc)   ARM_SET: ${ARM_SET}"
for s in "$@"; do
  echo "================================================================ $s"
  pixi run -e clic python "$s"
done
echo "Done!"
