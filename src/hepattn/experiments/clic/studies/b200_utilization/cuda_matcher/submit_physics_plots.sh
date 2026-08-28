#!/bin/bash
# Jet-level physics comparisons (Tier 1 of STUDY.md §4.3) for the two matcher A/Bs.
#
# A batch job rather than an interactive run because jet clustering wants cores: the vscode
# allocation this study is usually driven from has ONE, and `compute_jets(n_procs=8)` on one
# core turns a ten-minute job into an hour. CPU-only -- the performance module never touches a
# GPU, it reads the .root files the eval jobs already wrote.
#
# Takes any number of script paths, relative to the clic experiment dir, and runs them in order:
#   sbatch studies/b200_utilization/cuda_matcher/submit_physics_plots.sh \
#     studies/b200_utilization/cuda_matcher/plot_v6_phase4_jet_iqr.py
#
# Used for plot_v6_phase4_jet_iqr.py (v6 Phase 4, both arms converged) and
# plot_v7_matcher_ab_jet_iqr.py (v7 device vs host at the epoch-169 pair).

#SBATCH --job-name=clic-physics-plots
#SBATCH -p hpg-default
#SBATCH --account=avery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/slurm_logs/slurm-%j.%x.out

set -euo pipefail
cd /blue/avery/m.mazza/projects/fastml/hepattn/src/hepattn/experiments/clic/
export TMPDIR=/var/tmp/

echo "Hostname: $(hostname)   cores: $(nproc)"
for s in "$@"; do
  echo "================================================================ $s"
  pixi run -e clic python "$s"
done
echo "Done!"
