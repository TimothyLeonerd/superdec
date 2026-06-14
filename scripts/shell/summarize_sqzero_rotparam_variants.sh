#!/usr/bin/env bash
set -euo pipefail
EVAL=${1:-/hnvme/workspace/v123be13-WS/superdec_eval}
for d in "$EVAL"/eps_scale_trans_rot_*_16min_*; do
  [ -f "$d/eps_optimization_summary.csv" ] || continue
  echo
  echo "### $(basename "$d")"
  column -s, -t < "$d/eps_optimization_summary.csv" || cat "$d/eps_optimization_summary.csv"
done
