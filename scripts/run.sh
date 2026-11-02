#!/bin/zsh
# Usage: scripts/run.sh <video> [height_in_meters] [orthodox|southpaw]
# Runs Sports2D on one person, then plots joint angles, center of mass and punches,
# and renders a 0.5x slow-motion video with skeleton, CoM and trunk lean.
set -e
ROOT="${0:A:h:h}"
VIDEO="${1:A}"
HEIGHT="${2:-1.75}"
STANCE="${3:-orthodox}"
STEM="${VIDEO:t:r}"
OUT="$ROOT/results/$STEM"

# --save_graphs false: Sports2D 0.8.x crashes saving graphs without showing them; analyze.py draws our own
"$ROOT/.venv/bin/sports2d" -i "$VIDEO" -n 1 -H "$HEIGHT" \
  -R false -G false --save_graphs false \
  --person_ordering_method highest_likelihood -r "$OUT"

D="$OUT/${STEM}_Sports2D"
"$ROOT/.venv/bin/python" -I "$ROOT/scripts/analyze.py" \
  "$D/${STEM}_Sports2D_m_person00.trc" "$D/${STEM}_Sports2D_angles_person00.mot" "$OUT/analysis"
# no -I: punch_analysis imports analyze.py from the same folder
"$ROOT/.venv/bin/python" "$ROOT/scripts/punch_analysis.py" \
  "$D/${STEM}_Sports2D_m_person00.trc" "$OUT/analysis"
"$ROOT/.venv/bin/python" "$ROOT/scripts/render_overlay.py" "$VIDEO" \
  "$D/${STEM}_Sports2D_px_person00.trc" "$OUT/${STEM}_slowmo_CoM.mp4" "$OUT/analysis/punches.txt" \
  --speed 0.5 --stance "$STANCE"
echo "\nVideo:    $D/${STEM}_Sports2D.mp4\nSlow-mo:  $OUT/${STEM}_slowmo_CoM.mp4\nAnalysis: $OUT/analysis"
