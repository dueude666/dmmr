#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO_ROOT}/baselines/ssas_unified/run_ssas_seed.py" \
  --exp_name quick_ssas_rcmix_fast \
  --session 1 \
  --subject_start 0 \
  --subject_end 3 \
  --seed 3 \
  --max_iter1 2 \
  --max_iter2 3 \
  --batch_size 50 \
  --use_source_weight_calibration \
  --source_weight_blend 0.3 \
  --source_weight_temperature 2.0 \
  --use_feature_mixstyle \
  --mixstyle_p 0.5 \
  --mixstyle_alpha 0.1
