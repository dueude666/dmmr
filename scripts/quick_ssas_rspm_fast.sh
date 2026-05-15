#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO_ROOT}/baselines/ssas_unified/run_ssas_seed.py" \
  --exp_name quick_ssas_rspm_fast \
  --session 1 \
  --subject_start 0 \
  --subject_end 3 \
  --seed 3 \
  --max_iter1 2 \
  --max_iter2 3 \
  --batch_size 50 \
  --use_rspm \
  --rspm_weight 0.05 \
  --rspm_temperature 0.2 \
  --rspm_momentum 0.9 \
  --rspm_target_conf_threshold 0.7 \
  --rspm_target_weight 0.3 \
  --rspm_reliability_tau 1.0
