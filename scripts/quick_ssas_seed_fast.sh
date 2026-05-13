#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO_ROOT}/baselines/ssas_unified/run_ssas_seed.py" \
  --exp_name quick_ssas_seed_fast \
  --session 1 \
  --subject_start 0 \
  --subject_end 3 \
  --seed 3 \
  --max_iter1 2 \
  --max_iter2 3 \
  --batch_size 50
