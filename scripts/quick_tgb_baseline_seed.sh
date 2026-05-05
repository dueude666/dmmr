#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED3_PATH="${SEED3_PATH:-$ROOT_DIR/../data/seed3_repo_layout/ExtractedFeatures}"
LOG_PATH="${ROOT_DIR}/logs/quick_tgb_baseline_seed.log"

mkdir -p "${ROOT_DIR}/logs"

"${PYTHON_BIN}" "${ROOT_DIR}/main.py" \
  --dataset_name seed3 \
  --session 1 \
  --seed 3 \
  --seed3_path "${SEED3_PATH}" \
  --subject_start 0 \
  --subject_end 3 \
  --epoch_preTraining 3 \
  --epoch_fineTuning 3 \
  --max_train_batches 24 \
  --num_workers_train 0 \
  --num_workers_test 0 \
  --use_tgb \
  --tgb_alpha_init 0.1 \
  --way outputs/quick_tgb_baseline_seed \
  --index run0 > "${LOG_PATH}" 2>&1

echo "log: ${LOG_PATH}"
