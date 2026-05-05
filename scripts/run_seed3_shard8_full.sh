#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED3_PATH="${SEED3_PATH:-/data/seed3_repo_layout/ExtractedFeatures}"
VARIANT="${VARIANT:-attnlstm_rspa_warmup}" # baseline | attnlstm_rspa_warmup
SHARD="${SHARD:-0}"                         # 0..7
SEED="${SEED:-3}"
SESSION="${SESSION:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/seed3_shard8"
mkdir -p "${LOG_DIR}"

case "${SHARD}" in
  0) SUBJECT_START=0;  SUBJECT_END=2 ;;
  1) SUBJECT_START=2;  SUBJECT_END=4 ;;
  2) SUBJECT_START=4;  SUBJECT_END=6 ;;
  3) SUBJECT_START=6;  SUBJECT_END=8 ;;
  4) SUBJECT_START=8;  SUBJECT_END=10 ;;
  5) SUBJECT_START=10; SUBJECT_END=12 ;;
  6) SUBJECT_START=12; SUBJECT_END=14 ;;
  7) SUBJECT_START=14; SUBJECT_END=15 ;;
  *) echo "invalid SHARD=${SHARD}, expected 0..7" >&2; exit 2 ;;
esac

TAG="seed${SEED}_s${SESSION}_shard8_${SHARD}"
COMMON_ARGS=(
  "${ROOT_DIR}/main.py"
  --dataset_name seed3
  --session "${SESSION}"
  --seed "${SEED}"
  --seed3_path "${SEED3_PATH}"
  --subject_start "${SUBJECT_START}"
  --subject_end "${SUBJECT_END}"
  --epoch_preTraining 300
  --epoch_fineTuning 500
  --batch_size 512
  --time_steps 30
  --num_workers_train 0
  --num_workers_test 0
)

if [[ "${VARIANT}" == "baseline" ]]; then
  WAY="outputs/seed3_full_baseline_shard8"
  LOG_PATH="${LOG_DIR}/baseline_${TAG}.log"
  EXTRA_ARGS=()
elif [[ "${VARIANT}" == "attnlstm_rspa_warmup" ]]; then
  WAY="outputs/seed3_full_attnlstm_rspa_warmup_shard8"
  LOG_PATH="${LOG_DIR}/attnlstm_rspa_warmup_${TAG}.log"
  EXTRA_ARGS=(
    --use_attn_lstm_readout
    --attn_lstm_alpha_init 0.3
    --attn_lstm_alpha_max 1.0
    --use_rspa
    --rspa_use_warmup
    --rspa_warmup_epochs 4
    --rspa_ramp_epochs 6
    --rspa_alpha_init 0.035
    --rspa_alpha_max 0.14
    --rspa_centered_adaptive_gate
    --rspa_centered_gate_delta 0.2
    --rspa_gate_output_init_std 0.02
  )
else
  echo "invalid VARIANT=${VARIANT}, expected baseline|attnlstm_rspa_warmup" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/${WAY}"
cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${COMMON_ARGS[@]}" --way "${WAY}" --index "${TAG}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"

