$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe"

& $python "$repo\baselines\ssas_unified\run_ssas_seed.py" `
  --exp_name quick_ssas_weightcal_light_fast `
  --session 1 `
  --subject_start 0 `
  --subject_end 3 `
  --seed 3 `
  --max_iter1 2 `
  --max_iter2 3 `
  --batch_size 50 `
  --use_source_weight_calibration `
  --source_weight_blend 0.15 `
  --source_weight_temperature 3.0
