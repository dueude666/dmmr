$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe"

& $python "$repo\baselines\ssas_unified\run_ssas_seed.py" `
  --exp_name quick_ssas_seed `
  --session 1 `
  --subject_start 0 `
  --subject_end 3 `
  --seed 3 `
  --max_iter1 20 `
  --max_iter2 35 `
  --batch_size 50
