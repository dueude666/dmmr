$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe"

& $python "$repo\baselines\ssas_unified\run_ssas_seed.py" `
  --exp_name smoke_ssas_seed `
  --session 1 `
  --subject_start 0 `
  --subject_end 1 `
  --seed 3 `
  --max_iter1 1 `
  --max_iter2 1 `
  --batch_size 50
