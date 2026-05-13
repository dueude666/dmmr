$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe"

& $python "$repo\baselines\ssas_unified\run_ssas_seed.py" `
  --exp_name quick_ssas_mixstyle_fast `
  --session 1 `
  --subject_start 0 `
  --subject_end 3 `
  --seed 3 `
  --max_iter1 2 `
  --max_iter2 3 `
  --batch_size 50 `
  --use_feature_mixstyle `
  --mixstyle_p 0.5 `
  --mixstyle_alpha 0.1
