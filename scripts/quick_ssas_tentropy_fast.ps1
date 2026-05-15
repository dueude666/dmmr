param(
    [string]$Python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

& $Python .\baselines\ssas_unified\run_ssas_seed.py `
  --exp_name quick_ssas_tentropy_w001_fast `
  --session 1 `
  --subject_start 0 `
  --subject_end 3 `
  --seed 3 `
  --max_iter1 2 `
  --max_iter2 3 `
  --batch_size 50 `
  --use_target_entropy `
  --target_entropy_weight 0.01
