param(
    [string]$Python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe",
    [string]$RawDir = "F:\egg baseline\data\seed_de_real_raw\ExtractedFeatures",
    [string]$PreparedDir = "F:\egg baseline\data\seed3_repo_layout\ExtractedFeatures",
    [string]$Session = "1",
    [int]$MaxSubjects = 1,
    [int]$EpochPreTraining = 1,
    [int]$EpochFineTuning = 1
)

$repoRoot = Split-Path -Parent $PSScriptRoot

& $Python "$repoRoot\tools\prepare_seed3_repo_layout.py" --raw-dir $RawDir --output-root $PreparedDir
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Push-Location $repoRoot
try {
    & $Python "$repoRoot\main.py" `
        --dataset_name seed3 `
        --session $Session `
        --seed3_path $PreparedDir `
        --epoch_preTraining $EpochPreTraining `
        --epoch_fineTuning $EpochFineTuning `
        --max_subjects $MaxSubjects `
        --num_workers_train 0 `
        --num_workers_test 0 `
        --way DMMR/seed3_sanity `
        --index run0
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
