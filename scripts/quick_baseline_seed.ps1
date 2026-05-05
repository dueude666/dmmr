param(
    [string]$Python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe",
    [string]$Seed3Path = "F:\egg baseline\data\seed3_repo_layout\ExtractedFeatures",
    [string]$LogPath = "F:\egg baseline\DMMR\logs\quick_baseline_seed.log"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

Push-Location $repoRoot
try {
    & $Python "$repoRoot\main.py" `
        --dataset_name seed3 `
        --session 1 `
        --seed 3 `
        --seed3_path $Seed3Path `
        --subject_start 0 `
        --subject_end 3 `
        --epoch_preTraining 3 `
        --epoch_fineTuning 3 `
        --max_train_batches 24 `
        --num_workers_train 0 `
        --num_workers_test 0 `
        --way outputs/quick_baseline_seed `
        --index run0 *>&1 | Tee-Object -FilePath $LogPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
