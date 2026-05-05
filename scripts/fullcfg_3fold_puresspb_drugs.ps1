param(
    [string]$Python = "F:\egg baseline\.venv-cu310sp\Scripts\python.exe",
    [string]$Seed3Path = "F:\egg baseline\data\seed3_repo_layout\ExtractedFeatures",
    [string]$LogPath = "F:\egg baseline\DMMR\logs\fullcfg_3fold_puresspb_drugs.log"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
New-Item -ItemType Directory -Force -Path "$repoRoot\outputs\fullcfg_3fold_puresspb_drugs" | Out-Null

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
        --use_sspb_v2 `
        --num_subjects_total 15 `
        --prompt_tau 2.0 `
        --prompt_alpha_max 0.2 `
        --prompt_beta_max 0.3 `
        --prompt_alpha_init 0.1 `
        --prompt_beta_init 0.1 `
        --prompt_dropout 0.0 `
        --use_zero_init_prompt_residual 0 `
        --prompt_gate_init 0.01 `
        --use_sspb_differential_lr 1 `
        --sspb_lr 0.005 `
        --use_prompt_ortho_loss 1 `
        --prompt_ortho_weight 0.05 `
        --way outputs/fullcfg_3fold_puresspb_drugs `
        --index run0 *>&1 | Tee-Object -FilePath $LogPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
