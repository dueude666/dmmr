param(
    [string]$Python = "python",
    [string]$Seed3Path = "F:\egg baseline\data\seed3_repo_layout\ExtractedFeatures",
    [ValidateSet("baseline", "attnlstm_rspa_warmup")]
    [string]$Variant = "attnlstm_rspa_warmup",
    [ValidateRange(0, 7)]
    [int]$Shard = 0,
    [int]$Seed = 3,
    [string]$Session = "1"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $repoRoot "logs\seed3_shard8"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$subjectRanges = @(
    @{ Start = 0; End = 2 },   # shard8_0: subject 0,1
    @{ Start = 2; End = 4 },   # shard8_1: subject 2,3
    @{ Start = 4; End = 6 },   # shard8_2: subject 4,5
    @{ Start = 6; End = 8 },   # shard8_3: subject 6,7
    @{ Start = 8; End = 10 },  # shard8_4: subject 8,9
    @{ Start = 10; End = 12 }, # shard8_5: subject 10,11
    @{ Start = 12; End = 14 }, # shard8_6: subject 12,13
    @{ Start = 14; End = 15 }  # shard8_7: subject 14
)

$subjectStart = $subjectRanges[$Shard].Start
$subjectEnd = $subjectRanges[$Shard].End
$tag = "seed${Seed}_s${Session}_shard8_${Shard}"

if ($Variant -eq "baseline") {
    $way = "outputs/seed3_full_baseline_shard8"
    $index = $tag
    $logPath = Join-Path $logsDir "baseline_${tag}.log"
    $extraArgs = @()
}
else {
    $way = "outputs/seed3_full_attnlstm_rspa_warmup_shard8"
    $index = $tag
    $logPath = Join-Path $logsDir "attnlstm_rspa_warmup_${tag}.log"
    $extraArgs = @(
        "--use_attn_lstm_readout",
        "--attn_lstm_alpha_init", "0.3",
        "--attn_lstm_alpha_max", "1.0",
        "--use_rspa",
        "--rspa_use_warmup",
        "--rspa_warmup_epochs", "4",
        "--rspa_ramp_epochs", "6",
        "--rspa_alpha_init", "0.035",
        "--rspa_alpha_max", "0.14",
        "--rspa_centered_adaptive_gate",
        "--rspa_centered_gate_delta", "0.2",
        "--rspa_gate_output_init_std", "0.02"
    )
}

Push-Location $repoRoot
try {
    $commonArgs = @(
        "$repoRoot\main.py",
        "--dataset_name", "seed3",
        "--session", $Session,
        "--seed", "$Seed",
        "--seed3_path", $Seed3Path,
        "--subject_start", "$subjectStart",
        "--subject_end", "$subjectEnd",
        "--epoch_preTraining", "300",
        "--epoch_fineTuning", "500",
        "--batch_size", "512",
        "--time_steps", "30",
        "--num_workers_train", "0",
        "--num_workers_test", "0",
        "--way", $way,
        "--index", $index
    )
    & $Python @commonArgs @extraArgs *>&1 | Tee-Object -FilePath $logPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

