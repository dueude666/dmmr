# Unified SSAS Reproduction

This wrapper runs the official SSAS implementation stored in `baselines/ssas_official/SSAS` with the SEED data layout used by this repository.

Official source:

- Repository: https://github.com/liuyici/SSAS
- Commit: `02fcad36c7a2c6ad6f65a058b71d37c291333c81`

The official SSAS code expects a flat SEED `ExtractedFeatures` directory. `run_ssas_seed.py` creates a cached flat session view from:

```text
F:/egg baseline/data/seed3_repo_layout/ExtractedFeatures/{session}/...
F:/egg baseline/data/seed3_repo_layout/ExtractedFeatures/label.mat
```

Quick reproduction command:

```powershell
& "F:\egg baseline\.venv-cu310sp\Scripts\python.exe" .\baselines\ssas_unified\run_ssas_seed.py `
  --exp_name quick_ssas_seed `
  --session 1 --subject_start 0 --subject_end 3 `
  --seed 3 --max_iter1 20 --max_iter2 35 --batch_size 50
```

Smoke command:

```powershell
& "F:\egg baseline\.venv-cu310sp\Scripts\python.exe" .\baselines\ssas_unified\run_ssas_seed.py `
  --exp_name smoke_ssas_seed `
  --session 1 --subject_start 0 --subject_end 1 `
  --seed 3 --max_iter1 1 --max_iter2 1 --batch_size 50
```

Outputs:

- Fold logs: `outputs/ssas/<exp_name>/target_XX/official_train_log.txt`
- Summary: `logs/<exp_name>_summary.txt` and `logs/<exp_name>_summary.json`

Improved quick experiment:

```powershell
.\scripts\quick_ssas_rcmix_fast.ps1
```

This enables:

- `--use_source_weight_calibration`: smooths SSAS source-selection weights to reduce pseudo-label overconfidence.
- `--use_feature_mixstyle`: migrates CV domain-generalization MixStyle to SSAS bottleneck features.

Both are disabled by default.
