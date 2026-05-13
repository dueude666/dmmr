# SSAS Reproduction Setup

## Source

Official code is vendored under:

```text
baselines/ssas_official/
```

Source repository:

```text
https://github.com/liuyici/SSAS
```

Cloned commit:

```text
02fcad36c7a2c6ad6f65a058b71d37c291333c81
```

## Local Compatibility

The official SSAS code expects SEED data in a flat `ExtractedFeatures` directory. The unified runner creates a cached flat view from the local repository-style layout:

```text
F:/egg baseline/data/seed3_repo_layout/ExtractedFeatures/label.mat
F:/egg baseline/data/seed3_repo_layout/ExtractedFeatures/1/*.mat
```

The only direct compatibility patch to official code is:

```text
baselines/ssas_official/SSAS/selection_domain_new.py
```

It replaces a hardcoded `E:/Research/MFA_LR_tsne/count/...` output path with `args.count_dir`. This does not change the SSAS model, source-selection logic, or losses.

## Fixed SSAS Parameters

Official reproduction defaults:

```text
dataset=seed
session=1
subject_start=0
subject_end=3
seed=3
max_iter1=20
max_iter2=35
batch_size=50
bottleneck_dim=128
lr_a=0.1
lr_b=0.1
radius=10
gamma=1
```

Fast development quick:

```text
max_iter1=2
max_iter2=3
```

Improved fast quick:

```text
use_source_weight_calibration=true
source_weight_blend=0.3
source_weight_temperature=2.0
use_feature_mixstyle=true
mixstyle_p=0.5
mixstyle_alpha=0.1
```

## Commands

Smoke:

```powershell
.\scripts\smoke_ssas_seed.ps1
```

Fast 3-fold quick:

```powershell
.\scripts\quick_ssas_seed_fast.ps1
```

Improved 3-fold fast quick:

```powershell
.\scripts\quick_ssas_rcmix_fast.ps1
```

Official-parameter 3-fold quick:

```powershell
.\scripts\quick_ssas_seed.ps1
```

Linux/server equivalent:

```bash
PYTHON=/path/to/python bash scripts/quick_ssas_seed.sh
```

## Outputs

Fold logs:

```text
outputs/ssas/<exp_name>/target_XX/official_train_log.txt
```

Summary:

```text
logs/<exp_name>_summary.txt
logs/<exp_name>_summary.json
```
