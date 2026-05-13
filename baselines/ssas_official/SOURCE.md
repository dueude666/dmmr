# SSAS Official Source

This directory contains the official SSAS implementation cloned for reproduction.

- Source repository: https://github.com/liuyici/SSAS
- Cloned commit: `02fcad36c7a2c6ad6f65a058b71d37c291333c81`
- Local integration note: the nested `.git` directory was removed so the code can be versioned inside this repository.

Local compatibility changes:

- `SSAS/selection_domain_new.py`: replaced a hardcoded `E:/Research/...` count-output path with `args.count_dir` / `args.output_dir/count`. This does not change the model or loss logic.
- `SSAS/selection_domain_new.py` and `SSAS/solvers.py`: added optional `FeatureMixStyle` hooks controlled by `args.use_feature_mixstyle`. The default is off, so official SSAS behavior is unchanged.

The official code expects SEED `ExtractedFeatures` in a flat session directory containing `label.mat` and session `.mat` files. The unified runner under `baselines/ssas_unified/` prepares this view from the repository-style data layout.
