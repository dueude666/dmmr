# SSAS Quick Results

Protocol for all rows:

```text
dataset=SEED
session=1
subject_start=0
subject_end=3
seed=3
max_iter1=2
max_iter2=3
batch_size=50
```

This is a fast feasibility protocol, not the official full SSAS reproduction setting.

| Experiment | Method | Each fold acc | Avg | Std | Decision |
|---|---|---:|---:|---:|---|
| `quick_ssas_seed_fast` | Official SSAS fast baseline | 0.9232, 0.8218, 0.8215 | 0.8555 | 0.0479 | Baseline for fast feasibility |
| `quick_ssas_weightcal_fast` | SSAS + source-weight calibration | 0.8983, 0.8218, 0.8339 | 0.8513 | 0.0336 | Std improves, avg slightly drops |
| `quick_ssas_weightcal_light_fast` | SSAS + lighter source-weight calibration | 0.9220, 0.8169, 0.8200 | 0.8530 | 0.0488 | No clear gain |
| `quick_ssas_mixstyle_fast` | SSAS + Feature MixStyle | 0.9220, 0.8068, 0.7763 | 0.8350 | 0.0628 | Not recommended |
| `quick_ssas_rcmix_fast` | SSAS + source-weight calibration + MixStyle | 0.8983, 0.8218, 0.8339 | 0.8513 | 0.0336 | Same as weight calibration in this run; MixStyle not useful |

Current conclusion:

```text
Best avg: official SSAS fast baseline.
Best std: SSAS + source-weight calibration.
MixStyle is not suitable for the current SSAS quick setting.
```

Next recommended improvement direction:

```text
Keep SSAS official baseline intact. Replace feature MixStyle with a prototype-level reliability loss or calibrated source prototype memory, because current source-weight calibration helps stability but hurts the first fold's high accuracy.
```
