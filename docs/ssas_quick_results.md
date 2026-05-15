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
| `quick_ssas_rspm_fast` | SSAS + Reliability-guided Source Prototype Memory | 0.9051, 0.8313, 0.8109 | 0.8491 | 0.0405 | Std improves, avg drops |
| `quick_ssas_tentropy_w001_fast` | SSAS + target entropy minimization | 0.9190, 0.8437, 0.8471 | 0.8699 | 0.0347 | Best current quick result |

Current conclusion:

```text
Best avg: SSAS + target entropy minimization.
Best std among avg-improving methods: SSAS + target entropy minimization.
MixStyle and RSPM are not suitable as the main direction in the current SSAS quick setting.
```

Next recommended improvement direction:

```text
Keep SSAS official baseline intact. Use target entropy minimization as the current main improvement because it improves both avg and std in the 3-fold quick protocol and completed 10-fold validation without instability.
```

## 10-Fold Validation

Protocol:

```text
dataset=SEED
session=1
subject_start=0
subject_end=10
seed=3
max_iter1=2
max_iter2=3
batch_size=50
```

| Experiment | Method | Each fold acc | Avg | Std | Decision |
|---|---|---:|---:|---:|---|
| `tenfold_ssas_tentropy_w001` | SSAS + target entropy minimization | 0.9190, 0.8437, 0.8471, 0.9857, 0.8851, 0.9736, 0.8490, 0.8083, 0.8798, 0.8493 | 0.8841 | 0.0555 | Passed 10-fold feasibility |

Command:

```powershell
.\scripts\tenfold_ssas_tentropy_w001.ps1
```

## Implemented Next Direction

`Reliability-guided Source Prototype Memory (RSPM)` has been added as an optional SSAS DGMA-stage loss.

Command:

```powershell
.\scripts\quick_ssas_rspm_fast.ps1
```

Core idea:

```text
source bottleneck features -> EMA global class prototypes + source-domain class prototypes
source reliability = distance(source-class prototype, global-class prototype)
loss = reliability-weighted source class-center contrast + confidence-gated target pseudo-label center alignment
```

`Target Entropy Minimization` has been added as an optional SSAS DGMA-stage loss.

Command:

```powershell
.\scripts\quick_ssas_tentropy_fast.ps1
```

Core idea:

```text
Use unlabeled target batches already present in SSAS DGMA training.
Add a small entropy penalty on target logits: total_loss += 0.01 * H(p_target).
This encourages confident target decision boundaries without changing SSAS architecture.
```
