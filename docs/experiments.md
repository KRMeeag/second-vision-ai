# Experiments

The reproducibility record for every trained model. `models/current/README.md` names
this file as the thing that makes trained artifacts disposable: weights, ONNX and HEF
are all gitignored because *this file plus the pinned dataset and config is enough to
rebuild them.* An entry that does not carry enough to do that is incomplete.

Populated by `scripts/train/evaluate.py`, which prints a ready-made table row at the
end of every evaluation.

## The pre-registered target

**mAP@0.5 ≥ 0.70** (test split, FP32, 640px), **≥ 0.65** for the INT8 HEF. Per-class
floors: Person and Vehicle ≥ 0.75; Motorcycle, Tricycle and Bicycle ≥ 0.65. **Pole and
Potholes are pre-registered as expected-weak** and carry no floor.

Declared **before the first training run** — see **DEC-122** for the full criterion, its
literature justification, and the honest gaps. Nothing in the code gates on it: a run
under target is a result to report, not a failure to hide.

## Results

Every row is measured on the **frozen test split** (6,403 images), which is byte-identical
across all cap conditions (DEC-119). Rows are therefore directly comparable.

| run | precision | split | mAP@0.5 | mAP@0.5:0.95 | FP vs background | criterion |
|---|---|---|---:|---:|---:|---|
| _no runs yet_ | | | | | | |

`FP vs background` is the sum of the confusion matrix's background column at **conf 0.25**
— not the mAP threshold. The dataset contains zero background images by design (DEC-120),
so this column is the substitute measure, and 0.25 is also what `hef_deploy` ships with.

## Per-run detail

Copy this block per run. Everything in it is needed to reproduce the weights.

```markdown
### <run name>  ·  <date>

| | |
|---|---|
| Condition | cap4500 — train 30,123 / val 6,644 / test 6,403, box ratio 11.95 |
| Model | yolov8s.pt (COCO-pretrained), imgsz 640, nc=15 |
| Schedule | 100 epochs, batch 32, patience off, close_mosaic 10 |
| Git SHA | `<from provenance.json>` |
| ultralytics | 8.4.118 (pinned) |
| Hardware | RTX 4090 |
| Run dir | `runs/detect/<name>/` — holds `provenance.json`, `pip_freeze.txt`, `args.yaml` |
| Per-class CSV | `eval_fp32_test_per_class.csv` (15 rows, always — see below) |
| Weights | <where the .pt actually lives; not in git> |
| ONNX / HEF | <export date + location, or "not exported"> |

**Notes:** <what was unusual, what broke, what you would change>
```

`pip_freeze.txt` in the run directory — not `requirements-train.txt` — is the
authoritative environment record, because it captures the pod template's torch build
alongside everything pip resolved (DEC-120). **Cite that file in the thesis.**

## Reading the per-class CSV

Every CSV has **exactly 15 rows in canonical class order**, whether or not a class
appeared in the split. This is deliberate and is enforced by `evaluate.py`:
`metrics.summary()` silently omits classes with no test instances, so raw ultralytics
output would produce different row sets per condition and the ablation's CSVs would
stop lining up.

A class absent from the split has **empty** metric cells — not `0.0`. The distinction
matters: `0.0` reads as "the model detected nothing", whereas empty means "never
tested, no claim made". `present_in_split` and `instances` say which case you are in.

Do not substitute `box.maps` for these numbers. It seeds every class with the *overall*
mAP and only overwrites classes that were present, so an untested class comes back
looking like an average result (`ultralytics/utils/metrics.py:1000-1005`).
