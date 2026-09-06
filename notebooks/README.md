# notebooks/

Interactive FiftyOne notebooks for inspecting data at different pipeline stages. All scripted/repeatable work lives in `scripts/` — these are for looking at data, not producing it.

| Notebook | Stage | Needs network? | What it browses |
|---|---|---|---|
| `fiftyone_preview.ipynb` | Before pulling | Yes | A class at real volume, straight from the Open Images Zoo — nothing saved to disk. Tune filters before committing to a full pull. Also has an ad-hoc "candidate" section (2026-08-20) for previewing a class that isn't in `config/classes.yaml` yet at all — e.g. weighing a brand-new class before deciding whether to add it. |
| `fiftyone_explore.ipynb` | After a raw pull | No | An already-acquired raw per-class export (`dataset/raw/open_images/<class>/`, COCO-style). |
| `fiftyone_review_processed.ipynb` | After Stage 5.2 conversion | No | Converted intermediate-schema output (`dataset/processed/<source>/`, canonical class ids, flat `images/`+`labels/`) — what Stage 5.3/5.5 curation actually works on. |
| `fiftyone_final_dataset.ipynb` | After Stage 5.9 | No | The **final training dataset** (`dataset/final/{train,val,test}/`) as one browsable pool, with `source` and `split` as real sidebar-filterable fields — the source is otherwise only a `<source>__` filename prefix that the App can't filter on. **Read-only by design:** `split.py` regenerates `dataset/final/` wholesale on every cascade run, so edits made here would be destroyed; corrections belong in `fiftyone_review_processed.ipynb`, which writes to `labels_reviewed/`. Also prints per-source × per-split and per-class × per-split breakdowns, and cross-checks its counts against `split_report.json`. |
| `fiftyone_test.ipynb` | — | Yes | Historical scratch notebook (the original experiment that surfaced `IsDepiction`). Not part of the maintained workflow — kept for reference only. |

Each notebook has a markdown cell at the top with the same information plus usage notes. See `docs/DECISIONS.md` and `docs/PLAN.md` for why the pipeline is staged this way.
