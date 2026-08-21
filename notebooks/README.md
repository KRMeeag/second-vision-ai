# notebooks/

Interactive FiftyOne notebooks for inspecting data at different pipeline stages. All scripted/repeatable work lives in `scripts/` — these are for looking at data, not producing it.

| Notebook | Stage | Needs network? | What it browses |
|---|---|---|---|
| `fiftyone_preview.ipynb` | Before pulling | Yes | A class at real volume, straight from the Open Images Zoo — nothing saved to disk. Tune filters before committing to a full pull. Also has an ad-hoc "candidate" section (2026-08-20) for previewing a class that isn't in `config/classes.yaml` yet at all — e.g. weighing a brand-new class before deciding whether to add it. |
| `fiftyone_explore.ipynb` | After a raw pull | No | An already-acquired raw per-class export (`dataset/raw/open_images/<class>/`, COCO-style). |
| `fiftyone_review_processed.ipynb` | After Stage 5.2 conversion | No | Converted intermediate-schema output (`dataset/processed/<source>/`, canonical class ids, flat `images/`+`labels/`) — what Stage 5.3/5.5 curation actually works on. |
| `fiftyone_test.ipynb` | — | Yes | Historical scratch notebook (the original experiment that surfaced `IsDepiction`). Not part of the maintained workflow — kept for reference only. |

Each notebook has a markdown cell at the top with the same information plus usage notes. See `docs/DECISIONS.md` and `docs/PLAN.md` for why the pipeline is staged this way.
