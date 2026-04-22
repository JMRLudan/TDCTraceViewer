# GRPO / SFT rollout viewer

FastAPI browser for:

- **GRPO rollout runs** — per-prompt 24-sample groups from `traces/` folders
  in any number of sibling `grpo-*` run directories.
- **SFT compare archives** — per-step-per-task sampled traces from folders
  laid out as `archive/step_{N}/{TASK}.json` with a sibling
  `archive/step_{N}/summary.json`.

One server instance shows both side-by-side with a run / archive switcher.

## Run

```
bash launch.sh              # default port 8765
bash launch.sh 9000         # pick a port
```

Requires `uv` on PATH (see https://astral.sh/uv). Logs land in
`logs/uvicorn_<timestamp>.log`. Open <http://127.0.0.1:8765/>.

## What's indexed

By default the viewer scans **both** `RUN_DIR.parent` (siblings of this
run's folder) and `RUN_DIR/peers/` (bundled peers inside this run's own
folder, for when the parent isn't writable). It looks for:

| Peer kind       | Detection                                                      |
| --------------- | -------------------------------------------------------------- |
| GRPO run        | dir name matches `grpo-*` and contains a `traces/` subdir      |
| SFT compare     | dir contains `step_N/summary.json` with a `"tasks"` key        |

Override discovery with env vars (colon-separated absolute paths):

```
GRPO_RUN_DIRS=/path/run1:/path/run2
SFT_COMPARE_DIRS=/path/archive1:/path/archive2
```

Within a GRPO run, four on-disk layouts are unified onto one `Group` shape
(one prompt + N samples). See `TRACES_STRUCTURE.md` for the spec.

## Endpoints

| path                                                  | what                                              |
| ----------------------------------------------------- | ------------------------------------------------- |
| `/`                                                   | landing page if >1 run or any SFT; else redirect  |
| `/runs/{run_id}/`                                     | group table for one run (filter by step/type/ds)  |
| `/runs/{run_id}/group/{group_id}`                     | 24-sample grid with parsed harmony segments       |
| `/sft/{archive_id}/`                                  | step × task grid of sampled counts                |
| `/sft/{archive_id}/step/{step}/{task}`                | sample grid for one (step, task), plus best/worst |
| `/annotations/`                                       | Claude-authored per-sample walkthroughs           |
| `/annotations/{id}`                                   | annotation detail with notes inline over segments |
| `/analysis/`                                          | failure-analysis dashboard (all runs overlaid)    |
| `/analysis/?run=RID`                                  | single-run analysis view                          |
| `/api/runs`, `/api/runs/{id}/groups`, `/api/runs/{id}/group/{gid}` | JSON equivalents                |
| `/api/sft`, `/api/sft/{id}/step/{s}/{task}`           | JSON equivalents                                  |
| `/api/annotations`, `/api/annotations/{id}`           | JSON equivalents                                  |
| `/api/analysis[?run=RID][&include_samples=true]`      | per-run, per-step aggregates                      |
| `/api/reindex`                                        | rescan discovery without restart                  |
| `/healthz`                                            | `{ok, n_runs, n_sft, total_groups}`               |
| `/group/{id}` (legacy)                                | redirects to default run's group                  |

## Harmony format

Two surface variants are normalized into the same segmented view:

- **markers-stripped** (GRPO traces) — role/channel transitions appear as
  bare tokens like `assistantanalysis`, `assistant to=functions.X commentary`.
- **full-marker** (SFT responses) — transitions wrapped in
  `<|start|>...<|channel|>...<|message|>...<|end|>`; normalized to the
  stripped form before parsing.

Resulting segment kinds: `analysis`, `commentary`, `final`, `tool_call`
(with JSON args extracted), `tool_response` (with JSON result extracted),
and `raw` for anything that doesn't match.

## Failure analysis tab

`/analysis/` aggregates per-sample metrics across all GRPO runs and plots
them against training step. Derived per sample from the parsed response:

- `n_tool_calls`, `tools_used`
- `used_knn` (did it call `compare_similar_mols`)
- `top_pos_sim` / `top_neg_sim` — top-1 similarity on each side of the first
  `compare_similar_mols` result
- `knn_vote` — `B` if top pos > top neg, `A` if top neg > top pos (TDC
  convention: (B) = has-property)
- `final_answer` — letter from `Answer: (X)` in the last final segment
- `gold` — letter from the group-level `label` field (`(A)` / `(B)`)
- `model_correct` / `knn_correct` / `obeys_knn`

Seven charts on the dashboard: model accuracy, KNN obedience rate, KNN
retrieval correctness, KNN call rate, mean #tool-calls, per-tool call rate,
and a stacked outcome-bucket histogram (obey/disobey × knn_right/knn_wrong
× no_knn × no_answer). A per-step table below the charts lists raw counts.
Rates are cached per run; `GET /api/reindex` invalidates them.

## Claude Annotations

The `annotations/` subfolder holds per-sample walkthroughs authored by
Claude when Josh directs it at a specific trace. Each annotation is one
JSON file keyed by source (`grpo` run+group+sample_idx or `sft`
archive+step+task+sample_idx). The detail page re-parses the underlying
sample and interleaves the notes inline above the segments they reference.

Schema and conventions: see `annotations/README.md`.

## Files

```
viewer/
  app.py                   # FastAPI app: Registry + TraceIndex + SFTArchive + AnnotationStore + parser + analysis
  TRACES_STRUCTURE.md      # spec for the on-disk trace layouts
  annotations/             # Claude-authored walkthroughs (JSON per sample)
    README.md              # annotation schema + filename convention
  templates/
    landing.html           # multi-run/archive landing page
    index.html             # GRPO run's group table
    group.html             # GRPO group detail (24 samples)
    sft_archive.html       # SFT step × task grid
    sft_task.html          # SFT sample grid (with best/worst highlights)
    annotations_list.html  # list of all annotations
    annotation_detail.html # one annotation, notes inline with segments
    analysis.html          # failure-analysis dashboard (Chart.js)
    _segment.html          # shared segment renderer
  static/style.css
  launch.sh                # uv-based launcher
  logs/                    # uvicorn stdout/stderr (timestamped)
  cache/
    intermediate_decisions/  # committed: gpt-5-nano extractions per sample
    llm_tracker_logs/        # gitignored: cost/latency CSV per run
```

## Intermediate decisions

Each sample with ≥1 tool call exposes an **intermediate decisions** button
that shows the model's running A/B decision after each tool response, as
extracted by gpt-5-nano from the analysis+commentary window between tool
calls. Results include a confidence label (committed / tentative / none)
and an evidence quote from the original reasoning.

Extractions are cached to disk at `cache/intermediate_decisions/<hash>.json`
(sha256 of `response_text`, first 32 chars). The cache is committed to the
repo so the deployed instance serves everything from disk without an API key.

The deployed viewer is **read-only** — it never calls an LLM. The
extractor module (`intermediate_decisions.py`), the backfill script, and
the LLM cost tracker are kept local (excluded from the repo via
`.gitignore`). To run the backfill locally:

```
OPENAI_API_KEY=sk-... python3 backfill_intermediate_decisions.py \
  --concurrency 50 --max-cost 10.00
```

To enable live on-demand extraction in local dev (POST routes + UI hook):

```
VIEWER_ENABLE_LIVE_EXTRACTION=1 OPENAI_API_KEY=sk-... bash launch.sh
```

## Deploy to Render

This viewer is deployable to Render.com as a public read-only web service.
See `render.yaml` for the blueprint — Python 3.11, `uvicorn app:app`,
with a 1GB persistent disk mounted at `/data`.

**No API keys or LLM code** ship to the Render instance. The
intermediate-decisions panel serves pre-computed extractions only. Fresh
samples without a cache entry display a "no cached extraction" message
rather than calling an LLM.

Data strategy: only the viewer code and the intermediate-decisions cache
ship in the repo. Traces + SFT archives are uploaded separately to
`/data` on the Render disk:

```
/data/
  grpo/<run-id>/traces/*.json
  grpo/<run-id>/step_*/ ...
  sft/<archive-id>/step_*/ ...
```

`GRPO_RUN_DIRS=/data/grpo` and `SFT_COMPARE_DIRS=/data/sft` are set in
the blueprint — each path is scanned one level down for valid run /
archive subdirs. Upload data via Render's SSH shell or by rsyncing to a
staging box that has access to the disk.
