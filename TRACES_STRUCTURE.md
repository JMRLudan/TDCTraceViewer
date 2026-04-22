# Trace file structure

Written when the viewer was first built. Use this as the spec when adding a
new trace source: every new layout either (a) maps cleanly onto the unified
`Group` shape already wired through the UI, or (b) needs a new loader +
`_rescan()` filename rule plus — if the response payload changes — parser
updates.

---

## Unified in-memory shape

Everything the viewer renders lives as `app.Group` (see `app.py`):

```python
@dataclass
class Group:
    group_id: str                 # stable id, used in URLs
    step: int                     # training step this rollout belongs to
    group_type: str               # "mixed" | "needle" | "step_groups" | "first_ever" | ...
    datasource: str | None        # task name (e.g. "AMES", "BBB_Martins"); may be _train-suffixed
    source_file: str              # filename relative to traces/
    prompt: str                   # full harmony-formatted prompt (with <|start|>… markers)
    label: str                    # gold answer — usually "(A)" or "(B)"
    samples: list[dict]           # 1..N samples, each: {"reward": float, "response_text": str, "score": float?, ...}
    group_rewards: list[float] | None
    dataset_idx: int | None
```

`group_id` is built from `{group_type}_step{step}` (with `_{i}` suffix where
multiple groups share the same key, e.g. the 5 groups inside a
`step{N}_groups.json`). URLs: `/group/{group_id}`.

A new trace source needs to produce `Group` instances; everything else
(templates, filters, API) already works.

---

## Current on-disk layouts in `traces/`

### 1. `group_trace_step{N}_mixed.json` — "mixed" (32 files)

One prompt × 24 samples per file.

```jsonc
{
  "step": 0,
  "type": "mixed",
  "group_rewards": [0.2, 0.2, 1.2, ...],      // length 24
  "prompt": "<|start|>system<|message|>...",
  "label": "(B)",
  "samples": [
    { "reward": 0.2, "response_text": "analysisWe need to predict..." },
    ...                                           // 24 total
  ]
}
```

`datasource` is **not** present — inferred from the `Context:` line of the
prompt via `_CONTEXT_RULES`.

Steps covered: every 4th step, 0–124.

### 2. `group_trace_step{N}_needle.json` — "needle" (23 files)

Identical shape to `mixed`, just `"type": "needle"`. These are the "needle"
eval prompts run alongside the regular mixed rollouts. Same 24-sample
structure, same fields.

Steps covered: subset of every-4th-step, 4–124 (no needle for step 0).

### 3. `step{N}_groups.json` — "step_groups" (25 files)

Wraps **5 prompts** per file, each prompt getting its own 24-sample group.

```jsonc
{
  "step": 5,
  "groups": [
    {
      "dataset_idx": 123,
      "datasource": "AMES_train",           // explicit, usually _train-suffixed
      "global_step": 5,
      "prompt": "<|start|>system|...",
      "label": "(A)",
      "samples": [
        { "reward": 1.0, "score": 1.0, "response_text": "..." },
        ...                                      // 24 total
      ]
    },
    ...                                           // 5 groups
  ]
}
```

Note: samples here have an extra `"score"` field alongside `"reward"`.
`datasource` carries a `_train` suffix.

Steps covered: every 5th step, 5–125.

### 4. `first_ever_trace.json` — "first_ever" (1 file)

Sanity-check dump from very early training. Different shape:

```jsonc
{
  "engine_idx": 7,
  "trace": {
    "prompt": "<|start|>system|...",
    "label": "(A)",
    "reward": 1.2,
    "scores": 1.0,
    "action_ranges": [[512,610], [752,1175], [1280,1526]],
    "rollout_log_probs": [...],
    "extra_logs": {...}
  },
  "decoded": {
    "full_text": "<full prompt+response>",
    "sections": [...],
    "response_text": "analysisWe need to predict..."
  }
}
```

Loader pulls `decoded.response_text` as the single sample's response. Only
1 sample, so `n_samples` is 1 for this row.

---

## Response-text format (harmony, markers-stripped)

`response_text` is the assistant's completion in OpenAI "harmony" format
with the `<|start|>` / `<|message|>` / `<|end|>` markers **removed**, so
role/channel transitions appear inline as bare tokens. The parser
(`app.parse_response`) finds them with a single regex and slices the
response into typed `Segment`s.

| On-disk marker                                         | Segment `kind`    | Fields populated                        |
| ------------------------------------------------------ | ----------------- | --------------------------------------- |
| `^analysis` (very first token, implicit role=assistant)| `analysis`        | `text`                                  |
| `assistantanalysis`                                    | `analysis`        | `text`                                  |
| `assistantcommentary`                                  | `commentary`      | `text`                                  |
| `assistantfinal`                                       | `final`           | `text`                                  |
| `assistant to=functions.<name>commentary` `json{...}`  | `tool_call`       | `tool=<name>`, `parsed=<args dict>`     |
| `functions.<name> to=assistantcommentary {...}`        | `tool_response`   | `tool=<name>`, `parsed=<result dict>`   |

The usual per-sample sequence is: `analysis → tool_call(get_mol_...) → tool_response → analysis → tool_call(compare_similar_mols) → tool_response → analysis → final`. When a sample goes off the rails you'll see extra analysis chunks or duplicated tool calls.

Tool-call JSON extraction is brace-depth based (handles escaped quotes &
nested braces), with tolerance for a leading `json` token before the opening
brace.

---

## Adding a new trace source

If someone drops in e.g. a `grpo_rl_epoch2/...` folder or a new filename
pattern, extend the indexer. Three possible cases:

### Case A — same shape, new filename

Just add a regex branch in `TraceIndex._rescan()`:

```python
elif re.match(r"mynewlayout_step\d+\.json$", f):
    self.groups.append(_load_group_from_flat(p))
```

if the file already has `{step, type, prompt, label, samples, group_rewards}`.

### Case B — different fields, but still one prompt per file

Write a new loader that produces a single `Group`, then wire it into
`_rescan()`. Template + parser don't need changes as long as
`samples[i]["response_text"]` exists and is harmony-format.

### Case C — different response format (no harmony markers)

Extend `parse_response`. Either add new regex alternations to `_MARKER_RE`,
or short-circuit for the new format before the main loop. Existing kinds
(`analysis`, `commentary`, `final`, `tool_call`, `tool_response`, `raw`)
already have CSS styling — add a new kind only if one of those doesn't fit.

### Dropdown hygiene

`/api/reindex` re-scans without a server restart. No caching beyond the
in-memory `TraceIndex`, so a reindex picks up new files immediately.

---

---

## SFT compare archives (a sibling shape, not in `traces/`)

Added 2026-04-21. Lives outside any `traces/` folder as its own root:

```
<archive_root>/
  step_{N}/
    summary.json                    # {global_step, sampled_every, tasks: {TASK: {num_results, num_sampled, file}}}
    {TASK}.json                     # one file per TDC task
```

Per-task JSON:

```jsonc
{
  "task": "AMES",
  "num_results": 721,
  "sampled_every": 50,
  "samples": [
    { "index": 0, "prompt": "<|start|>...",
      "response": "<|channel|>analysis<|message|>...<|end|>...",
      "label": "(B)", "reward": 1.0, "score": 1.0, "extra_logs": {...} },
    ...
  ],
  "best_correct": { ...same shape as a sample... },
  "worst_wrong":  { ...same shape as a sample... }
}
```

Key differences vs GRPO traces:

1. Response uses **full harmony markers** (`<|channel|>...<|message|>...<|end|>`).
   The parser normalizes these to the markers-stripped form via
   `_normalize_harmony()` before segmentation, so the same segment taxonomy
   (`analysis`, `commentary`, `final`, `tool_call`, `tool_response`) applies.
2. Prompt is included per-sample (not shared across samples like GRPO groups).
3. No `group_rewards` list — per-sample `reward` and `score` only.
4. Extra `best_correct` + `worst_wrong` singletons outside the sampled subset
   get highlighted at the top of the per-task page.

Indexed lazily: `SFTArchive` caches only the step→task summary map; task
JSON is opened on detail-page hit.

Discovery rule: any sibling of the viewer's run dir with at least one
`step_*/summary.json` containing a `"tasks"` key (overridable by
`SFT_COMPARE_DIRS` env var).

---

## File inventory (as of 2026-04-21)

- `traces/` — 111 MB, 81 files total across the 4 layouts above.
- Sibling folders in the run dir that might eventually feed the viewer:
  - `eval_traces/` — 1 representative correct + 1 wrong example per task per checkpoint (8 checkpoints). Shape: `{global_step, datasources: {TASK: {correct: {prompt,response,reward}, wrong: {prompt,response,reward}}}}`.
  - `eval_metrics/` — per-task `{accuracy, macro_f1, count}` + `avg_macro_f1` + `knn_eval` stats per checkpoint. Not a "trace" but could feed a step-overlay on the group list.
  - `tool_usage_eval/`, `tool_heavy_eval/`, `knn_eval_reversals/`, `eval_unparseable/` — aggregate counters per checkpoint.
  - `vllm_stats/` — JSONL timeseries for rollout/eval throughput and per-iteration train timing.
