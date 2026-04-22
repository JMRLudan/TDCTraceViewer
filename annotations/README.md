# Claude annotations

One JSON file per annotated sample. Drop files into this directory; the
viewer picks them up on next page load (or via `GET /api/reindex`).

## Schema

```json
{
  "id": "grpo_0420_1223_mixed_step0_s0",
  "title": "Initial AMES prediction — correct via neighbor similarity",
  "verdict": "correct",
  "tags": ["tool-use", "neighbor-evidence"],
  "created_at": "2026-04-21T18:00:00Z",

  "source": {
    "kind": "grpo",
    "run_id": "0420_1223",
    "group_id": "mixed_step0",
    "sample_idx": 0
  },

  "summary": "Top-level paragraph describing overall behavior.",

  "segment_notes": [
    { "segment_idx": 0, "heading": "Framing", "note": "Model reframes the task as..." },
    { "segment_idx": 1, "heading": "Tool choice", "note": "Calls compare_similar_mols because..." },
    { "segment_idx": 4, "heading": "Final answer", "note": "Commits to (B) because..." }
  ],

  "conclusion": "Closing paragraph: why the sample is correct / wrong / interesting."
}
```

## Source shapes

### GRPO sample

```json
{"kind": "grpo", "run_id": "0420_1223", "group_id": "mixed_step0", "sample_idx": 0}
```

### SFT sample (indexed)

```json
{"kind": "sft", "archive_id": "gpt_5_4_mini_compare_only_sft",
 "step": 0, "task": "AMES", "sample_idx": 0}
```

### SFT highlight (best_correct / worst_wrong singleton)

```json
{"kind": "sft", "archive_id": "gpt_5_4_mini_compare_only_sft",
 "step": 0, "task": "AMES", "sample_idx": "best_correct"}
```

## Filename

Any `*.json` in this directory is loaded. The filename stem is used as a
fallback `id` if the JSON doesn't carry one. Suggested naming:

```
grpo_{run_id}_{group_id}_s{sample_idx}.json
sft_{archive_id}_step{N}_{task}_s{sample_idx}.json
```

Files beginning with `_` or `.` are ignored by convention (use them for
drafts or notes). Anything ending in `.md` is ignored by the loader.

## Field notes

- `id` is used in the URL: `/annotations/{id}`.
- `verdict` is rendered as a colored chip (`correct`, `wrong`, `interesting`;
  any other value renders uncolored).
- `segment_notes[].segment_idx` indexes into the parsed segment list (0 =
  first analysis chunk, 1 = first tool call, etc.). An out-of-range or
  non-integer `segment_idx` falls out as an "other notes" block at the
  bottom of the page — useful for free-standing observations.
- `tags` is a free-form list of strings shown as chips.
- `created_at` is ISO-8601 UTC. Surfaces on the list and detail pages.
