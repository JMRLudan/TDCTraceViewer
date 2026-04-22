"""Intermediate-decision cache reader — READ-ONLY.

This module is what ships in the public/Render deploy. It can look up
pre-computed intermediate-decision extractions from the on-disk cache,
but it cannot run the LLM extractor. That keeps the public viewer from
needing an OpenAI API key or being able to burn credits via its HTTP
surface.

The full extractor (``intermediate_decisions.py``) is kept out of the
public repo via .gitignore. Backfill and local on-demand extraction
still use the full module on Josh's laptop.

Cache layout is identical to the writer:
    viewer/cache/intermediate_decisions/<sha256(response_text)[:32]>.json

Payload schema (v1):
    {
      "decisions": [
        {"after_tool_idx": int, "tool": str|None,
         "decision": "A"|"B"|None,
         "confidence": "committed"|"tentative"|"none",
         "evidence": str, "is_last_before_final": bool},
        ...
      ],
      "final_answer": "A"|"B"|None,
      "n_windows": int,
      "version": 1
    }
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "cache" / "intermediate_decisions"


def _response_hash(response_text: str) -> str:
    return hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:32]


def cache_path(response_text: str) -> Path:
    return CACHE_DIR / f"{_response_hash(response_text)}.json"


def cache_get(response_text: str) -> dict | None:
    """Return cached payload if present, else None. Never calls an LLM."""
    p = cache_path(response_text)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None
