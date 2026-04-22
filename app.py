"""
GRPO rollout viewer — FastAPI app.

Serves a browser for GRPO rollout groups (24 samples per prompt) AND for
SFT-style compare archives (per-step-per-task sampled traces). A single
instance can index multiple GRPO runs and multiple SFT archives side by
side.

Launch
------
    uv run --with fastapi --with uvicorn --with jinja2 \
        python -m uvicorn app:app --host 127.0.0.1 --port 8765

Discovery
---------
By default we pick up:
  * this viewer's own run (RUN_DIR = ../) if it has a traces/ subdir
  * any sibling dir of RUN_DIR matching ``grpo-*`` that contains traces/
  * any sibling dir of RUN_DIR containing step_*/summary.json with a
    "tasks" key (SFT compare archives)

Override with env vars (colon-separated paths). Each path can either point
at a run/archive directly, or at a parent that contains one or more
valid run/archive subdirs — the latter is scanned one level deep:
  * GRPO_RUN_DIRS=/path/to/run1:/path/to/run2        # direct
  * GRPO_RUN_DIRS=/data/grpo                          # scan for children
  * SFT_COMPARE_DIRS=/path/to/archive1:/path/to/archive2
  * SFT_COMPARE_DIRS=/data/sft                        # scan for children
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent                         # the grpo-tdc-gptoss-... folder this viewer lives in
TRACES_DIR = RUN_DIR / "traces"
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

# ---------------------------------------------------------------------------
# Harmony-format response parser
# ---------------------------------------------------------------------------
#
# Two surface forms show up in this project:
#   1. "markers-stripped" harmony — the GRPO trace files. Role/channel
#      transitions appear as bare tokens like ``assistantanalysis``,
#      ``assistant to=functions.X commentary``, ``functions.X to=assistant
#      commentary``, etc. (no ``<|start|>`` / ``<|message|>`` markers).
#   2. "full-marker" harmony — the SFT compare archives. Role/channel
#      transitions are wrapped in ``<|start|>…<|channel|>…<|message|>``
#      markers with ``<|end|>`` / ``<|call|>`` / ``<|return|>`` terminators.
#
# We normalize (2) into (1) before parsing — see ``_normalize_harmony()``.

_MARKER_RE = re.compile(
    r"(?P<tool_call>assistant\s*to=functions\.(?P<tc_name>[A-Za-z0-9_]+)\s*commentary)"
    r"|(?P<tool_resp>functions\.(?P<tr_name>[A-Za-z0-9_]+)\s*to=assistant\s*commentary)"
    r"|(?P<a_analysis>assistantanalysis)"
    r"|(?P<a_commentary>assistantcommentary)"
    r"|(?P<a_final>assistantfinal)"
)


@dataclass
class Segment:
    kind: str                     # analysis | commentary | final | tool_call | tool_response
    text: str                     # body after the header
    tool: str | None = None       # tool name if kind in {tool_call, tool_response}
    parsed: Any = None            # parsed JSON payload when extractable


def _try_extract_json(text: str) -> tuple[Any, str]:
    """Pull the first JSON object out of ``text`` if present.

    Returns ``(parsed_obj_or_None, leftover_text_after_object)``. Brace-depth
    scan so embedded braces in strings don't confuse the boundary.
    """
    s = text.lstrip()
    # Some tool-call headers have a "json" prefix (from the <|constrain|>json token).
    if s.startswith("json"):
        s = s[4:].lstrip()
    if not s.startswith("{"):
        return None, text

    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[: i + 1]
                    try:
                        return json.loads(candidate), s[i + 1 :]
                    except json.JSONDecodeError:
                        return None, text
    return None, text


def _normalize_harmony(text: str) -> str:
    """Collapse full-marker harmony into the markers-stripped form.

    Idempotent — safe to run on already-stripped text. The point is to make
    one parser work for both GRPO traces (already stripped) and SFT
    compare archives (still have the markers).
    """
    if not text or "<|" not in text:
        return text

    out = text
    # Tool-call header
    # <|end|>?<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>
    out = re.sub(
        r"(?:<\|end\|>\s*)?<\|start\|>assistant<\|channel\|>commentary\s*to=functions\.([A-Za-z0-9_]+)\s*(?:<\|constrain\|>[a-zA-Z0-9_]+\s*)?<\|message\|>",
        r"assistant to=functions.\1 commentary ",
        out,
    )
    # Tool-response header
    # <|call|>?<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>
    out = re.sub(
        r"(?:<\|call\|>\s*)?<\|start\|>functions\.([A-Za-z0-9_]+)\s+to=assistant<\|channel\|>commentary<\|message\|>",
        r"functions.\1 to=assistant commentary ",
        out,
    )
    # Assistant channel continuation: <|end|>?<|start|>assistant<|channel|>(analysis|commentary|final)<|message|>
    out = re.sub(
        r"(?:<\|end\|>\s*)?<\|start\|>assistant<\|channel\|>(analysis|commentary|final)<\|message\|>",
        r"assistant\1",
        out,
    )
    # Leading channel declaration with no <|start|> prefix (first segment in SFT responses)
    out = re.sub(
        r"^\s*<\|channel\|>(analysis|commentary|final)<\|message\|>",
        r"\1",
        out,
    )
    # Strip any remaining harmony tokens
    out = re.sub(r"<\|[a-z_]+\|>", "", out)
    return out


def parse_response(text: str) -> list[Segment]:
    """Split a harmony-format response into typed segments."""
    if not text:
        return []

    text = _normalize_harmony(text)
    segs: list[Segment] = []

    # First chunk is the implicit "analysis" channel if the text starts with
    # "analysis" (the prompt ended in <|start|>assistant, so the first channel
    # is bare). Otherwise treat the initial chunk as raw text until the first
    # marker.
    first_match = _MARKER_RE.search(text)
    cursor = 0
    if text.lstrip().startswith(("analysis", "commentary", "final")):
        lead = text[: first_match.start()] if first_match else text
        body = lead.lstrip()
        kind = "analysis"
        for k in ("analysis", "commentary", "final"):
            if body.startswith(k):
                kind = k
                body = body[len(k):]
                break
        segs.append(Segment(kind=kind, text=body.strip()))
        cursor = first_match.start() if first_match else len(text)
    elif first_match and first_match.start() > 0:
        lead = text[: first_match.start()]
        if lead.strip():
            segs.append(Segment(kind="raw", text=lead.strip()))
        cursor = first_match.start()

    while cursor < len(text):
        m = _MARKER_RE.search(text, cursor)
        if not m:
            tail = text[cursor:].strip()
            if tail:
                segs.append(Segment(kind="raw", text=tail))
            break

        header_end = m.end()
        next_m = _MARKER_RE.search(text, header_end)
        body_end = next_m.start() if next_m else len(text)
        body = text[header_end:body_end].strip()

        if m.group("tool_call"):
            parsed, leftover = _try_extract_json(body)
            tool = m.group("tc_name")
            segs.append(
                Segment(
                    kind="tool_call",
                    text=leftover.strip(),
                    tool=tool,
                    parsed=parsed,
                )
            )
        elif m.group("tool_resp"):
            parsed, leftover = _try_extract_json(body)
            tool = m.group("tr_name")
            segs.append(
                Segment(
                    kind="tool_response",
                    text=leftover.strip(),
                    tool=tool,
                    parsed=parsed,
                )
            )
        elif m.group("a_analysis"):
            segs.append(Segment(kind="analysis", text=body))
        elif m.group("a_commentary"):
            segs.append(Segment(kind="commentary", text=body))
        elif m.group("a_final"):
            segs.append(Segment(kind="final", text=body))

        cursor = body_end

    return segs


# ---------------------------------------------------------------------------
# GRPO trace indexer
# ---------------------------------------------------------------------------


class Group:
    """Unified representation of one prompt + N samples (typically 24).

    Samples are loaded lazily from ``_source_path`` on first access to keep
    the base registry footprint small. On a run with ~10k samples and ~5-10KB
    of response text per sample, eager loading was costing ~240MB RAM just
    to hold response text; lazy loading defers that cost until a group is
    actually viewed and lets the flip-index builder drop each group's
    samples after processing.
    """

    def __init__(
        self,
        run_id: str,
        group_id: str,
        step: int,
        group_type: str,
        datasource: str | None,
        source_file: str,
        prompt: str,
        label: str,
        samples: list[dict] | None = None,
        group_rewards: list[float] | None = None,
        dataset_idx: int | None = None,
        source_path: str | None = None,
        bundle_index: int | None = None,
        sample_count: int | None = None,
        sample_rewards: list[float] | None = None,
    ) -> None:
        self.run_id = run_id
        self.group_id = group_id
        self.step = step
        self.group_type = group_type
        self.datasource = datasource
        self.source_file = source_file
        self.prompt = prompt
        self.label = label
        self.group_rewards = group_rewards
        self.dataset_idx = dataset_idx
        self._source_path = source_path
        self._bundle_index = bundle_index
        if samples is not None:
            # eager path (small groups like first_ever); also derive count/rewards
            self._samples_cache: list[dict] | None = samples
            self._sample_count = len(samples)
            self._sample_rewards = [s.get("reward", 0.0) for s in samples]
        else:
            self._samples_cache = None
            self._sample_count = int(sample_count or 0)
            self._sample_rewards = list(sample_rewards or [])

    @property
    def uid(self) -> str:
        """Globally unique id across runs, used in URLs."""
        return f"{self.run_id}::{self.group_id}"

    @property
    def samples(self) -> list[dict]:
        # If samples were provided eagerly at construction (e.g. first_ever),
        # keep that cache — it's a single-sample group. For every other
        # group, reload from disk on each access: caching here would
        # accumulate 24 samples × ~10KB per group view with no bound, and
        # previously OOM-killed the 512Mi Render instance after a browsing
        # session of ~100 groups. Reload is ~5–15ms and the OS FS cache
        # keeps it hot.
        if self._samples_cache is not None:
            return self._samples_cache
        return self._load_samples()

    @samples.setter
    def samples(self, value: list[dict]) -> None:
        self._samples_cache = value
        if value is not None:
            self._sample_count = len(value)
            self._sample_rewards = [s.get("reward", 0.0) for s in value]

    def unload_samples(self) -> None:
        """Drop any eager cache; lazy groups are already non-caching."""
        self._samples_cache = None

    @property
    def mean_reward(self) -> float:
        if self.group_rewards:
            return statistics.fmean(self.group_rewards)
        if self._sample_rewards:
            return statistics.fmean(self._sample_rewards)
        if self._samples_cache:
            rewards = [s.get("reward", 0.0) for s in self._samples_cache]
            return statistics.fmean(rewards) if rewards else 0.0
        return 0.0

    @property
    def n_samples(self) -> int:
        if self._samples_cache is not None:
            return len(self._samples_cache)
        return self._sample_count

    def _load_samples(self) -> list[dict]:
        """Re-read the source file to pull the per-sample list.

        Mirrors the parsing in ``_load_group_from_flat`` /
        ``_load_groups_from_bundle`` so lazy loads match what eager loads
        would have produced.
        """
        if not self._source_path:
            return []
        try:
            with open(self._source_path) as f:
                d = json.load(f)
        except Exception:
            return []
        name = Path(self._source_path).name
        if name == "first_ever_trace.json":
            tr = d.get("trace", d)
            decoded = d.get("decoded", {}) or {}
            resp = (
                decoded.get("response_text")
                or tr.get("response_text")
                or tr.get("response")
                or ""
            )
            return [{"reward": tr.get("reward", 0.0), "response_text": resp}]
        if isinstance(d.get("groups"), list):
            gi = self._bundle_index or 0
            try:
                return d["groups"][gi].get("samples") or []
            except Exception:
                return []
        return d.get("samples") or []


# Context -> task name (rough inference from opening phrase of the Context line).
_CONTEXT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Mutagenicity means", re.I), "AMES"),
    (re.compile(r"P-glycoprotein \(Pgp\)", re.I), "Pgp_Broccatelli"),
    (re.compile(r"Blood-Brain Barrier|BBB", re.I), "BBB_Martins"),
    (re.compile(r"HIA|Human intestinal absorption", re.I), "HIA_Hou"),
    (re.compile(r"Oral bioavailability", re.I), "Bioavailability_Ma"),
    (re.compile(r"Membrane permeability|PAMPA", re.I), "PAMPA_NCATS"),
    (re.compile(r"hERG", re.I), "hERG"),
    (re.compile(r"Drug-induced liver injury|DILI", re.I), "DILI"),
    (re.compile(r"skin reactions", re.I), "Skin_Reaction"),
    (re.compile(r"Carcinogen", re.I), "Carcinogens_Lagunin"),
    (re.compile(r"ClinTox|clinical trial tox", re.I), "ClinTox"),
    (re.compile(r"CYP2C9.*substrate", re.I), "CYP2C9_Substrate_CarbonMangels"),
    (re.compile(r"CYP2D6.*substrate", re.I), "CYP2D6_Substrate_CarbonMangels"),
    (re.compile(r"CYP3A4.*substrate", re.I), "CYP3A4_Substrate_CarbonMangels"),
    (re.compile(r"3CLPro|3CL protease", re.I), "SARSCoV2_3CLPro_Diamond"),
    (re.compile(r"SARS-CoV-2.*in vitro|Vitro_Touret", re.I), "SARSCoV2_Vitro_Touret"),
]


def infer_datasource(prompt: str) -> str | None:
    m = re.search(r"Context:\s*(.+)", prompt)
    if not m:
        # fall back to scanning the whole prompt body for the task cue
        ctx = prompt[:4000]
    else:
        ctx = m.group(1)[:600]
    for pat, name in _CONTEXT_RULES:
        if pat.search(ctx):
            return name
    return None


def _load_group_from_flat(run_id: str, path: Path) -> Group:
    d = json.load(open(path))
    step = int(d.get("step", -1))
    gtype = d.get("type", "mixed")
    if path.name == "first_ever_trace.json":
        tr = d.get("trace", d)
        prompt = tr["prompt"]
        label = tr.get("label", "")
        decoded = d.get("decoded", {}) or {}
        response = (
            decoded.get("response_text")
            or tr.get("response_text")
            or tr.get("response")
            or ""
        )
        samples = [{"reward": tr.get("reward", 0.0), "response_text": response}]
        return Group(
            run_id=run_id,
            group_id="first_ever",
            step=0,
            group_type="first_ever",
            datasource=infer_datasource(prompt),
            source_file=path.name,
            prompt=prompt,
            label=label,
            samples=samples,
        )
    samples_list = d["samples"]
    rewards = [s.get("reward", 0.0) for s in samples_list]
    return Group(
        run_id=run_id,
        group_id=f"{gtype}_step{step}",
        step=step,
        group_type=gtype,
        datasource=infer_datasource(d["prompt"]),
        source_file=path.name,
        prompt=d["prompt"],
        label=d.get("label", ""),
        samples=None,  # lazy: drop response_text bodies from RAM
        group_rewards=d.get("group_rewards"),
        source_path=str(path),
        sample_count=len(samples_list),
        sample_rewards=rewards,
    )


def _load_groups_from_bundle(run_id: str, path: Path) -> list[Group]:
    d = json.load(open(path))
    step = int(d.get("step", -1))
    out = []
    for i, g in enumerate(d.get("groups", [])):
        prompt = g["prompt"]
        ds = g.get("datasource") or infer_datasource(prompt)
        samples_list = g["samples"]
        rewards = [s.get("reward", 0.0) for s in samples_list]
        out.append(
            Group(
                run_id=run_id,
                group_id=f"grp_step{step}_{i}",
                step=int(g.get("global_step", step)),
                group_type="step_groups",
                datasource=ds,
                source_file=path.name,
                prompt=prompt,
                label=g.get("label", ""),
                samples=None,  # lazy: drop response_text bodies from RAM
                dataset_idx=g.get("dataset_idx"),
                group_rewards=rewards,
                source_path=str(path),
                bundle_index=i,
                sample_count=len(samples_list),
                sample_rewards=rewards,
            )
        )
    return out


class TraceIndex:
    """One GRPO run's traces/ indexed as Group instances."""

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.traces_dir = run_dir / "traces"
        self.groups: list[Group] = []
        self._by_id: dict[str, Group] = {}
        self._rescan()

    def _rescan(self) -> None:
        self.groups.clear()
        if not self.traces_dir.exists():
            return
        for f in sorted(os.listdir(self.traces_dir)):
            p = self.traces_dir / f
            if not f.endswith(".json"):
                continue
            try:
                if f == "first_ever_trace.json":
                    self.groups.append(_load_group_from_flat(self.run_id, p))
                elif re.match(r"group_trace_step\d+_(mixed|needle)\.json$", f):
                    self.groups.append(_load_group_from_flat(self.run_id, p))
                elif re.match(r"step\d+_groups\.json$", f):
                    self.groups.extend(_load_groups_from_bundle(self.run_id, p))
            except Exception as exc:  # pragma: no cover
                print(f"[indexer:{self.run_id}] skipping {f}: {exc}")
        # de-dup ids within a run
        seen: dict[str, int] = {}
        for g in self.groups:
            base = g.group_id
            n = seen.get(base, 0)
            seen[base] = n + 1
            if n:
                g.group_id = f"{base}__{n}"
        self._by_id = {g.group_id: g for g in self.groups}

    def all_steps(self) -> list[int]:
        return sorted({g.step for g in self.groups})

    def all_datasources(self) -> list[str]:
        return sorted({g.datasource for g in self.groups if g.datasource})

    def all_types(self) -> list[str]:
        return sorted({g.group_type for g in self.groups})

    def filter(
        self,
        step: int | None = None,
        group_type: str | None = None,
        datasource: str | None = None,
    ) -> list[Group]:
        out = self.groups
        if step is not None:
            out = [g for g in out if g.step == step]
        if group_type:
            out = [g for g in out if g.group_type == group_type]
        if datasource:
            out = [g for g in out if g.datasource == datasource]
        return out

    def get(self, group_id: str) -> Group | None:
        return self._by_id.get(group_id)


# ---------------------------------------------------------------------------
# SFT compare archive indexer
# ---------------------------------------------------------------------------
#
# Shape:
#   archive_dir/
#     step_{N}/
#       summary.json              {global_step, sampled_every, tasks: {TASK: {...}}}
#       {TASK}.json               {task, num_results, sampled_every,
#                                  samples: [{index, prompt, response,
#                                             label, reward, score, extra_logs}],
#                                  best_correct: {...},
#                                  worst_wrong: {...}}
#
# Per-task JSON is compact enough that we can load lazily on detail pages
# and only cache the step/task summary grid.


@dataclass
class SFTArchive:
    archive_id: str
    root: Path
    # step -> {task -> {num_results, num_sampled, file}}
    summaries: dict[int, dict[str, Any]]

    @property
    def steps(self) -> list[int]:
        return sorted(self.summaries.keys())

    def tasks_at(self, step: int) -> list[str]:
        return sorted(self.summaries.get(step, {}).keys())

    def all_tasks(self) -> list[str]:
        out: set[str] = set()
        for m in self.summaries.values():
            out.update(m.keys())
        return sorted(out)

    def task_file(self, step: int, task: str) -> Path:
        return self.root / f"step_{step}" / f"{task}.json"

    def load_task(self, step: int, task: str) -> dict | None:
        p = self.task_file(step, task)
        if not p.exists():
            return None
        return json.load(open(p))


def _index_sft_archive(root: Path) -> SFTArchive | None:
    """Discover ``step_*/summary.json`` files under ``root`` and build summary map."""
    summaries: dict[int, dict[str, Any]] = {}
    for entry in sorted(os.listdir(root)):
        m = re.match(r"^step_(\d+)$", entry)
        if not m:
            continue
        step = int(m.group(1))
        summary_path = root / entry / "summary.json"
        if not summary_path.exists():
            continue
        try:
            doc = json.load(open(summary_path))
        except Exception as exc:  # pragma: no cover
            print(f"[sft:{root.name}] skipping step {step}: {exc}")
            continue
        tasks = doc.get("tasks") or {}
        if not tasks:
            continue
        summaries[step] = tasks
    if not summaries:
        return None
    return SFTArchive(archive_id=root.name, root=root, summaries=summaries)


# ---------------------------------------------------------------------------
# Run / archive discovery
# ---------------------------------------------------------------------------


def _run_id_from_path(p: Path) -> str:
    """Short run id: trailing MMDD_HHMM timestamp if present, else last dash chunk."""
    m = re.search(r"(\d{4}_\d{4})$", p.name)
    if m:
        return m.group(1)
    return p.name.split("-")[-1] if "-" in p.name else p.name


def _is_grpo_run(p: Path) -> bool:
    return p.is_dir() and (p / "traces").is_dir() and any(
        re.match(r"^(group_trace_step\d+_(mixed|needle)|step\d+_groups|first_ever_trace)\.json$", f)
        for f in os.listdir(p / "traces")
        if (p / "traces" / f).is_file()
    )


def _is_sft_archive(p: Path) -> bool:
    if not p.is_dir():
        return False
    # need at least one step_N/summary.json
    try:
        for entry in os.listdir(p):
            if re.match(r"^step_\d+$", entry) and (p / entry / "summary.json").exists():
                try:
                    d = json.load(open(p / entry / "summary.json"))
                    if isinstance(d, dict) and "tasks" in d:
                        return True
                except Exception:
                    continue
    except OSError:
        return False
    return False


def _search_dirs() -> list[Path]:
    """Dirs we scan for peer runs / archives.

    By default: RUN_DIR.parent (siblings of this run) AND RUN_DIR/peers/
    (peers bundled inside this run's folder — useful when the parent isn't
    writable, e.g. when the viewer is launched from a mounted subfolder
    that doesn't have write access to its parent).
    """
    dirs = [RUN_DIR.parent, RUN_DIR / "peers"]
    return [d for d in dirs if d.exists()]


def _expand_env_paths(env_val: str, predicate) -> list[Path]:
    """Resolve colon-separated env-var paths.

    Each entry can be either:
      * a direct match (predicate(p) is True) — included as-is, OR
      * a scan directory — we look one level down for children that satisfy
        the predicate. This lets ``GRPO_RUN_DIRS=/data/grpo`` pick up any
        run dirs placed under /data/grpo on a Render persistent disk.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in env_val.split(":"):
        raw = raw.strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            continue
        if predicate(p):
            if p not in seen:
                found.append(p)
                seen.add(p)
            continue
        if p.is_dir():
            try:
                children = sorted(p.iterdir())
            except OSError:
                continue
            for c in children:
                cr = c.resolve()
                if cr in seen:
                    continue
                try:
                    if predicate(c):
                        found.append(c)
                        seen.add(cr)
                except Exception:
                    continue
    return found


def _discover_grpo_runs() -> list[Path]:
    env = os.environ.get("GRPO_RUN_DIRS")
    if env:
        return _expand_env_paths(env, _is_grpo_run)
    found: list[Path] = []
    seen: set[Path] = set()
    # the viewer's own run first
    if _is_grpo_run(RUN_DIR):
        found.append(RUN_DIR)
        seen.add(RUN_DIR.resolve())
    for search in _search_dirs():
        for entry in sorted(os.listdir(search)):
            p = search / entry
            rp = p.resolve()
            if rp in seen or rp == RUN_DIR.resolve():
                continue
            if not entry.startswith("grpo"):
                continue
            if _is_grpo_run(p):
                found.append(p)
                seen.add(rp)
    return found


def _discover_sft_archives() -> list[Path]:
    env = os.environ.get("SFT_COMPARE_DIRS")
    if env:
        return _expand_env_paths(env, _is_sft_archive)
    found: list[Path] = []
    seen: set[Path] = set()
    for search in _search_dirs():
        for entry in sorted(os.listdir(search)):
            p = search / entry
            rp = p.resolve()
            if rp in seen or rp == RUN_DIR.resolve():
                continue
            try:
                if _is_sft_archive(p):
                    found.append(p)
                    seen.add(rp)
            except Exception:
                continue
    return found


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


class Registry:
    """Top-level registry aggregating GRPO runs + SFT archives."""

    def __init__(self) -> None:
        self.runs: dict[str, TraceIndex] = {}
        self.run_order: list[str] = []
        self.sft: dict[str, SFTArchive] = {}
        self.default_run_id: str | None = None
        self.rescan()

    def rescan(self) -> None:
        self.runs.clear()
        self.run_order.clear()
        self.sft.clear()

        for rd in _discover_grpo_runs():
            rid = _run_id_from_path(rd)
            # tiebreak on collision
            base = rid
            n = 1
            while rid in self.runs:
                n += 1
                rid = f"{base}_{n}"
            ti = TraceIndex(rid, rd)
            self.runs[rid] = ti
            self.run_order.append(rid)

        for sd in _discover_sft_archives():
            arc = _index_sft_archive(sd)
            if not arc:
                continue
            aid = arc.archive_id
            if aid in self.sft:
                n = 2
                while f"{aid}_{n}" in self.sft:
                    n += 1
                aid = f"{aid}_{n}"
                arc.archive_id = aid
            self.sft[aid] = arc

        # prefer the viewer's own run as default if present, else first
        own_id = _run_id_from_path(RUN_DIR)
        self.default_run_id = own_id if own_id in self.runs else (self.run_order[0] if self.run_order else None)

    def resolve_uid(self, uid: str) -> Group | None:
        if "::" in uid:
            rid, gid = uid.split("::", 1)
        else:
            rid, gid = self.default_run_id or "", uid
        ti = self.runs.get(rid)
        if not ti:
            return None
        return ti.get(gid)


REG = Registry()


# ---------------------------------------------------------------------------
# Claude annotations
# ---------------------------------------------------------------------------
#
# Free-form per-sample annotations authored by Claude to explain what the
# model is doing at each step. Stored as flat JSON under viewer/annotations/,
# one file per annotated sample. Schema lives in annotations/README.md.

ANNOTATIONS_DIR = HERE / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Annotation:
    id: str
    title: str
    source: dict                  # see schema in annotations/README.md
    summary: str
    verdict: str                  # "correct" | "wrong" | "interesting" | ""
    segment_notes: list[dict]     # [{segment_idx, heading, note}]
    conclusion: str
    tags: list[str]
    created_at: str
    path: Path

    @property
    def source_label(self) -> str:
        s = self.source or {}
        kind = s.get("kind")
        if kind == "grpo":
            return f"grpo {s.get('run_id')} / {s.get('group_id')} / sample #{s.get('sample_idx')}"
        if kind == "sft":
            return f"sft {s.get('archive_id')} / step {s.get('step')} / {s.get('task')} / sample #{s.get('sample_idx')}"
        return "?"

    @property
    def source_url(self) -> str | None:
        s = self.source or {}
        kind = s.get("kind")
        if kind == "grpo":
            return f"/runs/{s.get('run_id')}/group/{s.get('group_id')}"
        if kind == "sft":
            return f"/sft/{s.get('archive_id')}/step/{s.get('step')}/{s.get('task')}"
        return None


class AnnotationStore:
    """Flat directory of annotation JSON files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.items: dict[str, Annotation] = {}
        self.rescan()

    def rescan(self) -> None:
        self.items.clear()
        if not self.root.exists():
            return
        for entry in sorted(os.listdir(self.root)):
            if not entry.endswith(".json"):
                continue
            if entry.startswith((".", "_")):
                continue
            p = self.root / entry
            try:
                d = json.load(open(p))
            except Exception as exc:  # pragma: no cover
                print(f"[annotations] skipping {entry}: {exc}")
                continue
            ann_id = d.get("id") or entry[:-5]
            self.items[ann_id] = Annotation(
                id=ann_id,
                title=d.get("title", ann_id),
                source=d.get("source") or {},
                summary=d.get("summary", ""),
                verdict=d.get("verdict", ""),
                segment_notes=d.get("segment_notes") or [],
                conclusion=d.get("conclusion", ""),
                tags=list(d.get("tags") or []),
                created_at=d.get("created_at", ""),
                path=p,
            )

    def list(self) -> list[Annotation]:
        return sorted(self.items.values(), key=lambda a: (a.created_at, a.id))

    def get(self, ann_id: str) -> Annotation | None:
        return self.items.get(ann_id)


ANN_STORE = AnnotationStore(ANNOTATIONS_DIR)


def _resolve_annotated_sample(ann: Annotation) -> dict | None:
    """Fetch the underlying sample dict the annotation targets.

    Returns a dict with keys: prompt, response, reward, score, label, idx,
    plus source_label / source_url for template rendering. Returns None if
    the target can't be resolved (e.g. source went away).
    """
    s = ann.source or {}
    kind = s.get("kind")
    if kind == "grpo":
        ti = REG.runs.get(s.get("run_id", ""))
        if ti is None:
            return None
        g = ti.get(s.get("group_id", ""))
        if g is None:
            return None
        idx = s.get("sample_idx")
        if not isinstance(idx, int) or not (0 <= idx < len(g.samples)):
            return None
        smp = g.samples[idx]
        return {
            "prompt": g.prompt,
            "response": smp.get("response_text", ""),
            "reward": smp.get("reward"),
            "score": smp.get("score"),
            "label": g.label,
            "idx": idx,
            "datasource": g.datasource,
            "step": g.step,
            "group_type": g.group_type,
        }
    if kind == "sft":
        arc = REG.sft.get(s.get("archive_id", ""))
        if arc is None:
            return None
        doc = arc.load_task(s.get("step"), s.get("task", ""))
        if doc is None:
            return None
        idx = s.get("sample_idx")
        pool = None
        if isinstance(idx, int):
            pool = doc.get("samples") or []
            if not (0 <= idx < len(pool)):
                return None
            smp = pool[idx]
        elif idx in ("best_correct", "worst_wrong"):
            smp = doc.get(idx)
            if not smp:
                return None
        else:
            return None
        return {
            "prompt": smp.get("prompt", ""),
            "response": smp.get("response", ""),
            "reward": smp.get("reward"),
            "score": smp.get("score"),
            "label": smp.get("label"),
            "idx": idx,
            "datasource": s.get("task"),
            "step": s.get("step"),
            "group_type": "sft",
        }
    return None


# ---------------------------------------------------------------------------
# Failure analysis (KNN obedience / retrieval quality / tool-call tracking)
# ---------------------------------------------------------------------------
#
# Per-sample metrics we pull out of each GRPO sample's response_text:
#
#   n_tool_calls     — count of ``assistant to=functions.X`` headers
#   tools_used       — set of tool names that appeared as tool_calls
#   used_knn         — True iff ``compare_similar_mols`` was called
#   neighbors        — list of {sim, pred, section} across ALL retrieved
#                      neighbors from the first compare_similar_mols response.
#                      The tool returns 3 "positive" + 3 "negative" neighbors
#                      per call; section membership maps 1:1 to the neighbor's
#                      own label (positive = "is X", negative = "is not X"),
#                      which under TDC convention means B and A respectively.
#   top_pos_sim      — max similarity over the positive neighbors
#   top_neg_sim      — max similarity over the negative neighbors
#   knn_margin       — sum(pos sims) − sum(neg sims); signed retrieval signal
#   knn_top3_pos     — how many of the top-3 highest-sim neighbors are positive
#   knn_vote_top1    — class letter of the single highest-sim neighbor
#   knn_vote_top3    — top-3 majority, tie-break by similarity sum within top-3
#   knn_vote_weighted— sign(knn_margin); majority weighted by similarity
#   knn_vote         — PRIMARY vote used for obey/correctness. Defaults to
#                      knn_vote_weighted (soft vote over all 6 neighbors).
#                      We expose the other definitions too so you can compare.
#   final_answer     — "A" / "B" / None, from ``Answer: (X)`` in the final
#                      segment.
#   gold             — "A" / "B" / None, from the group-level ``label``
#                      field (stored as ``(A)`` / ``(B)`` in the trace file).
#   model_correct    — final_answer == gold
#   knn_correct      — knn_vote == gold
#   obeys_knn        — final_answer == knn_vote   (defined only when both)
#
# We aggregate per (run_id, step) into the rates plotted on the analysis page.
# Cached on each TraceIndex and invalidated on rescan.

_ANS_RE = re.compile(r"Answer:\s*\(([A-Za-z])\)")
_GOLD_RE = re.compile(r"\(([A-Za-z])\)")
_SIM_RE = re.compile(r"similarity:\s*([0-9]*\.?[0-9]+)")


def _parse_neighbors(body: str) -> list[dict]:
    """Extract every ``Neighbor N`` block from a compare_similar_mols result.

    Returns a list of ``{sim: float, pred: "A"|"B", section: str}``, ordered
    as they appear in the response. Under TDC convention, ``(B)`` = positive
    class (has-property), so "positive neighbors" → pred "B" and "negative
    neighbors" → pred "A".

    Works on either the raw tool-response text or the stringified JSON
    payload — we just need to find the ``positive neighbors:`` /
    ``negative neighbors:`` section headers.
    """
    if not body:
        return []

    pos_idx = body.find("positive neighbors:")
    neg_idx = body.find("negative neighbors:")
    if pos_idx < 0 and neg_idx < 0:
        return []

    neighbors: list[dict] = []

    def parse_section(text: str, section: str) -> None:
        pred = "B" if section == "positive" else "A"
        for m in _SIM_RE.finditer(text):
            try:
                sim = float(m.group(1))
            except ValueError:
                continue
            neighbors.append({"sim": sim, "pred": pred, "section": section})

    if pos_idx >= 0 and neg_idx >= 0:
        if pos_idx < neg_idx:
            parse_section(body[pos_idx:neg_idx], "positive")
            parse_section(body[neg_idx:], "negative")
        else:
            parse_section(body[neg_idx:pos_idx], "negative")
            parse_section(body[pos_idx:], "positive")
    elif pos_idx >= 0:
        parse_section(body[pos_idx:], "positive")
    else:
        parse_section(body[neg_idx:], "negative")

    return neighbors


def _votes_from_neighbors(neighbors: list[dict]) -> dict:
    """Derive multiple KNN-vote definitions + retrieval-quality signals.

    Returns a dict with:
      top_pos_sim, top_neg_sim     — max similarity per class (None if empty)
      knn_margin                   — sum(pos sims) − sum(neg sims)
      knn_top3_pos                 — count of B (positive) neighbors in top-3
      knn_vote_top1                — class of the single highest-sim neighbor
      knn_vote_top3                — top-3 majority, sim-sum tie-break
      knn_vote_weighted            — sign(margin): class with greater Σ sim
      n_neighbors
    """
    if not neighbors:
        return {
            "top_pos_sim": None, "top_neg_sim": None,
            "knn_margin": None, "knn_top3_pos": None,
            "knn_vote_top1": None, "knn_vote_top3": None,
            "knn_vote_weighted": None, "n_neighbors": 0,
        }

    ranked = sorted(neighbors, key=lambda n: -n["sim"])

    # Top-1 overall
    vote_top1 = ranked[0]["pred"]

    # Top-3 majority
    top3 = ranked[:3]
    pos_in_top3 = sum(1 for n in top3 if n["pred"] == "B")
    neg_in_top3 = len(top3) - pos_in_top3
    if pos_in_top3 > neg_in_top3:
        vote_top3: str | None = "B"
    elif neg_in_top3 > pos_in_top3:
        vote_top3 = "A"
    else:
        # unlikely with 3 neighbors but handle anyway — tie-break by sim sum
        pos_sim = sum(n["sim"] for n in top3 if n["pred"] == "B")
        neg_sim = sum(n["sim"] for n in top3 if n["pred"] == "A")
        vote_top3 = "B" if pos_sim > neg_sim else "A" if neg_sim > pos_sim else None

    # Similarity-weighted over ALL neighbors
    pos_sim_all = sum(n["sim"] for n in neighbors if n["pred"] == "B")
    neg_sim_all = sum(n["sim"] for n in neighbors if n["pred"] == "A")
    margin = pos_sim_all - neg_sim_all
    if margin > 0:
        vote_weighted: str | None = "B"
    elif margin < 0:
        vote_weighted = "A"
    else:
        vote_weighted = None

    pos_sims = [n["sim"] for n in neighbors if n["pred"] == "B"]
    neg_sims = [n["sim"] for n in neighbors if n["pred"] == "A"]

    return {
        "top_pos_sim": max(pos_sims) if pos_sims else None,
        "top_neg_sim": max(neg_sims) if neg_sims else None,
        "knn_margin": margin,
        "knn_top3_pos": pos_in_top3,
        "knn_vote_top1": vote_top1,
        "knn_vote_top3": vote_top3,
        "knn_vote_weighted": vote_weighted,
        "n_neighbors": len(neighbors),
    }


def _parse_gold_letter(label: str | None) -> str | None:
    if not label:
        return None
    m = _GOLD_RE.search(label)
    if not m:
        return None
    return m.group(1).upper()


def _parse_final_letter(final_segments: list[Segment]) -> str | None:
    """Pick the letter out of the last ``final`` segment if present."""
    # iterate in reverse; the last final chunk is authoritative
    for seg in reversed(final_segments):
        m = _ANS_RE.search(seg.text or "")
        if m:
            return m.group(1).upper()
    return None


def extract_sample_metrics(segs: list[Segment], gold_letter: str | None) -> dict:
    """Compute per-sample analysis metrics from parsed segments.

    Primary ``knn_vote`` is the similarity-weighted vote over ALL retrieved
    neighbors (3 positive + 3 negative per call). Alternative vote
    definitions (top-1, top-3 majority) are exposed alongside so you can
    compare how obey/retrieval-quality rates depend on how you define the
    vote.
    """
    tool_calls = [s for s in segs if s.kind == "tool_call"]
    tool_resps = [s for s in segs if s.kind == "tool_response"]
    finals = [s for s in segs if s.kind == "final"]

    tools_used = sorted({s.tool for s in tool_calls if s.tool})
    used_knn = any(s.tool == "compare_similar_mols" for s in tool_calls)

    # Pull neighbors from the FIRST compare_similar_mols response (the one
    # that's actually grounding the decision in almost every sample we've
    # seen — repeat calls are rare).
    neighbors: list[dict] = []
    for tr in tool_resps:
        if tr.tool != "compare_similar_mols":
            continue
        body: str = tr.text or ""
        if isinstance(tr.parsed, dict) and isinstance(tr.parsed.get("result"), str):
            body = tr.parsed["result"]
        ns = _parse_neighbors(body)
        if ns:
            neighbors = ns
            break

    votes = _votes_from_neighbors(neighbors)
    # Primary vote: similarity-weighted over all neighbors.
    knn_vote = votes["knn_vote_weighted"]

    final_answer = _parse_final_letter(finals)
    model_correct = (final_answer is not None and gold_letter is not None
                     and final_answer == gold_letter)
    knn_correct = (knn_vote is not None and gold_letter is not None
                   and knn_vote == gold_letter)
    obeys_knn = (final_answer is not None and knn_vote is not None
                 and final_answer == knn_vote)

    return {
        "n_tool_calls": len(tool_calls),
        "tools_used": tools_used,
        "used_knn": used_knn,
        "neighbors": neighbors,
        "n_neighbors": votes["n_neighbors"],
        "top_pos_sim": votes["top_pos_sim"],
        "top_neg_sim": votes["top_neg_sim"],
        "knn_margin": votes["knn_margin"],
        "knn_top3_pos": votes["knn_top3_pos"],
        "knn_vote": knn_vote,
        "knn_vote_top1": votes["knn_vote_top1"],
        "knn_vote_top3": votes["knn_vote_top3"],
        "knn_vote_weighted": votes["knn_vote_weighted"],
        "final_answer": final_answer,
        "gold": gold_letter,
        "model_correct": bool(model_correct),
        "knn_correct": bool(knn_correct) if knn_vote is not None and gold_letter is not None else None,
        "obeys_knn": bool(obeys_knn) if (final_answer is not None and knn_vote is not None) else None,
    }


# Bucket categories for the stacked-outcome chart.
#   obey_right     — model followed KNN, KNN was right       (contributes to correct)
#   disobey_right  — model overruled KNN, model was right    (contributes to correct)
#   obey_wrong     — model followed KNN, KNN was wrong       (wrong w/ KNN to blame)
#   disobey_wrong  — model overruled KNN, KNN was right      (wrong w/ model to blame)
#   no_knn         — model never retrieved                    (retrieval skipped)
#   no_answer      — couldn't parse final letter              (format failure)
#   na             — other (missing gold / ambiguous)
def _bucket(m: dict) -> str:
    if m.get("final_answer") is None:
        return "no_answer"
    if m.get("gold") is None:
        return "na"
    if not m.get("used_knn") or m.get("knn_vote") is None:
        return "no_knn"
    obeys = m.get("obeys_knn")
    knn_right = m.get("knn_correct")
    if obeys and knn_right:
        return "obey_right"
    if obeys and not knn_right:
        return "obey_wrong"
    if (obeys is False) and knn_right:
        return "disobey_wrong"   # model overruled a correct KNN → wrong
    if (obeys is False) and not knn_right:
        return "disobey_right"   # model overruled a wrong KNN → right
    return "na"


_BUCKETS = ["obey_right", "disobey_right", "obey_wrong", "disobey_wrong", "no_knn", "no_answer", "na"]


def _empty_bucket_counts() -> dict[str, int]:
    return {b: 0 for b in _BUCKETS}


_VOTE_DEFS = ("weighted", "top1", "top3")


def _extraction_for_template(segs: list[Segment], gold_letter: str | None) -> dict:
    """Lean view of what fed the extraction: the raw retrieved neighbors'
    labels + similarities (in the order the tool returned them), plus the
    final answer and the gold answer. Nothing derived — this is for
    eyeballing, not plotting. The analysis dashboard uses
    ``extract_sample_metrics`` directly for the derived vote / obey / bucket
    fields.
    """
    # Pull neighbors from the FIRST compare_similar_mols response — same
    # rule the analysis uses.
    neighbors: list[dict] = []
    for tr in segs:
        if tr.kind != "tool_response" or tr.tool != "compare_similar_mols":
            continue
        body: str = tr.text or ""
        if isinstance(tr.parsed, dict) and isinstance(tr.parsed.get("result"), str):
            body = tr.parsed["result"]
        ns = _parse_neighbors(body)
        if ns:
            neighbors = ns
            break

    finals = [s for s in segs if s.kind == "final"]
    final_answer = _parse_final_letter(finals)

    similarities = [round(n["sim"], 3) for n in neighbors]
    labels = [n["pred"] for n in neighbors]

    return {
        "final_answer": final_answer,
        "gold": gold_letter,
        "similarities": similarities,
        "labels": labels,
        "n_neighbors": len(neighbors),
    }


def _empty_step_agg() -> dict:
    per_vote = {v: {"n_with_vote": 0, "n_knn_correct": 0,
                    "n_obey_denom": 0, "n_obey": 0} for v in _VOTE_DEFS}
    return {
        "n_samples": 0,
        "n_with_final": 0,
        "n_with_gold": 0,
        "n_used_knn": 0,
        "n_knn_with_vote": 0,
        "n_knn_correct": 0,
        "n_model_correct": 0,
        "n_obey_knn": 0,              # of samples where both final & knn exist (primary vote)
        "n_obey_denom": 0,
        "sum_tool_calls": 0,
        "tool_counts": {},            # tool_name -> count of samples that called it
        "buckets": _empty_bucket_counts(),
        "pos_sims": [],               # max-sim per sample (one per sample)
        "neg_sims": [],
        "margins": [],                # knn_margin per sample
        "top3_pos_counts": [],        # how many of top-3 were positive, per sample
        "per_vote": per_vote,
    }


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _compute_run_analysis(ti: TraceIndex) -> dict:
    """Build per-step aggregates for one GRPO run. Expensive; cache on the
    TraceIndex, invalidated on rescan."""
    by_step: dict[int, dict] = {}
    by_step_ds: dict[tuple[int, str], dict] = {}

    per_sample_rows: list[dict] = []   # optional raw data (small)

    for g in ti.groups:
        gold = _parse_gold_letter(g.label)
        step = int(g.step)
        ds = g.datasource or "?"
        agg = by_step.setdefault(step, _empty_step_agg())
        agg_ds = by_step_ds.setdefault((step, ds), _empty_step_agg())
        for i, smp in enumerate(g.samples):
            resp = smp.get("response_text", "")
            segs = parse_response(resp)
            m = extract_sample_metrics(segs, gold)
            # per-sample row: drop the raw neighbors list (big) from the
            # canonical sample table; we keep derived fields only.
            sample_row = {
                "run_id": ti.run_id,
                "step": step,
                "datasource": ds,
                "group_id": g.group_id,
                "sample_idx": i,
                **{k: v for k, v in m.items() if k != "neighbors"},
                "reward": smp.get("reward"),
            }
            per_sample_rows.append(sample_row)
            for A in (agg, agg_ds):
                A["n_samples"] += 1
                if m["final_answer"] is not None:
                    A["n_with_final"] += 1
                if gold is not None:
                    A["n_with_gold"] += 1
                if m["used_knn"]:
                    A["n_used_knn"] += 1
                if m["knn_vote"] is not None:
                    A["n_knn_with_vote"] += 1
                if m["knn_correct"]:
                    A["n_knn_correct"] += 1
                if m["model_correct"]:
                    A["n_model_correct"] += 1
                if m["obeys_knn"] is not None:
                    A["n_obey_denom"] += 1
                    if m["obeys_knn"]:
                        A["n_obey_knn"] += 1
                A["sum_tool_calls"] += m["n_tool_calls"]
                for t in m["tools_used"]:
                    A["tool_counts"][t] = A["tool_counts"].get(t, 0) + 1
                A["buckets"][_bucket(m)] += 1
                if m["top_pos_sim"] is not None:
                    A["pos_sims"].append(m["top_pos_sim"])
                if m["top_neg_sim"] is not None:
                    A["neg_sims"].append(m["top_neg_sim"])
                if m["knn_margin"] is not None:
                    A["margins"].append(m["knn_margin"])
                if m["knn_top3_pos"] is not None:
                    A["top3_pos_counts"].append(m["knn_top3_pos"])
                # track per-vote-definition obey/correctness rates
                final = m["final_answer"]
                for vdef in _VOTE_DEFS:
                    v = m[f"knn_vote_{vdef}"]
                    if v is None:
                        continue
                    pv = A["per_vote"][vdef]
                    pv["n_with_vote"] += 1
                    if gold is not None and v == gold:
                        pv["n_knn_correct"] += 1
                    if final is not None:
                        pv["n_obey_denom"] += 1
                        if v == final:
                            pv["n_obey"] += 1

    # Finalize: build sorted step list + derived rates
    all_tools: set[str] = set()
    for A in by_step.values():
        all_tools.update(A["tool_counts"].keys())
    tool_list = sorted(all_tools)

    def _finalize(A: dict) -> dict:
        n = A["n_samples"]
        per_vote_rates = {}
        for vdef in _VOTE_DEFS:
            pv = A["per_vote"][vdef]
            per_vote_rates[vdef] = {
                "knn_correct_rate": (pv["n_knn_correct"] / pv["n_with_vote"])
                                     if pv["n_with_vote"] else None,
                "obey_rate": (pv["n_obey"] / pv["n_obey_denom"])
                              if pv["n_obey_denom"] else None,
                "n_with_vote": pv["n_with_vote"],
                "n_obey_denom": pv["n_obey_denom"],
            }
        out = {
            "n_samples": n,
            "n_model_correct": A["n_model_correct"],
            "accuracy": (A["n_model_correct"] / n) if n else None,
            "knn_call_rate": (A["n_used_knn"] / n) if n else None,
            "knn_correct_rate": (A["n_knn_correct"] / A["n_knn_with_vote"])
                                if A["n_knn_with_vote"] else None,
            "obey_rate": (A["n_obey_knn"] / A["n_obey_denom"])
                          if A["n_obey_denom"] else None,
            "mean_tool_calls": (A["sum_tool_calls"] / n) if n else None,
            "tool_rates": {t: (A["tool_counts"].get(t, 0) / n) if n else 0.0
                           for t in tool_list},
            "buckets": dict(A["buckets"]),
            "mean_top_pos_sim": _mean(A["pos_sims"]),
            "mean_top_neg_sim": _mean(A["neg_sims"]),
            "mean_knn_margin": _mean(A["margins"]),
            "mean_top3_pos_count": _mean(A["top3_pos_counts"]),
            "n_pos_sim": len(A["pos_sims"]),
            "n_neg_sim": len(A["neg_sims"]),
            "per_vote": per_vote_rates,
        }
        return out

    steps_sorted = sorted(by_step.keys())
    by_step_out = {s: _finalize(by_step[s]) for s in steps_sorted}

    ds_list = sorted({ds for (_, ds) in by_step_ds.keys()})
    by_step_ds_out: dict[str, dict[int, dict]] = {}
    for ds in ds_list:
        by_step_ds_out[ds] = {}
        for s in steps_sorted:
            a = by_step_ds.get((s, ds))
            if a is not None:
                by_step_ds_out[ds][s] = _finalize(a)

    return {
        "run_id": ti.run_id,
        "steps": steps_sorted,
        "tools": tool_list,
        "buckets": _BUCKETS,
        "vote_defs": list(_VOTE_DEFS),
        "by_step": by_step_out,
        "datasources": ds_list,
        "by_step_ds": by_step_ds_out,
        "samples": per_sample_rows,
        "n_total": len(per_sample_rows),
    }


def get_run_analysis(ti: TraceIndex) -> dict:
    """Compute-or-return cached analysis for a run."""
    cached = getattr(ti, "_analysis", None)
    if cached is not None:
        return cached
    res = _compute_run_analysis(ti)
    ti._analysis = res  # type: ignore[attr-defined]
    return res


def _invalidate_analysis_cache() -> None:
    for ti in REG.runs.values():
        if hasattr(ti, "_analysis"):
            try:
                delattr(ti, "_analysis")
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="GRPO / SFT rollout viewer")

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _render_segments_for_template(segs: list[Segment]) -> list[dict]:
    out = []
    for s in segs:
        entry = {"kind": s.kind, "text": s.text, "tool": s.tool}
        if s.parsed is not None:
            try:
                entry["parsed_pretty"] = json.dumps(s.parsed, indent=2)
            except Exception:
                entry["parsed_pretty"] = str(s.parsed)
        out.append(entry)
    return out


def _group_summary(g: Group) -> dict:
    # Use cached per-sample rewards to avoid triggering a lazy sample load.
    # _sample_rewards was captured when the group was first indexed; it has
    # one float per sample and is enough for all the list-page stats. This
    # must NOT touch g.samples — list pages iterate every group, so loading
    # full sample text here would blow up memory on the 512Mi Render box.
    rewards = g._sample_rewards or g.group_rewards or []
    return {
        "uid": g.uid,
        "run_id": g.run_id,
        "group_id": g.group_id,
        "step": g.step,
        "type": g.group_type,
        "datasource": g.datasource or "?",
        "label": g.label,
        "n_samples": g.n_samples,
        "mean_reward": round(statistics.fmean(rewards), 3) if rewards else 0.0,
        "min_reward": round(min(rewards), 3) if rewards else 0.0,
        "max_reward": round(max(rewards), 3) if rewards else 0.0,
        "n_correct": sum(1 for r in rewards if r > 0.5),
        "source_file": g.source_file,
    }


def _nav_context() -> dict:
    """Shared nav/context passed to every template."""
    return {
        "runs": [
            {
                "run_id": rid,
                "run_dir": str(REG.runs[rid].run_dir),
                "n_groups": len(REG.runs[rid].groups),
            }
            for rid in REG.run_order
        ],
        "sft_archives": [
            {
                "archive_id": aid,
                "root": str(arc.root),
                "n_steps": len(arc.steps),
                "n_tasks": len(arc.all_tasks()),
            }
            for aid, arc in REG.sft.items()
        ],
        "default_run_id": REG.default_run_id,
        "n_annotations": len(ANN_STORE.items),
        "has_analysis": bool(REG.runs),
    }


# ---- Sample-level helpers for SFT samples ---------------------------------


def _parse_sample(prompt: str, response: str, reward: float,
                  score: float | None, idx: int | None = None,
                  extra: dict | None = None) -> dict:
    segs = parse_response(response)
    extra = extra or {}
    gold_letter = _parse_gold_letter(extra.get("label"))
    return {
        "idx": idx,
        "reward": reward,
        "score": score,
        "n_chars": len(response or ""),
        "segments": _render_segments_for_template(segs),
        "extraction": _extraction_for_template(segs, gold_letter),
        "raw": response,
        "prompt": prompt,
        "extra": extra,
    }


# ===========================================================================
# Routes: landing / health
# ===========================================================================


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    """Landing page when there's more than one run or any SFT archive;
    otherwise redirect to the single run's group list for back-compat."""
    has_multiple_surfaces = (len(REG.runs) > 1) or bool(REG.sft)
    if not has_multiple_surfaces and REG.default_run_id:
        return RedirectResponse(url=f"/runs/{REG.default_run_id}/")
    return templates.TemplateResponse(
        request,
        "landing.html",
        {**_nav_context()},
    )


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "n_runs": len(REG.runs),
        "n_sft": len(REG.sft),
        "total_groups": sum(len(ti.groups) for ti in REG.runs.values()),
    }


@app.get("/api/reindex")
def api_reindex():
    REG.rescan()
    ANN_STORE.rescan()
    _invalidate_analysis_cache()
    _invalidate_flip_index()
    return {
        "n_runs": len(REG.runs),
        "n_sft": len(REG.sft),
        "n_annotations": len(ANN_STORE.items),
        "runs": [rid for rid in REG.run_order],
        "sft": list(REG.sft.keys()),
    }


# ===========================================================================
# Routes: GRPO rollout lists + details
# ===========================================================================


@app.get("/runs/{run_id}/", response_class=HTMLResponse)
def run_index(
    request: Request,
    run_id: str,
    step: str | None = None,
    type: str | None = None,   # noqa: A002
    datasource: str | None = None,
    sort: str = "step",
):
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404, f"unknown run: {run_id}")

    step_i: int | None = int(step) if step not in (None, "", "None") else None
    type = type or None
    datasource = datasource or None

    groups = ti.filter(step=step_i, group_type=type, datasource=datasource)
    summaries = [_group_summary(g) for g in groups]

    if sort == "mean_reward":
        summaries.sort(key=lambda s: s["mean_reward"])
    elif sort == "mean_reward_desc":
        summaries.sort(key=lambda s: s["mean_reward"], reverse=True)
    elif sort == "spread":
        summaries.sort(key=lambda s: s["max_reward"] - s["min_reward"], reverse=True)
    else:
        summaries.sort(key=lambda s: (s["step"], s["type"], s["datasource"] or ""))

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            **_nav_context(),
            "active_run_id": run_id,
            "run_dir": str(ti.run_dir),
            "summaries": summaries,
            "total_all": len(ti.groups),
            "total_shown": len(summaries),
            "steps": ti.all_steps(),
            "types": ti.all_types(),
            "datasources": ti.all_datasources(),
            "current": {"step": step_i, "type": type, "datasource": datasource, "sort": sort},
        },
    )


@app.get("/runs/{run_id}/group/{group_id}", response_class=HTMLResponse)
def run_group_detail(request: Request, run_id: str, group_id: str):
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404, f"unknown run: {run_id}")
    g = ti.get(group_id)
    if g is None:
        raise HTTPException(404, f"unknown group_id: {group_id}")

    gold_letter = _parse_gold_letter(g.label)
    samples = []
    for i, s in enumerate(g.samples):
        reward = s.get("reward", 0.0)
        response_text = s.get("response_text", "")
        segs = parse_response(response_text)
        samples.append(
            {
                "idx": i,
                "reward": reward,
                "score": s.get("score"),
                "n_chars": len(response_text),
                "segments": _render_segments_for_template(segs),
                "extraction": _extraction_for_template(segs, gold_letter),
                "raw": response_text,
            }
        )
    samples.sort(key=lambda s: -s["reward"])

    rewards = [s["reward"] for s in samples]
    stats = {
        "mean": round(statistics.fmean(rewards), 3) if rewards else 0.0,
        "median": round(statistics.median(rewards), 3) if rewards else 0.0,
        "min": round(min(rewards), 3) if rewards else 0.0,
        "max": round(max(rewards), 3) if rewards else 0.0,
        "stdev": round(statistics.pstdev(rewards), 3) if rewards else 0.0,
        "n_correct": sum(1 for r in rewards if r > 0.5),
    }

    return templates.TemplateResponse(
        request,
        "group.html",
        {
            **_nav_context(),
            "active_run_id": run_id,
            "group": g,
            "summary": _group_summary(g),
            "samples": samples,
            "stats": stats,
            "back_url": f"/runs/{run_id}/",
        },
    )


# Back-compat: /group/{group_id} — resolve against default run.
@app.get("/group/{group_id}")
def legacy_group(group_id: str):
    if REG.default_run_id is None:
        raise HTTPException(404, "no runs indexed")
    return RedirectResponse(url=f"/runs/{REG.default_run_id}/group/{group_id}")


@app.get("/api/runs")
def api_runs():
    return [
        {
            "run_id": rid,
            "run_dir": str(REG.runs[rid].run_dir),
            "n_groups": len(REG.runs[rid].groups),
            "steps": REG.runs[rid].all_steps(),
            "types": REG.runs[rid].all_types(),
            "datasources": REG.runs[rid].all_datasources(),
        }
        for rid in REG.run_order
    ]


@app.get("/api/runs/{run_id}/groups")
def api_run_groups(run_id: str):
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404)
    return [_group_summary(g) for g in ti.groups]


@app.get("/api/runs/{run_id}/group/{group_id}")
def api_run_group(run_id: str, group_id: str):
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404)
    g = ti.get(group_id)
    if g is None:
        raise HTTPException(404)
    return JSONResponse(
        {
            "summary": _group_summary(g),
            "prompt": g.prompt,
            "samples": [
                {
                    "idx": i,
                    "reward": s.get("reward"),
                    "score": s.get("score"),
                    "response_text": s.get("response_text", ""),
                    "segments": [
                        {
                            "kind": x.kind,
                            "tool": x.tool,
                            "text": x.text,
                            "parsed": x.parsed,
                        }
                        for x in parse_response(s.get("response_text", ""))
                    ],
                }
                for i, s in enumerate(g.samples)
            ],
        }
    )


# ===========================================================================
# Routes: SFT compare archives
# ===========================================================================


@app.get("/sft/{archive_id}/", response_class=HTMLResponse)
def sft_archive(
    request: Request,
    archive_id: str,
    step: str | None = None,
):
    arc = REG.sft.get(archive_id)
    if arc is None:
        raise HTTPException(404, f"unknown sft archive: {archive_id}")

    step_i: int | None = int(step) if step not in (None, "", "None") else None
    tasks = arc.all_tasks()

    # Build a matrix of step x task with small-sample stats loaded lazily
    rows = []
    for s in arc.steps:
        if step_i is not None and s != step_i:
            continue
        row = {"step": s, "cells": []}
        task_map = arc.summaries.get(s, {})
        for t in tasks:
            info = task_map.get(t)
            row["cells"].append({
                "task": t,
                "present": bool(info),
                "num_results": info.get("num_results") if info else None,
                "num_sampled": info.get("num_sampled") if info else None,
            })
        rows.append(row)

    return templates.TemplateResponse(
        request,
        "sft_archive.html",
        {
            **_nav_context(),
            "active_sft_id": archive_id,
            "archive": {
                "archive_id": archive_id,
                "root": str(arc.root),
                "n_steps": len(arc.steps),
                "n_tasks": len(tasks),
            },
            "steps": arc.steps,
            "tasks": tasks,
            "rows": rows,
            "current_step": step_i,
        },
    )


@app.get("/sft/{archive_id}/step/{step}/{task}", response_class=HTMLResponse)
def sft_task_detail(request: Request, archive_id: str, step: int, task: str):
    arc = REG.sft.get(archive_id)
    if arc is None:
        raise HTTPException(404, f"unknown sft archive: {archive_id}")
    doc = arc.load_task(step, task)
    if doc is None:
        raise HTTPException(404, f"{task} not found at step {step}")

    # top-level samples (sorted sampled subset, typically ~every 50th)
    samples_in = doc.get("samples", [])
    parsed_samples = []
    for s in samples_in:
        parsed_samples.append(_parse_sample(
            prompt=s.get("prompt", ""),
            response=s.get("response", ""),
            reward=s.get("reward", 0.0),
            score=s.get("score"),
            idx=s.get("index"),
            extra={"label": s.get("label")},
        ))

    # best_correct + worst_wrong singletons — dress them up as "special" samples
    extras: list[dict] = []
    for kind, key in (("best_correct", "best_correct"), ("worst_wrong", "worst_wrong")):
        item = doc.get(key)
        if not item:
            continue
        extras.append({
            **_parse_sample(
                prompt=item.get("prompt", ""),
                response=item.get("response", ""),
                reward=item.get("reward", 0.0),
                score=item.get("score"),
                idx=None,
                extra={"label": item.get("label"), "highlight": kind},
            ),
            "highlight": kind,
        })

    rewards = [s["reward"] for s in parsed_samples if s["reward"] is not None]
    stats = {
        "mean": round(statistics.fmean(rewards), 3) if rewards else 0.0,
        "median": round(statistics.median(rewards), 3) if rewards else 0.0,
        "min": round(min(rewards), 3) if rewards else 0.0,
        "max": round(max(rewards), 3) if rewards else 0.0,
        "stdev": round(statistics.pstdev(rewards), 3) if rewards else 0.0,
        "n_correct": sum(1 for r in rewards if r > 0.5),
        "n_samples": len(parsed_samples),
        "num_results": doc.get("num_results"),
        "sampled_every": doc.get("sampled_every"),
    }

    # navigation: prev/next step with this task present
    steps_with_task = [s for s in arc.steps if task in arc.summaries.get(s, {})]
    try:
        cur_idx = steps_with_task.index(step)
    except ValueError:
        cur_idx = -1
    prev_step = steps_with_task[cur_idx - 1] if cur_idx > 0 else None
    next_step = steps_with_task[cur_idx + 1] if 0 <= cur_idx < len(steps_with_task) - 1 else None

    return templates.TemplateResponse(
        request,
        "sft_task.html",
        {
            **_nav_context(),
            "active_sft_id": archive_id,
            "archive_id": archive_id,
            "task": task,
            "step": step,
            "steps_with_task": steps_with_task,
            "prev_step": prev_step,
            "next_step": next_step,
            "samples": parsed_samples,
            "extras": extras,
            "stats": stats,
            "doc_header": {
                "num_results": doc.get("num_results"),
                "sampled_every": doc.get("sampled_every"),
            },
            "back_url": f"/sft/{archive_id}/",
        },
    )


@app.get("/api/sft")
def api_sft():
    return [
        {
            "archive_id": aid,
            "root": str(arc.root),
            "steps": arc.steps,
            "tasks": arc.all_tasks(),
        }
        for aid, arc in REG.sft.items()
    ]


@app.get("/api/sft/{archive_id}/step/{step}/{task}")
def api_sft_task(archive_id: str, step: int, task: str):
    arc = REG.sft.get(archive_id)
    if arc is None:
        raise HTTPException(404)
    doc = arc.load_task(step, task)
    if doc is None:
        raise HTTPException(404)
    return JSONResponse(doc)


# ===========================================================================
# Routes: Claude annotations
# ===========================================================================


@app.get("/annotations/", response_class=HTMLResponse)
def annotations_list(request: Request):
    # lazy rescan so new files drop in without restart
    ANN_STORE.rescan()
    anns = ANN_STORE.list()
    rows = []
    for a in anns:
        rows.append({
            "id": a.id,
            "title": a.title,
            "verdict": a.verdict,
            "summary": a.summary,
            "source_label": a.source_label,
            "source_url": a.source_url,
            "tags": a.tags,
            "created_at": a.created_at,
            "n_notes": len(a.segment_notes),
        })
    return templates.TemplateResponse(
        request,
        "annotations_list.html",
        {
            **_nav_context(),
            "active_tab": "annotations",
            "rows": rows,
            "empty": not rows,
            "annotations_dir": str(ANNOTATIONS_DIR),
        },
    )


@app.get("/annotations/{ann_id}", response_class=HTMLResponse)
def annotation_detail(request: Request, ann_id: str):
    ANN_STORE.rescan()
    ann = ANN_STORE.get(ann_id)
    if ann is None:
        raise HTTPException(404, f"unknown annotation: {ann_id}")

    target = _resolve_annotated_sample(ann)
    segments: list[dict] = []
    notes_by_idx: dict[int, dict] = {}
    for n in ann.segment_notes:
        idx = n.get("segment_idx")
        if isinstance(idx, int):
            notes_by_idx[idx] = n

    if target is not None:
        raw_segs = parse_response(target.get("response", ""))
        for i, s in enumerate(raw_segs):
            entry = {"kind": s.kind, "text": s.text, "tool": s.tool}
            if s.parsed is not None:
                try:
                    entry["parsed_pretty"] = json.dumps(s.parsed, indent=2)
                except Exception:
                    entry["parsed_pretty"] = str(s.parsed)
            entry["note"] = notes_by_idx.get(i)
            segments.append(entry)

    # orphan notes: ones whose segment_idx is out of range or missing, plus
    # non-integer "positional" notes like {"segment_idx": "conclusion"}
    seen_idxs = {n.get("segment_idx") for n in ann.segment_notes
                 if isinstance(n.get("segment_idx"), int) and n.get("segment_idx") < len(segments)}
    orphan_notes = [n for n in ann.segment_notes
                    if n.get("segment_idx") not in seen_idxs
                    or not isinstance(n.get("segment_idx"), int)]

    return templates.TemplateResponse(
        request,
        "annotation_detail.html",
        {
            **_nav_context(),
            "active_tab": "annotations",
            "ann": {
                "id": ann.id,
                "title": ann.title,
                "summary": ann.summary,
                "verdict": ann.verdict,
                "conclusion": ann.conclusion,
                "tags": ann.tags,
                "created_at": ann.created_at,
                "source_label": ann.source_label,
                "source_url": ann.source_url,
                "source_kind": (ann.source or {}).get("kind"),
                "path": str(ann.path),
            },
            "target": target,
            "segments": segments,
            "orphan_notes": orphan_notes,
            "found": target is not None,
        },
    )


@app.get("/api/annotations")
def api_annotations():
    ANN_STORE.rescan()
    return [
        {
            "id": a.id,
            "title": a.title,
            "verdict": a.verdict,
            "source": a.source,
            "tags": a.tags,
            "n_notes": len(a.segment_notes),
            "created_at": a.created_at,
        }
        for a in ANN_STORE.list()
    ]


@app.get("/api/annotations/{ann_id}")
def api_annotation(ann_id: str):
    ANN_STORE.rescan()
    ann = ANN_STORE.get(ann_id)
    if ann is None:
        raise HTTPException(404)
    return JSONResponse(json.load(open(ann.path)))


# ===========================================================================
# Routes: Failure analysis (KNN obedience + tool-call tracking over time)
# ===========================================================================


def _analyses_for_runs(run_ids: list[str] | None = None) -> list[dict]:
    rids = run_ids or list(REG.run_order)
    out = []
    for rid in rids:
        ti = REG.runs.get(rid)
        if ti is None:
            continue
        out.append(get_run_analysis(ti))
    return out


@app.get("/analysis/", response_class=HTMLResponse)
def analysis_page(request: Request, run: str | None = None):
    """Failure-analysis dashboard: KNN obedience, retrieval quality, tool
    usage, and model accuracy over training step. Multi-run if none
    selected; single run if `?run=...` is provided."""
    if not REG.runs:
        return templates.TemplateResponse(
            request,
            "analysis.html",
            {
                **_nav_context(),
                "active_tab": "analysis",
                "analyses": [],
                "selected_run": None,
                "empty": True,
            },
        )

    if run and run in REG.runs:
        analyses = [get_run_analysis(REG.runs[run])]
        selected = run
    else:
        analyses = _analyses_for_runs()
        selected = None

    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            **_nav_context(),
            "active_tab": "analysis",
            "analyses": analyses,
            "selected_run": selected,
            "empty": False,
        },
    )


@app.get("/api/analysis")
def api_analysis(run: str | None = None, include_samples: bool = False):
    analyses = _analyses_for_runs([run] if run else None)
    if not include_samples:
        for a in analyses:
            a = {k: v for k, v in a.items() if k != "samples"}
    return JSONResponse(
        [
            {k: v for k, v in a.items() if include_samples or k != "samples"}
            for a in analyses
        ]
    )


# ===========================================================================
# Routes: Intermediate decisions (gpt-5-nano)
# ===========================================================================
#
# The public/Render deploy is READ-ONLY — we serve cache hits only, and
# do not import any LLM-calling code. Local dev can opt in to live
# on-demand extraction by setting VIEWER_ENABLE_LIVE_EXTRACTION=1, which
# attempts to import the full extractor module (kept out of the public
# repo via .gitignore).

from intermediate_decisions_read import cache_get as _id_cache_get  # noqa: E402

_LIVE_EXTRACTION_ENABLED = os.environ.get("VIEWER_ENABLE_LIVE_EXTRACTION") == "1"
_id_extract_async = None
_id_build_windows = None
if _LIVE_EXTRACTION_ENABLED:
    try:
        from intermediate_decisions import (  # noqa: E402
            extract_for_sample_async as _id_extract_async,
            build_windows as _id_build_windows,
        )
    except Exception as _exc:  # pragma: no cover - import guard
        import logging
        logging.getLogger(__name__).warning(
            "VIEWER_ENABLE_LIVE_EXTRACTION=1 but intermediate_decisions "
            "module not importable (%s) — POST routes will 503.", _exc,
        )
        _id_extract_async = None
        _id_build_windows = None


async def _intermediate_decisions_for(
    response_text: str,
    *,
    task: str | None,
    source_kind: str,
    source_id: str,
    sample_idx: int,
    gold_letter: str | None,
) -> dict:
    if _id_extract_async is None:
        raise HTTPException(
            503,
            "Live extraction is disabled on this deploy. "
            "Set VIEWER_ENABLE_LIVE_EXTRACTION=1 and install the extractor module.",
        )
    segs = parse_response(response_text)
    final_letter = _parse_final_letter([s for s in segs if s.kind == "final"])
    return await _id_extract_async(
        segs,
        response_text,
        task=task,
        source_kind=source_kind,
        source_id=source_id,
        sample_idx=sample_idx,
        final_answer=final_letter,
    )


@app.get("/api/intermediate_decisions/grpo/{run_id}/{group_id}/{sample_idx}")
async def api_id_grpo_get(run_id: str, group_id: str, sample_idx: int):
    """GET — returns cached result only (no API call)."""
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404, f"unknown run: {run_id}")
    g = ti.get(group_id)
    if g is None:
        raise HTTPException(404, f"unknown group: {group_id}")
    if not (0 <= sample_idx < len(g.samples)):
        raise HTTPException(404, "sample_idx out of range")
    smp = g.samples[sample_idx]
    response_text = smp.get("response_text", "")
    cached = _id_cache_get(response_text)
    if cached is None:
        return JSONResponse({
            "cached": False,
            "live_extraction_available": _id_extract_async is not None,
        })
    return JSONResponse({"cached": True, **cached})


@app.post("/api/intermediate_decisions/grpo/{run_id}/{group_id}/{sample_idx}")
async def api_id_grpo_post(run_id: str, group_id: str, sample_idx: int):
    """POST — runs extraction (uses cache if available, else calls gpt-5-nano).

    503s when VIEWER_ENABLE_LIVE_EXTRACTION is not set.
    """
    if _id_extract_async is None:
        raise HTTPException(503, "Live extraction disabled on this deploy.")
    ti = REG.runs.get(run_id)
    if ti is None:
        raise HTTPException(404, f"unknown run: {run_id}")
    g = ti.get(group_id)
    if g is None:
        raise HTTPException(404, f"unknown group: {group_id}")
    if not (0 <= sample_idx < len(g.samples)):
        raise HTTPException(404, "sample_idx out of range")
    smp = g.samples[sample_idx]
    payload = await _intermediate_decisions_for(
        smp.get("response_text", ""),
        task=g.datasource,
        source_kind="grpo",
        source_id=f"{run_id}::{group_id}",
        sample_idx=sample_idx,
        gold_letter=_parse_gold_letter(g.label),
    )
    return JSONResponse(payload)


@app.get("/api/intermediate_decisions/sft/{archive_id}/step/{step}/{task}/{sample_ref}")
async def api_id_sft_get(archive_id: str, step: int, task: str, sample_ref: str):
    arc = REG.sft.get(archive_id)
    if arc is None:
        raise HTTPException(404)
    doc = arc.load_task(step, task)
    if doc is None:
        raise HTTPException(404)
    response_text, _ = _sft_sample_lookup(doc, sample_ref)
    if response_text is None:
        raise HTTPException(404)
    cached = _id_cache_get(response_text)
    if cached is None:
        return JSONResponse({
            "cached": False,
            "live_extraction_available": _id_extract_async is not None,
        })
    return JSONResponse({"cached": True, **cached})


@app.post("/api/intermediate_decisions/sft/{archive_id}/step/{step}/{task}/{sample_ref}")
async def api_id_sft_post(archive_id: str, step: int, task: str, sample_ref: str):
    if _id_extract_async is None:
        raise HTTPException(503, "Live extraction disabled on this deploy.")
    arc = REG.sft.get(archive_id)
    if arc is None:
        raise HTTPException(404)
    doc = arc.load_task(step, task)
    if doc is None:
        raise HTTPException(404)
    response_text, sample_dict = _sft_sample_lookup(doc, sample_ref)
    if response_text is None:
        raise HTTPException(404)
    payload = await _intermediate_decisions_for(
        response_text,
        task=task,
        source_kind="sft",
        source_id=f"{archive_id}::step{step}::{task}::{sample_ref}",
        sample_idx=-1,
        gold_letter=_parse_gold_letter(sample_dict.get("label") if sample_dict else None),
    )
    return JSONResponse(payload)


def _sft_sample_lookup(doc: dict, sample_ref: str) -> tuple[str | None, dict | None]:
    """sample_ref is either an integer index into doc['samples'] or
    'best_correct' / 'worst_wrong'."""
    if sample_ref in ("best_correct", "worst_wrong"):
        item = doc.get(sample_ref)
        if not item:
            return None, None
        return item.get("response", ""), item
    try:
        idx = int(sample_ref)
    except ValueError:
        return None, None
    pool = doc.get("samples") or []
    # 'index' field carries the original sample id; fall back to positional
    for s in pool:
        if s.get("index") == idx:
            return s.get("response", ""), s
    if 0 <= idx < len(pool):
        return pool[idx].get("response", ""), pool[idx]
    return None, None


# ===========================================================================
# Routes: Answer-flip pattern analysis
# ===========================================================================
#
# For each sample we look up its cached intermediate-decision sequence and
# build a "flip pattern" — the run-length-compressed sequence of A/B
# decisions across tool-call windows (so AABBA → A→B→A). Aggregating these
# across a filter (source/run/step/task) lets us see how often the model
# changes its mind from one tool call to the next.
#
# Index is built lazily, in-memory, from the on-disk decision cache (cheap:
# one file read per sample, sha256 keyed). Invalidated when the run
# registry rescans.

_FLIP_INDEX: list[dict] | None = None


def _flip_pattern_from_decisions(
    decisions: list[dict],
) -> tuple[str | None, int, int, dict[str, int]]:
    """Return (compressed pattern, n_decisions, n_flips, transitions) for one sample.

    ``transitions`` counts adjacent-decision pairs in the raw (uncompressed)
    sequence: {"AA": int, "AB": int, "BA": int, "BB": int}. AA/BB are stays
    (non-reversals), AB/BA are flips. Summing transitions over the sample
    always gives n_decisions - 1 (or 0 if only one decision).

    Pattern is run-length-compressed (consecutive duplicate decisions
    merged) so the table reads as a flip trajectory. Decisions with
    decision == None are skipped (the extractor couldn't pin down a
    letter for that window).
    """
    empty_trans = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
    if not decisions:
        return None, 0, 0, empty_trans
    # defensive sort by after_tool_idx in case the cache ever reorders
    ds = sorted(decisions, key=lambda d: d.get("after_tool_idx") or 0)
    letters = [d.get("decision") for d in ds if d.get("decision") in ("A", "B")]
    if not letters:
        return None, 0, 0, empty_trans
    compressed: list[str] = []
    for letter in letters:
        if not compressed or compressed[-1] != letter:
            compressed.append(letter)
    pattern = "→".join(compressed)
    n_flips = max(0, len(compressed) - 1)
    transitions = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
    for a, b in zip(letters, letters[1:]):
        transitions[a + b] += 1
    return pattern, len(letters), n_flips, transitions


def _build_flip_index() -> list[dict]:
    """Walk every indexed sample, look up its cached decisions, build
    a flat record list. Records carry the filter-able fields and the
    derived pattern so /api/analysis/flip_patterns can aggregate cheaply.

    Memory discipline: after reading a GRPO group's samples we call
    ``g.unload_samples()`` so the builder never holds more than one
    group's response_text at a time. Essential on the 512Mi Render
    instance where eager-loading all samples once cost ~240MB.
    """
    out: list[dict] = []
    empty_trans = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}

    # GRPO samples
    for run_id in REG.run_order:
        ti = REG.runs.get(run_id)
        if ti is None:
            continue
        for g in ti.groups:
            try:
                for idx, smp in enumerate(g.samples):
                    response_text = smp.get("response_text", "")
                    if not response_text:
                        continue
                    cached = _id_cache_get(response_text)
                    if cached is None:
                        out.append({
                            "source": "grpo",
                            "run_id": run_id,
                            "archive_id": None,
                            "step": g.step,
                            "task": g.datasource,
                            "group_id": g.group_id,
                            "group_type": g.group_type,
                            "sample_idx": idx,
                            "pattern": None,
                            "n_decisions": 0,
                            "n_flips": 0,
                            "transitions": dict(empty_trans),
                            "final_answer": None,
                            "cached": False,
                        })
                        continue
                    decisions = cached.get("decisions") or []
                    pattern, n_dec, n_flips, trans = _flip_pattern_from_decisions(decisions)
                    out.append({
                        "source": "grpo",
                        "run_id": run_id,
                        "archive_id": None,
                        "step": g.step,
                        "task": g.datasource,
                        "group_id": g.group_id,
                        "group_type": g.group_type,
                        "sample_idx": idx,
                        "pattern": pattern,
                        "n_decisions": n_dec,
                        "n_flips": n_flips,
                        "transitions": trans,
                        "final_answer": cached.get("final_answer"),
                        "cached": True,
                    })
            finally:
                # drop the loaded response_text payload before moving on
                g.unload_samples()

    # SFT samples — load each step/task doc lazily; samples include
    # named items (best_correct, worst_wrong) and the per-sample pool.
    for archive_id, arc in REG.sft.items():
        for step in arc.steps:
            for task in arc.tasks_at(step):
                doc = arc.load_task(step, task)
                if doc is None:
                    continue
                items: list[tuple[str, dict]] = []
                for ref in ("best_correct", "worst_wrong"):
                    s = doc.get(ref)
                    if isinstance(s, dict) and s.get("response"):
                        items.append((ref, s))
                for s in (doc.get("samples") or []):
                    if not isinstance(s, dict):
                        continue
                    if not s.get("response"):
                        continue
                    sid = s.get("index")
                    items.append((str(sid) if sid is not None else "?", s))
                for ref, s in items:
                    response_text = s.get("response", "")
                    if not response_text:
                        continue
                    cached = _id_cache_get(response_text)
                    if cached is None:
                        out.append({
                            "source": "sft",
                            "run_id": None,
                            "archive_id": archive_id,
                            "step": step,
                            "task": task,
                            "group_id": None,
                            "group_type": None,
                            "sample_idx": ref,
                            "pattern": None,
                            "n_decisions": 0,
                            "n_flips": 0,
                            "transitions": dict(empty_trans),
                            "final_answer": None,
                            "cached": False,
                        })
                        continue
                    decisions = cached.get("decisions") or []
                    pattern, n_dec, n_flips, trans = _flip_pattern_from_decisions(decisions)
                    out.append({
                        "source": "sft",
                        "run_id": None,
                        "archive_id": archive_id,
                        "step": step,
                        "task": task,
                        "group_id": None,
                        "group_type": None,
                        "sample_idx": ref,
                        "pattern": pattern,
                        "n_decisions": n_dec,
                        "n_flips": n_flips,
                        "transitions": trans,
                        "final_answer": cached.get("final_answer"),
                        "cached": True,
                    })
    return out


# On-disk cache location for precomputed flip index. Populated offline
# and shipped with the data bundle; otherwise the in-process builder
# will compute it on first request (slow + memory heavy).
_FLIP_INDEX_FILE = Path(os.environ.get("FLIP_INDEX_FILE", "/data/flip_index.json"))


def _load_flip_index_from_disk() -> list[dict] | None:
    """Return the precomputed flip index if present on disk, else None."""
    try:
        if _FLIP_INDEX_FILE.exists():
            with open(_FLIP_INDEX_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                # upgrade older records (pre-transitions) transparently
                empty = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
                for r in data:
                    if "transitions" not in r or not isinstance(r.get("transitions"), dict):
                        r["transitions"] = dict(empty)
                return data
    except Exception as exc:
        print(f"[flip_index] failed to load {_FLIP_INDEX_FILE}: {exc}")
    return None


def _get_flip_index() -> list[dict]:
    global _FLIP_INDEX
    if _FLIP_INDEX is None:
        disk = _load_flip_index_from_disk()
        if disk is not None:
            print(f"[flip_index] loaded {len(disk)} records from {_FLIP_INDEX_FILE}")
            _FLIP_INDEX = disk
        else:
            print(f"[flip_index] building in-process (no cache at {_FLIP_INDEX_FILE})")
            _FLIP_INDEX = _build_flip_index()
    return _FLIP_INDEX


def _invalidate_flip_index() -> None:
    global _FLIP_INDEX
    _FLIP_INDEX = None


@app.get("/api/analysis/flip_patterns")
def api_flip_patterns(
    source: str | None = None,
    run: str | None = None,
    archive: str | None = None,
    step: int | None = None,
    task: str | None = None,
    group_type: str | None = None,
):
    """Aggregate answer-flip pattern counts under a filter.

    Returns:
        {
          "filters_applied": {...},
          "total_traces": int,             # samples that match filter
          "n_with_decisions": int,         # subset that had >= 1 decision
          "n_without_decisions": int,
          "pattern_counts": [{"pattern": str, "count": int, "share": float}, ...],
          "flip_count_distribution": {"0": int, "1": int, ...},
          "available_filters": {
            "sources": [...], "runs": [...], "archives": [...],
            "steps": [...], "tasks": [...], "group_types": [...],
          },
        }
    """
    idx = _get_flip_index()

    def matches(rec: dict) -> bool:
        if source and source != "all" and rec["source"] != source:
            return False
        if run and rec.get("run_id") != run:
            return False
        if archive and rec.get("archive_id") != archive:
            return False
        if step is not None and rec.get("step") != step:
            return False
        if task and rec.get("task") != task:
            return False
        if group_type and rec.get("group_type") != group_type:
            return False
        return True

    filtered = [r for r in idx if matches(r)]
    n_total = len(filtered)
    n_with = sum(1 for r in filtered if r["pattern"])
    n_without = n_total - n_with

    pattern_counts: dict[str, int] = {}
    flip_dist: dict[int, int] = {}
    transition_counts = {"AA": 0, "AB": 0, "BA": 0, "BB": 0}
    for r in filtered:
        if r["pattern"]:
            pattern_counts[r["pattern"]] = pattern_counts.get(r["pattern"], 0) + 1
            flip_dist[r["n_flips"]] = flip_dist.get(r["n_flips"], 0) + 1
        tr = r.get("transitions") or {}
        for k in transition_counts:
            v = tr.get(k)
            if isinstance(v, int):
                transition_counts[k] += v

    # Transition summary: adjacent-decision pairs across the filter.
    # stays = AA + BB (model sticks), flips = AB + BA (model reverses).
    trans_total = sum(transition_counts.values())
    stays = transition_counts["AA"] + transition_counts["BB"]
    flips = transition_counts["AB"] + transition_counts["BA"]
    transition_summary = {
        "total_pairs": trans_total,
        "stays": stays,
        "flips": flips,
        "stay_share": (stays / trans_total) if trans_total else 0.0,
        "flip_share": (flips / trans_total) if trans_total else 0.0,
    }

    pattern_rows = sorted(
        (
            {
                "pattern": p,
                "count": c,
                "share": (c / n_with) if n_with else 0.0,
            }
            for p, c in pattern_counts.items()
        ),
        key=lambda d: (-d["count"], d["pattern"]),
    )

    # available filter values — derived from the full index so the UI
    # can show every choice even when the current filter zeros things out
    sources = sorted({r["source"] for r in idx})
    runs = sorted({r["run_id"] for r in idx if r.get("run_id")})
    archives = sorted({r["archive_id"] for r in idx if r.get("archive_id")})
    steps = sorted({r["step"] for r in idx if r.get("step") is not None})
    tasks = sorted({r["task"] for r in idx if r.get("task")})
    group_types = sorted({r["group_type"] for r in idx if r.get("group_type")})

    return JSONResponse({
        "filters_applied": {
            "source": source or "all",
            "run": run,
            "archive": archive,
            "step": step,
            "task": task,
            "group_type": group_type,
        },
        "total_traces": n_total,
        "n_with_decisions": n_with,
        "n_without_decisions": n_without,
        "pattern_counts": pattern_rows,
        "flip_count_distribution": {str(k): flip_dist[k] for k in sorted(flip_dist)},
        "transition_counts": transition_counts,
        "transition_summary": transition_summary,
        "available_filters": {
            "sources": sources,
            "runs": runs,
            "archives": archives,
            "steps": steps,
            "tasks": tasks,
            "group_types": group_types,
        },
    })
