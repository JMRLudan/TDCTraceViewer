#!/usr/bin/env bash
# Launch the GRPO / SFT rollout viewer via uv (no system pip required).
# Usage: bash launch.sh [PORT]   (default port: 8765)
#
# Auto-discovers runs + archives as siblings of this viewer's parent folder.
# Override explicitly with env vars before invoking:
#   GRPO_RUN_DIRS=/path/a:/path/b    SFT_COMPARE_DIRS=/path/c   bash launch.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PORT="${1:-8765}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found on PATH. install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/uvicorn_${STAMP}.log"

echo "starting viewer on http://127.0.0.1:${PORT}  (logs: $LOG)"
exec uv run --with fastapi --with uvicorn --with jinja2 \
  python -m uvicorn app:app --host 127.0.0.1 --port "${PORT}" --log-level info \
  2>&1 | tee -a "$LOG"
