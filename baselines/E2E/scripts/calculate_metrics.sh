#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_ROOT"

MOVE_TYPE="2D"      # 2D or 3D
EVAL_TYPE="test"   # test, val_seen, val_unseen

PY_FILE="$PROJ_ROOT/src/result_analysis.py"
RESULT_ROOT="$PROJ_ROOT/logs"

python "$PY_FILE" \
  --result_root "$RESULT_ROOT" \
  --move_type "$MOVE_TYPE" \
  --eval_type "$EVAL_TYPE"