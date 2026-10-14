#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-core}"
SETUP_SCRIPT="${ROOT_DIR}/scripts/setup_uv.sh"
UV_BIN="${UV_BIN:-uv}"
PYTHON_WARNINGS_FILTERS="${PYTHONWARNINGS:-}"

SWIG_WARNING_FILTERS="ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning,ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning,ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning"

if [[ -n "$PYTHON_WARNINGS_FILTERS" ]]; then
  export PYTHONWARNINGS="${PYTHON_WARNINGS_FILTERS},${SWIG_WARNING_FILTERS}"
else
  export PYTHONWARNINGS="${SWIG_WARNING_FILTERS}"
fi

case "$MODE" in
  core)
    "$SETUP_SCRIPT" core
    "$UV_BIN" run --python .venv/bin/python pytest
    ;;
  full)
    "$SETUP_SCRIPT" full
    "$UV_BIN" run --python .venv/bin/python pytest
    ;;
  integration)
    "$SETUP_SCRIPT" integration
    "$UV_BIN" run --python .venv/bin/python pytest -m integration
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [core|full|integration]" >&2
    exit 1
    ;;
esac
