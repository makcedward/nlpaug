#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODE="${1:-core}"

case "$MODE" in
  core)
    EXTRAS=".[dev]"
    NLTK_DATA=""
    ;;
  full)
    EXTRAS=".[dev,audio,nltk,transformers]"
    NLTK_DATA="punkt punkt_tab"
    ;;
  integration)
    EXTRAS=".[dev,audio,nltk,transformers,lambada,word-embs]"
    NLTK_DATA="punkt punkt_tab"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [core|full|integration]" >&2
    exit 1
    ;;
esac

"$UV_BIN" venv --python "$PYTHON_VERSION" --allow-existing
"$UV_BIN" pip install -p .venv/bin/python -e "$EXTRAS"

if [[ -n "$NLTK_DATA" ]]; then
  for package_name in $NLTK_DATA; do
    "$UV_BIN" run --python .venv/bin/python python - <<PY
import nltk
nltk.download("${package_name}")
PY
  done
fi

echo "Environment ready in .venv for mode: $MODE"
