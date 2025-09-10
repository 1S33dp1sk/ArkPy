#!/usr/bin/env bash
# Bash shim that prefers .venv if present.
# Usage: ./scripts/install_arkpy.sh [--editable|--system|--user|--upgrade|...]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
INSTALLER="$ROOT/scripts/install_arkpy.py"

choose_py() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"; return 0
  fi
  if command -v python3 >/dev/null 2>&1; then echo "python3"; return 0; fi
  if command -v python  >/dev/null 2>&1; then echo "python";  return 0; fi
  if command -v py      >/dev/null 2>&1; then echo "py -3";    return 0; fi
  echo "ERR" && return 1
}

PY_BIN="$(choose_py)" || { echo "Python not found. Install Python 3.9+ or create .venv first." >&2; exit 1; }

echo "== ArkPy installer (sh) =="
echo "root: $ROOT"
echo "py:   $PY_BIN"
# shellcheck disable=SC2086
exec $PY_BIN "$INSTALLER" "$@"
