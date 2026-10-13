#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python_bin="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo 'Create .venv and install requirements.txt first; see README.md.' >&2
  exit 1
fi

"$python_bin" scripts/annotation/extract.py
"$python_bin" scripts/annotation/add_winrates.py
