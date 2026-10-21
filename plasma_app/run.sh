#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/plasma-app-matplotlib"
exec "${PYTHON:-python3}" -m streamlit run "$APP_DIR/app.py" --server.address 127.0.0.1 --browser.gatherUsageStats false "$@"
