#!/usr/bin/env bash
# Run CairnIQ API from the project root.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
    echo "Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

HOST="${CAIRNIQ_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "Starting CairnIQ API at http://${HOST}:${PORT}"
source .venv/bin/activate

uvicorn server:app --reload --host "$HOST" --port "$PORT"
