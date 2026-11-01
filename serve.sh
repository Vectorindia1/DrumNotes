#!/usr/bin/env bash
# Start the DrumScribe website.  ./serve.sh            -> http://127.0.0.1:8000 (this computer only)
#                                HOST=0.0.0.0 ./serve.sh -> reachable from other devices on your network
# Limits/retention are environment variables, see README.md.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/uvicorn backend.api.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
