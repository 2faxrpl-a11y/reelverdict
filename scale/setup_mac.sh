#!/usr/bin/env bash
# One-time setup for the SCALE pipeline on macOS (Apple Silicon or Intel).
#   cd scale && ./setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list python@3.13 >/dev/null 2>&1 || brew install python@3.13

PY="$(brew --prefix python@3.13)/bin/python3.13"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m unittest discover tests
.venv/bin/python -m scale_engine lint

cat <<'EOF'

SCALE is ready. Next:
  source .venv/bin/activate
  python -m scale_engine build ep001 --preview --tts say     # free scratch voice, ~2 min
  export ELEVENLABS_API_KEY=...                              # then paste voice IDs into config/cast.yaml
  python -m scale_engine build ep001 --tts elevenlabs        # full 1080x1920 master
EOF
