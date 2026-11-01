#!/usr/bin/env bash
# DrumScribe setup: CPU-only, safe to rerun. Does not download models or use sudo.
set -u
cd "$(dirname "$0")"
ok(){ echo "[OK]   $*"; }; warn(){ echo "[WARN] $*"; }; fail(){ echo "[FAIL] $*"; rc=1; }
rc=0

echo "OS: $(uname -sr)"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 15 ] && ok "disk ${free_gb} GB free" || warn "only ${free_gb} GB free (need ~15 for venv + weights)"
command -v uv >/dev/null && ok "uv" || fail "uv missing (https://docs.astral.sh/uv/)"

[ -x .venv/bin/python ] || uv venv --python 3.12 .venv -q || fail "venv creation"
PY=.venv/bin/python
# torch must come from the CPU index; PyPI resolves to a CUDA build
if ! $PY -c "import torch,sys;sys.exit('+cpu' not in torch.__version__)" 2>/dev/null; then
  uv pip install -q --python $PY --index-url https://download.pytorch.org/whl/cpu torch torchaudio && ok "torch cpu" || fail "torch install"
else ok "torch $($PY -c 'import torch;print(torch.__version__)')"; fi
uv pip install -q --python $PY librosa soundfile numpy scipy fastapi uvicorn pytest ruff pretty_midi pyfluidsynth httpx && ok "python deps" || fail "python deps"

if [ -d third_party/ADTOF-pytorch ]; then
  uv pip install -q --python $PY --no-deps -e third_party/ADTOF-pytorch && ok "ADTOF-pytorch (NON-COMMERCIAL)" || fail "ADTOF install"
else warn "third_party/ADTOF-pytorch missing: git clone https://github.com/xavriley/ADTOF-pytorch third_party/ADTOF-pytorch (commit 85c192e, see MODEL_REGISTRY.md)"; fi

$PY -c "import demucs" 2>/dev/null && ok "demucs (optional, M2)" || warn "demucs not installed (optional): uv pip install demucs, then python scripts/download_models.py"
[ -n "$(ls /usr/share/sounds/sf3/MuseScore_General*.sf3 /usr/share/sounds/sf2/TimGM6mb.sf2 2>/dev/null)" ] && ok "drum soundfont" || warn "no drum soundfont: sudo apt install musescore-general-soundfont-small (player falls back to a tonal render)"
[ -f models/weights/labels.joblib ] && ok "learned ride/crash + open-hat labels" || warn "no label model (falls back to heuristics): $PY scripts/fetch_mdb.py && PYTHONPATH=. $PY scripts/train_labels.py  (~230 MB download, ~10 min)"
command -v mscore3 >/dev/null && ok "MuseScore $(mscore3 --version 2>/dev/null | head -1)" || warn "mscore3 missing: sudo apt install musescore3"
command -v ffmpeg >/dev/null && ok ffmpeg || fail "ffmpeg missing"

$PY scripts/check_environment.py >/dev/null 2>&1 && ok "decode smoke test" || fail "decode smoke test"
[ $rc -eq 0 ] && echo "SETUP SUCCESS" || echo "SETUP FINISHED WITH FAILURES"
exit $rc
