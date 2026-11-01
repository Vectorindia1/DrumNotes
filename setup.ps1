# UNVERIFIED: never executed (no Windows machine). Mirrors setup.sh; MuseScore 3 and ffmpeg must be on PATH.
$ErrorActionPreference = "Continue"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Host "[FAIL] uv missing"; exit 1 }
if (-not (Test-Path .venv\Scripts\python.exe)) { uv venv --python 3.12 .venv }
$py = ".venv\Scripts\python.exe"
uv pip install --python $py --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv pip install --python $py librosa soundfile numpy scipy fastapi uvicorn pytest ruff pretty_midi pyfluidsynth httpx
if (Test-Path third_party\ADTOF-pytorch) { uv pip install --python $py --no-deps -e third_party\ADTOF-pytorch }
foreach ($b in "mscore3","MuseScore3","ffmpeg") { if (Get-Command $b -ErrorAction SilentlyContinue) { Write-Host "[OK] $b" } else { Write-Host "[WARN] $b missing" } }
& $py scripts\check_environment.py
