# ENVIRONMENT_REPORT (audited 2026-10-30)

| Item | Finding |
|---|---|
| OS | Kali GNU/Linux Rolling, kernel 7.0.12, x86_64 |
| CPU / RAM | 8 cores / 11 GiB total (~3 GiB available at audit time) |
| Disk | /home: 58 GB free of 401 GB (85% used) |
| GPU / CUDA | **None** (no nvidia-smi). CPU-only path required |
| Python | 3.14.6 system (`/usr/bin/python3`); `uv` present at ~/.local/bin/uv |
| Installed | numpy 2.4.6, scipy 1.17.1, fastapi 0.135.3, uvicorn 0.38.0, pytest 9.0.3 |
| Missing | torch, torchaudio, librosa, soundfile, ruff, demucs, pretty_midi, music21, mido |
| Binaries present | ffmpeg, git 2.53, node 24.18, npm, docker 29.8 (**daemon not running**), xvfb-run, pdftoppm |
| Binaries missing | MuseScore (any), sox |
| MuseScore | apt candidate `musescore3` 3.2.3 (kali-rolling); not installed; needs sudo |
| Network | pypi.org 200, huggingface.co 200, github.com 200, musescore.org 302, download.pytorch.org root 403 (but `/whl/cpu/torch/` 200) |
| PyPI torch | 2.15.0 listed for this interpreter (CPU wheel availability from PyPI/CPU index not yet confirmed by install) |
| Weight caches | ~/.cache/huggingface has unrelated models only; no drum/Demucs weights |

## Limitations (environment, not app bugs)
- RAM is tight: ~3 GiB available. Chunked processing required; Demucs on CPU will be slow.
- Python 3.14 is very new: ADTOF (tested on 3.10) and its dependencies (madmom, TF) are unlikely to build. Use a uv-managed venv with an older Python (3.11/3.12) if wheels are missing.
- Docker is UNVERIFIED (daemon down); no Dockerfile will be produced.

## Update
- `.venv` created with uv Python 3.12.12 (system 3.14 unused). torch **2.15.0+cpu** must come from `--index-url https://download.pytorch.org/whl/cpu`; plain PyPI resolves to a +cu132 build (cuda unavailable here).
- Decode smoke test passes (WAV+MP3, librosa 1.0.0).
- `sudo` requires a password: MuseScore install is a user action.
