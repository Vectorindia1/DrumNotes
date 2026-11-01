# DrumScribe AI — operating manual
Objective: full-mix drum transcription -> validated MusicXML/PDF/MIDI/JSON. Spec: DrumScribe_AI_Revised_PRD_v2.pdf.

Every session: read PROGRESS.md and ENVIRONMENT_REPORT.md first. Update PROGRESS.md at the end.

Rules
- CPU only (no GPU). Don't install large deps/models without checking size, network, disk.
- torch must come from `--index-url https://download.pytorch.org/whl/cpu` (PyPI gives a CUDA build).
- Never claim untested things work; never fabricate benchmarks; synthetic accuracy != real-mix accuracy.
- Preserve raw timestamps (quantization never overwrites them); report ambiguous tempo/meter as alternatives.
- ADTOF is CC BY-NC-SA: dev prototype only, never bundle for commercial use (see LICENSE_AUDIT.md).
- Don't commit weights, generated audio, or secrets. Commit after each passing milestone. Stop and ask when blocked.

Commands (all verified working)
- `./serve.sh` (website on :8000; PORT/HOST env vars; see README.md). Browser test: `node scripts/ui_e2e.mjs URL AUDIO SHOTS_DIR`
- `.venv/bin/python scripts/check_environment.py`
- `.venv/bin/pytest backend/tests -q`
- `.venv/bin/ruff check backend scripts`
- `.venv/bin/python -m backend.pipeline AUDIO --tempo BPM` (AUDIO = file or direct audio-file URL; YouTube/Spotify links are refused on purpose, see backend/intake.py; audio -> outputs/{musicxml,pdf,midi,json}; use --tempo, unhinted tempo is unreliable; --offset SEC makes the grid exact; --beat1 N overrides the downbeat guess; --separate = Demucs drum stem, ~13 min per 6.5 min of audio on this CPU; --player builds the blue-playhead viewer; --kit picks the sampled drum kit for the score audio)
- `PYTHONPATH=. .venv/bin/python scripts/run_benchmarks.py` (category A synthetic; B/C need data/benchmarks)
- `PYTHONPATH=. .venv/bin/python scripts/compare_separation.py` (original vs Demucs stem; needs data/stems for the real-clip part)
- `PYTHONPATH=. .venv/bin/python scripts/validate_pdf.py PDF [MUSICXML [BARS]]`
- MuseScore render: `QT_QPA_PLATFORM=offscreen mscore3 -o out.pdf in.musicxml` (mscore3 3.2.3, /usr/bin)
