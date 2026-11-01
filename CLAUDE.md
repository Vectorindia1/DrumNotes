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
- `.venv/bin/python scripts/check_environment.py`
- `.venv/bin/pytest backend/tests -q`
- `.venv/bin/ruff check backend scripts`
- MuseScore render: `QT_QPA_PLATFORM=offscreen mscore3 -o out.pdf in.musicxml` (mscore3 3.2.3, /usr/bin)
