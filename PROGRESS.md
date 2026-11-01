# PROGRESS

**Current milestone:** M0 complete (see below); M1 next
**Git:** M0 committed.

## Done / verified
- Step 1 Audit: repo contains only the PRD PDF; environment inventoried (see ENVIRONMENT_REPORT.md). VERIFIED.
- Step 2 Report: ENVIRONMENT_REPORT, MODEL_REGISTRY, LICENSE_AUDIT written. Licensing findings from GitHub API/LICENSE file. VERIFIED (read from source).

- `.venv` (uv, Python 3.12.12): torch 2.15.0+cpu, torchaudio, librosa 1.0.0, soundfile, numpy, scipy, fastapi, uvicorn, pytest, ruff. VERIFIED (imports; cuda=False).
- `scripts/check_environment.py`: WAV + MP3 decode incl. sample-0 event, tool presence. TESTED, passes. (No pyproject/lockfile yet.)

- MuseScore 3.2.3 headless render: **VERIFIED** — generated 4-bar rock beat MusicXML -> 1-page A4 PDF, rasterized and visually inspected (perc clef, 4/4, tempo, hi-hat x, kick, snare correct). `backend/tests/test_render.py` 2 passed; ruff clean.
- pyproject.toml, requirements-lock.txt, CLAUDE.md written. Lock pins torch 2.15.0+cpu.
- ADTOF-pytorch installed (user-approved), strict weight load OK, run on synthetic clip: kick/snare F1 1.00, hat 0.99 @50ms (SYNTHETIC ONLY, EXPERIMENTAL). Sample-0 defect in upstream peak picker found + worked around; details in MODEL_REGISTRY.md.

- `setup.sh`: TESTED — rerun on existing venv succeeds (8 s, all checks OK). Fresh-venv path (venv creation, torch download) NOT exercised this session.
- `setup.ps1`: UNVERIFIED (never run).

## Unverified / not started
- Docker: UNVERIFIED (daemon down; no Dockerfile)
- Demucs/source separation: NOT STARTED (optional, M2)

## Blocked / needs user
1. **Step 4 Data:** no audio or ground-truth annotations exist. Real-mix MVP acceptance test is BLOCKED until a licensed 60–120 s full-mix clip with kick/snare timestamps is provided (or we get ENST/IDMT/Slakh and a mixed clip with annotations). Synthetic tests only until then.
2. (resolved) ADTOF-pytorch installed by user.
3. License decision: user confirmed prototype use of ADTOF (non-commercial).

## Next three actions
1. M1 pipeline (with ADTOF wrapper: zero-pad fix, strict weight load, sample-0 regression test): events JSON (raw timestamps) -> beat/tempo -> quantize -> MusicXML -> PDF -> MIDI, with tests.
3. Obtain real full-mix clip + annotations to unblock the acceptance test.

## M0 completion report (PRD 18.2)
- Implemented/Tested/Verified: environment audit, CPU venv, decode smoke test, MuseScore headless PDF render, ADTOF-pytorch synthetic run, setup.sh.
- Unverified: setup.ps1, Docker, fresh-venv setup path.
- Blocked: real-mix acceptance test (no licensed clip/annotations); production model approval (NC licence).
- Known limitations: 8 cores / ~3 GiB free RAM; ADTOF 5 classes, NC-SA weights; upstream sample-0 peak-picker defect (workaround in scripts/adtof_smoke.py).
