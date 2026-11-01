# MODEL_REGISTRY

Status values per PRD §20. No model has been installed or executed.

| Model | Source | License | Status | Notes |
|---|---|---|---|---|
| ADTOF (original, Keras/TF + madmom) | github.com/MZehren/ADTOF | **CC BY-NC-SA 4.0 — NON-COMMERCIAL — NOT APPROVED FOR PRODUCTION** | NOT STARTED | Tested on Python 3.10 only; TF/madmom heavy. Prefer PyTorch port. |
| ADTOF-pytorch | github.com/xavriley/ADTOF-pytorch | **No LICENSE file in repo; derived from ADTOF weights (NC-SA)** — NON-COMMERCIAL — NOT APPROVED FOR PRODUCTION | EXPERIMENTAL (installed editable at third_party/ADTOF-pytorch, commit 85c192e; synthetic smoke test only) | Deps: torch, librosa, pretty_midi. Claims -0.2% F vs original on MDBDrums++. Weights are converted ADTOF weights. |
| Demucs | github.com/adefossez/demucs | Code MIT; weight/dataset licenses to verify | NOT STARTED | Optional (PRD §5.2/§11); not needed for M1. |
| librosa / NumPy / SciPy / PyTorch | PyPI | ISC / BSD / BSD / BSD | NOT STARTED (numpy, scipy installed) | |
| MuseScore | apt `musescore3` 3.2.3 | GPL-3.0 (CLI invoked as external process) | BLOCKED (needs sudo) | |

Production approval (PRD §5.3): none granted.

## ADTOF-pytorch source inspection (read-only, not executed)
- Frame-RNN (CNN + 3 BiRNN layers), bundled `adtof_frame_rnn_pytorch_weights.pth` (cloned to /tmp/adtofpt, ~16 MB repo), deps: torch, librosa, pretty_midi, numpy; Python >=3.9.
- Output: **5 classes only** (MIDI 35 kick, 38 snare, 47 toms, 42 hi-hat, 49 cymbals) at 100 fps, per-class peak-picking thresholds.
- No ghost-note, accent, open/closed hi-hat, ride/crash, or articulation output. Those need other models/heuristics (M4).
- Trained on drum-in-mix audio (ADTOF dataset, per upstream README), so full mixes are plausible input; **unverified here** — no run yet.
- Reported upstream F-measure 88.5 on MDBDrums++ (third-party claim, not reproduced).

## ADTOF-pytorch execution record (2026-11-01)
- Repo commit `85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9` (2025-11-11). Weights `adtof_frame_rnn_pytorch_weights.pth` sha256 `1bc986e596ec47ba0b44916f87cd4a39f0b2bec23596df3fb5d0e87749217320` (checkpoint nests tensors under `model_weights`).
- Strict load: 0 missing / 0 unexpected keys. The package itself loads with `strict=False` (silent mismatch risk), so our wrapper must keep the strict check.
- CPU inference on a 20 s clip: works (cuda unavailable).
- **SYNTHETIC ONLY** (`scripts/adtof_smoke.py`, programmatic kick/snare/hat, 8 bars @100 BPM, tol ±50 ms): kick P/R/F1 1.00/1.00/1.00 (24 ref), snare 1.00/1.00/1.00 (16), hi-hat 1.00/0.984/0.992 (64). Toms/cymbals not exercised. This is NOT real-mix accuracy (PRD 7.3 category A).
- **Defect found (sample-0):** upstream PeakPicker edge-pads the moving average, so an onset at frame 0 is cancelled although the model fires (kick act 0.51 at frame 0). Workaround: pad 10 zero frames before picking and shift times back; recovers the kick at t=0. A hi-hat simultaneous with the t=0 kick (act 0.235 vs threshold 0.22) is still missed. Needs a regression test in the M1 wrapper.
- Limitations: 5 classes only, no ghost notes/accents/open-hat; unverified on full mixes; non-commercial weights.
