# MODEL_REGISTRY

Status values per PRD §20. No model has been installed or executed.

| Model | Source | License | Status | Notes |
|---|---|---|---|---|
| ADTOF (original, Keras/TF + madmom) | github.com/MZehren/ADTOF | **CC BY-NC-SA 4.0 — NON-COMMERCIAL — NOT APPROVED FOR PRODUCTION** | NOT STARTED | Tested on Python 3.10 only; TF/madmom heavy. Prefer PyTorch port. |
| ADTOF-pytorch | github.com/xavriley/ADTOF-pytorch | **No LICENSE file in repo; derived from ADTOF weights (NC-SA)** — NON-COMMERCIAL — NOT APPROVED FOR PRODUCTION | EXPERIMENTAL (installed editable at third_party/ADTOF-pytorch, commit 85c192e; synthetic smoke test only) | Deps: torch, librosa, pretty_midi. Claims -0.2% F vs original on MDBDrums++. Weights are converted ADTOF weights. |
| Demucs htdemucs | github.com/adefossez/demucs (PyPI demucs 4.1.0) | Code MIT. **Weights' own licence / training-data terms not stated by the repo: UNVERIFIED for commercial use** | EXPERIMENTAL (executed, see below) | Optional (PRD §5.2/§11). Not required by the pipeline (`--separate` opt-in). |
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
- **SYNTHETIC ONLY** (`scripts/run_benchmarks.py` category A; earlier ad-hoc script removed; programmatic kick/snare/hat): kick P/R/F1 1.00/1.00/1.00 (24 ref), snare 1.00/1.00/1.00 (16), hi-hat 1.00/0.984/0.992 (64). Toms/cymbals not exercised. This is NOT real-mix accuracy (PRD 7.3 category A).
- **Defect found (sample-0):** upstream PeakPicker edge-pads the moving average, so an onset at frame 0 is cancelled although the model fires (kick act 0.51 at frame 0). Workaround now in `backend/detection/adtof.py` (pad 10 zero frames, shift back), covered by `test_detection.py`. A weak hi-hat (act 0.235 vs thr 0.22) simultaneous with a t=0 kick can still be missed; strong activations at frame 0 are kept (kick+snare+hat chord at sample 0 detected in the stacked synthetic test).
- Stacked synthetic pattern (kick+snare+hat only): snare recall 5/8 (activation 0.12–0.15 below threshold 0.24); kick and hat 8/8.
- Detection times sit ~10 ms (one frame) before synthetic truth; resolution is 10 ms (100 fps).
- Limitations: 5 classes only, no ghost notes/accents/open-hat; unverified on full mixes; non-commercial weights.

## Demucs htdemucs execution record (2026-11-01)
- Install: `uv pip install demucs` (4.1.0; torch 2.15.0+cpu / torchaudio 2.11.1 left unchanged; deps einops, julius, lameenc, sphn, safetensors, huggingface-hub, pyyaml, tqdm).
- Weights: Hugging Face repo `adefossez/HTDemucs`, `955717e8.safetensors`, 84,025,440 bytes, sha256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`, cached in ~/.cache/huggingface/hub (not in git). Loaded as a 1-model bag; sources drums/bass/other/vocals, 44.1 kHz.
- Hardware: CPU only, 8 cores, ~2 GiB free RAM at the time. Settings: segment 5 s (model trained on 7.8 s), overlap 0.25, shifts 0. Peak RSS ~1.9 GB. Speed: 397.5 s of audio took ~13-15 min (≈2.2 s of compute per second of audio, slower while other work ran).

## Playback sound sources (2026-11-01)
| Item | Source | Licence | Status |
|---|---|---|---|
| MuseScore_General_Lite.sf3 (drum kits, bank 128) | Debian `musescore-general-soundfont-small` -> /usr/share/sounds/sf3 | MIT (per package copyright file) | used for score audio, EXPERIMENTAL |
| TimGM6mb.sf2 | Debian `timgm6mb-soundfont` | GPL-2 | fallback only |
| pyfluidsynth 1.4.0 (+ system libfluidsynth.so.3) | PyPI | LGPL-2.1 | used, dev dependency |

## ADT_STR (2026-11-01) - EVALUATED AND REJECTED (see bottom)
- huggingface.co/Pierfrancesco/adt-str (paper arXiv 2601.09520, code github.com/pier-maker92/ADT_STR). Seq2seq transformer, 69 M params, 24 kHz, 2.56 s chunks, 10 ms resolution, trained on synthetic audio rendered from MIDI; reports state of the art on ENST and MDB test sets (third-party claim, not reproduced).
- 26 classes incl. closed/pedal/open hi-hat, crash, ride (+bell), china, splash, tambourine, shaker, cowbell, triangle, floor/mid/high tom, side stick, clap: exactly the distinctions ADTOF cannot make (ADTOF merges tambourine into hi-hat and all cymbals into one class).
- Licence: CC BY-SA 4.0 per the GitHub README; the HF model card states none -> UNVERIFIED.
- Downloaded: code + setting-tau-0.8 only, third_party/adt-str (git-ignored), model.safetensors sha256 f94d1773d929166e0a3f63d305975aa6667512f8a71d7706872e1f8b23ca930b (289 MB). Code read; scan found no shell/network/eval calls (one torch.load fallback, unused with safetensors). Deps added: transformers 5.22.1, omegaconf.
- Running it was blocked by the auto-mode permission classifier (external code). `scripts/try_adtstr.py AUDIO` runs it for evaluation once the user allows/runs it.

### ADT_STR evaluation result (2026-11-01): REJECTED
- Executed with user permission. Original decoder: up to 27x slower than real time on this CPU. Bug found: decoder self-attention masks the future with an additive -1e4, but attention scores reach ~1.26e6, so future tokens leak (logits for a position change with sequence length; a true -inf mask gives zero difference). The model was trained with the leak. `backend/detection/adtstr.py` = correct-causal cached decoder, 0.6x real time, not token-identical (cannot be, given the leak).
- On MDB full mixes AND drum-only stems, both decoders mislabel hi-hats massively (e.g. Hendrix 64 closed hats -> 0 hats; Rock hats -> cowbell/claves/tambourine; loudness normalisation turns hats into "tambourine"). Not used anywhere. Kept only for reproducibility of the evaluation.

## Learned labels (2026-11-01) - IN USE
- `backend/detection/labels.py`, model `models/weights/labels.joblib` (git-ignored, ~1 MB), trained by `scripts/train_labels.py` on MDB Drums full mixes (CC BY-NC-SA -> model non-commercial). sklearn HistGradientBoosting on 21 features per hit (band energies, spectral shape, decay to the next hit, ADTOF activations, spacing in beats).
- Leave-one-song-out (23 songs): ride vs crash accuracy 0.551 -> 0.903 (ride F1 0.654 -> 0.942, ~81 % of crashes found); open hi-hat F1 0.556 -> 0.680 (false opens 161 -> 69; threshold 0.6 chosen on held-out predictions, slightly optimistic). Splash/china/tambourine: too few examples to learn.
