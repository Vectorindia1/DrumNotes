"""Drum-stem extraction with Demucs htdemucs (OPTIONAL; the pipeline works without it, PRD 5.2).

CPU-only and memory-conscious: audio is processed in short overlapping segments. Output is cached.
Weights terms: code is MIT; the weights' own licence/training-data terms are not separately stated
by the repo, so they are UNVERIFIED for commercial use (see MODEL_REGISTRY.md).
"""
import pathlib

import librosa
import numpy as np
import soundfile as sf
import torch

MODEL_NAME = "htdemucs"
SEGMENT_S = 5  # seconds per chunk (model trained on 7.8 s); smaller = less RAM


def load_model():
    from demucs.pretrained import get_model
    return get_model(MODEL_NAME).eval()


def separate_drums(audio_path, out_path, segment=SEGMENT_S, overlap=0.25, shifts=0, progress=False):
    """Write the drum stem (stereo wav) to out_path; return the path. Reuses an existing file."""
    from demucs.apply import apply_model
    out_path = pathlib.Path(out_path)
    if out_path.exists() and out_path.stat().st_mtime >= pathlib.Path(audio_path).stat().st_mtime:
        return out_path
    model = load_model()
    y, _ = librosa.load(str(audio_path), sr=model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    mix = torch.from_numpy(y)[None]
    ref = mix.mean(dim=1)  # demucs normalisation
    mix = (mix - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, mix, shifts=shifts, split=True, overlap=overlap, segment=segment,
                          device="cpu", progress=progress)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    drums = out[model.sources.index("drums")].numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, drums.T, model.samplerate)
    return out_path
