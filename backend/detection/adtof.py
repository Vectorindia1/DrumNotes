"""ADTOF-pytorch wrapper. NON-COMMERCIAL weights (CC BY-NC-SA), see MODEL_REGISTRY.md.

Differences from upstream `transcribe_to_midi`: weights are loaded strictly, activations are
zero-padded before peak picking so an onset in frame 0 survives, and events keep their
activation and raw time instead of becoming fixed-length MIDI notes.
"""
from functools import lru_cache

import numpy as np
import torch
from adtof_pytorch import (
    FRAME_RNN_THRESHOLDS,
    LABELS_5,
    PeakPicker,
    calculate_n_bins,
    create_frame_rnn_model,
    get_default_weights_path,
    load_audio_for_model,
)

from backend.events import Event

SAMPLE_RATE = 44100  # ADTOF resamples everything to this
FPS = 100
PAD_FRAMES = 10  # >= the picker's pre_avg window (0.1 s)
LABEL_TO_INSTRUMENT = {35: "kick", 38: "snare", 47: "tom", 42: "hihat", 49: "cymbal"}


@lru_cache(maxsize=1)
def load_model():
    model = create_frame_rnn_model(calculate_n_bins()).eval()
    ckpt = torch.load(get_default_weights_path(), map_location="cpu")
    result = model.load_state_dict(ckpt["model_weights"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys, result
    return model


def activations(audio_path):
    """Frame activations [time, 5] at 100 fps (frame i is centred on time i/100)."""
    with torch.no_grad():
        return load_model()(load_audio_for_model(str(audio_path))).numpy()[0]


def pick(pred, thresholds=None):
    """Peak-pick activations -> {midi_label: [seconds]} with first-frame onsets preserved."""
    thresholds = FRAME_RNN_THRESHOLDS if thresholds is None else thresholds
    padded = np.concatenate([np.zeros((PAD_FRAMES, pred.shape[1]), pred.dtype), pred])
    raw = PeakPicker(thresholds=thresholds, fps=FPS).pick(padded[None], labels=LABELS_5)[0]
    shift = PAD_FRAMES / FPS
    return {k: [round(t - shift, 6) for t in v if t >= shift - 1e-9] for k, v in raw.items()}


def detect(audio_path, thresholds=None, return_activations=False):
    pred = activations(audio_path)
    peaks = pick(pred, thresholds)
    events = []
    for col, label in enumerate(LABELS_5):
        for t in peaks.get(label, []):
            frame = min(round(t * FPS), len(pred) - 1)
            act = float(pred[frame, col])
            events.append(
                Event(
                    id="",
                    time_s=t,
                    sample=round(t * SAMPLE_RATE),
                    instrument=LABEL_TO_INSTRUMENT[label],
                    confidence=round(act, 4),
                    model_output={"model": "adtof-pytorch", "label": label, "activation": round(act, 4)},
                )
            )
    events.sort(key=lambda e: (e.time_s, e.instrument))
    for i, e in enumerate(events):
        e.id = f"e{i:05d}"
    meta = {
        "model": "adtof-pytorch Frame_RNN (NON-COMMERCIAL)",
        "thresholds": list(FRAME_RNN_THRESHOLDS if thresholds is None else thresholds),
        "frames": len(pred),
        "fps": FPS,
    }
    return (events, meta, pred) if return_activations else (events, meta)
