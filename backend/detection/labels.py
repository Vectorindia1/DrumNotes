"""Learned labels for detected hits: ride vs crash (cymbals) and open vs closed (hi-hats).

Replaces the hand-set thresholds in articulation.py, which were right for only 54 % of cymbals on the MDB
Drums benchmark. Features per hit: band energies and spectral shape at the onset, how the energy decays
(cut at the next hit of the same family), ADTOF's own activations at the hit, and rhythmic context (spacing to
neighbouring hits of the same family, in beats). Models are trained by scripts/train_labels.py on MDB Drums
(CC BY-NC-SA, so the trained models are non-commercial, like ADTOF) and are only used if the file exists.
"""
import pathlib

import numpy as np

MODEL_PATH = pathlib.Path(__file__).resolve().parents[2] / "models" / "weights" / "labels.joblib"
SR = 44100
BANDS = ((2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000), (11000, 16000))


def _spec(y, t, dur):
    a = int(max(t - 0.005, 0) * SR)
    seg = y[a:a + int(dur * SR)]
    if len(seg) < 512:
        seg = np.pad(seg, (0, 512 - len(seg)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
    return mag, np.fft.rfftfreq(len(seg), 1 / SR)


def hit_features(y, pred, t, family_times, beat_s):
    """One feature vector for a hit at time t. family_times: sorted times of hits of the same family."""
    mag, f = _spec(y, t, 0.06)
    hf = (f >= 2000) & (f <= 16000)
    band = [np.log10(mag[(f >= lo) & (f < hi)].mean()) for lo, hi in BANDS]
    total = np.log10(mag[hf].mean())
    centroid = float((f[hf] * mag[hf]).sum() / mag[hf].sum()) / 1000
    flatness = float(np.exp(np.log(mag[hf]).mean()) / mag[hf].mean())
    peakiness = float(np.log10(mag[hf].max() / np.median(mag[hf])))
    i, j = np.searchsorted(family_times, t - 1e-6), np.searchsorted(family_times, t + 1e-6)
    prev_gap = t - family_times[i - 1] if i > 0 else 4.0
    next_gap = family_times[j] - t if j < len(family_times) else 4.0
    # decay: energy in 40-ms windows after the hit, relative to the first, up to the next hit (or 400 ms)
    decay = []
    a = int(max(t - 0.005, 0) * SR)
    first = np.mean(y[a:a + int(0.04 * SR)] ** 2) + 1e-12
    for k in (1, 2, 4, 8):
        s = int((t + 0.04 * k) * SR)
        if 0.04 * k + 0.04 > next_gap or s + int(0.04 * SR) > len(y):
            decay.append(np.nan)
        else:
            decay.append(10 * np.log10((np.mean(y[s:s + int(0.04 * SR)] ** 2) + 1e-12) / first))
    frame = min(round(t * 100), len(pred) - 1)
    act = list(pred[max(frame - 1, 0):frame + 2].max(0)) if len(pred) else [0.0] * 5
    return [*[b - total for b in band], total, centroid, flatness, peakiness, *decay,
            min(prev_gap / beat_s, 8), min(next_gap / beat_s, 8), *act]


def features_for(events, y, pred, beat_s, instrument):
    times = np.array(sorted(e.time_s for e in events if e.instrument in ("hihat", "cymbal") and e.grace_of is None))
    sel = [e for e in events if e.instrument == instrument and e.grace_of is None]
    fam = np.array(sorted(e.time_s for e in sel)) if instrument == "cymbal" else times
    return sel, np.array([hit_features(y, pred, e.time_s, fam, beat_s) for e in sel]) if sel else np.zeros((0, 21))


RARE = ("splash", "china", "tambourine")


def decide(proba, classes, thresholds):
    """Most likely common class, unless a rare class (splash/china/tambourine) clears its own threshold:
    a false splash or tambourine is worse than a missed one."""
    common = [i for i, c in enumerate(classes) if c not in RARE]
    out = []
    for p in proba:
        rare = [(p[i] / thresholds[c], c) for i, c in enumerate(classes) if c in thresholds and p[i] >= thresholds[c]]
        out.append(max(rare)[1] if rare else classes[common[int(np.argmax(p[common]))]])
    return out


def apply(events, y, pred, beat_s):
    """Relabel cymbals (ride/crash/splash/china) and hi-hats (open/closed); either may turn out to be a
    tambourine (instrument "percussion"). Uses the multi-class models from scripts/train_cymbal_types.py when
    present, else the ride/crash + open/closed models. Returns counts or None if no model file exists."""
    if not MODEL_PATH.exists():
        return None
    import joblib
    models = joblib.load(MODEL_PATH)
    out = {}
    feats = {inst: features_for(events, y, pred, beat_s, inst) for inst in ("cymbal", "hihat")}  # before any relabel
    for inst in ("cymbal", "hihat"):
        sel, X = feats[inst]
        if not len(sel):
            continue
        Xn = np.nan_to_num(X, nan=-60.0)
        multi = models.get(f"{inst}_multi")
        if multi is not None:
            proba = multi.predict_proba(Xn)
            names = decide(proba, list(multi.classes_), models.get(f"{inst}_thresholds", {}))
            conf = proba.max(1)
        elif inst in models:
            p = models[inst].predict_proba(Xn)[:, 1]
            if inst == "cymbal":  # keep splash/china only if the heuristics found them
                names = [e.articulation if e.articulation in ("splash", "china") else ("ride" if pi >= 0.5 else "crash")
                         for e, pi in zip(sel, p, strict=True)]
            else:
                names = ["hihat_open" if pi >= models.get("hihat_threshold", 0.5) else "hihat" for pi in p]
            conf = np.maximum(p, 1 - p)
        else:
            continue
        for e, name, c in zip(sel, names, conf, strict=True):
            if name == "tambourine":
                e.instrument, e.articulation = "percussion", "tambourine"
            elif inst == "cymbal":
                e.articulation = name
            else:
                e.articulation = "open" if name == "hihat_open" else None
            e.articulation_confidence = round(float(c), 2)
            e.articulation_features = {**(e.articulation_features or {}), "label_source": "learned"}
        out[inst] = dict(zip(*np.unique(names, return_counts=True), strict=True)) if names else {}
        out[inst] = {str(k): int(v) for k, v in out[inst].items()}
    return out
