"""Train the ride/crash and open/closed hi-hat classifiers on MDB Drums and report leave-one-song-out scores.

    PYTHONPATH=. .venv/bin/python scripts/train_labels.py     # needs data/benchmarks/mdb; writes models/weights/labels.joblib

Each song is scored by a model that never saw it. The comparison 'heuristic' is the current hand-set rule on
the same hits. MDB Drums is CC BY-NC-SA: the saved models are non-commercial."""
import json
import pathlib
import pickle
import sys
import tempfile

import joblib
import librosa
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from eval_mdb import ROOT, TRUTH

from backend.detection import adtof, labels
from backend.pipeline import run

CACHE = pathlib.Path("outputs/label_features")
TARGET = {"cymbal": {"RDC": 1, "RDB": 1, "CRC": 0, "CHC": 0, "SPC": 0}, "hihat": {"OHH": 1, "CHH": 0, "PHH": 0}}


def song_rows(audio):
    cache = CACHE / f"{audio.stem}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    truth = []
    for line in (ROOT / "annotations/subclass" / f"{audio.stem.replace('_MIX', '')}_subclass.txt").read_text().split("\n"):
        p = line.split()
        if len(p) == 2 and p[1] in TRUTH:
            truth.append((float(p[0]), p[1]))
    with tempfile.TemporaryDirectory() as d:
        _, rep, ev = run(audio, d, player=False)
    y, _ = librosa.load(str(audio), sr=labels.SR, mono=True)
    pred = adtof.activations(audio)
    beat = 60.0 / rep["rhythm"]["tempo_bpm"]
    rows = {}
    for inst in ("cymbal", "hihat"):
        sel, X = labels.features_for(ev, y, pred, beat, inst)
        lab, heur = [], []
        for e in sel:
            cands = [(abs(t - e.time_s), c) for t, c in truth if c in TARGET[inst] and abs(t - e.time_s) <= 0.05]
            lab.append(TARGET[inst][min(cands)[1]] if cands else -1)
            heur.append(1 if (e.articulation == ("ride" if inst == "cymbal" else "open")) else 0)
        rows[inst] = (X, np.array(lab), np.array(heur))
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(rows))
    return rows


def stats(pr, y):
    """accuracy and F1 of the positive class (ride / open) for 0/1 predictions."""
    tp, fp, fn = int(((pr == 1) & (y == 1)).sum()), int(((pr == 1) & (y == 0)).sum()), int(((pr == 0) & (y == 1)).sum())
    return {"accuracy": round(float((pr == y).mean()), 3), "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0, "tp": tp, "fp": fp, "fn": fn}


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0)


if __name__ == "__main__":
    songs = sorted((ROOT / "audio/full_mix").glob("*_MIX.wav"))
    data = {a.stem: song_rows(a) for a in songs}
    print("features cached for", len(data), "songs")
    report, final = {}, {}
    for inst in ("cymbal", "hihat"):
        X_all, y_all, g_all, h_all = [], [], [], []
        for s, rows in data.items():
            X, y, h = rows[inst]
            m = y >= 0
            if m.any():
                X_all.append(X[m]); y_all.append(y[m]); h_all.append(h[m]); g_all += [s] * int(m.sum())
        X, y, h, g = np.nan_to_num(np.vstack(X_all), nan=-60.0), np.concatenate(y_all), np.concatenate(h_all), np.array(g_all)
        p = np.zeros(len(y))
        for s in np.unique(g):  # leave one song out
            tr, te = g != s, g == s
            if len(np.unique(y[tr])) < 2:
                continue
            p[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        best_t = 0.5
        if inst == "hihat":  # open hats are rare: threshold chosen on the leave-one-song-out predictions
            best_t = max(np.arange(0.2, 0.9, 0.05), key=lambda t: stats((p >= t).astype(int), y)["f1"])
        report[inst] = {"n": len(y), "positives": int(y.sum()), "heuristic": stats(h, y), "learned_LOSO": stats((p >= best_t).astype(int), y),
                        "threshold": round(float(best_t), 2)}
        final[inst] = model().fit(X, y)
        if inst == "hihat":
            final["hihat_threshold"] = float(best_t)
        print(inst, json.dumps(report[inst]))
    labels.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, labels.MODEL_PATH)
    pathlib.Path("outputs/json/label_training.json").write_text(json.dumps(report, indent=1))
    print("saved", labels.MODEL_PATH)
