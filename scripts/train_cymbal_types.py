"""Learn splash / china / tambourine (plus ride, crash, open/closed hi-hat) for detected cymbal and hi-hat hits.

    PYTHONPATH=. .venv/bin/python scripts/train_cymbal_types.py     # needs data/benchmarks/mdb + data/samples

Real labelled examples are rare (MDB Drums: 10 splash, 15 china, 32 tambourine), so training adds hits made from
real recorded one-shots (scripts/fetch_samples.py: CrocellKit CC BY 4.0, VSCO-2-CE CC0) mixed into the real MDB
songs at moments with no annotated hit, at random level and slight pitch change. The full pipeline then runs on
each mix, so the features are exactly what the website computes.

Evaluation is leave-one-song-out and uses ONLY the real hand-labelled hits of the held-out song; mixes built on
that song's audio are left out of its training set. Rare classes are only predicted when their probability
clears a threshold (one global value per class, chosen on those out-of-song predictions: mildly optimistic).
MDB Drums is CC BY-NC-SA, so a model trained on it is non-commercial, like ADTOF.
"""
import json
import pathlib
import pickle
import sys
import tempfile
import zlib

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "scripts")
from eval_mdb import ROOT

from backend.detection import adtof, articulation, labels
from backend.pipeline import run

CACHE = pathlib.Path("outputs/label_features_fine")
SAMPLES = pathlib.Path("data/samples")
TRUTH = {"RDC": "ride", "RDB": "ride", "CRC": "crash", "SPC": "splash", "CHC": "china",
         "CHH": "hihat", "PHH": "hihat", "OHH": "hihat_open", "TMB": "tambourine"}
SAMPLE_CLASS = {"crash": "crash", "ride": "ride", "splash": "splash", "china": "china", "hihat": "hihat",
                "hihat_open": "hihat_open", "tambourine": "tambourine", "sleighbells": "tambourine"}
FAMILIES = {"cymbal": ("ride", "crash", "splash", "china", "tambourine"), "hihat": ("hihat", "hihat_open", "tambourine")}
RARE = ("splash", "china", "tambourine")
AUGMENTS = 6
# Only the hi-hat family model is saved. Measured (leave-one-song-out, real hits only): it finds 7 of 21 real
# tambourine hits (precision 0.54; 0 without the samples) at unchanged open-hat F1 (0.677). The cymbal model is NOT
# used: splash 0/7, china 3/14 at precision 0.11, and crash F1 below the existing ride/crash model.
SHIP = ("hihat",)  # synthetic mixes per song


def annotations(song):
    out = []
    for line in (ROOT / "annotations/subclass" / f"{song}_subclass.txt").read_text().split("\n"):
        p = line.split()
        if len(p) == 2:
            out.append((float(p[0]), p[1]))
    return out


def rows_for(audio, truth, beat=None):
    """Pipeline events on `audio` -> ({family: (X, labels)}, beat_s); label = nearest truth within 50 ms, '' if none.
    With `beat` given (a mix of a song already analysed) only detection + onset refinement run: the same event
    times and features as the full pipeline, without engraving a score."""
    y, _ = librosa.load(str(audio), sr=labels.SR, mono=True)
    if beat is None:
        with tempfile.TemporaryDirectory() as d:
            _, rep, ev = run(audio, d, player=False)
        pred = adtof.activations(audio)
        beat = 60.0 / rep["rhythm"]["tempo_bpm"]
    else:
        ev, _, pred = adtof.detect(audio, return_activations=True)
        ev, _ = articulation.annotate(ev, y, pred, adtof.pick, env=articulation.Env(articulation.band_power(y)))
    out = {}
    for fam in FAMILIES:
        sel, X = labels.features_for(ev, y, pred, beat, fam)
        lab = []
        for e in sel:
            near = [(abs(t - e.time_s), c) for t, c in truth if abs(t - e.time_s) <= 0.05]
            lab.append(min(near)[1] if near else "")
        out[fam] = (X, np.array(lab, dtype=object))
    return out, beat


def make_mix(audio, song, k, rng, pool):
    """Real song + one-shots at hit-free moments. Returns (mix path, [(time, class)])."""
    y, sr = librosa.load(str(audio), sr=labels.SR, mono=True)
    busy = np.array(sorted(t for t, _ in annotations(song)))
    times, t = [], 2.0
    while t < len(y) / sr - 2:
        t += rng.uniform(0.4, 1.0)
        i = np.searchsorted(busy, t)
        if all(abs(busy[j] - t) > 0.08 for j in (i - 1, i) if 0 <= j < len(busy)):
            times.append(t)
    classes = sorted(pool)
    truth = []
    for t in times:
        cls = classes[rng.integers(len(classes))]
        x, xsr = sf.read(pool[cls][rng.integers(len(pool[cls]))])
        x = x[max(int(np.argmax(np.abs(x) > 0.1 * np.abs(x).max())) - xsr // 500, 0):]  # start at the attack (-2 ms)
        x = librosa.resample(x, orig_sr=xsr, target_sr=int(sr * rng.uniform(0.94, 1.06)))  # slight pitch/size change
        a = int(t * sr)
        win = y[max(a - sr // 4, 0):a + sr // 4]
        rms_mix = np.sqrt(np.mean(win ** 2)) + 1e-6
        rms_hit = np.sqrt(np.mean(x[: sr // 10] ** 2)) + 1e-9
        x = x * rms_mix / rms_hit * 10 ** (rng.uniform(-10, 2) / 20)
        n = min(len(x), len(y) - a)
        y[a:a + n] += x[:n]
        truth.append((t, SAMPLE_CLASS[cls]))
    path = CACHE / f"{song}.mix{k}.wav"
    sf.write(path, y / max(1.0, np.abs(y).max()), sr)
    return path, truth


def song_data(audio, pool):
    song = audio.stem.replace("_MIX", "")
    real_cache = CACHE / f"{song}.real.pkl"
    if real_cache.exists():
        real, beat = pickle.loads(real_cache.read_bytes())
    else:
        real, beat = rows_for(audio, [(t, TRUTH[c]) for t, c in annotations(song) if c in TRUTH])
        real_cache.write_bytes(pickle.dumps((real, beat)))
    synth = []
    rng = np.random.default_rng(zlib.crc32(song.encode()))
    for k in range(AUGMENTS):
        cache = CACHE / f"{song}.synth{k}.pkl"
        if cache.exists():
            synth.append(pickle.loads(cache.read_bytes()))
            continue
        path, inserted = make_mix(audio, song, k, rng, pool)
        rows, _ = rows_for(path, inserted, beat)  # only inserted hits are labelled; the song's own hits get ''
        path.unlink()
        cache.write_bytes(pickle.dumps(rows))
        synth.append(rows)
    return {"real": real, "synth": synth}


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0)


def decide(proba, classes, thr):
    return np.array(labels.decide(proba, classes, thr), dtype=object)


def prf(pred, y, c):
    tp, fp, fn = int(((pred == c) & (y == c)).sum()), int(((pred == c) & (y != c)).sum()), int(((pred != c) & (y == c)).sum())
    return {"n": int((y == c).sum()), "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0, "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0}


def stack(parts):
    X = [p[0][p[1] != ""] for p in parts if len(p[0])]
    y = [p[1][p[1] != ""] for p in parts if len(p[0])]
    return (np.nan_to_num(np.vstack(X), nan=-60.0), np.concatenate(y)) if X else (np.zeros((0, 21)), np.array([], dtype=object))


if __name__ == "__main__":
    SAVE = labels.MODEL_PATH
    labels.MODEL_PATH = pathlib.Path("/nonexistent/labels.joblib")  # features from raw detector families, as at inference
    CACHE.mkdir(parents=True, exist_ok=True)
    pool = {d.name: sorted(d.glob("*.wav")) for d in SAMPLES.iterdir() if d.is_dir() and d.name in SAMPLE_CLASS}
    print("samples:", {k: len(v) for k, v in pool.items()}, flush=True)
    songs = sorted((ROOT / "audio/full_mix").glob("*_MIX.wav"))
    data = {}
    for a in songs:
        data[a.stem] = song_data(a, pool)
        print("features", a.stem, flush=True)
    report, final = {}, {}
    for fam, classes in FAMILIES.items():
        def ok(lab, classes=classes):
            return np.isin(lab, classes)
        real = {s: (d["real"][fam][0][ok(d["real"][fam][1])], d["real"][fam][1][ok(d["real"][fam][1])]) for s, d in data.items()}
        synth = {s: [(r[fam][0][ok(r[fam][1])], r[fam][1][ok(r[fam][1])]) for r in d["synth"]] for s, d in data.items()}
        res = {}
        for name, use_synth in (("real only", False), ("real + samples", True)):
            P, Y = [], []
            for s in data:  # leave one song out; test on its REAL hits only
                parts = [real[o] for o in data if o != s] + ([x for o in data if o != s for x in synth[o]] if use_synth else [])
                Xtr, ytr = stack(parts)
                Xte, yte = stack([real[s]])
                if not len(yte):
                    continue
                m = model().fit(Xtr, ytr)
                p = np.zeros((len(yte), len(classes)))
                p[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(Xte)
                P.append(p)
                Y.append(yte)
            P, Y = np.vstack(P), np.concatenate(Y)
            thr = {}
            for c in RARE:
                if c in classes:
                    thr[c] = 1.01  # off
                    best = prf(decide(P, classes, thr), Y, c)["f1"]
                    for t in np.arange(0.2, 0.96, 0.05):
                        f = prf(decide(P, classes, {**thr, c: t}), Y, c)["f1"]
                        if f > best:
                            best, thr[c] = f, float(round(t, 2))
            pred = decide(P, classes, thr)
            res[name] = {"accuracy": round(float((pred == Y).mean()), 3), "thresholds": thr,
                         "per_class": {c: prf(pred, Y, c) for c in classes},
                         "confusion": {f"{t}->{p}": int(((Y == t) & (pred == p)).sum()) for t in classes for p in classes
                                       if ((Y == t) & (pred == p)).sum()}}
            print(fam, name, json.dumps(res[name]), flush=True)
        report[fam] = res
        if fam not in SHIP:
            continue
        Xall, yall = stack(list(real.values()) + [x for v in synth.values() for x in v])
        final[f"{fam}_multi"] = model().fit(Xall, yall)
        final[f"{fam}_thresholds"] = res["real + samples"]["thresholds"]
    old = joblib.load(SAVE) if SAVE.exists() else {}
    joblib.dump({**old, **final}, SAVE)
    pathlib.Path("outputs/json/cymbal_types_training.json").write_text(json.dumps(report, indent=1))
    print("saved", SAVE)
