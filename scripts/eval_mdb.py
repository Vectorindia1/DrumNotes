"""Real-recording benchmark on MDB Drums full mixes (23 songs, hand-annotated), category C (PRD 7.3).

    PYTHONPATH=. .venv/bin/python scripts/eval_mdb.py [--adtstr]   # needs data/benchmarks/mdb (git-ignored)

Systems, scored against the subclass annotations with one-to-one onset matching at +-50 ms:
  adtof     the raw detector (5 coarse classes only)
  ours      the full pipeline (refinement, hi-hat rebuild, groove fill, articulation labels)
  adtstr    ADT_STR (26 classes), from cached hits in outputs/adtstr/<song>.hits.pkl (scripts/try_adtstr.py)
Fine classes (hi-hat closed/open, ride, crash, china, splash, ghost snare, tambourine) are scored only for
systems that output them. 'cymbal type' is the share of correctly typed cymbals among truth cymbals that
some cymbal detection matched: it isolates the labelling question from detection. MDB Drums is CC BY-NC-SA:
evaluation only.
"""
import argparse
import collections
import json
import pathlib
import pickle
import tempfile

from backend.detection import adtof
from backend.pipeline import run
from backend.validation.metrics import match

ROOT = pathlib.Path("data/benchmarks/mdb")
TOL = 0.05
TRUTH = {  # MDB subclass label -> (coarse, fine)
    "KD": ("kick", "kick"), "SD": ("snare", "snare"), "SDB": ("snare", "snare"), "SDD": ("snare", "snare"),
    "SDF": ("snare", "snare"), "SDNS": ("snare", "snare"), "SST": ("snare", "snare"), "SDG": ("snare", "ghost"),
    "LFT": ("tom", "tom"), "MHT": ("tom", "tom"), "HFT": ("tom", "tom"), "HIT": ("tom", "tom"),
    "CHH": ("hihat", "hat_closed"), "PHH": ("hihat", "hat_closed"), "OHH": ("hihat", "hat_open"),
    "RDC": ("cymbal", "ride"), "RDB": ("cymbal", "ride"), "CRC": ("cymbal", "crash"),
    "CHC": ("cymbal", "china"), "SPC": ("cymbal", "splash"), "TMB": ("tambourine", "tambourine"),
}
ADTSTR = {35: ("kick", "kick"), 36: ("kick", "kick"), 37: ("snare", "snare"), 38: ("snare", "snare"), 39: ("snare", "snare"),
          40: ("snare", "snare"), 41: ("tom", "tom"), 45: ("tom", "tom"), 47: ("tom", "tom"), 42: ("hihat", "hat_closed"),
          43: ("hihat", "hat_closed"), 44: ("hihat", "hat_open"), 46: ("cymbal", "crash"), 48: ("cymbal", "ride"),
          49: ("cymbal", "china"), 50: ("tambourine", "tambourine"), 51: ("cymbal", "splash")}
COARSE = ("kick", "snare", "tom", "hihat", "cymbal")  # "percussion" (tambourine/shaker) is scored as the fine class only
FINE = ("hat_closed", "hat_open", "ride", "crash", "china", "splash", "ghost", "tambourine")


def truth_of(song):
    out = []
    for line in (ROOT / "annotations/subclass" / f"{song}_subclass.txt").read_text().split("\n"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in TRUTH:
            out.append((float(parts[0]), *TRUTH[parts[1]]))
    return out


def ours_fine(e):
    if e.instrument == "hihat":
        return "hat_open" if e.articulation == "open" else "hat_closed"
    if e.instrument == "cymbal":
        return e.articulation if e.articulation in ("ride", "crash", "china", "splash") else "crash"
    if e.instrument == "snare":
        return "ghost" if e.articulation == "ghost" else "snare"
    if e.instrument == "percussion":
        return "tambourine"
    return e.instrument


def systems_for(song, audio, use_adtstr, adtstr_dir="outputs/adtstr"):
    raw, _ = adtof.detect(audio)
    out = {"adtof": [(e.time_s, e.instrument, None) for e in raw]}
    with tempfile.TemporaryDirectory() as d:
        _, _, ev = run(audio, d, player=False)
    out["ours"] = [(e.time_s, e.instrument, ours_fine(e)) for e in ev if e.notated and e.grace_of is None]
    cache = pathlib.Path(adtstr_dir) / f"{audio.stem}.hits.pkl"
    if use_adtstr and cache.exists():
        hits = pickle.loads(cache.read_bytes())
        out["adtstr"] = [(t, *ADTSTR[p]) for t, p, _ in hits if p in ADTSTR]
        import copy

        from backend.detection.adtstr import relabel
        ev2 = copy.deepcopy(ev)
        relabel(ev2, hits)
        out["hybrid"] = [(e.time_s, "tambourine" if e.instrument == "percussion" and e.articulation == "tambourine" else e.instrument,
                          "tambourine" if e.instrument == "percussion" and e.articulation == "tambourine" else ours_fine(e))
                         for e in ev2 if e.notated and e.grace_of is None]
    return out


def tally(counts, key, det, ref):
    tp = len(match(det, ref, TOL))
    c = counts[key]
    c["tp"] += tp
    c["fp"] += len(det) - tp
    c["fn"] += len(ref) - tp


def f1(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adtstr", action="store_true", help="include ADT_STR (cached hits needed)")
    ap.add_argument("--songs", nargs="*", help="limit to these song names")
    ap.add_argument("--adtstr-dir", default="outputs/adtstr", help="cache dir: outputs/adtstr (fast) or outputs/adtstr_orig")
    a = ap.parse_args()
    counts = collections.defaultdict(lambda: collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    confusion = collections.defaultdict(collections.Counter)
    songs = sorted(p.name.replace("_MIX.wav", "") for p in (ROOT / "audio/full_mix").glob("*_MIX.wav"))
    for song in songs if not a.songs else [s for s in songs if s in a.songs]:
        audio = ROOT / "audio/full_mix" / f"{song}_MIX.wav"
        truth = truth_of(song)
        for name, det in systems_for(song, audio, a.adtstr, a.adtstr_dir).items():
            for c in COARSE:
                tally(counts[name], c, [t for t, k, _ in det if k == c], [t for t, k, _ in truth if k == c])
            if name != "adtof":
                for f in FINE:
                    tally(counts[name], f, [t for t, _, ff in det if ff == f], [t for t, _, ff in truth if ff == f])
                # cymbal typing among detected truth cymbals
                dets = [(t, ff) for t, k, ff in det if k == "cymbal"]
                for t, k, ff in truth:
                    if k == "cymbal":
                        near = [d for d in dets if abs(d[0] - t) <= TOL]
                        if near:
                            confusion[name][(ff, min(near, key=lambda d: abs(d[0] - t))[1])] += 1
        print("done", song, flush=True)
    res = {name: {k: {"precision_recall_f1": f1(v), **v} for k, v in c.items()} for name, c in counts.items()}
    for name, conf in confusion.items():
        total = sum(conf.values())
        res[name]["cymbal_type_accuracy"] = round(sum(v for (t, p), v in conf.items() if t == p) / total, 3) if total else None
        res[name]["cymbal_confusion"] = {f"{t}->{p}": v for (t, p), v in sorted(conf.items())}
    pathlib.Path("outputs/json").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs/json/mdb_benchmark.json").write_text(json.dumps(res, indent=1))
    print(f"\nsongs: {len(a.songs) if a.songs else 23}; ADT_STR cache: {a.adtstr_dir if a.adtstr else 'not used'}")
    names = list(res)
    print(f"\nF1 at +-50 ms on 23 real songs (MDB Drums full mixes). Reference counts in brackets.\n{'class':12s}" + "".join(f"{n:>10s}" for n in names))
    for k in (*COARSE, *FINE):
        ref = next((res[n][k]["tp"] + res[n][k]["fn"] for n in names if k in res[n]), 0)
        print(f"{k:12s}" + "".join(f"{res[n][k]['precision_recall_f1'][2]:10.3f}" if k in res[n] else f"{'-':>10s}" for n in names) + f"   [{ref}]")
    for n in names:
        if "cymbal_type_accuracy" in res[n]:
            print(f"cymbal type accuracy ({n}): {res[n]['cymbal_type_accuracy']}  {res[n]['cymbal_confusion']}")
