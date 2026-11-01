"""Cache ADT_STR hits for the MDB full mixes: fast (cached decoder) for all songs, original for the shortest N.
    PYTHONPATH=. .venv/bin/python scripts/run_adtstr_mdb.py [--original N]"""
import argparse
import pathlib
import pickle
import sys
import time

import librosa
import soundfile as sf
import torch

from backend.detection import adtstr

ap = argparse.ArgumentParser()
ap.add_argument("--original", type=int, default=5)
a = ap.parse_args()
songs = sorted(pathlib.Path("data/benchmarks/mdb/audio/full_mix").glob("*_MIX.wav"), key=lambda p: sf.info(str(p)).duration)
fast_dir, orig_dir = pathlib.Path("outputs/adtstr"), pathlib.Path("outputs/adtstr_orig")
fast_dir.mkdir(parents=True, exist_ok=True)
orig_dir.mkdir(parents=True, exist_ok=True)
tr = adtstr.load()
for i, p in enumerate(songs):
    y, _ = librosa.load(str(p), sr=tr.sample_rate, mono=True)
    out = fast_dir / f"{p.stem}.hits.pkl"
    if not out.exists():
        t = time.time()
        hits = adtstr.transcribe(y, tr.sample_rate)
        out.write_bytes(pickle.dumps([(float(o), int(c), float(v)) for o, c, v in hits]))
        print(f"fast {p.stem}: {len(y)/tr.sample_rate:.0f}s audio in {time.time()-t:.0f}s, {len(hits)} hits", flush=True)
    out = orig_dir / f"{p.stem}.hits.pkl"
    if i < a.original and not out.exists():
        t = time.time()
        notes = tr._transcribe_waveform(torch.from_numpy(y)[None]).numpy()
        out.write_bytes(pickle.dumps([(float(n[0]), int(n[2]), float(n[3])) for n in notes]))
        print(f"original {p.stem}: {len(y)/tr.sample_rate:.0f}s audio in {time.time()-t:.0f}s, {len(notes)} hits", flush=True)
sys.exit(0)
