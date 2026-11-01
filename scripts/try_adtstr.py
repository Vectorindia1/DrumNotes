"""Run the third-party ADT_STR model (third_party/adt-str, 26 drum classes incl. ride/crash/splash/china,
open/closed/pedal hi-hat, tambourine, shaker, cowbell) on one audio file and print hits per class.

    .venv/bin/python scripts/try_adtstr.py AUDIO [--out DIR]

Evaluation only: nothing in the pipeline uses it yet. Code + weights: huggingface.co/Pierfrancesco/adt-str
(setting-tau-0.8, sha256 f94d1773...ca930b); licence per its GitHub README CC BY-SA 4.0 (not stated on the model card).
"""
import argparse
import collections
import pathlib
import pickle
import resource
import sys
import time

import librosa
import pretty_midi
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "adt-str"
sys.path.insert(0, str(ROOT))
from adt_transcriber import ADTTranscriber
from utils.mapping_utils import MappingUtils

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default="outputs/adtstr")
    a = ap.parse_args()
    t0 = time.time()
    tr = ADTTranscriber.from_pretrained(str(ROOT), variant="setting-tau-0.8", device="cpu")
    y, _ = librosa.load(a.audio, sr=tr.sample_rate, mono=True)
    t1 = time.time()
    paths = tr.transcribe(torch.from_numpy(y)[None], sample_rate=tr.sample_rate, lengths=[len(y)], output_dir=a.out)
    midi_path = pathlib.Path(paths[0] if isinstance(paths, (list, tuple)) else paths)
    print(f"model loaded in {t1 - t0:.1f}s; {len(y) / tr.sample_rate:.0f}s of audio transcribed in {time.time() - t1:.0f}s; "
          f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
    notes = [n for i in pretty_midi.PrettyMIDI(str(midi_path)).instruments for n in i.notes]
    names = MappingUtils().GM_reduced_name_convention
    print(f"MIDI: {midi_path}\nhits by class:")
    for pitch, k in sorted(collections.Counter(n.pitch for n in notes).items(), key=lambda kv: -kv[1]):
        print(f"  {names.get(pitch, pitch):22s} {k}")
    out = pathlib.Path(a.out) / (pathlib.Path(a.audio).stem + ".hits.pkl")
    out.write_bytes(pickle.dumps([(n.start, n.pitch, n.velocity) for n in notes]))
    print("raw hits saved to", out)
