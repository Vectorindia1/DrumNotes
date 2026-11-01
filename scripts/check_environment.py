"""Smoke test: torch CPU, WAV + MP3 decode (incl. sample-0 onset), tool presence."""
import pathlib
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch

sr = 44100
x = np.zeros(sr * 2, dtype=np.float32)
for t in (0, 1, 22050, 44100):  # hit at sample 0 and 1 must survive decode
    x[t : t + 200] = np.hanning(200).astype(np.float32)
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    sf.write(d / "t.wav", x, sr)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", d / "t.wav", d / "t.mp3"], check=True)
    for f in ("t.wav", "t.mp3"):
        y, s = librosa.load(d / f, sr=None, mono=True)
        assert s == sr and abs(len(y) - len(x)) < 5000, (f, s, len(y))
        assert y[:200].max() > 0.1, f"{f}: sample-0 event lost"
        print(f, "OK", s, len(y))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for b in ("ffmpeg", "mscore3", "musescore3", "mscore", "xvfb-run", "pdftoppm"):
    print(b, shutil.which(b) or "MISSING")
