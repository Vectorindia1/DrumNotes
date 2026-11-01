"""Fetch the free one-shot drum/percussion samples used to train the splash / china / tambourine labels.

    PYTHONPATH=. .venv/bin/python scripts/fetch_samples.py [--list]     # -> data/samples/<class>/*.wav (git-ignored)

Sources (both allow commercial use):
  CrocellKit 1.1 (DrumGizmo), CC BY 4.0, real acoustic kit: 2 chinas, 2 splashes, 3 crashes, ride, hi-hat.
      5.5 GB zip; only the needed files are read out of it with HTTP range requests (a few hundred MB).
  VSCO-2 Community Edition, CC0: three tambourines, two sets of sleigh bells.
  Soundfont drum kits (standard/room/power/jazz/brush): MuseScore General and TimGM6mb (already on the machine)
      and GeneralUser GS v2 (downloaded to data/soundfonts; its licence allows commercial use, sample origins
      "not 100 % certain" per its author). Identical renders are kept once.
Attribution for CrocellKit: "CrocellKit by DrumGizmo (drumgizmo.org), CC BY 4.0".
"""
import io
import pathlib
import re
import struct
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import soundfile as sf

OUT = pathlib.Path("data/samples")
CROCELL = "https://drumgizmo.org/kits/CrocellKit/CrocellKit1_1.zip"
VSCO = "https://raw.githubusercontent.com/sgossner/VSCO-2-CE/master/"


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def crocell_members():
    return zipfile.ZipFile(io.BufferedReader(HttpFile(CROCELL), buffer_size=1 << 18))


SECONDS = 0.8  # of each hit: enough for the onset spectrum and the 0.4-s decay features
CROCELL_CLASSES = {"ChinaL": "china", "ChinaR": "china", "SplashL": "splash", "SplashR": "splash",
                   "CrashL": "crash", "CrashR": "crash", "CrashRXtra": "crash", "RideR": "ride", "RideRBell": "ride",
                   "HihatClosed": "hihat", "HihatSemiOpen": "hihat_open", "HihatOpen": "hihat_open"}
PER_INSTRUMENT = 10  # velocity layers fetched per cymbal (of 9-48)
MICS = ("AmbLeft", "AmbRight", "OHLeft", "OHRight", "OHCenter")  # room + overheads: how cymbals sit in a mix
VSCO_FILES = {"tambourine": [f"Percussion/Tamb1-Hit_v{v}_rr{r}_Sum.wav" for v in (1, 2) for r in (1, 2)]
              + [f"Percussion/Tamb1-Shake_v1_rr{r}_Sum.wav" for r in (1, 2)]
              + [f"Miscellania Raw/Misc 2/tamb2_{i}.wav" for i in range(1, 5)]
              + [f"Miscellania Raw/Misc 2/tamb3_{i}.wav" for i in range(1, 9)],
              "sleighbells": [f"Miscellania Raw/Misc 2/sleighbell1_hit_{x}.wav" for x in ("1", "2", "3", "loud_1", "loud_2", "quiet")]
              + [f"Miscellania Raw/Misc 2/sleighbell2_hit_{x}.wav" for x in ("1", "2", "loud")]
              + [f"Percussion/Sleighbells_Hit_v1_rr{r}_Mid.wav" for r in (1, 2)]}


def wav_head(raw):
    """Decode the start of a (possibly truncated) PCM WAV: returns (float32 frames x channels, sr)."""
    pos, fmt = 12, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", raw[pos + 8:pos + 24])
            fmt = (ch, sr, bits)
        elif cid == b"data":
            ch, sr, bits = fmt
            data = raw[pos + 8:pos + 8 + size]
            w = bits // 8
            n = len(data) // (w * ch)
            b = np.frombuffer(data[:n * w * ch], np.uint8).reshape(n * ch, w)
            if w == 3:
                x = (b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8 | b[:, 2].astype(np.int8).astype(np.int32) << 16) / 2 ** 23
            elif w == 2:
                x = b.copy().view("<i2")[:, 0] / 2 ** 15
            else:
                x = b.copy().view("<f4")[:, 0] if bits == 32 else b.copy().view("<i4")[:, 0] / 2 ** 31
            return x.reshape(n, ch).astype(np.float32), sr
        pos += 8 + size + (size & 1)
    raise ValueError("no PCM data")


def fetch_crocell():
    z = crocell_members()
    for inst, cls in CROCELL_CLASSES.items():
        xml = z.read(f"CrocellKit/{inst}/{inst}.xml").decode()
        files = sorted(set(re.findall(r'file="(samples/[^"]+)"', xml)), key=lambda f: int(re.search(r"(\d+)-", f).group(1)))
        files = [files[i] for i in sorted({round(j) for j in np.linspace(0, len(files) - 1, PER_INSTRUMENT)})]  # soft..loud
        for f in files:
            dest = OUT / cls / f"crocell_{inst}_{pathlib.Path(f).stem}.wav"
            if dest.exists():
                continue
            chans = [int(c) - 1 for name, ff, c in re.findall(r'channel="([^"]+)" file="([^"]+)" filechannel="(\d+)"', xml)
                     if ff == f and name in MICS]
            with z.open(f"CrocellKit/{inst}/{f}") as fh:
                head = fh.read(64)
                raw = head + fh.read(int(SECONDS * 48000 * 3 * 15))  # 24-bit, 15 mics  # generous upper bound; decoded length is cut below
            x, sr = wav_head(raw)
            y = x[: int(SECONDS * sr), chans].mean(1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, y / (np.abs(y).max() + 1e-9) * 0.9, sr)
            print("ok", dest, sr, flush=True)


def fetch_vsco():
    for cls, files in VSCO_FILES.items():
        for f in files:
            dest = OUT / cls / f"vsco_{pathlib.Path(f).stem}.wav"
            if dest.exists():
                continue
            with urllib.request.urlopen(VSCO + urllib.parse.quote(f), timeout=120) as r:
                x, sr = sf.read(io.BytesIO(r.read()), always_2d=True)
            y = x[: int(SECONDS * sr)].mean(1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, y / (np.abs(y).max() + 1e-9) * 0.9, sr)
            print("ok", dest, sr, flush=True)


GENERALUSER = "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2"
GM_NOTES = {55: "splash", 52: "china", 54: "tambourine", 49: "crash", 57: "crash", 51: "ride", 59: "ride", 53: "ride",
            42: "hihat", 46: "hihat_open"}


def fetch_soundfonts():
    import hashlib

    from backend.player.score_audio import SOUNDFONTS, render
    gu = pathlib.Path("data/soundfonts/GeneralUser-GS.sf2")
    if not gu.exists():
        gu.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GENERALUSER, gu)
    seen = set()
    for font in [*SOUNDFONTS, str(gu)]:
        if not pathlib.Path(font).exists():
            continue
        for kit in ("standard", "room", "power", "jazz", "brush"):
            for note, cls in GM_NOTES.items():
                for vel in (60, 90, 120):
                    dest = OUT / cls / f"sf_{pathlib.Path(font).stem}_{kit}_{note}_{vel}.wav"
                    tmp = OUT / "_tmp.wav"
                    try:
                        render([(0.0, note, vel)], tmp, kit=kit, tail_s=SECONDS, soundfont=font)
                    except RuntimeError:
                        continue  # kit not in this soundfont
                    x, sr = sf.read(tmp)
                    x = x.mean(1) if x.ndim > 1 else x
                    x = x[: int(SECONDS * sr)]
                    if np.abs(x).max() < 1e-4:
                        continue
                    h = hashlib.md5((x / np.abs(x).max()).round(3).tobytes()).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(dest, x / np.abs(x).max() * 0.9, sr)
    (OUT / "_tmp.wav").unlink(missing_ok=True)


if __name__ == "__main__":
    if "--list" in sys.argv:
        for i in crocell_members().infolist():
            print(i.file_size, i.filename)
        raise SystemExit
    fetch_vsco()
    fetch_crocell()
    fetch_soundfonts()
    for d in sorted(OUT.iterdir()):
        print(d.name, len(list(d.glob("*.wav"))))
