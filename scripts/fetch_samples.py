"""Fetch the free one-shot drum/percussion samples used to train the splash / china / tambourine labels.

    PYTHONPATH=. .venv/bin/python scripts/fetch_samples.py [--list]     # -> data/samples/<class>/*.wav (git-ignored)

Sources (both allow commercial use):
  CrocellKit 1.1 (DrumGizmo), CC BY 4.0, real acoustic kit: 2 chinas, 2 splashes, 3 crashes, ride, hi-hat.
      5.5 GB zip; only the needed files are read out of it with HTTP range requests (a few hundred MB).
  VSCO-2 Community Edition, CC0: three tambourines, two sets of sleigh bells.
Attribution for CrocellKit: "CrocellKit by DrumGizmo (drumgizmo.org), CC BY 4.0".
"""
import io
import pathlib
import sys
import urllib.parse
import urllib.request
import zipfile

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
    return zipfile.ZipFile(io.BufferedReader(HttpFile(CROCELL), buffer_size=1 << 20))


if __name__ == "__main__":
    z = crocell_members()
    if "--list" in sys.argv:
        for i in z.infolist():
            print(i.file_size, i.filename)
        raise SystemExit
