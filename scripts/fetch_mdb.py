"""Download only what the benchmark and label training need from MDB Drums (CC BY-NC-SA 4.0, evaluation /
non-commercial use): the 23 full mixes (~230 MB) and the annotations, into data/benchmarks/mdb (git-ignored).
    .venv/bin/python scripts/fetch_mdb.py"""
import json
import pathlib
import urllib.parse
import urllib.request

API = "https://api.github.com/repos/CarlSouthall/MDBDrums/git/trees/master?recursive=1"
RAW = "https://raw.githubusercontent.com/CarlSouthall/MDBDrums/master/"
DEST = pathlib.Path("data/benchmarks/mdb")

if __name__ == "__main__":
    tree = json.load(urllib.request.urlopen(API, timeout=30))["tree"]
    want = [x["path"] for x in tree if x["type"] == "blob" and x["path"].startswith(("MDB Drums/audio/full_mix/", "MDB Drums/annotations/"))]
    for i, p in enumerate(want, 1):
        out = DEST / p.removeprefix("MDB Drums/")
        if out.exists() and out.stat().st_size:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW + urllib.parse.quote(p), out)
        print(f"{i}/{len(want)} {out}")
    print("done:", len(want), "files in", DEST)
