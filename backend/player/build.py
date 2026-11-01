"""Build a static, serverless score player: page images, MuseScore's own measure/segment positions,
the score rendered to audio by MuseScore (what the notation says), plus optional original/stem audio
mapped onto the score through the beat grid. Open <out>/index.html in a browser."""
import json
import pathlib
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from backend.player.score_audio import render, score_notes

HERE = pathlib.Path(__file__).parent
DPI = 110
SCORE_AUDIO_LATENCY_S = 0.003  # measured: standard-kit samples start 1.5-4 ms after the notated time


def _mscore(out_file, musicxml, *extra):
    import os
    r = subprocess.run(["mscore3", *extra, "-o", str(out_file), str(musicxml)], check=False, capture_output=True,
                       text=True, timeout=900, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    if r.returncode != 0:
        raise RuntimeError(f"mscore3 {out_file.name} failed: {r.stderr[-400:]}")


def _parse_positions(path):
    """MuseScore .mpos/.spos -> ({elid: {x,y,w,h,page}}, [(ms, elid)])."""
    root = ET.parse(path).getroot()
    els = {
        e.get("id"): {"x": float(e.get("x")), "y": float(e.get("y")), "w": float(e.get("sx")),
                      "h": float(e.get("sy")), "page": int(e.get("page"))}
        for e in root.iter("element")
    }
    ev = sorted((int(e.get("position")), e.get("elid")) for e in root.iter("event"))
    return els, ev


def measures_and_stops(mpos, spos, tempo, n_bars, page_px_width, beats_per_bar=4):
    """Per measure: start time (s), box, and (time, x) stops from segment positions, converted to
    page-image pixels. MuseScore's position units are not pixels; the scale is recovered from the
    page width (margins are symmetric, so page width = left margin + right edge of the staves)."""
    mels, mev = _parse_positions(mpos)
    sels, sev = _parse_positions(spos)
    unit = (min(b["x"] for b in mels.values()) + max(b["x"] + b["w"] for b in mels.values())) / page_px_width
    for d in (mels, sels):
        for b in d.values():
            for k in ("x", "y", "w", "h"):
                b[k] /= unit
    beat = 60.0 / tempo
    starts = [t / 1000 for t, _ in mev]
    boxes = [mels[i] for _, i in mev]
    ends = starts[1:] + [starts[-1] + beats_per_bar * beat]
    out = []
    for k, (t0, t1, b) in enumerate(zip(starts, ends, boxes, strict=True)):
        segs = [(t / 1000, sels[i]["x"]) for t, i in sev if t0 - 1e-6 <= t / 1000 < t1 - 1e-6 and sels[i]["page"] == b["page"]
                and b["y"] - 5 <= sels[i]["y"] + sels[i]["h"] / 2 <= b["y"] + b["h"] + 5]
        stops = {}
        for t, x in segs:  # keep first x for identical times
            stops.setdefault(round(t, 4), x)
        ts, xs = [t0], [min(stops.values(), default=b["x"]) if not stops else stops.get(round(t0, 4), min(stops.values()))]
        for t, x in sorted(stops.items()):
            if t > ts[-1] + 1e-4:
                ts.append(t)
                xs.append(x)
        if t1 > ts[-1]:
            ts.append(t1 - 1e-3)
            xs.append(b["x"] + b["w"])
        pad = 0.35 * b["h"]  # cover ledger-line cymbals above and the kick voice below
        out.append({"t": t0, "page": b["page"], "x": b["x"], "y": b["y"] - pad, "w": b["w"], "h": b["h"] + 2 * pad,
                    "stops": {"t": ts, "x": xs}})
    return out[: max(n_bars, 0) or None]


def build_player(musicxml, out_dir, grid, bars, title, tempo, audio_sources, kit="standard"):
    """bars: the notated bars (sets of note keys per slot). audio_sources: {name: (label, path)} extra
    recordings aligned to the score by `grid`. The score audio is the notation played on a sampled kit."""
    n_bars = len(bars)
    out = pathlib.Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = out / "_work"
    work.mkdir()
    _mscore(work / "score.png", musicxml, "-r", str(DPI))
    _mscore(work / "score.mpos", musicxml, "-r", str(DPI))
    _mscore(work / "score.spos", musicxml, "-r", str(DPI))
    pages = []
    for i, png in enumerate(sorted(work.glob("score-*.png")), 1):
        w, h = struct.unpack(">II", png.read_bytes()[16:24])  # IHDR width/height
        shutil.move(png, out / f"page{i}.png")
        pages.append({"src": f"page{i}.png", "w": w, "h": h})
    bpb = grid.beats_per_bar
    measures = measures_and_stops(work / "score.mpos", work / "score.spos", tempo, n_bars, pages[0]["w"], bpb)
    bar_starts = [m["t"] for m in measures] + [measures[-1]["t"] + bpb * 60.0 / tempo]
    beats_score = [bar_starts[i] + j * (bar_starts[i + 1] - bar_starts[i]) / bpb for i in range(len(measures)) for j in range(bpb)]
    try:
        notes = score_notes(bars, bar_starts)
        render(notes, out / "score.flac", kit=kit)
        sources = {"score": {"label": f"Score on a sampled {kit} drum kit", "file": "score.flac", "latency": SCORE_AUDIO_LATENCY_S}}
    except (ImportError, OSError, RuntimeError) as e:  # no fluidsynth/soundfont: fall back to MuseScore's tonal render
        print(f"WARNING: sampled drum kit unavailable ({e}); using MuseScore's piano-like render instead")
        _mscore(work / "score.wav", musicxml)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", work / "score.wav", "-b:a", "112k", out / "score.mp3"], check=True)
        sources = {"score": {"label": "Score (MuseScore fallback: piano-like, NOT a drum kit)", "file": "score.mp3", "latency": 0.036}}
    for name, (label, path) in audio_sources.items():
        dest = f"{name}.flac"  # lossless: mp3 adds ~25-50 ms of encoder/decoder delay that browsers may not remove
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-c:a", "flac", out / dest], check=True)
        beats = [grid.slot_time(4 * k) for k in range(len(beats_score))]
        assert all(np.diff(beats) > 0), "beat grid not increasing"
        sources[name] = {"label": label, "file": dest, "beats": beats}
    data = {"title": title, "tempo": round(tempo, 1), "pages": pages, "measures": measures,
            "beatsScore": beats_score, "sources": sources}
    (out / "data.js").write_text("window.SCORE = " + json.dumps(data) + ";")
    shutil.copy(HERE / "viewer.html", out / "index.html")
    shutil.copy(HERE / "viewer.js", out / "viewer.js")
    shutil.rmtree(work)
    return out / "index.html"
