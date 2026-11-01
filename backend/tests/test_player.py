import json
import pathlib
import shutil
import subprocess

import pytest

JS = pathlib.Path(__file__).parents[1] / "player" / "viewer.js"
SCRIPT = """
const M = require(process.argv[1]);
const D = {
  measures: [
    {t: 0, page: 0, x: 100, y: 50, w: 200, h: 60, stops: {t: [0, 1, 1.999], x: [120, 200, 300]}},
    {t: 2, page: 0, x: 300, y: 50, w: 200, h: 60, stops: {t: [2, 3, 3.999], x: [320, 400, 500]}},
    {t: 4, page: 1, x: 100, y: 80, w: 200, h: 60, stops: {t: [4, 5.999], x: [120, 300]}}],
  beatsScore: [0, 1, 2, 3, 4, 5],
  sources: {score: {latency: 0.04}, orig: {beats: [0.5, 1.7, 2.9, 4.1, 5.3, 6.5]}}};   // original runs 1.2 s per beat, 0.5 s late
const out = {
  mid: M.positionAt(D, 0.5), later: M.positionAt(D, 3.5), nextPage: M.positionAt(D, 4.5),
  toScore: M.toScoreTime(D, "orig", 2.9), toScoreMid: M.toScoreTime(D, "orig", 2.3),
  back: M.fromScoreTime(D, "orig", 2.0), scoreIdentity: M.toScoreTime(D, "score", 1.234), scoreBack: M.fromScoreTime(D, "score", 1.0),
  beforeStart: M.toScoreTime(D, "orig", 0.1), afterEnd: M.toScoreTime(D, "orig", 99)};
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(not shutil.which("node"), reason="node missing")
def test_playhead_position_maths():
    r = subprocess.run(["node", "-e", SCRIPT, str(JS)], capture_output=True, text=True, check=True)
    o = json.loads(r.stdout)
    assert o["mid"]["bar"] == 0 and o["mid"]["x"] == pytest.approx(160)  # halfway between stops 120 and 200
    assert o["later"]["bar"] == 1 and o["later"]["x"] == pytest.approx(450, abs=0.2)
    assert o["nextPage"]["page"] == 1 and o["nextPage"]["y"] == 80  # carries across the page turn
    assert o["toScore"] == pytest.approx(2.0)
    assert o["toScoreMid"] == pytest.approx(1 + (2.3 - 1.7) / 1.2)
    assert o["back"] == pytest.approx(2.9) and o["scoreIdentity"] == pytest.approx(1.194) and o["scoreBack"] == pytest.approx(1.04)  # latency 40 ms
    assert o["beforeStart"] == 0 and o["afterEnd"] == 5  # clamped, never extrapolated


@pytest.mark.skipif(not shutil.which("mscore3") or not shutil.which("ffmpeg"), reason="MuseScore/ffmpeg missing")
def test_builder_end_to_end_on_a_small_score(tmp_path):
    import numpy as np

    from backend.notation.drum_musicxml import bars_to_musicxml
    from backend.player.build import build_player
    from backend.rhythm.grid import Grid, _extend

    bars = []
    for b in range(10):
        bar = [set() for _ in range(16)]
        for s in range(0, 16, 2):
            bar[s].add("hihat")
        bar[0].add("kick")
        bar[8].add("snare_ghost" if b % 2 else "snare")
        bars.append(bar)
    xml = tmp_path / "s.musicxml"
    xml.write_text(bars_to_musicxml(bars, "t", 120))
    grid = Grid(_extend([0.0], 0.5, 20.0, 4), 4, "hint", 120.0, [])
    wav = tmp_path / "o.wav"
    import soundfile as sf
    sf.write(wav, np.zeros(44100 * 20, np.float32), 44100)
    idx = build_player(xml, tmp_path / "player", grid, bars, "t", 120.0, {"original": ("Original", wav)})
    data = json.loads(idx.parent.joinpath("data.js").read_text().removeprefix("window.SCORE = ").removesuffix(";"))
    ms = data["measures"]
    assert len(ms) == 10 and [m["t"] for m in ms] == pytest.approx([i * 2.0 for i in range(10)], abs=0.01)
    page_w = data["pages"][0]["w"]
    for m in ms:  # geometry is in page pixels and inside the page
        assert 0 <= m["x"] and m["x"] + m["w"] <= page_w + 1 and all(m["stops"]["x"][i] <= m["stops"]["x"][i + 1] for i in range(len(m["stops"]["x"]) - 1))
        assert m["x"] - 1 <= m["stops"]["x"][0] and m["stops"]["x"][-1] <= m["x"] + m["w"] + 1
    assert len(data["beatsScore"]) == 40 and set(data["sources"]) == {"score", "original"}
    assert all(idx.parent.joinpath(f).exists() for f in ("score.flac", "original.flac", "page1.png", "viewer.js", "index.html"))


@pytest.mark.skipif(not shutil.which("node"), reason="node missing")
def test_smoothed_clock_interpolates_between_coarse_updates():
    script = """
    const M = require(process.argv[1]); const st = {anchor: null};
    const out = [M.smoothTime(st, 10.0, 100.00, false, 1), M.smoothTime(st, 10.0, 100.02, false, 1),
                 M.smoothTime(st, 10.0, 100.50, false, 1),            // stalled clock: capped at +80 ms
                 M.smoothTime(st, 10.04, 100.52, false, 1),           // next coarse update re-anchors
                 M.smoothTime(st, 10.04, 100.54, false, 0.5),         // half speed
                 M.smoothTime(st, 33.0, 100.56, true, 1)];            // paused: raw time
    console.log(JSON.stringify(out));"""
    out = json.loads(subprocess.run(["node", "-e", script, str(JS)], capture_output=True, text=True, check=True).stdout)
    assert out == pytest.approx([10.0, 10.02, 10.08, 10.04, 10.05, 33.0])
