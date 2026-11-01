import os
import shutil
import subprocess
import xml.dom.minidom

import pytest

from backend.notation.drum_musicxml import bars_to_musicxml


def _rock_bar():
    bar = [set() for _ in range(16)]
    for s in range(0, 16, 2):
        bar[s].add("hihat")
    for s in (0, 8, 10):
        bar[s].add("kick")
    for s in (4, 12):
        bar[s].add("snare")
    return bar


def test_musicxml_wellformed_and_bar_count():
    doc = xml.dom.minidom.parseString(bars_to_musicxml([_rock_bar()] * 4))
    assert len(doc.getElementsByTagName("measure")) == 4


@pytest.mark.skipif(not shutil.which("mscore3"), reason="MuseScore missing")
def test_musescore_renders_pdf(tmp_path):
    src, out = tmp_path / "t.musicxml", tmp_path / "t.pdf"
    src.write_text(bars_to_musicxml([_rock_bar()] * 4))
    subprocess.run(["mscore3", "-o", out, src], check=True, timeout=120,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.read_bytes()[:4] == b"%PDF"
