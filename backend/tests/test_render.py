import os
import shutil
import subprocess
import xml.dom.minidom

import pytest

from backend.notation.drum_musicxml import ROCK_BAR, measures_to_xml


def test_musicxml_wellformed_and_bar_count():
    doc = xml.dom.minidom.parseString(measures_to_xml([ROCK_BAR] * 4))
    assert len(doc.getElementsByTagName("measure")) == 4


@pytest.mark.skipif(not shutil.which("mscore3"), reason="MuseScore missing")
def test_musescore_renders_pdf(tmp_path):
    src, out = tmp_path / "t.musicxml", tmp_path / "t.pdf"
    src.write_text(measures_to_xml([ROCK_BAR] * 4))
    subprocess.run(["mscore3", "-o", out, src], check=True, timeout=120,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.read_bytes()[:4] == b"%PDF"
