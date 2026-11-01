import random
import re

from backend.notation.drum_musicxml import KIT, bars_to_musicxml
from backend.validation.musicxml import validate_musicxml


def _write(tmp_path, bars, **kw):
    p = tmp_path / "s.musicxml"
    p.write_text(bars_to_musicxml(bars, **kw))
    return p


def test_random_bars_always_fill_every_voice(tmp_path):
    rng = random.Random(0)
    bars = [[{i for i in KIT if rng.random() < 0.25} for _ in range(16)] for _ in range(60)]
    assert validate_musicxml(_write(tmp_path, bars), expected_bars=60) == []


def test_other_meters(tmp_path):
    bars = [[{"kick"} if s % 4 == 0 else set() for s in range(12)]] * 3
    assert validate_musicxml(_write(tmp_path, bars, beats_per_bar=3), expected_bars=3) == []


def test_simultaneous_events_become_chords_and_no_tuplets(tmp_path):
    bar = [set() for _ in range(16)]
    bar[0] = {"kick", "snare", "hihat"}
    xml = _write(tmp_path, [bar]).read_text()
    assert xml.count("<chord/>") == 1  # snare+hihat chord in voice 1; kick alone in voice 2
    assert "time-modification" not in xml and "<tuplet" not in xml


def test_validator_catches_short_bar(tmp_path):
    p = _write(tmp_path, [[set() for _ in range(16)]])
    p.write_text(re.sub(r"<duration>16</duration>", "<duration>12</duration>", p.read_text()))
    assert validate_musicxml(p) != []


def test_compound_meters_write_valid_bars(tmp_path):
    from backend.notation.drum_musicxml import bars_to_musicxml
    from backend.validation.musicxml import validate_musicxml
    bars = [[{"hihat", "kick"} if s % 2 == 0 else set() for s in range(12)], [{"snare"} if s == 6 else set() for s in range(12)]]
    for sig, beat_type, mark in (("6/8", "8", "55"), ("12/8", "8", "109"), ("3/4", "4", "82")):
        xml = bars_to_musicxml(bars, "t", 82.0, 3, sig)
        assert f"<beat-type>{beat_type}</beat-type>" in xml and f"<per-minute>{mark}</per-minute>" in xml
        p = tmp_path / f"{sig.replace('/', '_')}.musicxml"
        p.write_text(xml)
        assert validate_musicxml(p, expected_bars=2) == [], sig
