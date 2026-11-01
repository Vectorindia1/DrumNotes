import numpy as np
import pytest
import soundfile as sf

from backend.events import Event
from backend.pipeline import run
from backend.rhythm import tempo as T
from backend.validation.synth import ROCK, SR, render


def song(bpm, bars=48, kick="dotted", jitter=0.0, seed=0):
    """Hits of a 4/4 groove. kick='dotted': 3+3+3+3+4 sixteenths (a pulse at 4/3 of the tempo);
    'four': four on the floor; 'rock': beats 1, 3 and the 'and' of 3."""
    rng = np.random.default_rng(seed)
    sixteenth = 60.0 / bpm / 4
    ev = []

    def add(inst, bar, slot):
        ev.append(Event("", (bar * 16 + slot) * sixteenth + rng.normal(0, jitter), 0, inst, 1.0, {}))
    kicks = {"dotted": (0, 3, 6, 9, 12), "four": (0, 4, 8, 12), "rock": (0, 8, 10)}[kick]
    for b in range(bars):
        for s in kicks:
            add("kick", b, s)
        for s in (4, 12):
            add("snare", b, s)
        for s in range(0, 16, 2):
            add("hihat", b, s)
    return sorted(ev, key=lambda e: e.time_s), bars * 16 * sixteenth + 1


def test_dotted_eighth_kick_does_not_fool_the_estimate():
    """The reported bug: a 114 BPM song whose kick recurs every 3 sixteenths was read as 152 BPM (4/3)."""
    ev, dur = song(114.0, kick="dotted")
    bpm, info = T.estimate_tempo(ev, dur)
    assert bpm == pytest.approx(114.0, rel=0.01)
    assert any(abs(a - 152.0) < 1.0 for a in info["alternatives"])  # the confusion is offered as a one-click alternative


@pytest.mark.parametrize("bpm,kick", [(80, "rock"), (95, "four"), (115, "dotted"), (128, "four"), (140, "rock"), (150, "four")])
def test_tempo_across_grooves(bpm, kick):
    ev, dur = song(float(bpm), kick=kick)
    assert T.estimate_tempo(ev, dur)[0] == pytest.approx(bpm, rel=0.015)


def test_refinement_pins_a_steady_song_to_a_fraction_of_a_percent():
    ev, dur = song(115.0)
    bpm, info = T.estimate_tempo(ev, dur)
    assert abs(bpm - 115.0) < 0.15 and info["grid_coherence"] > 0.9


def test_live_playing_is_still_in_the_right_region():
    ev, dur = song(112.0, jitter=0.025)  # ~25 ms of human timing noise
    assert T.estimate_tempo(ev, dur)[0] == pytest.approx(112.0, rel=0.03)


def test_too_little_to_go_on_returns_nothing():
    ev, dur = song(110.0, bars=1)
    assert T.estimate_tempo(ev, dur) is None
    assert T.estimate_tempo([], 60.0) is None


def test_typed_tempo_is_corrected_only_when_the_drums_clearly_disagree():
    ev, _ = song(115.0)
    bpm, note = T.snap_hint(ev, 114.0)  # the real report: user said 114, the audio is locked to 115
    assert bpm == pytest.approx(115.0, abs=0.15) and "lock to" in note and "114" in note
    assert T.snap_hint(ev, 115.0) == (115.0, None)  # already right: untouched
    assert T.snap_hint(ev, 100.0) == (100.0, None)  # far away (>3 %): the user's choice stands
    rng = np.random.default_rng(3)
    noise = [Event("", float(t), 0, "kick", 1.0, {}) for t in np.sort(rng.uniform(0, 90, 400))]
    assert T.snap_hint(noise, 114.0) == (114.0, None)  # nothing locked: never invent a correction


@pytest.mark.parametrize("bpm", [90, 140])
def test_pipeline_without_a_tempo_finds_it_from_the_audio(tmp_path, bpm):
    audio, _ = render(ROCK, bpm, 16)
    wav = tmp_path / "a.wav"
    sf.write(wav, audio, SR)
    _, report, _ = run(wav, tmp_path / "out")
    assert report["rhythm"]["tempo_bpm"] == pytest.approx(bpm, rel=0.015)
    assert report["rhythm"]["tempo_basis"].startswith("drum pattern") and report["rhythm"]["alternatives_unresolved"]


def compound_song(meter, q=82.0, bars=40, start=0.37):
    """12-slot bars at quarter tempo q (slot = q's 16th). '6/8': kick 1, snare on the 4th eighth (slot 6), hats on
    eighths (the reported 'Yeh Fitoor Mera' pattern). '12/8': shuffle, snare on slots 3 and 9, hats on every slot.
    '3/4': waltz, kick 1, snare on beats 2 and 3, hats on eighths."""
    slot = 60.0 / q / 4
    pat = {"6/8": ({0, 4, 10}, {6}, range(0, 12, 2)), "12/8": ({0, 8}, {3, 9}, range(12)),
           "3/4": ({0}, {4, 8}, range(0, 12, 2))}[meter]
    ev = [Event("", start + (b * 12 + s) * slot, 0, inst, 1.0, {})
          for b in range(bars) for inst, slots in zip(("kick", "snare", "hihat"), pat, strict=True) for s in slots]
    return sorted(ev, key=lambda e: e.time_s), start + bars * 12 * slot + 1


@pytest.mark.parametrize("meter", ["6/8", "12/8", "3/4"])
def test_meter_and_downbeat_from_the_pattern(meter):
    ev, _ = compound_song(meter)
    m, down, _ = T.detect_meter(ev, 82.0)
    assert m == meter
    assert (down - 0.37) / (60 / 82.0 * 3) == pytest.approx(round((down - 0.37) / (60 / 82.0 * 3)), abs=1e-6)  # on a bar line


def test_six_eight_read_as_four_four_triplets_is_relocked():
    """The bug: a 6/8 song was scored as 4/4 at 109.5 (= 82 x 4/3), putting every hat on the wrong slot."""
    ev, _ = compound_song("6/8")
    assert T.relock(ev, 82.0 * 4 / 3) == pytest.approx(82.0, rel=0.002)
    assert T.relock(ev, 55.0 * 1.0, (0.75, 1.5, 0.5)) == pytest.approx(82.0, rel=0.01)  # dotted quarter typed


def test_four_four_stays_four_four():
    for kick in ("dotted", "four", "rock"):
        ev, _ = song(110.0, kick=kick)
        assert T.relock(ev, 110.0) == 110.0
        assert T.detect_meter(ev, 110.0)[0] == "4/4"
