import numpy as np
import pytest

from backend.events import Event
from backend.reconstruction.bars import build_bars
from backend.rhythm.grid import Grid, _extend, quantize


def _grid(bpm=120.0, offset=0.0, duration=10.0):
    return Grid(_extend([offset], 60 / bpm, duration, 4), 4, "hint", bpm, [])


def _ev(t, inst="kick", conf=1.0):
    return Event("x", t, round(t * 44100), inst, conf, {})


def test_quantization_never_overwrites_raw_time():
    ev = [_ev(0.0), _ev(0.126), _ev(0.49)]
    raw = [e.time_s for e in ev]
    quantize(ev, _grid())
    assert [e.time_s for e in ev] == raw
    assert [(e.bar, e.slot) for e in ev] == [(0, 0), (0, 1), (0, 4)]  # 120 BPM: 16th = 0.125 s
    assert abs(ev[1].quantization_error_s - 0.001) < 1e-4


def test_events_in_second_bar_and_before_anchor():
    g = _grid(120, offset=0.3)  # hint offset = beat 1 of a bar; earlier material is a pickup bar
    ev = [_ev(0.0), _ev(0.3), _ev(2.3)]  # bar = 2.0 s
    quantize(ev, g)
    assert (ev[0].bar, ev[0].slot) == (0, 14)  # 0.3 s = 2.4 sixteenths before the anchor
    assert (ev[1].bar, ev[1].slot) == (1, 0) and (ev[2].bar, ev[2].slot) == (2, 0)


def test_unanchored_grid_starts_at_first_beat_before_zero():
    g = Grid(_extend([0.3], 0.5, 10.0, 4, anchor_is_downbeat=False), 4, "librosa.beat_track", 120, [])
    assert g.beat_times[0] == -0.2 and g.describe()["downbeat"].startswith("not detected")


def test_simultaneous_instruments_kept_and_duplicates_merged():
    ev = [_ev(0.0, "kick"), _ev(0.0, "snare"), _ev(0.004, "kick", conf=0.3)]
    quantize(ev, _grid())
    bars, collisions = build_bars(ev, 1, 16)
    assert bars[0][0] == {"kick", "snare"} and collisions == 1
    assert [e.notated for e in ev] == [True, True, False]


def test_bar_count_covers_duration():
    g = _grid(120)
    assert g.n_bars(8.0) == 4 and g.n_bars(8.01) == 5 and g.n_bars(0.1) == 1
    assert np.all(np.diff(g.beat_times) > 0)


def test_bar_phase_from_backbeat_and_override():
    g = Grid(_extend([0.0], 0.5, 40.0, 4, anchor_is_downbeat=False), 4, "librosa.beat_track", 120, [])
    # kick on tracked beat 2 of each 4, snare on tracked beats 3 and 1 -> beat 1 is tracked beat 2
    ev = []
    for bar in range(8):
        base = bar * 2.0
        ev += [_ev(base + 1.0, "kick"), _ev(base + 1.5, "snare"), _ev(base + 0.5, "snare")]
    from backend.rhythm.grid import fit_bar_phase
    fit_bar_phase(ev, g)
    quantize(ev, g)
    assert all(e.slot == 0 for e in ev if e.instrument == "kick") and "GUESSED" in g.phase_note
    g2 = Grid(_extend([0.0], 0.5, 40.0, 4, anchor_is_downbeat=False), 4, "librosa.beat_track", 120, [])
    fit_bar_phase(ev, g2, forced=0)
    assert g2.beat_shift == 0 and "user-specified" in g2.phase_note


def test_slot_time_extrapolates_and_matches_quantization_with_beat_shift():
    g = Grid(_extend([0.0], 0.5, 10.0, 4, anchor_is_downbeat=False), 4, "librosa.beat_track", 120, [], beat_shift=3)
    assert g.slot_time(0) == -1.5 and g.slot_time(12) == 0.0  # pre-roll of 3 beats before the first tracked beat
    far = g.slot_time(4 * 60)  # far past the tracked range
    assert far > g.slot_time(4 * 59) > 0
    ev = [_ev(0.0), _ev(1.0), _ev(7.77)]
    quantize(ev, g)
    beats = [g.slot_time(4 * k) for k in range(80)]
    assert np.all(np.diff(beats) > 0)
    for e in ev:  # the player's time->bar map must agree with the quantized bar/slot to within rounding
        pos16 = np.interp(e.time_s, beats, range(80)) * 4
        assert abs(pos16 - (e.bar * 16 + e.slot)) <= 0.5 + 1e-6


def test_calibrate_phase_removes_a_late_beat_tracker_bias():
    from backend.rhythm.grid import calibrate_phase
    g = Grid(_extend([0.046], 0.5, 40.0, 4, anchor_is_downbeat=False), 4, "librosa.beat_track", 120, [])
    ev = [_ev(0.125 * j) for j in range(0, 64, 2)]  # 8th-note hits on a 120 BPM grid that starts at exactly 0
    shift = calibrate_phase(ev, g)  # grid started 46 ms late -> must move 46 ms earlier
    assert shift == pytest.approx(-0.046, abs=0.002) and g.phase_shift_s == pytest.approx(shift)
    quantize(ev, g)
    assert max(abs(e.quantization_error_s) for e in ev) < 0.003
    exact = Grid(_extend([0.046], 0.5, 10.0, 4), 4, "hint", 120, [])
    assert calibrate_phase(ev, exact) == 0.0  # a user-supplied exact grid is never nudged
