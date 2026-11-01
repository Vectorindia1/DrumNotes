import numpy as np

from backend.detection import articulation as art
from backend.events import Event
from backend.reconstruction.bars import build_bars
from backend.rhythm.grid import Grid, _extend
from backend.rhythm.hats import regularize_hats
from backend.validation import synth

BPM, BARS = 115.0, 24
SL = 60 / BPM / 4


def song(odd_gain, even_gain=1.0, hat_bars=range(BARS), seed=0):
    """Audio + the true hat slots. Hats on every 16th; 8th positions at even_gain, the ticks between at
    odd_gain. Kick on slot 0, snare on 4 and 12 (both even) and on 7 (an odd slot: a bleed trap)."""
    rng = np.random.default_rng(seed)
    n = int((BARS * 16 * SL + 1) * synth.SR)
    x = rng.standard_normal(n).astype(np.float32) * 0.002
    truth = set()

    def put(b, s, voice, gain=1.0):
        a = round((b * 16 + s) * SL * synth.SR)
        v = voice(rng) * gain
        x[a:a + len(v)] += v[: n - a]
    for b in range(BARS):
        put(b, 0, synth._kick)
        for s in (4, 12, 7):
            put(b, s, synth._snare, 0.9)
        if b in hat_bars:
            for s in range(16):
                put(b, s, synth._hihat, even_gain if s % 2 == 0 else odd_gain)
                truth.add((b, s))
    return x, truth


def setup(odd_gain, drop=0.0, false_pos=(), seed=1, **kw):
    x, truth = song(odd_gain, **kw)
    grid = Grid(_extend([0.0], 60 / BPM, BARS * 16 * SL + 1, 4), 4, "hint", BPM, [])
    env = art.Env(art.band_power(x))
    rng = np.random.default_rng(seed)
    ev = []

    def add(inst, b, s, conf=0.6):
        ev.append(Event("", (b * 16 + s) * SL, 0, inst, conf, {}, bar=b, slot=s))
    for b in range(BARS):
        add("kick", b, 0)
        for s in (4, 12, 7):
            add("snare", b, s)
    for (b, s) in sorted(truth):
        if rng.random() >= drop:
            add("hihat", b, s)
    for (b, s) in false_pos:
        add("hihat", b, s, 0.5)
    return ev, grid, env


def notated(ev, new):
    return {(e.bar, e.slot) for e in [*ev, *new] if e.instrument == "hihat" and e.notated}


def interior(slots, lo=4, hi=BARS - 4):
    return {(b, s) for b, s in slots if lo <= b < hi}


def test_eighths_with_quiet_sixteenth_ticks_become_eighths_and_dropouts_are_filled():
    ev, grid, env = setup(odd_gain=0.45, drop=0.25)
    new, info = regularize_hats(ev, grid, env, BARS, "auto")
    got = interior(notated(ev, new))
    want = {(b, s) for b in range(4, BARS - 4) for s in range(0, 16, 2)}
    assert got == want, sorted(got ^ want)[:10]  # exactly the eighths: ticks gone, snare-bleed slot 7 gone, dropouts filled
    assert info["recovered_from_audio"] > 0 and info["filled_from_pattern"] > 0  # off-drum dropouts from audio, shared slots from the pattern
    assert info["dropped_quiet_ticks"] > 0 and info["bars_eighths_only"] > BARS * 0.7


def test_snare_bleed_on_an_in_between_slot_is_not_counted_as_a_hat_in_eighth_sections():
    ev, grid, env = setup(odd_gain=0.45)
    new, _ = regularize_hats(ev, grid, env, BARS, "auto")
    assert not any((b, 7) in notated(ev, new) for b in range(4, BARS - 4))


def test_true_flat_sixteenths_are_kept():
    ev, grid, env = setup(odd_gain=0.95, drop=0.25)
    new, info = regularize_hats(ev, grid, env, BARS, "auto")
    got = interior(notated(ev, new))
    assert len({s for _, s in got if s % 2 == 1 and s != 7}) >= 6  # the odd ticks survive
    assert len(got) >= 0.9 * 16 * 16 and info["bars_sixteenths"] > BARS * 0.7


def test_modes_override_the_automatic_choice():
    ev, grid, env = setup(odd_gain=0.95)
    new, _ = regularize_hats(ev, grid, env, BARS, "eighths")
    assert not any(s % 2 for _, s in interior(notated(ev, new)))
    ev, grid, env = setup(odd_gain=0.45)
    new, _ = regularize_hats(ev, grid, env, BARS, "sixteenths")
    assert sum(1 for _, s in interior(notated(ev, new)) if s % 2) >= 0.8 * 20 * 8


def test_detections_with_no_energy_behind_them_are_dropped_but_kept_in_the_data():
    ev, grid, env = setup(odd_gain=0.95, hat_bars=range(12), false_pos=[(b, s) for b in range(14, 22) for s in range(1, 16, 2)])
    new, info = regularize_hats(ev, grid, env, BARS, "auto")
    assert not any(b >= 14 for b, _ in notated(ev, new)), "hats reported in silent bars must not be notated"
    dead = [e for e in ev if e.instrument == "hihat" and e.bar >= 14]
    assert dead and all(not e.notated and e.articulation_features["dropped"] for e in dead)  # still in the list, with a reason
    assert info["dropped_no_energy"] + info["dropped_quiet_ticks"] == len(dead)  # all accounted for, none silently lost


def test_build_bars_and_midi_skip_dropped_hits():
    ev, grid, env = setup(odd_gain=0.45)
    new, _ = regularize_hats(ev, grid, env, BARS, "auto")
    bars, _ = build_bars([*ev, *new], BARS, 16)
    assert all("hihat" not in bars[b][1] for b in range(BARS))  # slot 1 ticks are not in the score
    assert all("hihat" in bars[b][2] for b in range(4, BARS - 4))


def test_a_short_clip_is_left_alone():
    ev, grid, env = setup(odd_gain=0.95)
    new, info = regularize_hats(ev, grid, env, 3, "auto")
    assert new == [] and "skipped" in info
