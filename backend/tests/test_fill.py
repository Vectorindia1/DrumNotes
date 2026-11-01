import numpy as np

from backend.detection import articulation as art
from backend.events import Event
from backend.rhythm.fill import fill_dropouts
from backend.rhythm.grid import Grid, _extend
from backend.validation import synth

BPM, BARS = 115.0, 24
SL = 60 / BPM / 4
GROOVE = {"kick": (0, 6, 10), "snare": (4, 12)}
BREAK = range(12, 14)  # two bars with no kick or snare at all (a break / fill)


def groove(drop=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int((BARS * 16 * SL + 1) * synth.SR)
    x = rng.standard_normal(n).astype(np.float32) * 0.002
    ev, truth = [], set()
    for b in range(BARS):
        if b in BREAK:
            continue
        for inst, slots in GROOVE.items():
            for s in slots:
                a = round((b * 16 + s) * SL * synth.SR)
                v = (synth._kick if inst == "kick" else synth._snare)(rng) * (0.8 + 0.4 * rng.random())
                x[a:a + len(v)] += v[: n - a]
                truth.add((inst, b, s))
    for inst, b, s in sorted(truth):
        if rng.random() >= drop:
            ev.append(Event("", (b * 16 + s) * SL, 0, inst, 0.7, {}, bar=b, slot=s))
    grid = Grid(_extend([0.0], 60 / BPM, BARS * 16 * SL + 1, 4), 4, "hint", BPM, [])
    return ev, truth, grid, art.Env(art.band_power(x))


def filled(ev, grid, env):
    out = set()
    for inst in GROOVE:
        out |= {(inst, e.bar, e.slot) for e in fill_dropouts(ev, grid, env, BARS, inst)}
    return out


def test_dropouts_inside_the_groove_are_filled():
    ev, truth, grid, env = groove(drop=0.25)
    missing = truth - {(e.instrument, e.bar, e.slot) for e in ev}
    got = filled(ev, grid, env)
    assert len(missing) > 15
    assert len(got & missing) >= 0.85 * len(missing), (len(got & missing), len(missing))  # most dropouts recovered
    assert not (got - truth), sorted(got - truth)[:8]  # and nothing invented


def test_a_break_where_the_audio_is_quiet_is_never_filled():
    ev, _, grid, env = groove(drop=0.25)
    assert not any(b in BREAK for _, b, _ in filled(ev, grid, env))


def test_a_clean_groove_is_left_alone():
    ev, _, grid, env = groove(drop=0.0)
    assert filled(ev, grid, env) == set()


def test_filled_hits_say_where_they_came_from():
    ev, _, grid, env = groove(drop=0.25)
    new = fill_dropouts(ev, grid, env, BARS, "kick")
    assert new and all(e.provenance == "filled-from-pattern" and e.articulation_features["pattern_share"] >= 0.5 for e in new)


def test_no_pattern_no_fill():
    ev, _, grid, env = groove(drop=0.0)
    ev = [e for e in ev if e.bar < 3]  # too few bars to call anything a pattern
    assert fill_dropouts(ev, grid, env, BARS, "kick") == []
