"""Fill detector dropouts inside a repeating kick/snare groove, only where the AUDIO agrees.

A drummer's kick/snare pattern repeats bar after bar, and the detector (ADTOF) occasionally misses one
hit. A slot is filled when (a) the same slot has this instrument in at least PATTERN_SHARE of the
neighbouring bars and (b) the instrument's own frequency band shows a hit-sized onset there (at least half
as loud as the typical marked hit around it). Fills, breaks and variations where the audio is quiet are
never 'corrected'. Nothing is removed. Added hits are marked provenance="filled-from-pattern".
"""
import numpy as np

from backend.events import Event

BAND = {"kick": "kick", "snare": "snare"}
WINDOW_BARS = 3
PATTERN_SHARE = 0.5
MIN_BARS = 3  # neighbouring playing bars needed to call it a pattern
MIN_PER_BAR = 1  # a bar counts as "playing" if the detector found the instrument in it at all


def _strength(env, band, t):
    return max(env.strength(band, t + d) for d in (-0.008, 0.0, 0.008))


def fill_dropouts(events, grid, env, n_bars, instrument):
    """Returns the list of new events (not appended)."""
    spb = grid.slots_per_bar
    have = {}
    for e in events:
        if e.instrument == instrument and e.grace_of is None and e.notated and e.bar is not None and e.articulation not in ("ghost", "grace"):
            have.setdefault(e.bar, {})[e.slot] = e
    playing = [b for b in range(n_bars) if len(have.get(b, {})) >= MIN_PER_BAR]
    if len(playing) < MIN_BARS + 1:
        return []
    band = BAND[instrument]
    new = []
    for b in range(n_bars):
        win = [x for x in playing if x != b and abs(x - b) <= WINDOW_BARS]
        if len(win) < MIN_BARS or b not in playing and not any(abs(b - x) <= 1 for x in playing):
            continue
        for slot in range(spb):
            if slot in have.get(b, {}):
                continue
            present = sum(slot in have[x] for x in win)
            share = present / len(win)
            if share < PATTERN_SHARE or present < 2:  # one neighbouring bar is not a pattern
                continue
            marked = [_strength(env, band, grid.slot_time(x * spb + slot)) for x in win if slot in have[x]]
            t = grid.slot_time(b * spb + slot)
            s = _strength(env, band, t)
            if s >= 0.5 * np.median(marked):  # hit-sized: at least half of the typical marked hit
                new.append(Event(
                    id="", time_s=round(t, 5), sample=round(t * 44100), instrument=instrument, confidence=0.35,
                    model_output={"source": f"filled from the repeating {instrument} pattern; audio onset agrees"},
                    bar=b, slot=slot, quantization_error_s=0.0, provenance="filled-from-pattern",
                    articulation_features={"pattern_share": round(float(share), 2), "strength_vs_marked": round(float(s / np.median(marked)), 2)}))
    return new
