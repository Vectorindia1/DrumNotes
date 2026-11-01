"""Beat grid + 16th-note quantization (M1: straight 16ths only; swing/tuplets are M3).

Tempo comes from a user hint or librosa beat tracking on the ORIGINAL mix. Beat times are
extended backwards/forwards with the median beat period so events near t=0 and the tail land
on the grid. With a tempo hint AND an offset the grid is exact and the offset is beat 1. With a tempo
hint alone the beat tracker is run at that fixed tempo to find the phase. Which beat is beat 1 is
then guessed from the snare backbeat (fit_bar_phase), which is a heuristic and is reported as such.
Raw timestamps are never modified.
"""
import math
from dataclasses import dataclass

import librosa
import numpy as np

from backend.rhythm.tempo import detect_meter, estimate_tempo, refine, relock, snap_hint

SLOTS_PER_BEAT = 4


class TempoUndetermined(RuntimeError):
    pass


@dataclass
class Grid:
    beat_times: np.ndarray  # extended; beat_times[0] <= 0, last >= duration
    beats_per_bar: int
    method: str  # "hint" | "librosa.beat_track"
    tempo_bpm: float
    tempo_alternatives: list  # unranked half/double hypotheses; M1 does not resolve them
    beat_shift: int = 0  # whole beats prepended so that bar boundaries fall on the guessed downbeat
    phase_note: str = ""
    phase_shift_s: float = 0.0  # grid shifted by this to line up with the measured attacks
    tempo_basis: str = ""
    tempo_note: str | None = None  # e.g. a typed tempo that was corrected to what the drums lock to
    meter: str = ""  # "6/8" / "3/4" when detected from the pattern; "" = beats_per_bar/4

    @property
    def time_signature(self):
        return self.meter or f"{self.beats_per_bar}/4"

    @property
    def marking(self):
        """The tempo as printed on the score (6/8 and 12/8 count dotted quarters)."""
        f = {"6/8": 2 / 3, "12/8": 4 / 3}.get(self.meter)
        return f"dotted quarter = {self.tempo_bpm * f:.0f}" if f else f"quarter = {self.tempo_bpm:.0f}"

    @property
    def slots_per_bar(self):
        return self.beats_per_bar * SLOTS_PER_BEAT

    def beat_position(self, t):
        return float(np.interp(t, self.beat_times, np.arange(len(self.beat_times)))) + self.beat_shift

    def slot_time(self, index):
        """Time of a global 16th-slot index. Linear extrapolation outside the tracked range (e.g. the
        pre-roll created by beat_shift, or bars past the end) at the first/last beat period."""
        pos, bt = index / SLOTS_PER_BEAT - self.beat_shift, self.beat_times
        if pos < 0:
            return float(bt[0] + pos * (bt[1] - bt[0]))
        if pos > len(bt) - 1:
            return float(bt[-1] + (pos - (len(bt) - 1)) * (bt[-1] - bt[-2]))
        return float(np.interp(pos, np.arange(len(bt)), bt))

    def n_bars(self, duration):
        end = math.ceil(self.beat_position(duration) * SLOTS_PER_BEAT - 1e-9)
        return max(1, math.ceil(end / self.slots_per_bar))

    def describe(self):
        return {
            "method": self.method,
            "tempo_bpm": round(self.tempo_bpm, 2),
            "tempo_basis": self.tempo_basis,
            "tempo_note": self.tempo_note,
            "alternatives_unresolved": [round(a, 2) for a in self.tempo_alternatives],
            "meter": f"{self.meter} (detected from the drum pattern)" if self.method == "pattern" else f"{self.time_signature} (assumed or chosen)",
            "time_signature": self.time_signature,
            "marking": self.marking,
            "phase_calibration_ms": round(self.phase_shift_s * 1000, 1),
            "downbeat": self.phase_note or "not detected: bar 1 starts at the first grid beat at/before t=0",
        }


def _extend(beats, period, duration, beats_per_bar, anchor_is_downbeat=True):
    """Extend backwards until the grid starts at/before t=0, and forwards past the end. If
    anchor_is_downbeat, the anchor (beats[0]) stays on a bar boundary (whole bars of pre-roll);
    otherwise bar 1 simply starts at the first grid beat <= 0. Index 0 = bar 1 beat 1."""
    beats, shift = list(beats), 0
    while beats[0] > 0 or (anchor_is_downbeat and shift % beats_per_bar):
        beats.insert(0, beats[0] - period)
        shift += 1
    while beats[-1] < duration + period:
        beats.append(beats[-1] + period)
    return np.array(beats)


def estimate_grid(audio_path, duration, tempo_hint=None, beats_per_bar=4, offset=None, events=None, meter="auto"):
    """With `events` (the detected hits) the tempo comes from the drum pattern, not the loudest pulse of the mix,
    and a typed tempo is checked against the hits (see tempo.py). meter: auto | 4/4 | 3/4 | 6/8 | 12/8. A 3/4,
    6/8 or 12/8 bar is 12 slots (beats_per_bar 3 of the grid's quarter tempo; in 12/8 a slot is notated as an eighth)."""
    tempo_note = tempo_basis = drum_alts = None
    typed = tempo_hint
    if events:
        if tempo_hint and offset is None:
            tempo_hint, tempo_note = snap_hint(events, tempo_hint)
            tempo_basis = "tempo entered by the user" if tempo_note is None else "tempo entered, corrected to the hits"
        elif not tempo_hint:
            found = estimate_tempo(events, duration)
            if found:
                tempo_hint, info = found
                tempo_basis = f"drum pattern (grid lock {info['grid_coherence']:.2f})"
                drum_alts = info["alternatives"]
    if events and tempo_hint and offset is None and meter != "4/4":
        q, lock = refine(events, relock(events, tempo_hint, (0.75, 1.5, 0.5) if typed else (0.75,)), 0.01)
        if lock >= 0.5:
            m, down, info = detect_meter(events, q, None if meter == "auto" else meter)
            if m != "4/4":
                grid = Grid(_extend([down], 60.0 / q, duration, 3), 3, "pattern", float(q),
                            sorted({round(tempo_hint, 1), round(q / 2, 1), round(q * 2, 1)} - {round(q, 1)}), meter=m)
                grid.tempo_basis = (tempo_basis or "") + f"; {m} found from the drum pattern" + (" (you chose it)" if meter != "auto" else "")
                why = {"6/8": "kick on 1, snare on the 4th eighth", "12/8": "a shuffle: triplet eighths, snare on 2 and 4",
                       "3/4": "kick on 1, snare on 2/3"}[m]
                grid.tempo_note = tempo_note or (f"The drums repeat every 12 steps ({why}), so the score is in {m}, {grid.marking}. "
                                                 "If that is wrong, choose the time signature under More options.")
                grid.phase_note = f"from the drum pattern ({why}): {info}"
                return grid
    compound = meter if meter in ("6/8", "12/8") else ""  # chosen, but the hits are not steady enough to place the bar
    if meter in ("3/4", "6/8", "12/8"):
        beats_per_bar = 3
    if tempo_hint and offset is not None:
        period = 60.0 / tempo_hint
        grid = Grid(_extend([offset], period, duration, beats_per_bar), beats_per_bar, "hint", float(tempo_hint), [], meter=compound)
        grid.phase_note = "hint offset taken as beat 1"
        grid.tempo_alternatives = [tempo_hint / 2, tempo_hint * 2]
        grid.tempo_basis = "tempo and offset entered by the user (exact grid)"
        return grid
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    kwargs = {"bpm": float(tempo_hint)} if tempo_hint else {}
    _, frames = librosa.beat.beat_track(y=y, sr=sr, **kwargs)
    times = librosa.frames_to_time(frames, sr=sr)
    if len(times) < 4:
        raise TempoUndetermined("beat tracker found <4 beats; pass tempo_hint")
    period = 60.0 / tempo_hint if tempo_hint else float(np.median(np.diff(times)))
    bpm = 60.0 / period
    method = "librosa.beat_track(bpm=hint)" if tempo_hint else "librosa.beat_track"
    alts = [bpm / 2, bpm * 2]
    if drum_alts is not None:  # tempo came from the drums: offer the usual ratio confusions + the beat tracker's own opinion
        alts = list(drum_alts)
        tracked = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
        if abs(tracked - bpm) / bpm > 0.03 and all(abs(tracked - a) / a > 0.03 for a in alts):
            alts = sorted([*alts, round(tracked, 1)])
    grid = Grid(_extend(list(times), period, duration, beats_per_bar, False), beats_per_bar, method, bpm, alts, meter=compound)
    grid.tempo_basis, grid.tempo_note = tempo_basis or "beat tracker on the mix", tempo_note
    return grid


def calibrate_phase(events, grid, instruments=("kick", "snare", "hihat")):
    """Shift the grid so hits sit on their slots on average. Beat trackers place beats tens of ms late
    relative to the attacks (measured 46 ms here), which also makes the playhead trail the audio.
    Uses a circular mean of the slot offsets (robust to wrap-around). Exact (hint+offset) grids are
    left alone. Returns the shift in seconds."""
    if grid.method == "hint":
        return 0.0
    ts = [e.time_s for e in events if e.instrument in instruments and e.grace_of is None]
    if len(ts) < 8:
        return 0.0
    slot = 60.0 / grid.tempo_bpm / SLOTS_PER_BEAT
    err = np.array([t - grid.slot_time(round(grid.beat_position(t) * SLOTS_PER_BEAT)) for t in ts])
    shift = float(np.angle(np.exp(2j * np.pi * err / slot).mean()) / (2 * np.pi) * slot)
    grid.beat_times = grid.beat_times + shift
    grid.phase_shift_s += shift
    return shift


def fit_bar_phase(events, grid, forced=None):
    """Choose which tracked beat (0-3) is beat 1. forced = user override. Otherwise a heuristic:
    kick share on beat 1 plus snare share on beats 2 and 4. Weak on songs without a clear
    kick/snare backbeat (reported with its margin). No-op for exact (hint+offset) grids."""
    if grid.method == "hint" or grid.beats_per_bar != 4:
        return
    kick, snare = np.zeros(4), np.zeros(4)
    for e in events:
        if e.instrument in ("kick", "snare"):
            (kick if e.instrument == "kick" else snare)[round(grid.beat_position(e.time_s)) % 4] += 1
    if forced is not None:
        r, note = forced % 4, f"user-specified: tracked beat {forced % 4} is beat 1"
    elif kick.sum() + snare.sum() == 0:
        return
    else:
        ks, ss = kick / max(kick.sum(), 1), snare / max(snare.sum(), 1)
        scores = np.array([ks[r] + ss[(r + 1) % 4] + ss[(r + 3) % 4] for r in range(4)])
        r = int(np.argmax(scores))
        order = np.sort(scores)[::-1]
        note = (f"GUESSED (kick on beat 1 + snare on 2&4): scores {scores.round(2).tolist()}, margin "
                f"{order[0] - order[1]:.2f}" + (" - LOW CONFIDENCE, verify or pass --beat1" if order[0] - order[1] < 0.15 else ""))
    grid.beat_shift = (4 - r) % 4
    grid.phase_note = note


def quantize(events, grid):
    """Fill bar/slot/beat_position/quantization_error on each event; raw time_s untouched."""
    spb = grid.slots_per_bar
    for e in events:
        pos = grid.beat_position(e.time_s)
        idx = round(pos * SLOTS_PER_BEAT)
        e.beat_position = round(pos, 4)
        e.bar, e.slot = divmod(idx, spb)
        e.quantization_error_s = round(e.time_s - grid.slot_time(idx), 5)
    return events
