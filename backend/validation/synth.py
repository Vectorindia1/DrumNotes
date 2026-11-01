"""Synthetic drum audio with exact ground truth (benchmark category A only: NOT real-world evidence)."""
import numpy as np

SR = 44100
SLOT_TO_NAME = {"kick": "kick", "snare": "snare", "hihat": "hihat"}


def _kick(rng, n=int(0.25 * SR)):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)).cumsum() / SR) * np.exp(-t * 14)


def _snare(rng, n=int(0.2 * SR)):
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * 0.7 + np.sin(2 * np.pi * 190 * t) * 0.5) * np.exp(-t * 22)


def _hihat(rng, n=int(0.06 * SR)):
    t = np.arange(n) / SR
    return np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 70) * 0.35


VOICES = {"kick": _kick, "snare": _snare, "hihat": _hihat}


def render(pattern, bpm, bars, tail_s=1.0, seed=0):
    """pattern: list of (instrument, eighth_index 0-7) for one 4/4 bar; the first bar starts at
    sample 0. Returns (audio float32, truth {instrument: [seconds]})."""
    rng = np.random.default_rng(seed)
    step = 60.0 / bpm / 2
    x = np.zeros(int((bars * 8 * step + tail_s) * SR), np.float32)
    truth = {k: [] for k in VOICES}
    for b in range(bars):
        for inst, i in pattern:
            t = (b * 8 + i) * step
            truth[inst].append(t)
            s = VOICES[inst](rng)
            a = round(t * SR)
            x[a:a + len(s)] += s
    x /= np.abs(x).max() * 1.1
    return x, truth


ROCK = [("kick", 0), ("kick", 4), ("kick", 5), ("snare", 2), ("snare", 6)] + [("hihat", i) for i in range(8)]


def articulation_clip(seed=1):
    """Synthetic clip with known articulations for plumbing tests (NOT real-world evidence).
    Returns (audio, truth) where truth = [(time_s, instrument, articulation or None)] of the MAIN
    hits (a flam's grace note is 35 ms earlier)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(26 * SR), np.float32)
    truth = []

    def put(t, s, gain=1.0):
        a = round(t * SR)
        x[a:a + len(s)] += (s * gain)[: len(x) - a]

    def hat_open(n=int(0.5 * SR)):
        t = np.arange(n) / SR
        return np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 7) * 0.35

    def crash(n=int(2.0 * SR)):
        t = np.arange(n) / SR
        return np.diff(rng.standard_normal(n + 1)) * np.exp(-t * 2.2) * 0.5

    def tick(n=int(0.25 * SR)):  # ride-like short bright ping
        t = np.arange(n) / SR
        return (np.diff(rng.standard_normal(n + 1)) * 0.5 + np.sin(2 * np.pi * 5200 * t) * 0.3) * np.exp(-t * 14)

    for i in range(8):  # steady main snares, one ghost between every other pair, one flam at the end
        t = 1.0 + i * 0.5
        put(t, _snare(rng))
        truth.append((t, "snare", None))
    put(1.25, _snare(rng), 0.22)  # ghost
    truth.append((1.25, "snare", "ghost"))
    put(4.965, _snare(rng), 0.5)  # flam: grace 35 ms before the main stroke
    put(5.0 + 0.0, _snare(rng))
    truth.append((5.0, "snare", "flam"))
    for t in (7.0, 8.0):  # closed hats (short) then open hats (long ring)
        put(t, _hihat(rng))
        truth.append((t, "hihat", None))
    for t in (9.5, 11.0):
        put(t, hat_open())
        truth.append((t, "hihat", "open"))
    put(13.0, crash())
    truth.append((13.0, "cymbal", "crash"))
    for k in range(8):  # ride run
        put(16.0 + k * 0.4, tick())
        truth.append((16.0 + k * 0.4, "cymbal", "ride"))
    x /= np.abs(x).max() * 1.1
    return x, truth


def backing(n_samples, bpm, seed=3):
    """Synthetic non-drum instruments for mixture tests: saw bass on the beat + distorted power chords
    on eighths. Crude on purpose; it only has to overlap the drums in time and frequency."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n_samples, np.float32)
    step = 60.0 / bpm / 2
    notes = [55.0, 55.0, 65.4, 49.0]
    for i in range(int(n_samples / SR / step)):
        a, n = round(i * step * SR), int(step * SR * 0.95)
        seg = np.arange(n) / SR
        env = np.exp(-seg * 6)
        f = notes[(i // 2) % 4]
        if i % 2 == 0:  # bass note on the beat
            out[a:a + n] += (2 * ((seg * f) % 1) - 1)[: len(out) - a] * env * 0.8
        chord = sum(2 * ((seg * f * 2 * m * (1 + rng.normal(0, 0.001))) % 1) - 1 for m in (1, 1.5, 2))
        out[a:a + n] += (np.tanh(4 * chord) * np.exp(-seg * 9) * 0.5)[: len(out) - a]
    return out + rng.standard_normal(n_samples).astype(np.float32) * 0.01


def render_mix(pattern, bpm, bars, snr_db=0.0, seed=0):
    """Drums + synthetic backing at the given drum-to-rest level (dB RMS). Returns (mix, drums, truth)."""
    drums, truth = render(pattern, bpm, bars, seed=seed)
    rest = backing(len(drums), bpm)
    g = np.sqrt((drums**2).mean() / (rest**2).mean()) * 10 ** (-snr_db / 20)
    mix = drums + g * rest
    return (mix / (np.abs(mix).max() * 1.05)).astype(np.float32), drums, truth
