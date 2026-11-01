"""Tempo from the detected drum hits (4/4 assumed), instead of the loudest pulse in the mix.

Why: a beat tracker follows the strongest periodic pulse. A kick on dotted eighths (3 sixteenths) makes
114 BPM look like 152 BPM (4/3), a shuffle can look 3/2 faster, etc. Here every instrument votes on a
tempo T through its onset autocorrelation at the metrical lags of T: beat, 2 beats (snare backbeat),
a 4-beat BAR (the groove repeating: the cue a ratio-confusion cannot fake) and 2 bars. A mild prior
around 115 BPM breaks ties. The winner is refined to <1 % by sixteenth-grid coherence of the hits.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 100
LO, HI = 60.0, 200.0
LEVELS = ((1, 1.0), (2, 1.0), (4, 2.0), (8, 1.0))  # (beats, weight): bar-level evidence counts double
VOTERS = (("kick", 1.0), ("snare", 1.0), ("hihat", 0.5), ("cymbal", 0.25))
PRIOR_CENTRE, PRIOR_SIGMA_OCT = 115.0, 0.6
RATIOS = (0.5, 2 / 3, 0.75, 4 / 3, 1.5, 2.0)  # the usual confusions, offered to the user as alternatives


def _autocorrelations(events, duration):
    n = int(duration * FPS) + 1
    out = []
    for inst, w in VOTERS:
        ts = np.array([e.time_s for e in events if e.instrument == inst and e.grace_of is None])
        if len(ts) < 8:
            continue
        x = np.zeros(n)
        x[np.clip((ts * FPS).astype(int), 0, n - 1)] = 1
        x = gaussian_filter1d(x, 1.5) - gaussian_filter1d(x, 1.5).mean()  # +-15 ms tolerance
        f = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        out.append((w, ac / ac[0]))
    return out, n


def _salience(acs, n, bpm):
    s = 0.0
    for w, ac in acs:
        for beats, wm in LEVELS:
            i = round(beats * 60.0 / bpm * FPS)
            if i + 2 < n:
                s += w * wm * ac[i - 1:i + 2].max()
    return s


MIN_COHERENCE = 0.25  # below this the hits are not locked to any steady grid (live playing): do not "refine"


def _hit_times(events):
    return np.array([e.time_s for e in events if e.instrument in ("kick", "snare", "hihat") and e.grace_of is None])


def coherence(ts, bpm):
    """0..1: how exactly the hits sit on a 16th-note grid of this tempo (1 = all on the grid)."""
    return float(abs(np.exp(2j * np.pi * ts / (60.0 / bpm / 4)).mean())) if len(ts) else 0.0


def refine(events, bpm, span=0.02):
    """Sharpen to <1 %: the tempo whose 16th-note grid the hits fall on most coherently."""
    ts = _hit_times(events)
    best, best_r = bpm, coherence(ts, bpm)
    for cand in np.arange(bpm * (1 - span), bpm * (1 + span), bpm * 0.0004):
        r = coherence(ts, cand)
        if r > best_r:
            best, best_r = float(cand), r
    return best, best_r


def snap_hint(events, hint, span=0.03):
    """Check a user-supplied tempo against the hits. If the drums lock to a nearby tempo (within 3 %) and
    clearly NOT to the typed one, use the one the audio supports. Returns (bpm, note or None)."""
    ts = _hit_times(events)
    if len(ts) < 24:
        return hint, None
    best, r_best = refine(events, hint, span)
    r_hint = coherence(ts, hint)
    if r_best >= 0.4 and r_hint < 0.5 * r_best and abs(best - hint) / hint <= span:
        return round(best, 2), (f"You entered {hint:g} BPM, but the drums lock to {best:.1f} BPM (at {hint:g} the hits drift "
                                f"off the grid), so {best:.1f} was used.")
    return hint, None


def estimate_tempo(events, duration):
    """Return (bpm, info) or None when there is too little to go on."""
    acs, n = _autocorrelations(events, duration)
    if not acs or duration < 8:
        return None
    grid = np.arange(LO, HI, 0.25)
    raw = np.array([_salience(acs, n, t) for t in grid])
    prior = np.exp(-0.5 * (np.log2(grid / PRIOR_CENTRE) / PRIOR_SIGMA_OCT) ** 2)
    best = float(grid[int(np.argmax(raw * prior))])
    bpm, locked = refine(events, best)
    if locked < MIN_COHERENCE:
        bpm = best  # not locked to a steady grid: keep the coarse answer rather than chase noise
    alts = sorted({round(bpm * r, 1) for r in RATIOS if 50 <= bpm * r <= 220})
    return bpm, {"coarse_bpm": best, "grid_coherence": round(locked, 3), "alternatives": alts,
                 "voters": [round(w, 2) for w, _ in acs]}
