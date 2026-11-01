"""Tempo (and meter: 4/4, 3/4, 6/8) from the detected drum hits, instead of the loudest pulse in the mix.

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


def _slots(events, q, inst):
    """Hit times of `inst` as 16th-slot indices of a quarter tempo q (phase from all hits)."""
    ts, p = _hit_times(events), 60.0 / q / 4
    ph = float(np.angle(np.exp(2j * np.pi * ts / p).mean()) / (2 * np.pi) * p)
    x = np.array([e.time_s for e in events if e.instrument == inst and e.grace_of is None])
    return ph, p, np.round((x - ph) / p).astype(int)


def _recur(k, lag):
    s = set(k.tolist())
    return sum(x + lag in s for x in s) / max(len(s), 1)


def detect_meter(events, q, forced=None):
    """4/4, 3/4, 6/8 or 12/8 for hits on the 16th grid of quarter tempo q, and the downbeat time.
    A 12-slot bar is chosen only when kick/snare clearly repeat every 12 slots rather than every 16. Then where the
    snare sits decides: once at the half-bar = 6/8 (slots are 16ths); on slots 3 and 9 = 12/8, i.e. a shuffle/swing
    4/4 whose slots are triplet eighths; on slots 4 and 8 = 3/4. Returns (meter, downbeat_s or None, info)."""
    ph, p, kick = _slots(events, q, "kick")
    snare = _slots(events, q, "snare")[2]
    if forced == "4/4" or len(kick) + len(snare) < 24:
        return "4/4", None, {}
    rep = {lag: round(_recur(kick, lag) + _recur(snare, lag), 3) for lag in (12, 16)}
    if forced is None and rep[12] <= 1.15 * rep[16]:
        return "4/4", None, {"recurrence": rep}
    kh = np.bincount(kick % 12, minlength=12) / max(len(kick), 1)
    sh = np.bincount(snare % 12, minlength=12) / max(len(snare), 1)
    cands = [(kh[r] + sh[(r + 6) % 12], "6/8", r) for r in range(12)]
    cands += [(kh[r] + (sh[(r + 4) % 12] + sh[(r + 8) % 12]) / 2, "3/4", r) for r in range(12)]
    cands += [(kh[r] + (sh[(r + 3) % 12] + sh[(r + 9) % 12]) / 2, "12/8", r) for r in range(12)]
    if forced in ("3/4", "6/8", "12/8"):
        cands = [c for c in cands if c[1] == forced]
    cands.sort(reverse=True)
    (score, meter, r), runner = cands[0], next((c for c in cands[1:] if c[1] != cands[0][1]), None)
    margin = round(float(score - runner[0]), 2) if runner else None
    return meter, ph + r * p, {"recurrence": rep, "downbeat_score": round(float(score), 2), "margin_over_other_meter": margin}


def relock(events, bpm, ratios=(0.75,)):
    """The quarter tempo whose 16th grid the hits really lock to, when bpm's own grid clearly does not.
    0.75: hits on TRIPLETS of bpm (a 6/8 song read as 4/4); for a typed tempo also 1.5 (dotted quarter typed,
    as 6/8 is usually marked) and 0.5 (eighth notes typed). Returns bpm unchanged otherwise."""
    here = refine(events, bpm, 0.01)[1]  # refined: over minutes, a 0.1 % tempo error alone washes coherence out
    found = [refine(events, bpm * r, 0.01) for r in ratios]
    c = max(x[1] for x in found)
    q = max(x[0] for x in found if x[1] >= 0.9 * c)  # a coarser grid always fits at least as well: take the finest that fits
    return q if c >= 0.5 and here < 0.5 * c else bpm


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
