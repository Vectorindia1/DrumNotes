"""Articulation analysis on the original audio (EXPERIMENTAL, uncalibrated heuristics).

ADTOF only outputs kick/snare/tom/hi-hat/cymbal. This adds, from band-limited energy envelopes at
1.5 ms resolution and a few spectral features:
  flam       snare hit with a second onset 15-65 ms before it (grace note)
  ghost      snare hit far below the local main-hit level: stored as ghost_probability only, NOT marked in the score
             (on MDB Drums at most 44 % of such marks were real ghost notes, at any threshold)
  open       hi-hat whose high-frequency energy keeps ringing after the hit
  ride/crash/splash/china   cymbal type: ride from a regular run of hits, others from decay+spectrum
Every threshold below is a guess made without labelled data. Raw measurements are stored in
Event.articulation_features so thresholds can be recalibrated; nothing here is evidence of accuracy.
"""
import librosa
import numpy as np

from backend.events import Event

SR = 44100
N_FFT, HOP = 512, 64  # 11.6 ms window, 1.45 ms hop
BANDS = {"snare": (200, 4000), "hat": (6000, 16000), "cym": (4000, 16000), "kick": (30, 150), "tom": (80, 400)}
BAND_OF = {"kick": "kick", "snare": "snare", "tom": "tom", "hihat": "hat", "cymbal": "cym"}

FLAM_MIN_S, FLAM_MAX_S = 0.015, 0.065
FLAM_PICK_ACTIVATION = 0.08  # ADTOF snare activation needed for a grace hit (separate local max)
FLAM_REL_LEVEL = (0.15, 1.0)  # grace onset strength relative to the main hit
GHOST_RATIO = (0.35, 0.55)  # loudness ratio to main-hit level: <=0.35 -> p=1, >=0.55 -> p=0
OPEN_TAIL_RATIO = 0.10  # sustained HF power 80-170 ms after the hit, relative to its peak (closed ~0, open ~0.1+)
OPEN_MIN_WINDOW_S = 0.17  # need this long before the next hi-hat/cymbal to judge
RIDE_RUN_MIN, RIDE_MAX_IOI_S, RIDE_MAX_CV = 4, 0.7, 0.25
# splash/china cut-offs sit above the 90th percentile of the one real clip seen so far (flatness 0.72,
# centroid 8.6 kHz) so they stay rare; both also need a MEASURED decay (not cut off by the next hit)
SPLASH_MAX_DECAY_S, SPLASH_MIN_CENTROID = 0.30, 9000.0
CHINA_MIN_FLATNESS, CHINA_MIN_CENTROID, CHINA_DECAY_S = 0.75, 8000.0, (0.15, 0.9)


def band_power(y, sr=SR):
    """{band: power per frame}. Frame k covers samples [k*HOP, k*HOP+N_FFT)."""
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    masks = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in BANDS.items()}
    n_frames = 1 + (len(y) - N_FFT) // HOP
    out = {b: np.zeros(n_frames, np.float32) for b in BANDS}
    step = 30 * sr // HOP * HOP
    for start in range(0, max(len(y) - N_FFT, 1), step):
        seg = y[start:start + step + N_FFT]
        if len(seg) < N_FFT:
            break
        spec = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP, center=False)) ** 2
        k0 = start // HOP
        for b, m in masks.items():
            v = spec[m].sum(0)[: n_frames - k0]
            out[b][k0:k0 + len(v)] = v
    return out


class Env:
    def __init__(self, power):
        self.p = power

    def win(self, band, t0, t1):
        a = self.p[band]
        i0 = max(round((t0 * SR - N_FFT / 2) / HOP), 0)
        i1 = min(round((t1 * SR - N_FFT / 2) / HOP) + 1, len(a))
        return a[i0:i1] if i1 > i0 else a[:0]

    def strength(self, band, t, post=0.030):
        """Onset loudness (amplitude units): peak around t minus the quiet just before it."""
        peak = self.win(band, t - 0.010, t + post)
        base = self.win(band, t - 0.050, t - 0.010)
        if not len(peak):
            return 0.0
        return float(np.sqrt(max(peak.max() - (base.min() if len(base) else 0.0), 0.0)))


def _sorted(events, instrument):
    return sorted((e for e in events if e.instrument == instrument), key=lambda e: e.time_s)


def find_flams(snares, env, low_picks, pred):
    """Return [(grace_event, main_event)] and mark mains as flams. Evidence must be ADTOF's own snare
    activation (a separate low-level peak 15-65 ms before the hit, or ADTOF already emitted both)
    AND a matching onset in the snare-band envelope. Envelope-only evidence is NOT used: in a dense
    mix there is always some earlier transient."""
    pairs, taken = [], set()
    known = [e.time_s for e in snares]
    for i, e in enumerate(snares):
        prev = snares[i - 1] if i else None
        if prev is not None and prev.articulation != "grace" and FLAM_MIN_S <= e.time_s - prev.time_s <= FLAM_MAX_S:
            prev.articulation = "grace"
            e.articulation = "flam"
            e.articulation_features = {"route": "paired detections", "gap_ms": round((e.time_s - prev.time_s) * 1000, 1)}
            pairs.append((prev, e))
            continue
        cands = [p for p in low_picks if FLAM_MIN_S <= e.time_s - p <= FLAM_MAX_S and p not in taken
                 and all(abs(p - k) > 0.02 for k in known)]
        if not cands:
            continue
        p = max(cands, key=lambda q: pred[min(round(q * 100), len(pred) - 1), 1])
        gap = e.time_s - p
        rel = env.strength("snare", p, post=min(gap - 0.005, 0.02)) / max(env.strength("snare", e.time_s), 1e-9)
        if not FLAM_REL_LEVEL[0] <= rel <= FLAM_REL_LEVEL[1]:
            continue
        taken.add(p)
        act = float(pred[min(round(p * 100), len(pred) - 1), 1])
        grace = Event(
            id="", time_s=p, sample=round(p * SR), instrument="snare", confidence=round(act, 4),
            model_output={"model": "adtof-pytorch", "label": 38, "activation": round(act, 4), "low_threshold": FLAM_PICK_ACTIVATION},
            articulation="grace", articulation_features={"gap_ms": round(gap * 1000, 1), "rel_level": round(float(rel), 2)},
            provenance="detected-low-threshold",
        )
        e.articulation = "flam"
        e.articulation_features = {"route": "low-activation peak + envelope", "gap_ms": round(gap * 1000, 1),
                                   "grace_activation": round(act, 3), "grace_rel_level": round(float(rel), 2)}
        pairs.append((grace, e))
    return pairs


def classify_ghosts(snares, env):
    """Loudness of each non-grace snare vs the 75th percentile of its neighbours within +-10 s."""
    mains = [e for e in snares if e.articulation != "grace"]
    strengths = np.array([env.strength("snare", e.time_s) for e in mains])
    times = np.array([e.time_s for e in mains])
    for e, s in zip(mains, strengths, strict=True):
        ref = np.percentile(strengths[np.abs(times - e.time_s) <= 10.0], 75)
        ratio = s / ref if ref > 0 else 1.0
        e.ghost_probability = round(float(np.clip((GHOST_RATIO[1] - ratio) / (GHOST_RATIO[1] - GHOST_RATIO[0]), 0, 1)), 2)
        e.articulation_features = {**(e.articulation_features or {}), "snare_level_ratio": round(float(ratio), 2)}
    return strengths, times


def classify_hats(events, env):
    hats = _sorted(events, "hihat")
    cym_times = sorted(e.time_s for e in events if e.instrument in ("hihat", "cymbal"))
    for e in hats:
        later = [t for t in cym_times if t > e.time_s + 1e-6]
        room = (later[0] - e.time_s) if later else 1.0
        peak = env.win("hat", e.time_s - 0.005, e.time_s + 0.030)
        if not len(peak) or peak.max() <= 0:
            continue
        feats = {"room_to_next_ms": round(room * 1000)}
        if room >= OPEN_MIN_WINDOW_S:
            tails = [t for t in (env.win("hat", e.time_s + a, e.time_s + a + 0.03) for a in (0.08, 0.11, 0.14)) if len(t)]
            if not tails:  # the recording ends right after this hit: nothing to measure, leave it closed
                feats["tail_ratio"] = None
            else:
                ratio = min(float(t.mean()) for t in tails) / float(peak.max())
                feats["tail_ratio"] = round(ratio, 3)
                if ratio >= OPEN_TAIL_RATIO:
                    e.articulation, e.articulation_confidence = "open", round(min(ratio / (3 * OPEN_TAIL_RATIO), 1.0), 2)
        e.articulation_features = feats


def _spectral(y, t):
    a = int(max(t - 0.005, 0) * SR)
    seg = y[a:a + int(0.08 * SR)]
    if len(seg) < 512:
        return None
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    m = (f >= 2000) & (f <= 16000)
    return float((f[m] * mag[m]).sum() / mag[m].sum()), float(np.exp(np.log(mag[m]).mean()) / mag[m].mean())


def classify_cymbals(events, env, y):
    cyms = _sorted(events, "cymbal")
    if not cyms:
        return
    t = np.array([e.time_s for e in cyms])
    # ride: runs of >= RIDE_RUN_MIN hits with short, regular spacing
    start = 0
    for i in range(1, len(cyms) + 1):
        if i == len(cyms) or t[i] - t[i - 1] > 1.0:
            run = t[start:i]
            ioi = np.diff(run)
            if len(run) >= RIDE_RUN_MIN and np.median(ioi) <= RIDE_MAX_IOI_S:
                cv = float(np.std(ioi) / np.mean(ioi))
                if cv <= RIDE_MAX_CV:
                    for e in cyms[start:i]:
                        e.articulation, e.articulation_confidence = "ride", round(1 - cv / RIDE_MAX_CV * 0.5, 2)
            start = i
    for k, e in enumerate(cyms):
        nxt = t[k + 1] if k + 1 < len(cyms) else e.time_s + 2.0
        limit = min(e.time_s + 2.0, nxt - 0.01)
        db = 10 * np.log10(env.win("cym", e.time_s, limit) + 1e-12)
        decay, censored = None, True
        if len(db) > 10:
            peak_i = int(np.argmax(db[: max(int(0.05 * SR / HOP), 1)]))
            below = np.where(db[peak_i:] < db[peak_i] - 20)[0]
            if len(below):
                decay, censored = round(float(below[0] * HOP / SR), 3), False
        spec = _spectral(y, e.time_s)
        feats = {"decay20_s": decay, "decay_censored_by_next_hit": censored,
                 "centroid_hz": round(spec[0]) if spec else None, "flatness": round(spec[1], 3) if spec else None}
        e.articulation_features = feats
        if e.articulation == "ride" or spec is None:
            continue
        e.articulation = "crash"
        if decay is not None and decay <= SPLASH_MAX_DECAY_S and spec[0] >= SPLASH_MIN_CENTROID:
            e.articulation = "splash"
        elif (spec[1] >= CHINA_MIN_FLATNESS and spec[0] >= CHINA_MIN_CENTROID and decay is not None
              and CHINA_DECAY_S[0] <= decay <= CHINA_DECAY_S[1]):
            e.articulation = "china"


def annotate(events, y, pred=None, pick=None, env=None):
    """Mutates events (labels + features); returns (events incl. added flam graces, summary).
    pred/pick enable the low-threshold evidence for flams. Quiet low-threshold snares are NOT added as ghost
    notes any more: on MDB Drums 76 of 139 such additions were not real notes (full mix)."""
    env = env or Env(band_power(y))
    snares = _sorted(events, "snare")
    low = pick(pred, [2.0, FLAM_PICK_ACTIVATION, 2.0, 2.0, 2.0])[38] if pred is not None and pick else []
    pairs = find_flams(snares, env, low, pred)
    graces = [g for g, _ in pairs]
    classify_ghosts([e for e in snares if e.articulation != "grace"], env)
    classify_hats(events, env)
    classify_cymbals(events, env, y)
    refine_onsets(events + [g for g, _ in pairs if g not in events], env)
    new_graces = [g for g in graces if g not in events]
    events = sorted(events + new_graces, key=lambda e: (e.time_s, e.instrument))
    for i, e in enumerate(events):
        e.id = f"e{i:05d}"
    for g, main in pairs:
        g.grace_of = main.id
    return events, summary(events)


def summary(events):
    s = {}
    for e in (e for e in events if e.notated):  # what the score shows
        key = f"{e.instrument}:{e.articulation or 'plain'}"
        s[key] = s.get(key, 0) + 1
    return dict(sorted(s.items()))


# --- onset refinement -------------------------------------------------------------------------
# ADTOF works on 10 ms frames with a 46 ms window, so its event times lead the real attack by roughly
# 7-15 ms (measured). The attack is re-located as the sharpest rise of the instrument's own band at
# 1.45 ms resolution; REFINE_LAG_S[inst] converts "centre of the frame where the rise peaks" into the
# attack time and was calibrated on synthetic hits with exact truth (scripts/calibrate_onsets.py).
REFINE_WINDOW_S = 0.03
REFINE_LAG_S = {"kick": 0.0022, "snare": 0.0017, "tom": 0.0017, "hihat": 0.0015, "cymbal": 0.0015}  # tom/cymbal: copied from neighbours, not measured


def rise_time(env, band, t):
    """Frame-centre time of the sharpest amplitude rise within +-REFINE_WINDOW_S of t, or None."""
    t0 = t - REFINE_WINDOW_S
    a = np.sqrt(env.win(band, t0, t + REFINE_WINDOW_S))
    if len(a) < 8:
        return None
    flux = np.maximum(a[4:] - a[:-4], 0)
    k = int(np.argmax(flux))
    if flux[k] <= 0 or k in (0, len(flux) - 1):  # nothing rising, or the peak is at the window edge
        return None
    first = max(round((t0 * SR - N_FFT / 2) / HOP), 0)
    return ((first + k + 4) * HOP + N_FFT / 2) / SR


def refine_onsets(events, env):
    """Move each event to its measured attack; the detector's own time is kept in model_output."""
    n = 0
    for e in events:
        r = rise_time(env, BAND_OF[e.instrument], e.time_s)
        if r is None:
            continue
        e.model_output.setdefault("frame_time_s", e.time_s)
        e.time_s = round(r - REFINE_LAG_S[e.instrument], 5)
        e.sample = round(e.time_s * SR)
        n += 1
    return n
