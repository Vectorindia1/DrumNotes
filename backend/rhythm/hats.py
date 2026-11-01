"""Rebuild the hi-hat line from the AUDIO at every grid slot, not only where the detector fired.

Measured on a real song (see PROGRESS.md) the detector's hi-hat line had three faults: hats reported at
slots with no energy, dropouts inside a steady part, and snare bleed counted as hat. And it printed every
faint 16th tick although the section's eighths were clearly louder (what a listener hears as "8th-note
hats"). Here each slot is judged by its measured high-frequency onset strength:

  evidence   a slot is "verified" if its transient is >= EVIDENCE x the local off-grid noise floor. A hat is
             notated when its slot is verified in enough of the bars around it (VOTE_KEEP), and a slot the
             detector missed is filled when verified in most of them (VOTE_FILL) AND the detector reported a
             hat at that slot in a good share of them (PRIOR_SHARE): other high-frequency sounds (melody,
             vocals, noise) have transients near the grid too, so energy alone must never create a hi-hat
  level      the in-between 16th ticks are kept only where they are at least GHOST_RATIO as loud as the
             8th-position ticks around them (+-WINDOW_BARS bars); the 8ths only where they are at least
             EIGHTH_RATIO of the on-beat level. mode="eighths"/"sixteenths" overrides the 16th decision.
  bleed      slots where another drum plays cannot be verified from energy. There the detector's report is kept
             (if confident, subject to the level rule) only in bars with real hi-hat activity around them, and
             a missing report is filled when the same slot has a hat in at least half of the neighbouring
             bars (a hi-hat part repeats).
Dropped detections stay in the event list (notated=False, with the reason); nothing is deleted.
"""
import numpy as np
from scipy.ndimage import median_filter

from backend.events import Event

EVIDENCE = 1.5
TAIL_SHARE = 0.05
VOTE_KEEP = 0.4  # a verified slot is kept if it is verified in at least this share of the bars around it
VOTE_FILL = 0.7  # an unverified slot is filled if it is verified in at least this share of the bars around it
PRIOR_SHARE = 0.2  # ...AND the detector reported a hat at that slot in at least this share of them (energy alone is not a hi-hat)
WEAK_REPORT = 1.2  # a detector-reported hat needs at least this much over the noise floor
GHOST_RATIO = 0.8
EIGHTH_RATIO = 0.4
WINDOW_BARS = 3
MIN_SAMPLES = 6
OTHER = ("kick", "snare", "tom", "cymbal")


def _level(slot):
    return "q" if slot % 4 == 0 else "e" if slot % 4 == 2 else "s"


def _strength(env, t):
    return max(env.strength("hat", t + d) for d in (-0.008, 0.0, 0.008))


def regularize_hats(events, grid, env, n_bars, mode="auto"):
    """Mutates `events` (marks dropped ones, returns new recovered ones). Returns (new_events, summary)."""
    spb = grid.slots_per_bar
    n = n_bars * spb
    if n < 4 * spb:
        return [], {"skipped": "too short"}
    slot_len = grid.slot_time(1) - grid.slot_time(0)
    t_of = [grid.slot_time(g) for g in range(n)]
    S = np.array([_strength(env, t) for t in t_of])
    floor = median_filter(np.array([env.strength("hat", t + 0.5 * slot_len) for t in t_of]), size=2 * spb + 1, mode="nearest")
    floor = np.maximum(floor, 1e-6)
    over = S / floor

    other = {e.bar * spb + e.slot for e in events if e.instrument in OTHER and e.grace_of is None and e.bar is not None and e.bar * spb + e.slot < n}
    # the slot right AFTER another drum can hold only that drum's decaying tail: it counts as a hat only if it is
    # more than TAIL_SHARE of that drum's own hat-band level (tail ripples are ~1-3 %, real hats more)
    for h in sorted(other):
        g = h + 1
        if g < n and g not in other and S[g] < TAIL_SHARE * S[h]:
            over[g] = 0.0

    hats = {}
    for e in events:
        if e.instrument == "hihat" and e.grace_of is None and e.bar is not None and e.bar * spb + e.slot < n:
            g = e.bar * spb + e.slot
            if g not in hats or e.confidence > hats[g].confidence:
                hats[g] = e

    # per-bar decisions from the loudness of each metrical level among verified, pure (no other drum) slots
    allow8, allow16, ratios = [], [], []
    for b in range(n_bars):
        lo, hi = max(0, b - WINDOW_BARS) * spb, min(n_bars, b + WINDOW_BARS + 1) * spb
        lv = {"q": [], "e": [], "s": []}
        for g in range(lo, hi):
            if g not in other and over[g] >= EVIDENCE:
                lv[_level(g % spb)].append(S[g])
        mq, me, ms = (float(np.median(v)) if len(v) >= MIN_SAMPLES else None for v in (lv["q"], lv["e"], lv["s"]))
        ghost_ticks = ms is not None and me is not None and ms < GHOST_RATIO * me
        a16 = mode == "sixteenths" or (mode == "auto" and not ghost_ticks)
        a8 = not (mode == "auto" and mq is not None and me is not None and me < EIGHTH_RATIO * mq)
        allow8.append(a8)
        allow16.append(a16)
        ratios.append(ms / me if ms is not None and me else None)

    pure_hits = np.array([sum(1 for s_ in range(spb) if b * spb + s_ not in other and over[b * spb + s_] >= EVIDENCE) for b in range(n_bars)])
    active = [pure_hits[max(0, b - 1):b + 2].sum() >= 3 for b in range(n_bars)]  # real hi-hat energy in this bar +-1
    reported = {b: {e.slot for g_, e in hats.items() if g_ // spb == b} for b in range(n_bars)}

    def neighbours_have(b, slot):
        bars = [x for x in range(max(0, b - WINDOW_BARS), min(n_bars, b + WINDOW_BARS + 1)) if x != b and active[x]]
        return len(bars) >= 2 and sum(slot in reported[x] for x in bars) >= 0.5 * len(bars)

    verified = np.zeros((n_bars, spb), bool)
    for g in range(n):
        if g not in other and (over[g] >= EVIDENCE or (g in hats and over[g] >= WEAK_REPORT)):
            verified[g // spb, g % spb] = True

    def vote(b, slot):
        """Share of the active bars within +-WINDOW_BARS (this one included) where the slot is verified; None if too few."""
        bars = [x for x in range(max(0, b - WINDOW_BARS), min(n_bars, b + WINDOW_BARS + 1)) if active[x]]
        return float(np.mean([verified[x, slot] for x in bars])) if len(bars) >= 3 else None

    def report_share(b, slot):
        bars = [x for x in range(max(0, b - WINDOW_BARS), min(n_bars, b + WINDOW_BARS + 1)) if active[x]]
        return float(np.mean([slot in reported[x] for x in bars])) if bars else 0.0

    keep, filled, recovered = set(), set(), []
    for g in range(n):
        b, lvl = g // spb, _level(g % spb)
        if not (lvl == "q" or (lvl == "e" and allow8[b]) or (lvl == "s" and allow16[b])):
            continue
        if g in other:
            if not active[b]:
                continue
            if g in hats and hats[g].confidence >= 0.4:
                keep.add(g)
            elif g not in hats and neighbours_have(b, g % spb):
                keep.add(g)
                filled.add(g)
        elif active[b]:
            v = vote(b, g % spb)
            if g in hats:  # the detector reported it: keep if the audio backs it up
                if (verified[b, g % spb] and (v is None or v >= VOTE_KEEP)) or (v is not None and v >= VOTE_FILL):
                    keep.add(g)
            elif v is not None and v >= VOTE_FILL and report_share(b, g % spb) >= PRIOR_SHARE:
                keep.add(g)  # a dropout inside a pattern the detector does see
    dropped_weak = dropped_noise = 0
    for g, e in hats.items():
        if g not in keep:
            e.notated = False
            reason = "quiet in-between tick" if over[g] >= WEAK_REPORT else "no energy at this slot"
            e.articulation_features = {**(e.articulation_features or {}), "dropped": reason, "strength_over_floor": round(float(over[g]), 2)}
            dropped_weak += reason.startswith("quiet")
            dropped_noise += reason.startswith("no energy")
    for g in sorted(keep - set(hats)):
        t = t_of[g]
        r = art_rise(env, t)
        if g in filled:  # shares its slot with another drum: no audio verification, the pattern says so
            r = None
        e = Event(id="", time_s=round(r if r is not None else t, 5), sample=round((r if r is not None else t) * 44100),
                  instrument="hihat", confidence=0.3 if g in filled else round(min(0.99, float(over[g]) / 6), 3),
                  model_output={"source": "filled from neighbouring bars" if g in filled else "recovered from audio energy at the grid slot"},
                  bar=g // spb, slot=g % spb, quantization_error_s=round((r if r is not None else t) - t, 5),
                  articulation_features={"strength_over_floor": round(float(over[g]), 2)},
                  provenance="filled-from-pattern" if g in filled else "recovered-from-audio")
        recovered.append(e)
    present_bars = [b for b in range(n_bars) if any(b * spb + s in keep for s in range(spb))]
    return recovered, {
        "mode": mode, "hats_kept": len(keep & set(hats)), "recovered_from_audio": len(recovered) - len(filled),
        "filled_from_pattern": len(filled),
        "dropped_quiet_ticks": dropped_weak, "dropped_no_energy": dropped_noise,
        "bars_with_hats": len(present_bars),
        "bars_eighths_only": sum(1 for b in present_bars if not allow16[b]), "bars_sixteenths": sum(1 for b in present_bars if allow16[b]),
    }


def art_rise(env, t):
    from backend.detection.articulation import rise_time
    return rise_time(env, "hat", t)
