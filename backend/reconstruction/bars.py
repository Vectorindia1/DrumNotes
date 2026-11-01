"""Quantized events -> bars of 16th-note slots holding note keys. Same-key/same-slot duplicates are merged."""

CYMBALS = ("crash", "ride", "splash", "china")


def note_key(e):
    """Map (model instrument, articulation) to a notation key (see notation.drum_musicxml.KIT)."""
    a = e.articulation
    if e.instrument == "snare":
        return {"ghost": "snare_ghost", "flam": "snare_flam"}.get(a, "snare")
    if e.instrument == "hihat":
        return "hihat_open" if a == "open" else "hihat"
    if e.instrument == "cymbal":
        return a if a in CYMBALS else "crash"  # unclassified cymbal defaults to crash
    return e.instrument


def build_bars(events, n_bars, slots_per_bar):
    """Return (bars, collisions). bars[b][s] is the set of note keys sounding in that slot.
    Different keys in one slot are kept (simultaneous hits). A duplicate key in a slot keeps the
    higher-confidence event; the others get notated=False (still in the JSON). Flam grace notes
    are not placed on the grid: they are drawn from their main event's key."""
    events = [e for e in events if e.grace_of is None]
    n_bars = max(n_bars, max((e.bar for e in events), default=0) + 1)
    bars = [[set() for _ in range(slots_per_bar)] for _ in range(n_bars)]
    best, collisions = {}, 0
    for e in events:
        key = (e.bar, e.slot, note_key(e))
        if key in best:
            collisions += 1
            loser = e if best[key].confidence >= e.confidence else best[key]
            loser.notated = False
            best[key] = best[key] if loser is e else e
        else:
            best[key] = e
        bars[e.bar][e.slot].add(key[2])
    return bars, collisions
