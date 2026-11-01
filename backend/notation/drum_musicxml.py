"""Drum-set MusicXML writer: percussion clef, voice 1 = hands (stems up), voice 2 = kick (stems down).

Input is bars of 16th-note slots (4/4 style, divisions=4 per quarter). Only simple durations are
emitted (no tuplets/ties); note lengths are the largest aligned value up to the next onset and
the remainder is filled with aligned rests.
"""
from xml.sax.saxutils import escape

# note key -> (display-step, display-octave, GM midi-unpitched, parenthesised, label above, notehead)
# MuseScore honours <notehead> shapes only when each articulation is its own instrument (own MIDI
# number), so every key has one. Cymbals differ by shape + height (+ label for the rare ones).
KIT = {
    "kick": ("F", 4, 36, False, None, None),
    "tom": ("D", 5, 47, False, None, None),
    "snare": ("C", 5, 38, False, None, None),
    "snare_ghost": ("C", 5, 38, True, None, "normal"),
    "snare_flam": ("C", 5, 38, False, None, None),  # emitted with a preceding grace note
    "ride": ("F", 5, 51, False, None, "x"),
    "hihat": ("G", 5, 42, False, None, "x"),
    "hihat_open": ("G", 5, 46, False, None, "circle-x"),
    "crash": ("A", 5, 49, False, None, "x"),
    "splash": ("B", 5, 55, False, "splash", "inverted triangle"),
    "china": ("C", 6, 52, False, "china", "triangle"),
    "tambourine": ("E", 5, 54, False, "tamb.", "x"),
    "shaker": ("E", 5, 70, False, "shaker", "x"),
}
ORDER = list(KIT)  # low -> high within a chord
TYPE = {1: "16th", 2: "eighth", 4: "quarter", 8: "half", 16: "whole"}
FEET = {"kick"}
LEGEND = ("Key: x hi-hat | circled x open hi-hat | x top line ride | x ledger crash | "
          "down-triangle splash | up-triangle china | x 4th space tamb./shaker | (n) ghost | grace note flam")


def _rests(start, n):
    out = []
    while n > 0:
        size = next(s for s in (16, 8, 4, 2, 1) if s <= n and start % s == 0)
        out.append((start, size))
        start, n = start + size, n - size
    return out


def _rest_xml(voice, size):
    return f"<note><rest/><duration>{size}</duration><voice>{voice}</voice><type>{TYPE[size]}</type></note>"


def _note_xml(key, voice, size, chord, stem, grace=False):
    step, octave, midi, paren, _, head = KIT[key]
    return (
        "<note>" + ('<grace slash="yes"/>' if grace else "") + ("<chord/>" if chord else "")
        + f"<unpitched><display-step>{step}</display-step><display-octave>{octave}</display-octave></unpitched>"
        + ("" if grace else f"<duration>{size}</duration>")
        + f'<instrument id="P1-I{midi}"/><voice>{voice}</voice>'
        + f"<type>{'eighth' if grace else TYPE[size]}</type><stem>{stem}</stem>"
        + (f'<notehead{" parentheses=\"yes\"" if paren else ""}>{head}</notehead>' if head else "")
        + "</note>"
    )


def _label_xml(labels):
    return "".join(
        f'<direction placement="above"><direction-type><words font-size="8">{escape(t)}</words></direction-type></direction>'
        for t in labels
    )


def _voice_xml(slots, voice, stem):
    """slots: list of sets (note keys in this voice). Always sums to len(slots) divisions."""
    n = len(slots)
    onsets = [i for i, s in enumerate(slots) if s]
    if not onsets:
        return f'<note><rest measure="yes"/><duration>{n}</duration><voice>{voice}</voice></note>'
    xml = "".join(_rest_xml(voice, size) for _, size in _rests(0, onsets[0]))
    for k, i in enumerate(onsets):
        nxt = onsets[k + 1] if k + 1 < len(onsets) else n
        size = next(s for s in (4, 2, 1) if s <= nxt - i and i % s == 0)
        keys = sorted(slots[i], key=ORDER.index)
        xml += _label_xml(sorted({KIT[k][4] for k in keys if KIT[k][4]}))
        for j, key in enumerate(keys):
            if key == "snare_flam":
                xml += _note_xml(key, voice, size, False, stem, grace=True)
            xml += _note_xml(key, voice, size, j > 0, stem)
        xml += "".join(_rest_xml(voice, sz) for _, sz in _rests(i + size, nxt - i - size))
    return xml


def bars_to_musicxml(bars, title="Drum score", tempo=100, beats_per_bar=4):
    n = beats_per_bar * 4
    body = []
    for b, bar in enumerate(bars):
        assert len(bar) == n, f"bar {b} has {len(bar)} slots, expected {n}"
        attrs = ""
        if b == 0:
            attrs = (
                f"<attributes><divisions>4</divisions><time><beats>{beats_per_bar}</beats><beat-type>4</beat-type></time>"
                "<clef><sign>percussion</sign><line>2</line></clef></attributes>"
                '<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                f'<per-minute>{round(tempo)}</per-minute></metronome></direction-type><sound tempo="{tempo:.1f}"/></direction>'
            )
        hands = _voice_xml([s - FEET for s in bar], 1, "up")
        feet = ""
        if any(s & FEET for s in bar):
            feet = f"<backup><duration>{n}</duration></backup>" + _voice_xml([s & FEET for s in bar], 2, "down")
        body.append(f'<measure number="{b + 1}">{attrs}{hands}{feet}</measure>')
    midis = {v[2]: k for k, v in reversed(list(KIT.items()))}
    insts = "".join(
        f'<score-instrument id="P1-I{m}"><instrument-name>{k}</instrument-name></score-instrument>'
        f'<midi-instrument id="P1-I{m}"><midi-channel>10</midi-channel><midi-unpitched>{m + 1}</midi-unpitched></midi-instrument>'
        for m, k in midis.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
        f'<score-partwise version="3.1"><work><work-title>{escape(title)}</work-title></work>'
        f'<identification><rights>{escape(LEGEND)}</rights></identification>'
        f'<part-list><score-part id="P1"><part-name>Drum Set</part-name>{insts}</score-part></part-list>'
        f'<part id="P1">{"".join(body)}</part></score-partwise>'
    )
