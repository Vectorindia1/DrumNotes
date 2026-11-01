"""Minimal drum-set MusicXML writer (M0 render smoke test; real writer comes in M1)."""
from xml.sax.saxutils import escape

# instrument id -> (display-step, display-octave, notehead, midi-unpitched)
KIT = {
    "kick": ("F", 4, None, 36),
    "snare": ("C", 5, None, 38),
    "hihat": ("G", 5, "x", 42),
}


def _note(inst, dur, chord, typ):
    step, octv, head, midi = KIT[inst]
    return (
        "<note>" + ("<chord/>" if chord else "")
        + f"<unpitched><display-step>{step}</display-step><display-octave>{octv}</display-octave></unpitched>"
        + f"<duration>{dur}</duration><instrument id=\"P1-I{midi}\"/><voice>1</voice><type>{typ}</type>"
        + (f"<notehead>{head}</notehead>" if head else "")
        + f"<stem>{'down' if inst == 'kick' else 'up'}</stem></note>"
    )


def measures_to_xml(measures, title="Test", tempo=100):
    """measures: list of measures; measure = list of 8 eighth-note slots, each a list of instrument names."""
    body = []
    for i, m in enumerate(measures, 1):
        attrs = (
            "<attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
            "<clef><sign>percussion</sign></clef></attributes>"
            f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>{tempo}</per-minute></metronome></direction-type><sound tempo="{tempo}"/></direction>'
            if i == 1 else ""
        )
        notes = ""
        for slot in m:
            if not slot:
                notes += "<note><rest/><duration>1</duration><voice>1</voice><type>eighth</type></note>"
            for j, inst in enumerate(slot):
                notes += _note(inst, 1, j > 0, "eighth")
        body.append(f'<measure number="{i}">{attrs}{notes}</measure>')
    insts = "".join(
        f'<score-instrument id="P1-I{v[3]}"><instrument-name>{k}</instrument-name></score-instrument>'
        f'<midi-instrument id="P1-I{v[3]}"><midi-channel>10</midi-channel><midi-unpitched>{v[3]+1}</midi-unpitched></midi-instrument>'
        for k, v in KIT.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
        f'<score-partwise version="3.1"><work><work-title>{escape(title)}</work-title></work>'
        f'<part-list><score-part id="P1"><part-name>Drum Set</part-name>{insts}</score-part></part-list>'
        f'<part id="P1">{"".join(body)}</part></score-partwise>'
    )


ROCK_BAR = [["hihat", "kick"], ["hihat"], ["hihat", "snare"], ["hihat"],
            ["hihat", "kick"], ["hihat", "kick"], ["hihat", "snare"], ["hihat"]]

if __name__ == "__main__":
    import pathlib
    import sys
    pathlib.Path(sys.argv[1]).write_text(measures_to_xml([ROCK_BAR] * 4, "Rock beat"))
