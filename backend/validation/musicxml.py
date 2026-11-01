"""Structural MusicXML validation: well-formed, expected bar count, every voice fills every bar."""
import xml.etree.ElementTree as ET


def validate_musicxml(path, expected_bars=None):
    """Return a list of problems (empty = valid)."""
    problems = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f"not well-formed: {e}"]
    measures = root.findall("./part/measure")
    if not measures:
        return ["no measures"]
    if expected_bars is not None and len(measures) != expected_bars:
        problems.append(f"{len(measures)} measures, expected {expected_bars}")
    divisions = beats = None
    for m in measures:
        num = m.get("number")
        if (d := m.findtext("attributes/divisions")) is not None:
            divisions = int(d)
        if (b := m.findtext("attributes/time/beats")) is not None:
            beats = int(b)
        if divisions is None or beats is None:
            problems.append(f"measure {num}: no divisions/time signature in effect")
            continue
        per_voice = {}
        for el in m.findall("note"):
            if el.find("chord") is None and el.find("grace") is None:
                v = el.findtext("voice", "1")
                per_voice[v] = per_voice.get(v, 0) + int(el.findtext("duration"))
        for v, total in per_voice.items():
            if total != divisions * beats:
                problems.append(f"measure {num} voice {v}: duration {total} != {divisions * beats}")
    return problems
