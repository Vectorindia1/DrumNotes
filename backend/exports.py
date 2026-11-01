import json

import pretty_midi

from backend.events import SCHEMA_VERSION
from backend.reconstruction.bars import note_key

GM_PITCH = {"kick": 36, "snare": 38, "snare_ghost": 38, "snare_flam": 38, "tom": 47, "hihat": 42,
            "hihat_open": 46, "ride": 51, "crash": 49, "splash": 55, "china": 52, "tambourine": 54, "shaker": 70}


def write_midi(events, path, tempo):
    """MIDI at RAW detected times (not quantized). Velocity: 100, ghost notes 45, flam grace 60 (no
    real dynamics model yet)."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Drum Set")
    for e in (e for e in events if e.notated):  # hits dropped as inaudible ticks are not in the score or the MIDI
        vel = 60 if e.articulation == "grace" else 45 if e.articulation == "ghost" else 100
        pitch = 38 if e.articulation == "grace" else GM_PITCH[note_key(e)]
        drums.notes.append(pretty_midi.Note(vel, pitch, e.time_s, e.time_s + 0.1))
    midi.instruments.append(drums)
    midi.write(str(path))


def write_json(events, meta, path):
    doc = {"schema_version": SCHEMA_VERSION, **meta, "events": [e.to_dict() for e in events]}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
