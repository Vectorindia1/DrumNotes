"""Render the NOTATED drum hits through a real sampled drum kit (SoundFont, via libfluidsynth).

MuseScore's own command-line audio export plays the part with a tonal, piano-like patch (98 % harmonic
energy measured), so the score audio is rendered here instead: every note on the page becomes a GM drum
note at its exact bar/slot time (read from MuseScore's measure positions), ghost notes soft, flams as a
quiet grace note. Soundfont: MuseScore_General (MIT) if present, else TimGM6mb (GPL-2).
"""
import os

import numpy as np
import soundfile as sf

from backend.exports import GM_PITCH

SOUNDFONTS = ["/usr/share/sounds/sf3/MuseScore_General_Lite.sf3", "/usr/share/sounds/sf2/TimGM6mb.sf2"]
KITS = {"standard": 0, "room": 8, "power": 16, "electronic": 24, "808": 25, "jazz": 32, "brush": 40}
GHOST_VEL, GRACE_VEL, NORMAL_VEL, GRACE_OFFSET_S = 42, 55, 100, 0.035
SR = 44100
NOTE_OFF_AFTER_S = 0.6  # one-shot kit samples ignore it; it only frees the voice


def soundfont_path():
    return next((p for p in SOUNDFONTS if os.path.exists(p)), None)


def score_notes(bars, bar_starts):
    """[(time_s, gm_note, velocity)] for every note in `bars` (sets of note keys per slot).
    bar_starts has len(bars)+1 entries: the start time of each bar and the end of the last."""
    notes = []
    for b, bar in enumerate(bars):
        dur = bar_starts[b + 1] - bar_starts[b]
        for s, keys in enumerate(bar):
            t = bar_starts[b] + s / len(bar) * dur
            for key in keys:
                if key == "snare_flam":
                    notes.append((max(t - GRACE_OFFSET_S, 0.0), GM_PITCH["snare"], GRACE_VEL))
                notes.append((t, GM_PITCH[key], GHOST_VEL if key == "snare_ghost" else NORMAL_VEL))
    return sorted(notes)


def render(notes, out_path, kit="standard", sr=SR, tail_s=3.0, soundfont=None):
    """Write a stereo FLAC (lossless: no encoder delay to desynchronise the playhead)."""
    import fluidsynth  # pyfluidsynth; needs libfluidsynth on the system
    path = soundfont or soundfont_path()
    if path is None:
        raise RuntimeError("no drum SoundFont found (looked for " + ", ".join(SOUNDFONTS) + ")")
    if kit not in KITS:
        raise ValueError(f"unknown kit {kit!r}; choose from {sorted(KITS)}")
    synth = fluidsynth.Synth(gain=0.8, samplerate=float(sr))
    try:
        sfid = synth.sfload(path)
        if sfid < 0 or synth.program_select(9, sfid, 128, KITS[kit]) != 0:
            raise RuntimeError(f"kit {kit!r} (bank 128, preset {KITS[kit]}) missing in {path}")
        events = [(t, 1, p, v) for t, p, v in notes] + [(t + NOTE_OFF_AFTER_S, 0, p, 0) for t, p, _ in notes]
        end = int(((max((t for t, *_ in events), default=0)) + tail_s) * sr)
        chunks, cur = [], 0
        for t, on, pitch, vel in sorted(events, key=lambda e: (e[0], e[1])):
            at = min(round(t * sr), end)
            if at > cur:
                chunks.append(synth.get_samples(at - cur))
                cur = at
            synth.noteon(9, pitch, vel) if on else synth.noteoff(9, pitch)
        chunks.append(synth.get_samples(max(end - cur, 0)))
    finally:
        synth.delete()
    audio = np.concatenate(chunks).reshape(-1, 2).astype(np.float32) / 32768
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio *= 0.9 / peak
    sf.write(out_path, audio, sr)
    return out_path
