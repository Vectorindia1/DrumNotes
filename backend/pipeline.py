"""Full-mix audio -> events -> grid -> MusicXML -> PDF + MIDI + JSON + report (M1).

CLI: python -m backend.pipeline AUDIO [--tempo BPM] [--offset SEC] [--beats-per-bar N]
"""
import argparse
import copy
import json
import math
import pathlib
import sys
import traceback
from collections import Counter

import librosa
import numpy as np

from backend.detection import adtof, articulation
from backend.engraving.musescore import render_pdf
from backend.events import SCHEMA_VERSION
from backend.exports import write_json, write_midi
from backend.intake import UnsupportedSource, resolve_input
from backend.notation.drum_musicxml import bars_to_musicxml
from backend.reconstruction.bars import build_bars
from backend.rhythm.fill import fill_dropouts
from backend.rhythm.grid import calibrate_phase, estimate_grid, fit_bar_phase, quantize
from backend.rhythm.hats import regularize_hats
from backend.separation.demucs_sep import separate_drums
from backend.validation.musicxml import validate_musicxml
from backend.validation.pdf import validate_pdf


def grid_lock(events, grid):
    """0..1: how tightly the kick/snare/hi-hat hits sit on the 16th grid (1 = exactly)."""
    errs = np.array([e.quantization_error_s for e in events if e.instrument in ("kick", "snare", "hihat") and e.grace_of is None])
    slot = 60.0 / grid.tempo_bpm / 4
    return float(abs(np.exp(2j * np.pi * errs / slot).mean())) if len(errs) else 0.0


def fix_hihat_line(events, grid, env, n_bars, hat_detail):
    """Optional simplification (hat_detail="eighths"): drop in-between 16th hats that are much quieter than the eighths.
    Off by default: on the MDB Drums benchmark (23 real songs) every rebuild variant lowered hi-hat F1
    (0.855 as detected vs 0.734-0.842; it deleted real quiet 16ths, e.g. Disco 0.96 -> 0.58), see PROGRESS.md.
    Never adds hats. Never raises."""
    if hat_detail != "eighths":
        return {"applied": False, "reason": "hi-hat line as detected (default)"}
    lock = grid_lock(events, grid)
    if lock < 0.5:
        return {"applied": False, "reason": f"hits are not locked to a steady grid (lock {lock:.2f}); hi-hat line left as detected"}
    try:
        new, info = regularize_hats(events, grid, env, n_bars, "eighths", allow_additions=False)
    except Exception as exc:  # noqa: BLE001 - optional improvement, never fatal
        traceback.print_exc(file=sys.stderr)
        return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    events.extend(new)
    return {"applied": "skipped" not in info, "grid_lock": round(lock, 2), **info}


def fill_groove(events, grid, env, n_bars, lock):
    """Fill KICK dropouts inside a repeating groove when the audio agrees (rhythm/fill.py). Never raises.
    MDB Drums benchmark: kick recall 0.865 -> 0.886 (F1 0.845 -> 0.847); for snare it gave no gain and lower
    precision, so snare is not filled."""
    if lock < 0.5:
        return {"applied": False}
    try:
        added = {}
        for inst in ("kick",):
            new = fill_dropouts(events, grid, env, n_bars, inst)
            events.extend(new)
            added[inst] = len(new)
        return {"applied": True, **added}
    except Exception as exc:  # noqa: BLE001 - optional improvement, never fatal
        traceback.print_exc(file=sys.stderr)
        return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}


def run(audio_path, out_dir="outputs", tempo_hint=None, beats_per_bar=4, offset=None, title=None, beat1=None,
        separate=False, player=False, kit="standard", progress=None, stem_dir="data/stems", hat_detail="auto"):
    """progress(stage, fraction) is called as each stage starts (used by the web worker)."""
    def tick(stage, frac):
        if progress:
            progress(stage, frac)

    audio_path, out = pathlib.Path(audio_path), pathlib.Path(out_dir)
    tick("Reading the audio", 0.02)
    stem = audio_path.stem
    for sub in ("musicxml", "pdf", "midi", "json"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    duration = librosa.get_duration(path=str(audio_path))

    detect_path, stem_path = audio_path, None
    if separate:  # detection + articulation analysis on the Demucs drum stem; tempo still from the mix
        tick("Separating the drums from the mix", 0.05)
        stem_path = separate_drums(audio_path, pathlib.Path(stem_dir) / f"{stem}.drums.wav")
        detect_path = stem_path
    tick("Finding the drum hits", 0.55 if separate else 0.10)
    events, det_meta, pred = adtof.detect(detect_path, return_activations=True)
    tick("Telling ghost notes, open hi-hats and cymbals apart", 0.65 if separate else 0.30)
    y, _ = librosa.load(str(detect_path), sr=articulation.SR, mono=True)
    env = articulation.Env(articulation.band_power(y))
    try:  # optional extras (ghost/open/cymbal labels, refined times): a failure must not cost the score
        events, art_summary = articulation.annotate(copy.deepcopy(events), y, pred, adtof.pick, env=env)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        art_summary = {"error": f"{type(exc).__name__}: {exc}"[:200]}  # `events` is untouched: plain notation
    del y
    tick("Working out tempo and bar lines", 0.72 if separate else 0.45)
    grid = estimate_grid(audio_path, duration, tempo_hint, beats_per_bar, offset, events=events)
    calibrate_phase(events, grid)
    fit_bar_phase(events, grid, beat1)
    quantize(events, grid)
    n_bars = grid.n_bars(duration)
    hat_info = fix_hihat_line(events, grid, env, n_bars, hat_detail)
    fill_info = fill_groove(events, grid, env, n_bars, grid_lock(events, grid))
    events = sorted(events, key=lambda e: (e.time_s, e.instrument))
    for i, e in enumerate(events):
        e.id = f"e{i:05d}"
    bars, collisions = build_bars(events, n_bars, grid.slots_per_bar)

    paths = {
        "musicxml": out / "musicxml" / f"{stem}.musicxml",
        "pdf": out / "pdf" / f"{stem}.pdf",
        "midi": out / "midi" / f"{stem}.mid",
        "json": out / "json" / f"{stem}.events.json",
        "report": out / "json" / f"{stem}.report.json",
    }
    tick("Writing the notation", 0.78 if separate else 0.55)
    paths["musicxml"].write_text(bars_to_musicxml(bars, title or stem, grid.tempo_bpm, beats_per_bar))
    xml_problems = validate_musicxml(paths["musicxml"], expected_bars=len(bars))
    tick("Engraving the PDF", 0.82 if separate else 0.65)
    render_pdf(paths["musicxml"], paths["pdf"])
    pdf_info, pdf_problems = validate_pdf(paths["pdf"])
    write_midi(events, paths["midi"], grid.tempo_bpm)

    errs = [abs(e.quantization_error_s) for e in events]
    report = {
        "schema_version": SCHEMA_VERSION,
        "input": {"file": audio_path.name, "duration_s": round(duration, 3)},
        "detector": det_meta,
        "hihat": hat_info,
        "groove_fill": fill_info,
        "separation": {"used": separate, "model": "htdemucs (weights terms unverified)" if separate else None,
                       "stem": str(stem_path) if stem_path else None},
        "rhythm": grid.describe(),
        "counts": dict(Counter(e.instrument for e in events)),
        "articulations": {"status": ("FAILED: plain notation used" if "error" in art_summary
                                      else "EXPERIMENTAL, uncalibrated heuristics; verify by ear"), **art_summary},
        "n_events": len(events),
        "n_bars": len(bars),
        "quantization": {
            "grid": "16th notes only (no swing/tuplets in M1)",
            "max_error_ms": round(max(errs, default=0) * 1000, 1),
            "events_over_half_slot_ms": sum(e > 0.5 * 60 / grid.tempo_bpm / 4 for e in errs),
            "same_slot_duplicates_merged": collisions,
        },
        "validation": {
            "musicxml_problems": xml_problems,
            "pdf": pdf_info,
            "pdf_problems": pdf_problems,
            "pdf_scope": "valid file, pages>0, page 1 non-blank; per-bar PDF coverage is checked at the MusicXML level only",
            "bars_cover_duration": len(bars) >= math.ceil(grid.beat_position(duration) / beats_per_bar - 1e-9),
        },
        "ok": not (xml_problems or pdf_problems),
    }
    write_json(events, {"source": audio_path.name, "duration_s": duration, "rhythm": grid.describe()}, paths["json"])
    if player:
        tick("Building the playback player", 0.90 if separate else 0.80)
        from backend.player.build import build_player
        sources = {"original": ("Original recording", audio_path)}
        if stem_path:
            sources["stem"] = ("Separated drum stem", stem_path)
        paths["player"] = build_player(paths["musicxml"], out / "player", grid, bars, title or stem,
                                       grid.tempo_bpm, sources, kit=kit)
    paths["report"].write_text(json.dumps(report, indent=1))
    tick("Done", 1.0)
    return {k: str(v) for k, v in paths.items()}, report, events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="audio file path or a direct link to an audio file (YouTube/Spotify links are not supported)")
    ap.add_argument("--tempo", type=float)
    ap.add_argument("--offset", type=float, help="time of beat 1 (s); with --tempo makes the grid exact")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--beat1", type=int, help="which tracked beat (0-3) is beat 1; overrides the downbeat guess")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--separate", action="store_true", help="detect on the Demucs drum stem (slow on CPU)")
    ap.add_argument("--kit", default="standard", help="drum kit for the score audio: standard, room, power, electronic, 808, jazz, brush")
    ap.add_argument("--hats", default="auto", choices=("auto", "eighths"), help="hi-hat: auto = as detected (default); eighths = simplify by dropping quiet in-between 16ths")
    ap.add_argument("--player", action="store_true", help="also build the blue-playhead score player")
    a = ap.parse_args()
    try:
        a.audio = str(resolve_input(a.audio))
    except UnsupportedSource as e:
        raise SystemExit(f"error: {e}") from None
    files, rep, _ = run(a.audio, a.out, a.tempo, a.beats_per_bar, a.offset, beat1=a.beat1, separate=a.separate, player=a.player, kit=a.kit, hat_detail=a.hats)
    print(json.dumps({"files": files, "report": rep}, indent=1))
