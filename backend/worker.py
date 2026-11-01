"""Run ONE web job in its own process (a crash or out-of-memory kill cannot take the website down).

    python -m backend.worker JOB_DIR

Reads JOB_DIR/job.json, writes JOB_DIR/out/** and JOB_DIR/result.json. Progress goes to stdout as
`PROGRESS {json}` lines and a failure as `ERROR {json}`; everything else (library noise) goes to stderr.
"""
import json
import pathlib
import shutil
import sys

from backend.pipeline import run
from backend.rhythm.grid import TempoUndetermined

ART_LABELS = {"snare:ghost": "ghost_notes", "hihat:open": "open_hihats", "cymbal:ride": "ride", "cymbal:crash": "crash",
              "cymbal:splash": "splash", "cymbal:china": "china", "snare:flam": "flams"}


def emit(kind, **payload):
    print(kind + " " + json.dumps(payload), flush=True)


def summarise(report, params, title, duration):
    rhythm, arts, hat = report["rhythm"], report["articulations"], report.get("hihat", {})
    notes = []
    if hat.get("applied"):
        if hat["mode"] == "auto" and hat["dropped_quiet_ticks"] > 20 and hat["bars_eighths_only"] > 0:
            notes.append(f"Hi-hat: in {hat['bars_eighths_only']} bars the sixteenth notes between the eighths are much quieter than the eighths, "
                         "so only the eighths are written. Choose \"All sixteenths\" under More options to include them.")
        if hat["recovered_from_audio"] + hat["filled_from_pattern"] > 0:
            notes.append(f"Hi-hat: {hat['recovered_from_audio'] + hat['filled_from_pattern']} hits the detector missed were added from the audio "
                         f"and the repeating pattern, and {hat['dropped_no_energy'] + hat['dropped_quiet_ticks']} detections with no real sound behind them were removed.")
    elif hat.get("reason"):
        notes.append("Hi-hat line left as the detector found it: " + hat["reason"].split(" (")[0] + ".")
    if report["n_events"] < 20:
        notes.append("Very few drum hits were found. Is there drum audio in this file?")
    if rhythm.get("tempo_note"):
        notes.append(rhythm["tempo_note"])
    elif not params.get("tempo"):
        notes.append(f"The tempo ({rhythm['tempo_bpm']:.1f} BPM) was worked out from the drum pattern. If the score feels too fast "
                     "or too slow, try one of the other tempos above.")
    if "LOW CONFIDENCE" in rhythm["downbeat"]:
        notes.append("Where each bar starts is a best guess, so the bar lines may be shifted. The notes themselves are unaffected.")
    if "error" in arts:
        notes.append("Ghost notes, open hi-hats and cymbal types could not be analysed for this file, so the score shows plain notes.")
    else:
        notes.append("Ghost notes, open hi-hats and cymbal types are estimates. Flams are not detected yet. Check the score by ear.")
    return {
        "title": title, "duration_s": round(duration, 1), "tempo_bpm": round(rhythm["tempo_bpm"], 1),
        "tempo_detected": not params.get("tempo"), "tempo_alternatives": rhythm.get("alternatives_unresolved", [])[:6], "bars": report["n_bars"], "hits": report["counts"],
        "articulations": {ART_LABELS[k]: v for k, v in arts.items() if k in ART_LABELS and v},
        "notes": notes, "validation_ok": report["ok"],
    }


def friendly(exc):
    if isinstance(exc, TempoUndetermined):
        return "Couldn't find a steady beat in this recording. Enter the tempo under Advanced and try again."
    if isinstance(exc, MemoryError):
        return "The server ran out of memory. Try again without drum separation."
    return "Something went wrong while analysing this file."


def main(job_dir):
    job_dir = pathlib.Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    p = job["params"]
    out = job_dir / "out"
    shutil.rmtree(out, ignore_errors=True)
    audio = next(job_dir.glob("audio.*"))
    try:
        files, report, _ = run(audio, out, tempo_hint=p.get("tempo"), separate=p.get("separate", False), player=True,
                               kit=p.get("kit", "standard"), hat_detail=p.get("hat_detail", "auto"), title=job["title"], stem_dir=job_dir / "stems",
                               progress=lambda stage, frac: emit("PROGRESS", stage=stage, frac=frac))
        if not report["ok"]:
            raise RuntimeError("; ".join(report["validation"]["musicxml_problems"] + report["validation"]["pdf_problems"]))
        rel = {k: str(pathlib.Path(v).relative_to(out)) for k, v in files.items() if k != "report"}
        result = {"summary": summarise(report, p, job["title"], job["duration_s"]), "files": rel}
        (job_dir / "result.json").write_text(json.dumps(result))
        shutil.rmtree(job_dir / "stems", ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - last line of defence: report, then exit non-zero
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit("ERROR", message=friendly(exc), detail=f"{type(exc).__name__}: {exc}"[:300])
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
