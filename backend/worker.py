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
    rhythm, arts = report["rhythm"], report["articulations"]
    notes = []
    if report["n_events"] < 20:
        notes.append("Very few drum hits were found. Is there drum audio in this file?")
    if not params.get("tempo"):
        notes.append(f"The tempo was detected automatically ({rhythm['tempo_bpm']:.0f} BPM). If the score feels twice or "
                     "half as fast as the song, enter the tempo under Advanced and create it again.")
    if "LOW CONFIDENCE" in rhythm["downbeat"]:
        notes.append("Where each bar starts is a best guess, so the bar lines may be shifted. The notes themselves are unaffected.")
    notes.append("Ghost notes, open hi-hats and cymbal types are estimates. Flams are not detected yet. Check the score by ear.")
    return {
        "title": title, "duration_s": round(duration, 1), "tempo_bpm": round(rhythm["tempo_bpm"], 1),
        "tempo_detected": not params.get("tempo"), "bars": report["n_bars"], "hits": report["counts"],
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
                               kit=p.get("kit", "standard"), title=job["title"], stem_dir=job_dir / "stems",
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
