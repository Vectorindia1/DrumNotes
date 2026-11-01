"""DrumScribe web API. Run:  uvicorn backend.api.main:app --host 127.0.0.1 --port 8000

Flow: POST /api/upload (raw file body) -> POST /api/jobs/{id}/start -> poll GET /api/jobs/{id}
      -> GET /files/{id}/... (PDF, MusicXML, MIDI, JSON, the player). DELETE /api/jobs/{id} removes everything.
No accounts: a job is reachable only through its unguessable id.
"""
import os
import pathlib
import re
import shutil
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager

import anyio
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.api import jobs
from backend.intake import UnsupportedSource, readable
from backend.player.score_audio import KITS

STATIC = pathlib.Path(__file__).parent / "static"
FORMAT_EXT = {"WAV": ".wav", "FLAC": ".flac", "OGG": ".ogg", "MP3": ".mp3"}  # what libsndfile reads directly
MIN_SECONDS = 3.0
runner = None


@asynccontextmanager
async def lifespan(_app):
    global runner
    jobs.recover_after_restart()
    jobs.cleanup()
    jobs.start_cleanup_thread()
    runner = jobs.Runner()
    yield


app = FastAPI(title="DrumScribe", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                               "media-src 'self'; frame-ancestors 'self'")
    return resp


def _job_or_404(job_id):
    job = jobs.read(job_id) if jobs.ID_RE.match(job_id) else None
    if job is None:
        raise HTTPException(404, "That score doesn't exist (it may have been deleted or expired).")
    return job


def _public(job):
    keys = ("id", "title", "filename", "state", "stage", "progress", "error", "duration_s", "size_mb", "estimate",
            "params", "result", "created", "started")
    out = {k: job.get(k) for k in keys}
    if os.environ.get("DRUMSCRIBE_DEBUG") == "1":
        out["detail"] = job.get("detail")
    out["queue_position"] = runner.position(job["id"]) if runner else 0
    out["limits"] = {"max_minutes": jobs.MAX_MINUTES, "max_separate_minutes": jobs.MAX_SEPARATE_MINUTES,
                     "retention_hours": jobs.RETENTION_H}
    return out


def _title_from(filename):
    stem = pathlib.PurePath(filename).stem
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", stem).replace("_", " ")
    return re.sub(r"\s+", " ", stem).strip()[:80] or "Untitled"


def _mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _decoded_seconds(path):
    """(seconds the header claims, seconds that actually decode). A truncated mp3 keeps a header that
    describes the whole song, so only decoding tells the truth."""
    with sf.SoundFile(str(path)) as f:
        n = 0
        while len(block := f.read(1 << 16, dtype="float32", always_2d=True)):  # same decoder the analysis uses
            n += len(block)
        return f.frames / f.samplerate, n / f.samplerate


def _inspect(d, raw, filename):
    """Decode check + canonical file name. Returns (duration_s, audio_path). Raises ValueError (user-facing)."""
    try:
        ready = readable(raw, dest_dir=d)
        info = sf.info(str(ready))
        claimed, duration = _decoded_seconds(ready)
    except (UnsupportedSource, sf.LibsndfileError, RuntimeError):
        raise ValueError("We couldn't read that file as audio. Try an mp3, wav, flac, m4a, ogg or opus file.") from None
    if claimed > duration + max(2.0, 0.05 * claimed):
        raise ValueError(f"That file looks incomplete or damaged: it says it is {_mmss(claimed)} long but only {_mmss(duration)} of "
                         "audio could be read. If you copied or downloaded it, make sure that finished, then upload it again.")
    ext = FORMAT_EXT.get(info.format, ".audio") if ready == raw else ".wav"  # libsndfile and ffmpeg sniff content, not names
    audio = d / f"audio{ext}"
    ready.replace(audio)
    return duration, audio


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def health():
    return {"ok": True, "queue": runner.active_count() if runner else 0, "limits": {
        "max_upload_mb": jobs.MAX_UPLOAD_MB, "max_minutes": jobs.MAX_MINUTES, "retention_hours": jobs.RETENTION_H}}


@app.post("/api/upload")
async def upload(request: Request):
    filename = urllib.parse.unquote(request.headers.get("x-filename", "audio"))[:200]
    limit = int(jobs.MAX_UPLOAD_MB * 1024 * 1024)
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, f"That file is too large (limit {jobs.MAX_UPLOAD_MB:.0f} MB).")
    job_id = uuid.uuid4().hex
    d = jobs.DATA / job_id
    d.mkdir(parents=True)
    raw, size = d / "upload.bin", 0
    try:
        async with await anyio.open_file(raw, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"That file is too large (limit {jobs.MAX_UPLOAD_MB:.0f} MB).")
                await f.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty.")
        expected = int(request.headers.get("content-length") or 0)
        if expected and size != expected:
            raise HTTPException(400, "The upload was interrupted before it finished. Please try again.")
        try:
            duration, _ = await run_in_threadpool(_inspect, d, raw, filename)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if duration < MIN_SECONDS:
            raise HTTPException(422, "That recording is too short (minimum 3 seconds).")
        if duration > jobs.MAX_MINUTES * 60:
            raise HTTPException(413, f"That recording is {duration / 60:.1f} minutes long; the limit is {jobs.MAX_MINUTES:.0f} minutes.")
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    raw.unlink(missing_ok=True)
    job = {"id": job_id, "state": "uploaded", "filename": pathlib.PurePath(filename).name[:200], "title": _title_from(filename),
           "duration_s": round(duration, 1), "size_mb": round(size / 1e6, 1), "estimate": jobs.estimate_minutes(duration),
           "created": time.time(), "updated": time.time(), "params": {}}
    jobs.write(job_id, job)
    return _public(job)


class StartBody(BaseModel):
    rights: bool = False
    tempo: float | None = None
    separate: bool = False
    kit: str = "standard"


@app.post("/api/jobs/{job_id}/start")
async def start(job_id: str, body: StartBody):
    job = _job_or_404(job_id)
    if job["state"] in ("queued", "running"):
        raise HTTPException(409, "This score is already being created.")
    if not body.rights:
        raise HTTPException(400, "Please confirm you have the right to use this audio.")
    if body.tempo is not None and not 30 <= body.tempo <= 300:
        raise HTTPException(422, "Tempo must be between 30 and 300 BPM.")
    if body.kit not in KITS:
        raise HTTPException(422, f"Unknown drum kit. Choose one of: {', '.join(sorted(KITS))}.")
    if body.separate and job["duration_s"] > jobs.MAX_SEPARATE_MINUTES * 60:
        raise HTTPException(422, f"Drum separation is limited to {jobs.MAX_SEPARATE_MINUTES:.0f} minutes of audio. "
                                 "Turn it off or use a shorter file.")
    if runner.active_count() >= jobs.MAX_QUEUE:
        raise HTTPException(429, "The queue is full right now. Please try again in a few minutes.")
    d = jobs.job_dir(job_id)
    shutil.rmtree(d / "out", ignore_errors=True)
    shutil.rmtree(d / "stems", ignore_errors=True)
    updated = jobs.update(job_id, state="queued", stage="Waiting in line", progress=0.0, error=None, result=None,
                          params={"tempo": body.tempo, "separate": body.separate, "kit": body.kit})
    runner.submit(job_id)
    return _public(updated)


@app.get("/api/jobs/{job_id}")
async def status(job_id: str):
    return _public(_job_or_404(job_id))


@app.delete("/api/jobs/{job_id}")
async def delete(job_id: str):
    _job_or_404(job_id)
    runner.cancel(job_id)
    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
    return JSONResponse({"deleted": True})


@app.get("/files/{job_id}/{rel:path}")
async def files(job_id: str, rel: str, download: bool = False):
    job = _job_or_404(job_id)
    root = (jobs.job_dir(job_id) / "out").resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "File not found.")
    name = f"{job['title']}{target.suffix}" if download else None
    return FileResponse(target, filename=name, headers={"Cache-Control": "private, max-age=3600"})

