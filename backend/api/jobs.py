"""Job store (one folder per job, state in job.json), a single-worker queue and cleanup."""
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("DRUMSCRIBE_DATA") or ROOT / "data" / "jobs")
MAX_UPLOAD_MB = float(os.environ.get("DRUMSCRIBE_MAX_UPLOAD_MB", "200"))
MAX_MINUTES = float(os.environ.get("DRUMSCRIBE_MAX_MINUTES", "12"))
MAX_SEPARATE_MINUTES = float(os.environ.get("DRUMSCRIBE_MAX_SEPARATE_MINUTES", "8"))
RETENTION_H = float(os.environ.get("DRUMSCRIBE_RETENTION_HOURS", "72"))
TIMEOUT_MIN = float(os.environ.get("DRUMSCRIBE_JOB_TIMEOUT_MIN", "60"))
MAX_QUEUE = int(os.environ.get("DRUMSCRIBE_MAX_QUEUE", "20"))
ID_RE = re.compile(r"^[0-9a-f]{32}$")
_lock = threading.Lock()


def job_dir(job_id):
    if not ID_RE.match(job_id or ""):
        raise KeyError(job_id)
    return DATA / job_id


def read(job_id):
    try:
        return json.loads((job_dir(job_id) / "job.json").read_text())
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        return None


def write(job_id, data):
    d = job_dir(job_id)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(data))
    os.replace(tmp, d / "job.json")  # atomic: readers never see half a file


def update(job_id, **fields):
    with _lock:
        job = read(job_id)
        if job is None:
            return None
        job.update(fields, updated=time.time())
        write(job_id, job)
        return job


def estimate_minutes(duration_s):
    """Rough wall-clock estimates measured on the 8-core dev machine (about 0.2x and 2x real time)."""
    return {"standard": round(0.5 + duration_s * 0.25 / 60, 1), "separated": round(1 + duration_s * 2.2 / 60, 1)}


class Runner:
    """Runs queued jobs one at a time, each in a subprocess (isolated from the web server)."""

    def __init__(self):
        self.q = queue.Queue()
        self.waiting = []
        self.proc = None
        self.current = None
        self.cancelled = set()
        threading.Thread(target=self._loop, daemon=True).start()

    def active_count(self):
        return len(self.waiting) + (1 if self.current else 0)

    def position(self, job_id):
        return self.waiting.index(job_id) + 1 if job_id in self.waiting else 0

    def submit(self, job_id):
        self.waiting.append(job_id)
        self.q.put(job_id)

    def cancel(self, job_id):
        self.cancelled.add(job_id)
        if job_id in self.waiting:
            self.waiting.remove(job_id)
        if self.current == job_id and self.proc:
            self.proc.kill()

    def _loop(self):
        while True:
            job_id = self.q.get()
            if job_id in self.waiting:
                self.waiting.remove(job_id)
            if job_id in self.cancelled or (read(job_id) or {}).get("state") != "queued":
                self.cancelled.discard(job_id)
                continue
            self.current = job_id
            try:
                self._run(job_id)
            except Exception as exc:  # noqa: BLE001 - never let the worker thread die
                update(job_id, state="failed", error="Something went wrong on the server.", detail=str(exc)[:300])
            finally:
                self.current, self.proc = None, None

    def _run(self, job_id):
        d = job_dir(job_id)
        (d / "result.json").unlink(missing_ok=True)
        update(job_id, state="running", stage="Starting", progress=0.0, started=time.time(), error=None, detail=None)
        error = None
        with open(d / "worker.log", "w") as log:
            self.proc = subprocess.Popen([sys.executable, "-m", "backend.worker", str(d)], cwd=ROOT, text=True,
                                         stdout=subprocess.PIPE, stderr=log)
            timer = threading.Timer(TIMEOUT_MIN * 60, self.proc.kill)
            timer.start()
            for line in self.proc.stdout:
                kind, _, payload = line.partition(" ")
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if kind == "PROGRESS":
                    update(job_id, stage=data["stage"], progress=data["frac"])
                elif kind == "ERROR":
                    error = data
            code = self.proc.wait()
            timer.cancel()
        if job_id in self.cancelled:
            self.cancelled.discard(job_id)
            return
        result = d / "result.json"
        if code == 0 and result.exists():
            update(job_id, state="done", stage="Done", progress=1.0, result=json.loads(result.read_text()))
        elif code in (-9, 137) and not error:
            update(job_id, state="failed", error="The job ran out of time or memory. Try a shorter file, or turn off drum separation.")
        else:
            update(job_id, state="failed", error=(error or {}).get("message", "Something went wrong while analysing this file."),
                   detail=(error or {}).get("detail"))


def recover_after_restart():
    """Jobs left queued/running by a previous server process cannot continue."""
    DATA.mkdir(parents=True, exist_ok=True)
    for d in DATA.iterdir():
        job = read(d.name) if ID_RE.match(d.name) else None
        if job and job.get("state") in ("queued", "running"):
            update(d.name, state="failed", error="The server restarted while this was running. Please create it again.")


def cleanup(now=None):
    """Delete jobs not touched for RETENTION_H hours. Returns how many were removed."""
    now = now or time.time()
    removed = 0
    DATA.mkdir(parents=True, exist_ok=True)
    for d in DATA.iterdir():
        if ID_RE.match(d.name) and d.is_dir():
            job = read(d.name) or {}
            if job.get("state") not in ("queued", "running") and now - job.get("updated", d.stat().st_mtime) > RETENTION_H * 3600:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed


def start_cleanup_thread(every_s=1800):
    def loop():
        while True:
            time.sleep(every_s)
            cleanup()
    threading.Thread(target=loop, daemon=True).start()
