import io
import json
import time
import urllib.parse

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.api import jobs
from backend.api import main as api
from backend.validation.synth import ROCK, SR, render


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "DATA", tmp_path / "jobs")
    with TestClient(api.app) as c:
        yield c


def wav_bytes(audio, sr=SR):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def up(client, data, name="My Song_demo.wav"):
    return client.post("/api/upload", content=data, headers={"X-Filename": urllib.parse.quote(name)})


@pytest.fixture(scope="module")
def rock():
    audio, _ = render(ROCK, 100, 8)
    return wav_bytes(audio)


def wait_done(client, job_id, timeout=150):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_site_serves_with_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "DrumScribe" in r.text
    assert r.headers["x-content-type-options"] == "nosniff" and "frame-ancestors 'self'" in r.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200 and client.get("/static/style.css").status_code == 200
    assert client.get("/api/health").json()["limits"]["max_minutes"] == jobs.MAX_MINUTES


def test_upload_validation(client, monkeypatch, rock):
    r = up(client, rock)
    assert r.status_code == 200
    j = r.json()
    assert j["state"] == "uploaded" and j["title"] == "My Song demo" and 19 < j["duration_s"] < 22 and j["estimate"]["separated"] > j["estimate"]["standard"]
    assert client.get(f"/api/jobs/{j['id']}").json()["filename"] == "My Song_demo.wav"
    assert up(client, b"this is not audio" * 100).status_code == 422
    assert up(client, b"").status_code == 400
    assert up(client, wav_bytes(np.zeros(SR, np.float32))).status_code == 422  # 1 s: too short
    before = len(list(jobs.DATA.iterdir()))
    monkeypatch.setattr(jobs, "MAX_UPLOAD_MB", 0.01)
    assert up(client, rock).status_code == 413
    monkeypatch.setattr(jobs, "MAX_UPLOAD_MB", 200)
    monkeypatch.setattr(jobs, "MAX_MINUTES", 0.1)
    assert up(client, rock).status_code == 413
    assert len(list(jobs.DATA.iterdir())) == before  # rejected uploads leave nothing behind


def test_start_validation(client, monkeypatch, rock):
    jid = up(client, rock).json()["id"]
    start = lambda **b: client.post(f"/api/jobs/{jid}/start", json=b)
    assert start(rights=False).status_code == 400
    assert start(rights=True, tempo=900).status_code == 422
    assert start(rights=True, kit="theremin").status_code == 422
    assert start(rights=True, hat_detail="thirty-seconds").status_code == 422
    monkeypatch.setattr(jobs, "MAX_SEPARATE_MINUTES", 0.1)
    assert start(rights=True, separate=True).status_code == 422


def test_ids_and_paths_cannot_escape(client, rock):
    jid = up(client, rock).json()["id"]
    for bad in ("../../job.json", "..%2F..%2Fjob.json", "%2e%2e/%2e%2e/job.json", "upload.bin", "../audio.wav"):
        assert client.get(f"/files/{jid}/{bad}").status_code == 404
    assert client.get("/files/not-an-id/x").status_code == 404
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/jobs/../etc").status_code in (404, 405)
    assert client.delete("/api/jobs/" + "f" * 32).status_code == 404


def test_full_job_end_to_end_and_delete(client, rock):
    jid = up(client, rock, "Rock Beat.wav").json()["id"]
    r = client.post(f"/api/jobs/{jid}/start", json={"rights": True, "tempo": 100, "kit": "standard"})
    assert r.status_code == 200 and r.json()["state"] == "queued"
    j = wait_done(client, jid)
    assert j["state"] == "done", j.get("error")
    s = j["result"]["summary"]
    assert s["tempo_bpm"] == 100 and s["bars"] >= 8 and s["hits"]["kick"] > 15 and not s["tempo_detected"] and s["validation_ok"]
    f = j["result"]["files"]
    pdf = client.get(f"/files/{jid}/{f['pdf']}?download=true")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF" and "attachment" in pdf.headers["content-disposition"] and "Rock%20Beat.pdf" in pdf.headers["content-disposition"]
    assert client.get(f"/files/{jid}/{f['midi']}").content[:4] == b"MThd"
    assert json.loads(client.get(f"/files/{jid}/{f['json']}").text)["events"]
    page = client.get(f"/files/{jid}/{f['player']}")
    assert page.status_code == 200 and "data.js" in page.text
    assert client.get(f"/files/{jid}/player/data.js").status_code == 200
    rng = client.get(f"/files/{jid}/player/score.flac", headers={"Range": "bytes=0-99"})  # audio seeking needs Range
    assert rng.status_code == 206 and len(rng.content) == 100 and rng.headers["content-range"].startswith("bytes 0-99/")
    # re-running with other settings reuses the upload
    assert client.post(f"/api/jobs/{jid}/start", json={"rights": True, "tempo": 100, "kit": "jazz"}).status_code == 200
    assert wait_done(client, jid)["state"] == "done"
    assert client.delete(f"/api/jobs/{jid}").json() == {"deleted": True}
    assert client.get(f"/api/jobs/{jid}").status_code == 404 and not (jobs.DATA / jid).exists()


def test_unanalysable_audio_fails_with_a_helpful_message(client):
    jid = up(client, wav_bytes(np.zeros(SR * 6, np.float32)), "silence.wav").json()["id"]
    client.post(f"/api/jobs/{jid}/start", json={"rights": True})  # no tempo: the beat tracker has nothing to find
    j = wait_done(client, jid)
    assert j["state"] == "failed" and "tempo" in j["error"].lower() and "detail" not in j  # no internals leaked
    assert client.get(f"/files/{jid}/pdf/audio.pdf").status_code == 404


def test_cleanup_removes_only_old_finished_jobs(client, rock):
    old, fresh = up(client, rock).json()["id"], up(client, rock).json()["id"]
    busy = up(client, rock).json()["id"]
    stale = time.time() - (jobs.RETENTION_H + 1) * 3600
    for jid, extra in ((old, {}), (busy, {"state": "running"})):  # jobs.update() would re-stamp "updated"
        jobs.write(jid, {**jobs.read(jid), **extra, "updated": stale})
    assert jobs.cleanup() == 1
    assert not (jobs.DATA / old).exists() and (jobs.DATA / fresh).exists() and (jobs.DATA / busy).exists()


def test_jobs_interrupted_by_a_restart_are_marked_failed(tmp_path, monkeypatch, rock):
    monkeypatch.setattr(jobs, "DATA", tmp_path / "jobs")
    with TestClient(api.app) as c:
        jid = up(c, rock).json()["id"]
        jobs.update(jid, state="running")
    with TestClient(api.app) as c:  # a new server process starts
        j = c.get(f"/api/jobs/{jid}").json()
        assert j["state"] == "failed" and "restarted" in j["error"]


def test_truncated_upload_is_rejected_with_a_clear_message(client, tmp_path):
    """The real failure: a 16 KB prefix of an mp3 whose header still describes the whole song."""
    import subprocess
    audio, _ = render(ROCK, 100, 8)
    wav = tmp_path / "f.wav"
    sf.write(wav, audio, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", "128k", tmp_path / "f.mp3"], check=True)
    full = (tmp_path / "f.mp3").read_bytes()
    assert up(client, full, "song.mp3").status_code == 200
    before = len(list(jobs.DATA.iterdir()))
    r = up(client, full[:16384], "song.mp3")
    assert r.status_code == 422 and "incomplete or damaged" in r.json()["detail"] and "only 0:00" in r.json()["detail"]
    assert len(list(jobs.DATA.iterdir())) == before


def test_large_chunked_upload_arrives_complete(client):
    audio, _ = render(ROCK, 100, 40)  # ~100 s, ~8 MB
    data = wav_bytes(audio)
    r = client.post("/api/upload", content=(data[i:i + 8192] for i in range(0, len(data), 8192)),  # chunked, no Content-Length
                    headers={"X-Filename": "big.wav"})
    assert r.status_code == 200 and (jobs.DATA / r.json()["id"] / "audio.wav").stat().st_size == len(data)
    assert abs(r.json()["duration_s"] - len(audio) / SR) < 0.1


def test_technical_detail_is_hidden_unless_debug_is_on(client, monkeypatch, rock):
    jid = up(client, rock).json()["id"]
    jobs.update(jid, state="failed", error="Something went wrong.", detail="ValueError: boom")
    assert "detail" not in client.get(f"/api/jobs/{jid}").json()
    monkeypatch.setenv("DRUMSCRIBE_DEBUG", "1")
    assert client.get(f"/api/jobs/{jid}").json()["detail"] == "ValueError: boom"
