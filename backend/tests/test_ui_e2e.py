"""Boots the real server and drives the real website in headless Chromium (scripts/ui_e2e.mjs)."""
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import soundfile as sf

from backend.validation.synth import ROCK, SR, render

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (shutil.which("node") and shutil.which("chromium")), reason="node/chromium missing")


def test_website_end_to_end_in_a_real_browser(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    audio, _ = render(ROCK, 100, 4)
    wav = tmp_path / "beat.wav"
    sf.write(wav, audio, SR)
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.api.main:app", "--port", str(port), "--log-level", "warning"],
                              cwd=ROOT, env={**__import__("os").environ, "DRUMSCRIBE_DATA": str(tmp_path / "jobs")})
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        r = subprocess.run(["node", "scripts/ui_e2e.mjs", f"http://127.0.0.1:{port}", str(wav), str(tmp_path / "shots")],
                           cwd=ROOT, capture_output=True, text=True, timeout=420, check=False)
        assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-500:]
        assert "ALL UI CHECKS PASSED" in r.stdout
    finally:
        server.terminate()
        server.wait(timeout=20)
