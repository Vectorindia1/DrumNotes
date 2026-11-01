import functools
import http.server
import shutil
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf

from backend import intake
from backend.validation.synth import ROCK, SR, render


def _serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_local_path_passes_through_and_missing_file_is_an_error(tmp_path):
    f = tmp_path / "a.wav"
    sf.write(f, np.zeros(100, np.float32), SR)
    assert intake.resolve_input(str(f), dest_dir=tmp_path / "r") == f
    with pytest.raises(FileNotFoundError):
        intake.resolve_input(str(tmp_path / "nope.mp3"), dest_dir=tmp_path / "r")


@pytest.mark.parametrize("url,kind", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "YouTube"),
    ("https://youtu.be/dQw4w9WgXcQ", "YouTube"),
    ("https://music.youtube.com/watch?v=abc", "YouTube"),
    ("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC", "Spotify"),
])
def test_youtube_and_spotify_links_are_refused_with_instructions(monkeypatch, tmp_path, url, kind):
    monkeypatch.setattr(intake, "_title", lambda k, u: "Some Song - Some Artist")  # no network in tests
    with pytest.raises(intake.UnsupportedSource) as e:
        intake.resolve_input(url, dest_dir=tmp_path)
    msg = str(e.value)
    assert kind in msg and "Some Song - Some Artist" in msg and "cannot fetch audio" in msg
    assert "python -m backend.pipeline" in msg and not list(tmp_path.iterdir())  # nothing downloaded


def test_direct_audio_link_downloads_and_bad_links_are_rejected(tmp_path):
    served = tmp_path / "www"
    served.mkdir()
    sf.write(served / "clip.wav", np.zeros(SR, np.float32), SR)
    (served / "page.html").write_text("<html></html>")
    srv = _serve(served)
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        out = intake.resolve_input(f"{base}/clip.wav", dest_dir=tmp_path / "dl")
        assert out.read_bytes() == (served / "clip.wav").read_bytes()
        with pytest.raises(intake.UnsupportedSource, match="not an audio file"):
            intake.resolve_input(f"{base}/page.html", dest_dir=tmp_path / "dl")
        with pytest.raises(intake.UnsupportedSource, match="too large"):
            intake.resolve_input(f"{base}/clip.wav", dest_dir=tmp_path / "dl2", max_bytes=1000)
    finally:
        srv.shutdown()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
@pytest.mark.parametrize("ext,codec", [("m4a", "aac"), ("opus", "libopus"), ("ogg", "libvorbis"), ("flac", "flac"), ("mp3", "libmp3lame")])
def test_pipeline_reads_every_common_download_format(tmp_path, ext, codec):
    """Whatever the user downloads must decode: duration and the detector's frame count must agree."""
    import librosa

    from backend.detection import adtof
    audio, _ = render(ROCK, 100, 2)
    wav, f = tmp_path / "a.wav", tmp_path / f"a.{ext}"
    sf.write(wav, audio, SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", codec, f], check=True)
    f = intake.resolve_input(str(f), dest_dir=tmp_path / "ready")  # m4a etc. get converted to wav
    dur = librosa.get_duration(path=str(f))
    assert abs(dur - len(audio) / SR) < 0.15
    pred = adtof.activations(f)
    assert abs(len(pred) / 100 - dur) < 0.15
    assert pred[:, 0].max() > 0.3  # the kick at t = 0 is still seen after lossy coding


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_m4a_is_converted_and_garbage_is_reported(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(SR, np.float32), SR)
    m4a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-c:a", "aac", m4a], check=True)
    out = intake.resolve_input(str(m4a), dest_dir=tmp_path / "ready")
    assert out.suffix == ".wav" and sf.info(str(out)).samplerate == SR
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all" * 10)
    with pytest.raises(intake.UnsupportedSource, match="cannot decode"):
        intake.resolve_input(str(junk), dest_dir=tmp_path / "ready")
