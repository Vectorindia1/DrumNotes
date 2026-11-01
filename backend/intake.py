"""Turn the user's input (a local file or a link) into a local audio file.

Supported: local paths; direct links to an audio FILE (http/https). NOT supported, on purpose:
YouTube and Spotify links. Spotify audio is DRM-protected and its API gives metadata only; YouTube audio
can only be obtained by downloading it, which its terms forbid for other people's music, and the PRD
lists "bypassing Spotify, YouTube, or other platform restrictions" as out of scope (only authorized
integrations are allowed). For those links the track title is looked up through the public oEmbed
endpoints (metadata only) so the message can say what the link is and what to do instead.
"""
import json
import pathlib
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

import soundfile as sf

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".mpga", ".weba"}
MAX_BYTES = 300 * 1024 * 1024
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
SPOTIFY_HOSTS = {"open.spotify.com", "spotify.link", "play.spotify.com"}
OEMBED = {"youtube": "https://www.youtube.com/oembed?format=json&url=", "spotify": "https://open.spotify.com/oembed?url="}


class UnsupportedSource(RuntimeError):
    """The input cannot be turned into audio (message tells the user what to do instead)."""


def _title(kind, url):
    """Best-effort public metadata lookup; never raises."""
    try:
        with urllib.request.urlopen(OEMBED[kind] + urllib.parse.quote(url, safe=""), timeout=6) as r:
            d = json.load(r)
        return " - ".join(x for x in (d.get("title"), d.get("author_name")) if x)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _refuse(kind, url):
    name = {"youtube": "YouTube", "spotify": "Spotify"}[kind]
    title = _title(kind, url)
    what = f"That {name} link is \"{title}\"." if title else f"That is a {name} link."
    why = ("Spotify audio is DRM-protected and its API provides no audio." if kind == "spotify"
           else "YouTube audio can only be obtained by downloading it, which YouTube's terms do not allow for other people's music.")
    raise UnsupportedSource(
        f"{what} DrumScribe cannot fetch audio from {name} links. {why}\n"
        "Instead: use a copy you have the right to use (your own file, a purchase, or your own upload "
        "downloaded from YouTube Studio) and run\n"
        "  python -m backend.pipeline <that file> --tempo <bpm> --separate --player\n"
        "Direct links to an audio file (https://.../song.mp3) do work.")


def readable(path, dest_dir="data/incoming"):
    """Return `path` if libsndfile can decode it (wav, flac, ogg, opus, mp3); otherwise convert it to a
    wav with ffmpeg (m4a/aac, wma, webm, ...) and return the converted file."""
    path = pathlib.Path(path)
    try:
        sf.info(str(path))
        return path
    except (sf.LibsndfileError, RuntimeError):
        pass
    if not shutil.which("ffmpeg"):
        raise UnsupportedSource(f"cannot decode {path.name} and ffmpeg is not installed to convert it")
    out = pathlib.Path(dest_dir) / (path.stem + ".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-c:a", "pcm_s16le", str(out)],
                       check=False, capture_output=True, text=True)
    if r.returncode != 0:
        raise UnsupportedSource(f"cannot decode {path.name}: {r.stderr.strip()[-200:]}")
    return out


def resolve_input(arg, dest_dir="data/incoming", max_bytes=MAX_BYTES):
    """Return a local, decodable Path for a file path or a direct audio-file link."""
    return readable(_fetch(arg, dest_dir, max_bytes), dest_dir)


def _fetch(arg, dest_dir, max_bytes):
    parsed = urllib.parse.urlparse(str(arg))
    if parsed.scheme not in ("http", "https"):
        path = pathlib.Path(arg)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {arg}")
        return path
    host = (parsed.hostname or "").lower()
    if host in YOUTUBE_HOSTS:
        _refuse("youtube", str(arg))
    if host in SPOTIFY_HOSTS:
        _refuse("spotify", str(arg))
    ext = pathlib.PurePosixPath(parsed.path).suffix.lower()
    req = urllib.request.Request(str(arg), headers={"User-Agent": "DrumScribe/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not (ctype.startswith("audio/") or ctype == "application/ogg" or ext in AUDIO_EXT):
            raise UnsupportedSource(f"that link is not an audio file (Content-Type {ctype or 'unknown'}). "
                                    "Only direct links to audio files are supported.")
        length = int(r.headers.get("Content-Length") or 0)
        if length > max_bytes:
            raise UnsupportedSource(f"audio file too large ({length / 1e6:.0f} MB; limit {max_bytes / 1e6:.0f} MB)")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", pathlib.PurePosixPath(parsed.path).name or "download") or "download"
        if not pathlib.PurePosixPath(name).suffix:
            name += ext or ".audio"
        dest = pathlib.Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        out, got = dest / name, 0
        with open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > max_bytes:
                    f.close()
                    out.unlink(missing_ok=True)
                    raise UnsupportedSource(f"audio file larger than the {max_bytes / 1e6:.0f} MB limit")
                f.write(chunk)
    return out
