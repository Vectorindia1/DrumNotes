import os
import subprocess

TIMEOUT_S = 300


def render_pdf(musicxml_path, pdf_path):
    """Headless MuseScore 3 render. Raises on non-zero exit or missing output."""
    r = subprocess.run(
        ["mscore3", "-o", str(pdf_path), str(musicxml_path)],
        check=False, capture_output=True, text=True, timeout=TIMEOUT_S,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    if r.returncode != 0 or not os.path.exists(pdf_path):
        raise RuntimeError(f"mscore3 failed (exit {r.returncode}): {r.stderr[-500:]}")
