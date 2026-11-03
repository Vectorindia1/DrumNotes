# DrumScribe website in one CPU-only container (Hugging Face Spaces "Docker" SDK, or any Docker host).
#   docker build -t drumscribe . && docker run -p 7860:7860 drumscribe   -> http://localhost:7860
# NON-COMMERCIAL ONLY: ADTOF weights (CC BY-NC-SA), ADTOF-pytorch (no licence file) and the MDB-trained
# label model. See LICENSE_AUDIT.md before making an image or Space public.
FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg musescore3 musescore-general-soundfont-small libfluidsynth3 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

# torch must come from the CPU index (PyPI resolves to a CUDA build); dev tools and the local ADTOF line are skipped
COPY requirements-lock.txt .
RUN grep -v -E "^(adtof-pytorch|pytest|ruff)" requirements-lock.txt > /tmp/req.txt \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt
RUN git clone -q https://github.com/xavriley/ADTOF-pytorch third_party/ADTOF-pytorch \
    && git -C third_party/ADTOF-pytorch checkout -q 85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9 \
    && pip install --no-cache-dir --no-deps -e third_party/ADTOF-pytorch

COPY --chown=user . .
RUN mkdir -p data/jobs data/stems && chown -R user /app
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1
RUN python scripts/download_models.py  # Demucs htdemucs weights into the image (no download per job)

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
