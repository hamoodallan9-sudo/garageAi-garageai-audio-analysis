FROM python:3.13-slim

WORKDIR /app

# soundfile (librosa's audio backend) needs libsndfile at runtime on Linux.
# ffmpeg itself is not needed here: audio_converter.py uses the static binary
# bundled by the imageio-ffmpeg wheel, not a system install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch==...+cpu is only published on PyTorch's own index, not PyPI.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app ./app

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

# MODELS_DIR (car_test's processed_data output), GEMINI_API_KEY, and
# INTERNAL_API_KEY are environment/deployment specific -- pass them with
# `docker run -e` / `--env-file .env`, and mount MODELS_DIR as a volume, e.g.:
#   docker run -p 8001:8001 --env-file .env -v /path/to/processed_data:/models \
#     -e MODELS_DIR=/models garageai-audio-analysis

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
