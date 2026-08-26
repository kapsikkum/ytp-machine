# YTP Machine.
#
# The awkward dependency here is torch. The phoneme splicer uses torchaudio's
# CTC forced aligner (Wav2Vec2 BASE 960h) to find where inside a clip a sound
# actually starts and stops, so torch is needed to *serve*, not just to ingest.
# Installed from the CPU wheel index: the default index pulls the CUDA build,
# which is several gigabytes of driver payload that would never be used on a
# headless server with no GPU.
FROM docker.io/library/python:3.12-slim

# ffmpeg does all the cutting and concatenating; it is not optional.
# git is wanted by openai-whisper's install, curl by the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU torch first and on its own layer. Pinning it before the rest means the
# transitive torch requirement from openai-whisper is already satisfied, so pip
# will not go and fetch the CUDA build over the top of it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Bake the model weights and corpora in rather than fetching them on first use.
# Downloading ~400 MB the first time somebody presses GENERATE turns one
# request into a two-minute stall, and it fails outright on a host with no
# outbound access.
ENV TORCH_HOME=/opt/torch \
    NLTK_DATA=/opt/nltk
RUN mkdir -p "$TORCH_HOME" "$NLTK_DATA" \
 && python -c "import nltk; nltk.download('cmudict', download_dir='$NLTK_DATA')" \
 && python -c "import torchaudio; torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()" \
 && chmod -R a+rX "$TORCH_HOME" "$NLTK_DATA"

COPY main.py ./
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The corpus lives in a volume, not the image: it is ~110 MB of mp4 that
# changes on its own schedule, and baking it in would mean rebuilding the whole
# image to add one video. /app/downloads and /app/output are symlinks into that
# volume because the database stores clip paths relative to the project root
# and main.py mounts ./output as static files -- both expect those names here.
ENV MRS_DATA_DIR=/app/data \
    MRS_DB_PATH=/app/data/michael_rosen.db \
    PORT=8765
RUN mkdir -p /app/data \
 && ln -s /app/data/downloads /app/downloads \
 && ln -s /app/data/output    /app/output \
 && ln -s /app/data/app.log   /app/app.log

# Runs as a normal user; the volume is chowned by the entrypoint's caller.
RUN useradd --system --uid 1000 --create-home ytp \
 && chown -R ytp:ytp /app
USER ytp

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/stats" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
