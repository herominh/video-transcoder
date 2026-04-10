# ----------------------------------------------------------
# GPU variant:  docker build --build-arg BASE=gpu -t video-transcoder:gpu .
# CPU variant:  docker build -t video-transcoder:cpu .   (default)
# ----------------------------------------------------------
ARG BASE=cpu

# ---- GPU base ----
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    && curl -fsSL https://repo.jellyfin.org/jellyfin_team.gpg.key \
       | gpg --dearmor -o /usr/share/keyrings/jellyfin.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/jellyfin.gpg] https://repo.jellyfin.org/ubuntu jammy main" \
       > /etc/apt/sources.list.d/jellyfin.list \
    && apt-get update && apt-get install -y jellyfin-ffmpeg7 \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe \
    && rm -rf /var/lib/apt/lists/*

# ---- CPU base ----
FROM python:3.12-slim AS base-cpu

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Final stage ----
FROM base-${BASE} AS final

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ core/
COPY wrappers/ wrappers/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "wrappers/docker_server.py"]
