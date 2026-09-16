FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENJEV_BACKEND=cpu \
    OPENJEV_PUBLIC_DEMO=1 \
    OPENJEV_CPU_THREADS=2 \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsdl2-2.0-0 libgomp1 libopenal1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
WORKDIR /home/user/app
# CPU wheels avoid downloading CUDA libraries on a CPU-only host.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user:user pyproject.toml README.md LICENSE ./
COPY --chown=user:user src ./src
RUN pip install '.[hosted]' && chown user:user /home/user/app
USER user
# Public, pinned weights are downloaded at build time. Inference is offline.
RUN openjev setup
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')"
CMD ["openjev", "serve", "--host", "0.0.0.0", "--port", "7860"]
