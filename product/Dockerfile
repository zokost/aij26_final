FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    DETECTOR_DEVICE=cuda \
    DETECTOR_FAST=0 \
    OPENROUTER_EXPLANATIONS_ENABLED=0

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check

COPY src/detector2.py src/data.py src/model.py src/nc17_calibration.py src/nc17_features.py ./src/
COPY app/__init__.py app/core.py app/service.py app/explanations.py app/calibration.json ./app/
COPY web/index.html web/legacy.html web/*.css web/*.mjs ./web/
COPY web/assets/*.svg ./web/assets/
# Submission121: NC17 epoch32 and frozen lower-only calibration, square640 FP32.
COPY weights/121_NC17_e32/ ./weights/121_NC17_e32/
RUN printf '%s\n' \
    '40841b9c867d5c6f1ed8fcfacefb14787a377d58f770da86e67067f1d57a4cf1  weights/121_NC17_e32/model.pt' \
    '0bf2ba074b3609aefea2eba383ab512f5d635311315c76a79e3b08472023de25  weights/121_NC17_e32/policy.pkl' \
    'fc865212594a3e98eb3e9b1465d75225782a422bebc636f4a7f99ebc949b9758  weights/121_NC17_e32/encoder_transform.pkl' | sha256sum --check \
    && python -c "import app.service" \
    && useradd --create-home --uid 10001 detective

USER detective
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=6 \
    CMD python -c "import json, urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=4)); assert h['status']=='ok' and h['mode']=='inference'"
CMD ["python", "-m", "uvicorn", "app.service:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
