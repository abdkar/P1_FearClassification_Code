# Docker

Standard GPU-enabled container for the project.

## Build

```bash
docker build -t fear-classification:gpu -f docker/Dockerfile .
```

## Run (GPU required, single fold example)

```bash
docker run --gpus all --rm -v $(pwd):/app -w /app \
  -e TARGET_SUBJECT=HF_203 -e VERBOSE=1 \
  fear-classification:gpu
```

## MLflow UI (optional)
Launch tracking server (outside container) or map mlruns:

```bash
docker run --gpus all --rm -p 5000:5000 -v $(pwd):/app -w /app \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  fear-classification:gpu
```

Adjust environment variables as needed.

## CPU-only build

```bash
docker build -t fear-classification:cpu -f docker/Dockerfile.cpu .
```

Run CPU image:
```bash
docker run --rm -v $(pwd):/app -w /app \
  -e TARGET_SUBJECT=HF_203 -e VERBOSE=1 \
  fear-classification:cpu
```
