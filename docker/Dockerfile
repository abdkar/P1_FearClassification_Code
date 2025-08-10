# Base NVIDIA CUDA image with cudnn runtime suitable for TensorFlow GPU
FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set workdir
WORKDIR /app

# Copy requirement file first for layer caching
COPY requirements.txt ./

# Upgrade pip and install deps (use --no-cache-dir to reduce layer size)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter pynvml

# Copy project
COPY . .

# Environment defaults (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true \
    MPLBACKEND=Agg

# Expose MLflow and Jupyter optional ports
EXPOSE 5000 8888

# Healthcheck: simple python import test
HEALTHCHECK --interval=1m --timeout=10s --retries=3 CMD python -c "import tensorflow as tf, numpy as np; print(tf.__version__)" || exit 1

# Default command (override with docker run ...)
CMD ["python", "experiments/lopocv_real.py"]
