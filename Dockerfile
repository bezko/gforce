# G-Force Worker Image
# Pre-baked with PyTorch, Diffusers, and training dependencies
# Stored in Google Artifact Registry for fast VM boot times

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install ML dependencies (this layer is cached in the image)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    diffusers==0.25.0 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    xformers==0.0.23.post1 \
    bitsandbytes==0.41.3.post2 \
    safetensors>=0.4.0 \
    peft==0.7.1 \
    huggingface-hub>=0.20.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    google-cloud-storage>=2.10.0

# Create working directory
WORKDIR /app

# Copy worker script
COPY gforce/scripts/worker_init.py /app/worker_init.py

# Make it executable
RUN chmod +x /app/worker_init.py

# Set entrypoint
ENTRYPOINT ["python", "/app/worker_init.py"]
