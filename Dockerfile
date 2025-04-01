FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    git \
    libsndfile1 \
    && ln -s /usr/bin/python3 /usr/bin/python 

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TRANSFORMERS_OFFLINE=0

# Set the working directory
WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first
COPY requirements.txt .

RUN pip install --upgrade transformers

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 support
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Copy the application code
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "-u", "main.py"]
