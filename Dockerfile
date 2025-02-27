FROM python:3.10-slim

# Install system dependencies
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
    python3-dev \
    gcc \
    git \
    libsndfile1

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
# Add memory management settings
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TRANSFORMERS_OFFLINE=0

# Set the working directory
WORKDIR /app

# Upgrade pip and install basic dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the required port
EXPOSE 7860

# Run the application with increased timeout
CMD ["python", "-u", "main.py"]