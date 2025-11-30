# RunPod Gaussian Splatting Worker
# Docker image for 3DGS training on RunPod Serverless

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    colmap \
    wget \
    unzip \
    curl \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone Gaussian Splatting repository
RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git .

# Install Python dependencies for Gaussian Splatting
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    plyfile \
    tqdm \
    scipy \
    pillow \
    opencv-python \
    lpips

# Install submodules
RUN pip3 install submodules/diff-gaussian-rasterization
RUN pip3 install submodules/simple-knn

# Install RunPod SDK
RUN pip3 install --no-cache-dir runpod requests boto3

# Copy handler and helper scripts
COPY rp_handler.py ./rp_handler.py
COPY prepare_from_video.py ./prepare_from_video.py

# Make scripts executable
RUN chmod +x rp_handler.py prepare_from_video.py

# Set entrypoint
CMD ["python3", "-u", "rp_handler.py"]

