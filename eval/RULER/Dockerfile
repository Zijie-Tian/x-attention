# RULER Benchmark Execution Environment
# Base: NVIDIA PyTorch 24.08 (PyTorch 2.4.0, CUDA 12.x)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set working directory
WORKDIR /workspace/RULER

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements first for better caching
COPY requirements.txt .

# Install build dependencies
RUN pip install --no-cache-dir -q Cython

# Install youtokentome with no-build-isolation (needs Cython)
RUN pip install --no-cache-dir youtokentome --no-build-isolation

# Install all requirements including nemo-toolkit
# Note: torch/torchaudio/torchvision versions in requirements.txt should match base image
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy RULER scripts
COPY . .

# Set default command
CMD ["/bin/bash"]
