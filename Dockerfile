# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    sed \
    bsdextrautils \
    colmap \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Mamba)
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p $CONDA_DIR \
    && rm Miniforge3-$(uname)-$(uname -m).sh \
    && mamba clean -afy


# Set working directory
WORKDIR /app

# Copy dependency files and local source for caching
COPY environment.yml requirements.txt requirements_nerfstudio.txt pyproject.toml Makefile ./
COPY src ./src

# Create the mamba environment
# Note: Makefile's create-mamba-env command is used
RUN make create-mamba-env

# Copy the rest of the source code
COPY . .

# mamba/conda init and setting the shell
RUN conda init bash
RUN echo "conda activate morpheus" >> ~/.bashrc

# Default command
CMD ["bash"]
