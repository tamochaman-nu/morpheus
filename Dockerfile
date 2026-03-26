# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    column \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Mamba)
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3-$(uname)-$(uname -m).sh -b -p $CONDA_DIR \
    && rm Miniforge3-$(uname)-$(uname -m).sh \
    && mamba clean -afy

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY environment.yml requirements.txt requirements_nerfstudio.txt pyproject.toml Makefile ./

# Create the mamba environment
# Note: Makefile's create-mamba-env command is used
RUN make create-mamba-env

# Copy the rest of the source code
COPY . .

# Set the default shell to use the mamba environment
# mamba init and setting the shell
RUN mamba init bash
RUN echo "conda activate morpheus" >> ~/.bashrc

# Default command
CMD ["bash"]
