RED='\033[1;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NOCOLOR='\033[0m'

# Use Bash if it's available and fallback to regular Shell (needed to use Docker in GitLab CI Pipeline)
ifeq (, $(shell which bash))
SHELL = /bin/sh
else
SHELL = /bin/bash
endif

SYSTEM_NAME := $(shell uname)
SYSTEM_ARCHITECTURE := $(shell uname -m)
MAMBA_VERSION := 24.11.2-1
MAMBA_INSTALL_SCRIPT := Miniforge3-$(MAMBA_VERSION)-$(SYSTEM_NAME)-$(SYSTEM_ARCHITECTURE).sh

MAMBA_ENV_NAME := morpheus
PACKAGE_FOLDER := src
MORPHEUS_FOLDER := src/morpheus

#
# Conda/Mamba environment
#

# HELP: install-mamba: [Mamba] Install Mamba on a fresh environment
.PHONY: install-mamba
install-mamba:
	@echo "🏗  Installing Mamba..."
	@curl -L -O "https://github.com/conda-forge/miniforge/releases/download/$(MAMBA_VERSION)/$(MAMBA_INSTALL_SCRIPT)"
	@chmod +x "$(MAMBA_INSTALL_SCRIPT)"
	@./$(MAMBA_INSTALL_SCRIPT)
	@rm "$(MAMBA_INSTALL_SCRIPT)"

# HELP: create-mamba-env: [Mamba] Create new Mamba environment
.PHONY: create-mamba-env
create-mamba-env:
	@mamba env create -f environment.yml -n "$(MAMBA_ENV_NAME)"
	@echo -e $(GREEN)"✅ Mamba env created! ✅"$(NOCOLOR)
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install --upgrade pip "setuptools==69.5.1" wheel
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install xformers==0.0.23.post1 fvcore iopath --extra-index-url https://download.pytorch.org/whl/cu118
	@echo "🏗 Installing PIP dependencies..."
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install --timeout=60 --retries 10 -r requirements_nerfstudio.txt
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install --timeout=60 --retries 10 -r requirements.txt
	# Build pytorch3d from source at the end to avoid cache invalidation during experiments
	@mamba run -n "$(MAMBA_ENV_NAME)" env FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9" pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
	@mamba run -n "$(MAMBA_ENV_NAME)" pip install --timeout=60 --retries 10 -e .
	# Fix a silly issue regarding the openh264 lib:
	@mamba update -y -n "$(MAMBA_ENV_NAME)" -c conda-forge ffmpeg
	@echo -e $(GREEN)"✅ Done! All dependencies installed in the $(MAMBA_ENV_NAME) Mamba environment. ✅"$(NOCOLOR)


# HELP: download-input-data: [Data] Download data from Google Cloud Storage
.PHONY: download-input-data
download-input-data:
	@echo "🏗 Downloading scenes and configs from Google Cloud Storage..."
	@mkdir -p data
	@echo "🏗 We're downloading configs..."
	@wget -O data/configs.tar https://storage.googleapis.com/niantic-lon-static/research/morpheus/data/configs.tar
	@tar -xf data/configs.tar -C data/

	@echo "🏗 We're downloading scenes..."
	@wget -O data/scenes.tar https://storage.googleapis.com/niantic-lon-static/research/morpheus/data/scenes.tar
	@tar -xf data/scenes.tar -C data/


# HELP: download-models: [Models] Download pretrained models from Google Cloud Storage
.PHONY: download-models
download-models:
	@echo "🏗 Downloading models from Google Cloud Storage..."
	@mkdir -p data/models
	@echo "🏗 We're downloading the RGBD model..."
	@wget -O data/models/rgbd_diffusion_model.tar https://storage.googleapis.com/niantic-lon-static/research/morpheus/models/rgbd_diffusion_model.tar
	@tar -xf data/models/rgbd_diffusion_model.tar -C data/models

	@echo "🏗 We're downloading the warp_controlnet..."
	@wget -O data/models/warp_controlnet.tar https://storage.googleapis.com/niantic-lon-static/research/morpheus/models/warp_controlnet.tar
	@tar -xf data/models/warp_controlnet.tar -C data/models

# HELP: help: [Other] Help
.PHONY: help
help:
	@echo "🧑‍💻 Available commands:"
	@sed -n 's/^# HELP://p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/ - make/'

.PHONY: newline%
newline%:
	@echo ""
