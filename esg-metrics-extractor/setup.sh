#!/bin/bash
set -e

echo "Setting up ESG metrics extraction environment..."

# Create and activate virtual environment
python -m venv esg_env
source esg_env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.49.0 pillow numpy scikit-learn psutil pymupdf huggingface_hub

# Pre-download the model for offline use
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ibm-granite/granite-vision-3.3-2b')"

echo "Environment setup complete!"
