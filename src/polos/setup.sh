#!/bin/bash
set -e

echo "=== Installing Python 3.10 ==="
sudo apt-get update -qq
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

echo "=== Creating venv ==="
rm -rf /content/.venv-polos
python3.10 -m venv /content/.venv-polos

echo "=== Updating pip/setuptools ==="
/content/.venv-polos/bin/pip install "pip<24.1" setuptools==69.5.1 wheel

echo "=== Installing PyTorch stack ==="
/content/.venv-polos/bin/pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2

echo "=== Installing fairseq dependencies ==="
/content/.venv-polos/bin/pip install \
    PyYAML==5.3.1 \
    omegaconf==2.0.6 \
    hydra-core==1.0.7 \
    cython==0.29.36

echo "=== Installing fairseq ==="
/content/.venv-polos/bin/pip install fairseq==0.12.2 --no-build-isolation

echo "=== Installing remaining requirements ==="
/content/.venv-polos/bin/pip install -r /content/drive/MyDrive/polos-project/requirements-polos.txt

echo "=== Verifying POLoS ==="
/content/.venv-polos/bin/python -c "import polos; print('POLoS imported successfully')"

echo "=== Verifying GPU ==="
/content/.venv-polos/bin/python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "=== Setup complete ==="