#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'Requires Ubuntu/Debian Linux x86_64 and an NVIDIA GPU.' >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo 'Create .venv and install requirements.txt first; see README.md.' >&2
  exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
  apt-get update
  apt-get install -y curl unzip libzip4 libgomp1 ca-certificates
else
  sudo apt-get update
  sudo apt-get install -y curl unzip libzip4 libgomp1 ca-certificates
fi

# Keep KataGo's cuDNN 8 libraries separate from PyTorch's runtime.
.venv/bin/python -m pip install --upgrade --target local/katago/runtime \
  -r config/katago_requirements.txt

mkdir -p local/katago/bin local/katago/models
if [[ ! -x local/katago/bin/katago ]]; then
  curl -fL --retry 3 \
    https://github.com/lightvector/KataGo/releases/download/v1.18.2/katago-v1.18.2-cuda12.1-cudnn8.9.7-linux-x64.zip \
    -o local/katago/katago.zip.part
  mv local/katago/katago.zip.part local/katago/katago.zip
  unzip -jo local/katago/katago.zip '*katago' -d local/katago/bin
  chmod +x local/katago/bin/katago
fi

model_name=kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
model_file="local/katago/models/$model_name"
if [[ ! -s "$model_file" ]]; then
  curl -fL --retry 3 \
    "https://media.katagotraining.org/uploaded/networks/models/kata1/$model_name" \
    -o "$model_file.part"
  mv "$model_file.part" "$model_file"
fi

echo 'KataGo setup complete.'
