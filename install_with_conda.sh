#!/usr/bin/env bash
set -e

# ============ Config ============
ENV_NAME=${1:-chatterbox-env}
CHATTERBOX_DIR=${2:-chatterbox}
# ================================

echo "[*] Creating Conda env: ${ENV_NAME}"

# Create env with Python 3.10 (Chatterbox is 3.10-compatible)
conda create -y -n "$ENV_NAME" python=3.10

echo "[*] Activating environment: ${ENV_NAME}"
# IMPORTANT: Conda must be initialized in non-interactive scripts
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[*] Installing PyTorch + torchaudio (CUDA 12.1 wheels)"
# Conda doesn't support custom CUDA wheel URLs,
# so we install PyTorch via pip while inside the conda env
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0+cu121" \
  "torchaudio==2.4.0+cu121"

echo "[*] Installing Chatterbox dependencies"
pip install \
  accelerate \
  aiohttp \
  einops \
  sentencepiece \
  safetensors \
  soundfile \
  huggingface_hub \
  "transformers==4.44.2" \
  "diffusers==0.29.0" \
  resemble-perth \
  loguru \
  scipy \
  s3tokenizer \
  conformer \
  fastapi \
  uvicorn[standard]

if [ ! -d "$CHATTERBOX_DIR" ]; then
  echo "[*] Cloning chatterbox repo"
  git clone https://github.com/resemble-ai/chatterbox.git "$CHATTERBOX_DIR"
else
  echo "[*] chatterbox directory already exists, skipping clone"
fi

cd "$CHATTERBOX_DIR"

echo "[*] Patching src/chatterbox/__init__.py"
python - << 'PY'
from pathlib import Path

src_dir = Path("src")
init_path = src_dir / "chatterbox" / "__init__.py"
if not init_path.exists():
    raise SystemExit(f"Could not find {init_path}")

patched_code = """__version__ = "0.0.0"

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
"""

init_path.write_text(patched_code)
print("Patched", init_path)
PY

echo "[*] Done. To use the API:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${CHATTERBOX_DIR}"
echo "    python api_server.py"
