#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-mujoco-image}
YAML_PATH="environment/mujoco_image_conda.yaml"
REQ_PATH="environment/mujoco_image_requirements.txt"

if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
  echo "[INFO] Conda/Mamba detected, using conda environment flow"
  if command -v mamba >/dev/null 2>&1; then
    mamba env create -f "${YAML_PATH}" -n "${ENV_NAME}" || mamba env update -f "${YAML_PATH}" -n "${ENV_NAME}"
  else
    conda env create -f "${YAML_PATH}" -n "${ENV_NAME}" || conda env update -f "${YAML_PATH}" -n "${ENV_NAME}"
  fi

  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"

  python -m pip install -U pip wheel setuptools
  python -m pip install -e . || true
  python scripts/verify_env.py

  echo "[DONE] Conda environment '${ENV_NAME}' is ready."
  echo "Use: conda activate ${ENV_NAME}"
  exit 0
fi

echo "[WARN] conda/mamba not found, fallback to python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r "${REQ_PATH}"
echo "[INFO] Please install torch/torchvision wheel matching your CUDA manually."
python -m pip install -e . || true
python scripts/verify_env.py

echo "[DONE] venv ready at .venv"
echo "Use: source .venv/bin/activate"
