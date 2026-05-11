#!/usr/bin/env bash
# install.sh — bootstrap a complete TAVIS environment.
#
# Creates a fresh `tavis` conda env (Python 3.11) and installs the full
# pinned dependency stack: PyTorch + Isaac Sim 5.1 + IsaacLab v2.3.2 +
# IsaacLab-Arena @ 755e8cf3 + lerobot 0.4.3 + TAVIS itself.
#
# Override defaults via env vars:
#   TAVIS_ENV_NAME     (default: tavis)
#   TAVIS_INSTALL_DIR  (default: $HOME)   — where IsaacLab / IsaacLab-Arena clone to
#   TAVIS_TORCH_CUDA   (default: cu128)   — also valid: cu130
#
# Usage:
#   bash install.sh
#

set -eo pipefail

# ── Pins ──────────────────────────────────────────────────────────────────────
ENV_NAME="${TAVIS_ENV_NAME:-tavis}"
INSTALL_DIR="${TAVIS_INSTALL_DIR:-${HOME}}"
ISAACLAB_DIR="${INSTALL_DIR}/IsaacLab"
ARENA_DIR="${INSTALL_DIR}/IsaacLab-Arena"

PYTHON_VERSION="3.11"
TORCH_VERSION="2.10.0"
TORCH_CUDA="${TAVIS_TORCH_CUDA:-cu128}"
ISAACSIM_VERSION="5.1.0.0"
ISAACLAB_TAG="v2.3.2"
ARENA_SHA="755e8cf3"
LIGHTWHEEL_VERSION="1.0.1"
ONNXRUNTIME_VERSION="1.24.1"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found in PATH." >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "ERROR: git not found in PATH." >&2; exit 1; }

# Refuse to clobber an existing tavis conda env
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx -- "${ENV_NAME}"; then
    echo "ERROR: conda env '${ENV_NAME}' already exists." >&2
    echo "       Remove it (conda env remove -n ${ENV_NAME} -y) or set TAVIS_ENV_NAME=<other>." >&2
    exit 1
fi

# Refuse to overwrite existing IsaacLab / Arena directories
for d in "${ISAACLAB_DIR}" "${ARENA_DIR}"; do
    if [ -e "${d}" ]; then
        echo "ERROR: ${d} already exists." >&2
        echo "       Remove it, or set TAVIS_INSTALL_DIR=<dir> to clone elsewhere." >&2
        exit 1
    fi
done

# ── Create + activate env ─────────────────────────────────────────────────────
echo "==> Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Bootstrap uv (used for everything below) ──────────────────────────────────
echo "==> Installing uv..."
pip install --quiet uv

# ── 1. Torch (CUDA-pinned; must precede isaacsim) ─────────────────────────────
echo "==> Installing torch ${TORCH_VERSION} (${TORCH_CUDA})..."
uv pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# ── 2. Isaac Sim ──────────────────────────────────────────────────────────────
echo "==> Installing isaacsim ${ISAACSIM_VERSION}..."
uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

# ── 3. IsaacLab @ pinned tag ──────────────────────────────────────────────────
echo "==> Cloning IsaacLab @ ${ISAACLAB_TAG} into ${ISAACLAB_DIR}..."
git clone --depth 1 --branch "${ISAACLAB_TAG}" \
    https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
(cd "${ISAACLAB_DIR}" && ./isaaclab.sh --install)

# ── 4. IsaacLab-Arena @ pinned SHA ────────────────────────────────────────────
# Submodules (Arena's vendored IsaacLab + Isaac-GR00T) are intentionally NOT
# initialized: tavis uses the IsaacLab installed in step 3, and Isaac-GR00T
# is unused. Skipping them saves the GR00T clone (large) and avoids a second
# IsaacLab copy that would shadow nothing but waste disk.
echo "==> Cloning IsaacLab-Arena @ ${ARENA_SHA} into ${ARENA_DIR}..."
git clone https://github.com/isaac-sim/IsaacLab-Arena.git "${ARENA_DIR}"
git -C "${ARENA_DIR}" checkout "${ARENA_SHA}"
(cd "${ARENA_DIR}" && uv pip install -e .)
# Arena has undeclared import-time deps not listed in its pyproject:
#   - lightwheel_sdk: object_library.py (Microwave class)
#   - onnxruntime:    g1_homie_policy.py (whole-body controller)
echo "==> Installing Arena's undeclared deps (lightwheel-sdk, onnxruntime)..."
uv pip install "lightwheel-sdk==${LIGHTWHEEL_VERSION}" "onnxruntime==${ONNXRUNTIME_VERSION}"

# ── 5. TAVIS itself ───────────────────────────────────────────────────────────
echo "==> Installing tavis (editable, with [train] extra)..."
(cd "${SCRIPT_DIR}" && uv pip install -e ".[train]")

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "TAVIS install complete."
echo "  Activate the env:  conda activate ${ENV_NAME}"
echo "  Quick test:        python -c 'import tavis; print(tavis.__version__)'"
