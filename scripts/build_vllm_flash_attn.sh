#!/usr/bin/env bash
#
# Builds vLLM's flash-attention fork, which provides the paged attention kernel
# imported by distr_inference/attention.py:load_flash_attn_varlen_func.
#
# Why this is a source build and not a requirements.txt line:
#
#   * Upstream flash-attn requires the paged block size to be divisible by 256
#     (a kernel tiling assumption, not a tunable). We use 16, so we need vLLM's
#     fork, which implements the gather path for small pages.
#   * `vllm-flash-attn` on PyPI is abandoned: 2.6.2 (Sep 2024), pinned to
#     torch==2.4.0. It cannot resolve against any current torch.
#   * `vllm` vendors the same kernel but requires transformers>=5.5.3, which
#     conflicts with the <5 pin in requirements.txt.
#
# Building here takes just the kernel, compiled against whatever torch and CUDA
# toolkit this machine actually has, and leaves the dependency graph alone.
#
# Usage:  ./scripts/build_vllm_flash_attn.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pin the fork so builds are reproducible. `main` tracks current vLLM and
# expects a matching torch; bump this together with the torch pin.
FA_REPO="${FA_REPO:-https://github.com/vllm-project/flash-attention.git}"
FA_REF="${FA_REF:-main}"
BUILD_DIR="${BUILD_DIR:-$(mktemp -d)/flash-attention}"

# --- preflight -------------------------------------------------------------
# A CPU-only or driver-mismatched torch yields a wheel that imports fine and
# then fails at kernel launch, so fail loudly here rather than at test time.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        f"torch {torch.__version__} (cuda {torch.version.cuda}) reports no CUDA "
        "devices -- building against it would produce an unusable wheel. Most "
        "likely torch's CUDA build is newer than the installed driver; check "
        "`nvidia-smi` and reinstall torch from the matching download.pytorch.org index."
    )
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}")
PY

command -v nvcc >/dev/null || {
    echo "nvcc not found -- set CUDA_HOME or install a CUDA toolkit." >&2
    exit 1
}
nvcc --version | tail -1

pip install --quiet cmake ninja packaging psutil

# Compile only this machine's architecture; building all of them is the bulk of
# the wall-clock time.
TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; m, n = torch.cuda.get_device_capability(); print(f"{m}.{n}")')"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"   # ~2GB RAM per job
export NVCC_THREADS="${NVCC_THREADS:-4}"
echo "building for sm_${TORCH_CUDA_ARCH_LIST} with MAX_JOBS=${MAX_JOBS} (expect 20-60 min)"

# --- build -----------------------------------------------------------------
# CUTLASS arrives as a submodule; a non-recursive clone fails deep into the build.
git clone --recursive --depth 1 --branch "$FA_REF" "$FA_REPO" "$BUILD_DIR"
cd "$BUILD_DIR"

# --no-build-isolation so the build sees the torch already in this venv instead
# of pulling its own. --no-deps because the fork's setup.py still carries a
# stale `torch == 2.4.0` in install_requires and would otherwise downgrade us,
# silently invalidating everything built against the current torch.
pip wheel . --no-build-isolation --no-deps -w dist/
pip install --no-deps --force-reinstall dist/vllm_flash_attn-*.whl

# --- verify ----------------------------------------------------------------
# Resolve through the same path the library uses, and confirm the install did
# not move torch out from under us.
cd "$REPO_ROOT"
python - <<'PY'
import torch
from distr_inference.attention import load_flash_attn_varlen_func

fn = load_flash_attn_varlen_func()
print(f"resolved {fn.__module__}.{fn.__name__}")
print(f"torch {torch.__version__} (must still be the version you built against)")
PY

echo "ok -- built from ${FA_REPO}@${FA_REF}"
