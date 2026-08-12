#!/usr/bin/env bash
#
# Cloud Agent install script for Synet.
#
# Builds the core framework and the CPU-only test applications, then caches the
# compiled output outside /workspace so it survives the per-boot git checkout
# (which wipes untracked files such as build/). The companion start.sh restores
# the cache into /workspace/build on every boot.
#
# Idempotent: re-running reconfigures/rebuilds incrementally and refreshes the
# cache.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/.cursor/env.sh"

# The base image's default C/C++ compiler alternative resolves to clang, which
# cannot find libstdc++ in this image. Synet's build.sh hard-codes /usr/bin/c++,
# so point the c++/cc alternatives at GCC. Guarded so it is a no-op when the
# alternative is already set (e.g. baked into a snapshot) or sudo is unavailable.
if command -v sudo >/dev/null 2>&1; then
    sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100 || true
    sudo update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-13 100 || true
    sudo update-alternatives --set c++ /usr/bin/g++-13 || true
    sudo update-alternatives --set cc  /usr/bin/gcc-13 || true
fi

# Only the public dependencies (Simd, Cpl) are required for the core framework
# and the CPU test applications. The OpenVINO and ONNX Runtime submodules use
# SSH URLs to private forks and are intentionally left uninitialized here; enable
# them separately (SSH access + the Conan flow in README.md) if you need the
# test_inference_engine / test_onnx applications.
git submodule update --init 3rd/Simd 3rd/Cpl

# Build the core library plus every test application that does not depend on the
# private OpenVINO/ONNX Runtime submodules. The first mode compiles Simd (the
# slow part, several minutes); subsequent modes reuse the CMake cache and only
# rebuild their small test executable.
for mode in "${SYNET_CPU_TARGETS[@]}"; do
    echo "=== Building Synet test target: ${mode} ==="
    bash build.sh "${mode}"
done

# Cache the compiled output outside /workspace so future boots can restore it
# without recompiling (a fresh checkout wipes the untracked build/ directory).
echo "=== Caching build output to ${SYNET_BUILD_CACHE} ==="
mkdir -p "$(dirname "$SYNET_BUILD_CACHE")"
rm -rf "$SYNET_BUILD_CACHE"
cp -a "$ROOT/build" "$SYNET_BUILD_CACHE"

echo "=== Synet install complete. Built applications: ==="
ls -1 "$ROOT"/build/test_* 2>/dev/null || true
