#!/usr/bin/env bash
# Shared configuration for the Synet Cloud Agent environment scripts.

# CPU-only test targets built by install.sh (everything that does not require
# the private OpenVINO/ONNX Runtime submodules).
SYNET_CPU_TARGETS=(stability quantization optimizer bf16 multi_threads video)

# Location where the compiled build/ directory is cached, outside /workspace so
# it survives the per-boot git checkout that wipes untracked files.
SYNET_BUILD_CACHE="${HOME}/.cache/synet/build"
