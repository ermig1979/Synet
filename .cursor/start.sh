#!/usr/bin/env bash
#
# Cloud Agent start script for Synet.
#
# Runs on every boot after /workspace is checked out. The checkout wipes the
# untracked build/ directory, so restore the compiled artifacts from the cache
# populated by install.sh. Falls back to a full build if no cache exists (for
# example a just-in-time run where install.sh has not populated the cache).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/.cursor/env.sh"

# Submodule working trees normally survive via .git/modules, but restore them if
# a boot dropped them so the build stays reproducible.
if [ ! -f "$ROOT/3rd/Simd/prj/cmake/CMakeLists.txt" ] || [ ! -d "$ROOT/3rd/Cpl/src" ]; then
    git submodule update --init 3rd/Simd 3rd/Cpl
fi

restore_from_cache() {
    if [ -d "$SYNET_BUILD_CACHE" ] && ls "$SYNET_BUILD_CACHE"/test_* >/dev/null 2>&1; then
        echo "=== Restoring Synet build from cache ${SYNET_BUILD_CACHE} ==="
        if [ -e "$ROOT/build" ] && [ ! -d "$ROOT/build" ]; then
            rm -f "$ROOT/build"
        fi
        rm -rf "$ROOT/build"
        cp -a "$SYNET_BUILD_CACHE" "$ROOT/build"
        return 0
    fi
    return 1
}

if ls "$ROOT"/build/test_* >/dev/null 2>&1; then
    echo "=== Synet build already present in workspace ==="
elif restore_from_cache; then
    :
else
    echo "=== No cached build found; running full install ==="
    bash "$ROOT/.cursor/install.sh"
fi

echo "=== Synet applications available: ==="
ls -1 "$ROOT"/build/test_* 2>/dev/null || true
