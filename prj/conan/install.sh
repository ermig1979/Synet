#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Conan install ==="
OUTPUT=$(conan install "$PROJECT_ROOT/prj/conan" --build=missing "$@" 2>&1) && {
    echo "$OUTPUT"
    echo "=== Success ==="
    exit 0
}

echo "$OUTPUT"
echo ""
echo "=== Install failed. Exporting recipes from submodules... ==="

# Allow git operations in mounted directories (needed when running in Docker)
git config --global --add safe.directory '*' 2>/dev/null || true

SIMD_RECIPE="$PROJECT_ROOT/3rd/Simd/prj/conan/conanfile.py"
if [ -f "$SIMD_RECIPE" ]; then
    conan export "$SIMD_RECIPE"
else
    echo "ERROR: Simd recipe not found. Run: git submodule update --init 3rd/Simd"
    exit 1
fi

CPL_RECIPE="$PROJECT_ROOT/3rd/Cpl/prj/conan/conanfile.py"
if [ -f "$CPL_RECIPE" ]; then
    conan export "$CPL_RECIPE"
else
    echo "ERROR: Cpl recipe not found. Run: git submodule update --init 3rd/Cpl"
    exit 1
fi

echo "=== Retrying with local recipes ==="
conan install "$PROJECT_ROOT/prj/conan" --build=missing "$@"
echo "=== Success ==="
