#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME/libexec/tt-metalium}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/bin" "$ROOT/build/p150"
make -j"$(nproc)" -f "$HERE/Makefile" "$@"
