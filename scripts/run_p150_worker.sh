#!/usr/bin/env bash
set -euo pipefail

DSV41_ROOT="${DSV41_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DSV41_WEIGHTS_DIR="${DSV41_WEIGHTS_DIR:-$DSV41_ROOT}"
export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME/libexec/tt-metalium}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$DSV41_ROOT/tmp" "$DSV41_ROOT/cache/metal"
export TMPDIR="${TMPDIR:-$DSV41_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DSV41_ROOT/cache}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-$DSV41_ROOT/cache/metal}"

OVERLAY="${DSV41_OVERLAY:-$DSV41_WEIGHTS_DIR/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors}"
if [[ ! -f "$OVERLAY" && -f "$DSV41_WEIGHTS_DIR/hot_experts_wave8_5.safetensors" ]]; then
  OVERLAY="$DSV41_WEIGHTS_DIR/hot_experts_wave8_5.safetensors"
fi
PACKED_P150="${DSV41_PACKED_P150_DIR:-$DSV41_WEIGHTS_DIR/packed_p150}"

EXTRA_ARGS=()
HAS_OV_FLAG=0
HAS_L1_FLAG=0
HAS_RELAY_FLAG=0
HAS_KERNELS_FLAG=0
HAS_PACKED_FLAG=0
for arg in "$@"; do
  case "$arg" in
    --rarebal-overlay|--no-rarebal-overlay) HAS_OV_FLAG=1 ;;
    --p150-l1-shards)                       HAS_L1_FLAG=1 ;;
    --p150-x16-relay|--no-p150-x16-relay)   HAS_RELAY_FLAG=1 ;;
    --kernels-dir)                          HAS_KERNELS_FLAG=1 ;;
    --packed-root)                          HAS_PACKED_FLAG=1 ;;
  esac
done

if [[ "$HAS_OV_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--rarebal-overlay" "$OVERLAY")
fi
if [[ "$HAS_PACKED_FLAG" -eq 0 && -d "$PACKED_P150" ]]; then
  EXTRA_ARGS+=("--packed-root" "$PACKED_P150")
fi
if [[ "$HAS_KERNELS_FLAG" -eq 0 ]]; then
  EXTRA_ARGS+=("--kernels-dir" "$DSV41_ROOT/p150/kernels")
fi
if [[ "$HAS_L1_FLAG" -eq 0 ]]; then
  EXTRA_ARGS+=("--p150-l1-shards" "18")
fi
if [[ "$HAS_RELAY_FLAG" -eq 0 ]]; then
  EXTRA_ARGS+=("--p150-x16-relay")
fi

exec "$DSV41_ROOT/bin/p150_dsv41_worker" --max-layers 40 "${EXTRA_ARGS[@]}" "$@"
