#!/usr/bin/env bash
set -euo pipefail

DSV41_ROOT="${DSV41_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TT_METAL_HOME="${TT_METAL_HOME:-/opt/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME/libexec/tt-metalium}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$DSV41_ROOT/tmp" "$DSV41_ROOT/cache/metal"
export TMPDIR="${TMPDIR:-$DSV41_ROOT/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DSV41_ROOT/cache}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-$DSV41_ROOT/cache/metal}"

OVERLAY="${DSV41_OVERLAY:-$DSV41_ROOT/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors}"
if [[ ! -f "$OVERLAY" && -f "$DSV41_ROOT/hot_experts_wave8_5.safetensors" ]]; then
  OVERLAY="$DSV41_ROOT/hot_experts_wave8_5.safetensors"
fi
if [[ ! -f "$OVERLAY" && -f "$DSV41_ROOT/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors" ]]; then
  OVERLAY="$DSV41_ROOT/packed_rarebal_overlay_wave8_4/hot_experts_wave8_4_perm.safetensors"
fi
if [[ ! -f "$OVERLAY" && -f "$DSV41_ROOT/hot_experts_wave8_4_perm.safetensors" ]]; then
  OVERLAY="$DSV41_ROOT/hot_experts_wave8_4_perm.safetensors"
fi

EXTRA_ARGS=()
HAS_OV_FLAG=0
HAS_L1_FLAG=0
HAS_RELAY_FLAG=0
for arg in "$@"; do
  if [[ "$arg" == "--rarebal-overlay" || "$arg" == "--no-rarebal-overlay" ]]; then
    HAS_OV_FLAG=1
  elif [[ "$arg" == "--p150-l1-shards" ]]; then
    HAS_L1_FLAG=1
  elif [[ "$arg" == "--p150-x16-relay" || "$arg" == "--no-p150-x16-relay" ]]; then
    HAS_RELAY_FLAG=1
  fi
done
if [[ "$HAS_OV_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--rarebal-overlay" "$OVERLAY")
fi
if [[ "$HAS_L1_FLAG" -eq 0 ]]; then
  EXTRA_ARGS+=("--p150-l1-shards" "18")
fi
if [[ "$HAS_RELAY_FLAG" -eq 0 ]]; then
  EXTRA_ARGS+=("--p150-x16-relay")
fi

exec "$DSV41_ROOT/bin/p150_dsv41_worker" --model-dir "$DSV41_ROOT" "${EXTRA_ARGS[@]}" "$@"
