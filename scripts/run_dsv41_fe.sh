#!/usr/bin/env bash
set -euo pipefail
export OMP_WAIT_POLICY=PASSIVE
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"

DSV41_ROOT="${DSV41_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DSV41_WEIGHTS_DIR="${DSV41_WEIGHTS_DIR:-$DSV41_ROOT}"
DSV41_OVERLAY_DIR="${DSV41_OVERLAY_DIR:-$DSV41_WEIGHTS_DIR/wave9_2_overlays}"
DSV41_KV_CACHE_DIR="${DSV41_KV_CACHE_DIR:-$DSV41_WEIGHTS_DIR/nvme_kv_cache}"

OVERLAY="${DSV41_OVERLAY:-$DSV41_WEIGHTS_DIR/packed_rarebal_overlay_wave8_5/hot_experts_wave8_5.safetensors}"
if [[ ! -f "$OVERLAY" && -f "$DSV41_WEIGHTS_DIR/hot_experts_wave8_5.safetensors" ]]; then
  OVERLAY="$DSV41_WEIGHTS_DIR/hot_experts_wave8_5.safetensors"
fi
DEV3_OVERLAY="${DSV41_DEV3_OVERLAY:-$DSV41_OVERLAY_DIR/wave9_2_dev3_hlut3.safetensors}"
TP3_OVERLAY="${DSV41_TP3_OVERLAY:-$DSV41_OVERLAY_DIR/wave9_2_tp3_hlut3.safetensors}"
DEV3_ID="${DSV41_DEV3_ID:-3}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8888}"
NVME_KV_GB="${DSV41_NVME_KV_GB:-32}"
VIT_WEIGHTS="${DSV41_VIT_WEIGHTS:-$DSV41_WEIGHTS_DIR/packed_vit/dsv41_vit_bf16.safetensors}"
PORT="${DSV41_PORT:-8042}"
DEFAULT_EFFORT="${DSV41_DEFAULT_EFFORT:-65}"

EXTRA_ARGS=()
HAS_OV_FLAG=0
HAS_PERM_FLAG=0
HAS_DEV3_FLAG=0
HAS_TP3_FLAG=0
HAS_CSA2_SH8=0
for arg in "$@"; do
  case "$arg" in
    --rarebal-overlay|--no-rarebal-overlay) HAS_OV_FLAG=1 ;;
    --expert-perm-overlay|--no-expert-perm) HAS_PERM_FLAG=1 ;;
    --hlut3-dev3-overlay)                   HAS_DEV3_FLAG=1 ;;
    --hlut3-tp3-overlay)                    HAS_TP3_FLAG=1 ;;
    --csa2-shared-tree8|--no-csa2-shared-tree8) HAS_CSA2_SH8=1 ;;
  esac
done

if [[ "$HAS_OV_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--rarebal-overlay" "$OVERLAY")
fi
if [[ "$HAS_PERM_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--expert-perm-overlay" "$OVERLAY")
fi
if [[ "$HAS_DEV3_FLAG" -eq 0 && -f "$DEV3_OVERLAY" ]]; then
  EXTRA_ARGS+=("--hlut3-dev3-overlay" "$DEV3_OVERLAY" "--hlut3-dev3-id" "$DEV3_ID")
fi
if [[ "$HAS_TP3_FLAG" -eq 0 && -f "$TP3_OVERLAY" ]]; then
  EXTRA_ARGS+=("--hlut3-tp3-overlay" "$TP3_OVERLAY")
fi
if [[ "$HAS_CSA2_SH8" -eq 0 ]]; then
  EXTRA_ARGS+=("--csa2-shared-tree8")
fi

exec "$DSV41_ROOT/bin/dsv41_fe" \
  --model-dir "$DSV41_WEIGHTS_DIR" \
  --vit-weights "$VIT_WEIGHTS" \
  --nvme-kv-cache-dir "$DSV41_KV_CACHE_DIR" \
  --nvme-kv-cache-gb "$NVME_KV_GB" \
  --search-url "$SEARCH_URL" \
  --port "$PORT" \
  --max-seq-len 1048576 \
  --default-reasoning-effort "$DEFAULT_EFFORT" \
  --fused-mhc-norm \
  --fused-dspark-epilogue \
  --dual-stream-overlap \
  --dspark-tree-width auto \
  "${EXTRA_ARGS[@]}" "$@"
