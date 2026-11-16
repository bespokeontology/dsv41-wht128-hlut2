#!/usr/bin/env bash
set -euo pipefail
export OMP_WAIT_POLICY=PASSIVE
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-8}"

DSV41_ROOT="${DSV41_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8080}"
NVME_KV_DIR="${DSV41_NVME_KV_DIR:-$DSV41_ROOT/nvme_kv_cache}"
NVME_KV_GB="${DSV41_NVME_KV_GB:-32}"
VIT_WEIGHTS="${DSV41_VIT_WEIGHTS:-$DSV41_ROOT/packed_vit/dsv41_vit_bf16.safetensors}"
PORT="${DSV41_PORT:-8000}"

EXTRA_ARGS=()
HAS_OV_FLAG=0
HAS_PERM_FLAG=0
HAS_CSA2_SH8=0
for arg in "$@"; do
  if [[ "$arg" == "--rarebal-overlay" || "$arg" == "--no-rarebal-overlay" ]]; then
    HAS_OV_FLAG=1
  elif [[ "$arg" == "--expert-perm-overlay" || "$arg" == "--no-expert-perm" ]]; then
    HAS_PERM_FLAG=1
  elif [[ "$arg" == "--csa2-shared-tree8" || "$arg" == "--no-csa2-shared-tree8" ]]; then
    HAS_CSA2_SH8=1
  fi
done
if [[ "$HAS_OV_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--rarebal-overlay" "$OVERLAY")
fi
if [[ "$HAS_PERM_FLAG" -eq 0 && -f "$OVERLAY" ]]; then
  EXTRA_ARGS+=("--expert-perm-overlay" "$OVERLAY")
fi
if [[ "$HAS_CSA2_SH8" -eq 0 ]]; then
  EXTRA_ARGS+=("--csa2-shared-tree8")
fi

exec "$DSV41_ROOT/bin/dsv41_fe" \
  --model-dir "$DSV41_ROOT" \
  --vit-weights "$VIT_WEIGHTS" \
  --nvme-kv-cache-dir "$NVME_KV_DIR" \
  --nvme-kv-cache-gb "$NVME_KV_GB" \
  --search-url "$SEARCH_URL" \
  --port "$PORT" \
  --max-seq-len 1048576 \
  --dual-stream-overlap \
  --dspark-tree-width auto \
  "${EXTRA_ARGS[@]}" "$@"
