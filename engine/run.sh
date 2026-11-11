#!/usr/bin/env bash
# Run QEngine (bpx03/Qwen3.8-Flash-Next-QEngine) on 2x RTX 3090 Ti.
# Config = club-3090 `qengine/qwen38-flash-next-dual` recipe (models/qwen3.8-flash-next/
# qengine/compose/dual/qengine-int4/mtp.yml) with the knobs we tuned here.
# API on :8198, dashboard :9393. All knobs are env-overridable.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-./swift-qengine}   # point this at the converted checkpoint
NAME=${NAME:-qengine}
PORT=${PORT:-8198}
MON=${MON:-9393}
IMG=${QE_IMAGE:-ghcr.io/bpx03/qengine:1.3.5.7-vision}
GPUS=${QE_GPUS:-0,1}
# --- club-3090 recipe ---
SEQ=${FLASH_MAX_SEQS:-8}            # sequence slots (club-3090: 8; 4 = one user, 16 > 8 streams)
PAR=${FLASH_PAR:-auto}              # auto -> tp on 2 equal cards with P2P
KV=${FLASH_KV:-vram}                # vram pool (ram = many long conversations, -12% short)
COMM=${FLASH_COMM:-auto}           # auto keeps P2P on our clean x16 link (ce is the fallback)
ARCH=${FLASH_KV_ARCHIVE_GB:-48}    # host-RAM conversation archive (keep on: idle reclaim needs it)
MARGIN=${FLASH_EC_MARGIN_MB:-256}  # VRAM margin next to the expert cache (1024 if graph-OOM)
EFFORT=${QE_REASONING_EFFORT:-xhigh} # default reasoning effort for requests that send none
SERVED=${QE_SERVED_NAME:-qwen3.8-flash-next-qengine}   # served model id (/v1/models)
VISION=${QE_VISION:-1}             # 1 = accept image input (needs the :vision image + torch)
BUDGET=${QE_THINKING_BUDGET:-16384}  # thinking budget (flat; preserve_thinking is on by default in QEngine's template)
# Decode collective path (QEngine >= 1.3.5.6): auto = boot-timed pick, p2p / host = forced.
# Pass it per run: QE_DECODE_COMM=p2p bash run.sh
DECODE_COMM=${QE_DECODE_COMM:-auto}
# The transport probe is cached in the qengine-cache volume, keyed on the PCIe link
# string only -- an added NVLink bridge does NOT invalidate it. Set QE_COMM_CACHE=refresh
# (prefill probe) and/or QE_DECODE_COMM=remeasure (decode path) once after an interconnect change.
COMM_CACHE=${QE_COMM_CACHE:-}
P2P=${QE_P2P:-}
EXTRA_ENV=()
[ -n "$COMM_CACHE" ] && EXTRA_ENV+=(-e "QE_COMM_CACHE=$COMM_CACHE")
[ -n "$P2P" ] && EXTRA_ENV+=(-e "QE_P2P=$P2P")
[ -n "$COMM_CACHE" ] && DECODE_COMM=remeasure   # a fresh-boot measurement run measures decode too
# Publish the API on the docker bridge gateway too, so benchlocal sandboxes
# (default bridge + --add-host=host.docker.internal:host-gateway) can reach it.
# 127.0.0.1 keeps the LAN client path loopback-only; the bridge IP is private.
# CAVEAT (this host): INPUT policy is DROP and Docker's DNAT rule excludes
# -i docker0, so the bridge-published port is NOT actually reachable from a
# container here. Use QEngine's container IP on the bridge instead (see
# swift-qengine/run-hermes.sh). This binding is left in for hosts without the
# restrictive INPUT policy.
BRIDGE_IP=$(ip -4 addr show docker0 2>/dev/null | awk '/inet /{sub(/\/.*/,"",$2); print $2; exit}')
PUBLISH=(-p "127.0.0.1:${PORT}:8080" -p "127.0.0.1:${MON}:9393")
[ -n "${BRIDGE_IP}" ] && PUBLISH+=(-p "${BRIDGE_IP}:${PORT}:8080")
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --ulimit memlock=-1 \
  "${PUBLISH[@]}" \
  -v "$MODEL_DIR:/app/models:ro" -v qengine-cache:/root/.cache/qengine \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e QE_MODEL=/app/models -e QE_GPUS="$GPUS" \
  -e FLASH_MAX_SEQS="$SEQ" -e FLASH_PAR="$PAR" -e FLASH_KV="$KV" -e FLASH_COMM="$COMM" \
  -e FLASH_KV_ARCHIVE_GB="$ARCH" -e FLASH_EC_MARGIN_MB="$MARGIN" \
  -e QE_REASONING_EFFORT="$EFFORT" \
  -e QE_THINKING_BUDGET="$BUDGET" \
  -e QE_DECODE_COMM="$DECODE_COMM" \
  "${EXTRA_ENV[@]}" \
  -e QE_EXTRA_ARGS="--served-name ${SERVED} --vision ${VISION}" \
  "$IMG" >/dev/null
echo "qengine starting (seqs=$SEQ par=$PAR kv=$KV comm=$COMM decode=$DECODE_COMM archive=${ARCH}GB margin=${MARGIN}MB effort=$EFFORT); waiting for health..."
for i in $(seq 1 180); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${PORT}/health" || true)
  [ "$code" = 200 ] && { echo "qengine healthy on :${PORT}"; exit 0; }
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "qengine container exited:"; docker logs "$NAME" --tail 40 2>&1; exit 1
  fi
  sleep 5
done
echo "qengine health failed; last logs:"; docker logs "$NAME" --tail 40 2>&1; exit 1
