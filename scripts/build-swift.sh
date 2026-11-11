#!/bin/bash
# Swift-1.5 W4A16 -> abliterate -> QEngine checkpoint. The full pipeline we ran.
#
# Paths are environment variables; the defaults are relative to the current directory.
# Nothing here is host-specific.
#
#   SRC        the W4A16 checkpoint to convert            (default ./swift-w4a16)
#   ABL        where the abliterated copy is written      (default ./swift-abliterated)
#   DST        where the QEngine checkpoint is written    (default ./swift-qengine)
#   WORK       this repository                            (default: parent of this script)
#   DIRECTION  refusal_direction.safetensors              (default $WORK/abliteration/refusal_direction.safetensors)
#   BASE_QENGINE  an existing QEngine model dir; its shared n-gram table and expert prior
#                 are reused                                            (default ./qengine-base)
#   CONVERTER  tools/convert_flash_next.py (patched, see ../converter/)  (default: $WORK/../qengine-src/tools/...)
#   IMAGE      an image that has CUDA + torch (the QEngine image does)
#                                                                  (default ghcr.io/bpx03/qengine:1.3.5.7-vision)
#   CHECKPOINT_ID  recorded in the output's config       (default ukisai/Swift-1.5-Qwen3.8-Flash-Next)
#
# The abliteration step needs the refusal direction. We do not ship it: it is recovered
# from a (original, abliterated) tensor pair with ablation/direction.py. See the README.
set -euo pipefail

WORK=${WORK:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
SRC=${SRC:-$PWD/swift-w4a16}
ABL=${ABL:-$PWD/swift-abliterated}
DST=${DST:-$PWD/swift-qengine}
BASE_QENGINE=${BASE_QENGINE:-$PWD/qengine-base}
DIRECTION=${DIRECTION:-$WORK/abliteration/refusal_direction.safetensors}
CONVERTER=${CONVERTER:-$WORK/../qengine-src/tools/convert_flash_next.py}
IMAGE=${IMAGE:-ghcr.io/bpx03/qengine:1.3.5.7-vision}
CHECKPOINT_ID=${CHECKPOINT_ID:-ukisai/Swift-1.5-Qwen3.8-Flash-Next}
LOG=${LOG:-$PWD/swift-build.log}

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u) swift build: $SRC -> $ABL -> $DST"

for p in "$SRC" "$DIRECTION" "$CONVERTER"; do
  [ -e "$p" ] || { echo "ERROR: not found: $p" >&2; exit 1; }
done
du -sh "$SRC"

# --- 0. the checkpoint is complete? -----------------------------------------
python3 - "$SRC" <<'PY'
import glob, json, os, sys
S = sys.argv[1]
wm = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
files = {v for v in wm.values()}
miss = [f for f in sorted(files) if not os.path.exists(f"{S}/{f}")]
print("referenced", len(files), "missing", miss[:5] or "none",
      "| shards on disk", len(glob.glob(f"{S}/*.safetensors")))
if miss:
    sys.exit("incomplete checkpoint")
PY

# --- 1. abliterate on the GPU (through the engine image's CUDA torch) --------
# The engine and the abliteration step both want the cards, so stop the engine first.
if docker ps --format '{{.Names}}' | grep -qx qengine; then
  echo "=== stopping qengine to free the GPUs ==="
  docker stop qengine 2>&1 | tail -1
  sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
fi
# SRC and ABL must sit under one shared parent so they can be mounted together.
PARENT=$(cd -- "$(dirname -- "$SRC")" && pwd)
[ "$PARENT" = "$(cd -- "$(dirname -- "$ABL")" && pwd)" ] || {
  echo "ERROR: SRC and ABL must share a parent directory (they are mounted as /data)" >&2; exit 1; }
rm -rf "$ABL"
docker run --rm --gpus all --entrypoint python3 \
  -v "$PARENT":/data -v "$WORK":/work \
  "$IMAGE" \
  /work/ablation/ablate_swift.py "/data/$(basename "$SRC")" "/data/$(basename "$ABL")" \
  "/work/abliteration/$(basename "$DIRECTION")"
# NOTE: the direction file must be visible inside the mount, i.e. under $WORK.

# --- 2. convert to the QEngine layout ---------------------------------------
rm -rf "$DST"; mkdir -p "$DST"
QE_NGRAM_EXTERNAL=1 QE_CONV_THREADS=4 python3 "$CONVERTER" \
  --src "$ABL" --dst "$DST" --workers 12 --mtp-int4 \
  --checkpoint-id "$CHECKPOINT_ID"

# --- 3. n-gram: reuse the base INT8 table, plus the expert prior ------------
# QE_NGRAM_EXTERNAL=1 above makes the converter index ngram_int8.safetensors
# instead of symlinking the shards the FP8 table lives in.
if [ -e "$BASE_QENGINE/ngram_int8.safetensors" ]; then
  ln -f "$BASE_QENGINE/ngram_int8.safetensors" "$DST/ngram_int8.safetensors"
  cp -f "$BASE_QENGINE/expert_prior.bin" "$BASE_QENGINE/cold_experts.json" "$DST/" 2>/dev/null || true
else
  echo "WARNING: no ngram_int8.safetensors in $BASE_QENGINE — build it with tools/convert_ngram_fp8.py"
fi

# --- 4. sanity --------------------------------------------------------------
ls -la "$DST" | head -n 12
python3 - "$DST" <<'PY'
import json, os, sys
D = sys.argv[1]
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
print("index keys", len(wm), "| ngram_q8 ->", sorted({wm[k] for k in wm if 'ngram_q8' in k}))
print("has ngram_int8:", os.path.exists(f"{D}/ngram_int8.safetensors"))
PY
echo "SWIFT_BUILD_DONE"
