#!/bin/bash
# Convert an already-abliterated checkpoint to the QEngine layout (step 2 of
# build-swift.sh, on its own — useful when re-running only the conversion).
#
#   ABL          the abliterated checkpoint          (default ./swift-abliterated)
#   DST          output directory                    (default ./swift-qengine)
#   BASE_QENGINE an existing QEngine model dir       (default ./qengine-base)
#   CONVERTER    tools/convert_flash_next.py         (default ../qengine-src/tools/...)
#   CHECKPOINT_ID  recorded in the output's config   (default ukisai/Swift-1.5-Qwen3.8-Flash-Next)
set -euo pipefail

WORK=${WORK:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ABL=${ABL:-$PWD/swift-abliterated}
DST=${DST:-$PWD/swift-qengine}
BASE_QENGINE=${BASE_QENGINE:-$PWD/qengine-base}
CONVERTER=${CONVERTER:-$WORK/../qengine-src/tools/convert_flash_next.py}
CHECKPOINT_ID=${CHECKPOINT_ID:-ukisai/Swift-1.5-Qwen3.8-Flash-Next}
LOG=${LOG:-$PWD/swift-convert.log}

exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u) convert $ABL -> $DST ==="
[ -e "$CONVERTER" ] || { echo "ERROR: converter not found: $CONVERTER" >&2; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
QE_NGRAM_EXTERNAL=1 QE_CONV_THREADS=4 python3 "$CONVERTER" \
  --src "$ABL" --dst "$DST" --workers 12 --mtp-int4 \
  --checkpoint-id "$CHECKPOINT_ID"

echo "=== n-gram (reuse the base INT8 table; hardlink so it resolves inside the container mount) ==="
if [ -e "$BASE_QENGINE/ngram_int8.safetensors" ]; then
  ln -f "$BASE_QENGINE/ngram_int8.safetensors" "$DST/ngram_int8.safetensors"
  cp -f "$BASE_QENGINE/expert_prior.bin" "$BASE_QENGINE/cold_experts.json" "$DST/" 2>/dev/null || true
else
  echo "WARNING: no ngram_int8.safetensors in $BASE_QENGINE — build it with tools/convert_ngram_fp8.py"
fi

echo "=== sanity ==="
ls -la "$DST" | head -n 14
python3 - "$DST" <<'PY'
import json, os, sys
D = sys.argv[1]
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
print("index keys:", len(wm))
for pat in ("ngram_q8", "mtp.layers.0.mlp.experts.0.gate_proj", "mtp.layers.0.mlp.experts.gate_up_proj"):
    ks = [k for k in wm if pat in k]
    print(f"  {pat}: {len(ks)} -> {sorted({wm[k] for k in ks})[:3]}")
print("ngram_int8 exists:", os.path.exists(f"{D}/ngram_int8.safetensors"))
PY
echo "SWIFT_CONVERT_DONE"
