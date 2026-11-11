# converter/ — our patch to QEngine's Flash-Next converter

`convert_flash_next.patch` is a diff against **QEngine**'s `tools/convert_flash_next.py`.

| | |
|---|---|
| Base | `5028c50` (QEngine **1.3.5.7**) — verified byte-identical to upstream at that commit |
| Applies | cleanly with `git apply -p1` on top of `5028c50` (no rebase needed) |
| Size | +77 / −16 |
| Offered upstream | https://github.com/bpx03/QEngine/pull/23 |

```bash
git clone https://github.com/bpx03/QEngine
cd QEngine && git checkout 5028c50
git apply -p1 /path/to/convert_flash_next.patch
```

Once upstream merges (or if it takes a different shape), prefer upstream and ignore this directory.

## Why it exists

QEngine's converter targets Intel's AutoRound layout. Other publishers ship W4A16 exports that
differ in two ways, and both make the stock tool fail or emit a checkpoint the engine cannot load:

**1. Dense tensors that arrive already INT4.** In these exports the DeltaNet and `self_attn`
projections are quantised (auto_gptq packing, `.qweight`/`.qzeros`/`.scales`) where the engine
expects `.weight`. The patch dequantises them:

- `.linear_attn.*` → written as **INT8 g128** in the files (how the engine stores them);
- `self_attn.q/k/v/o_proj` → **BF16** `.weight`, which the engine quantises at load;
- the matching `.qzeros`/`.scales` are consumed and skipped rather than copied;
- every other non-`.weight` tensor (`A_log`, `dt_bias`, …) is copied as-is.

**2. Fused MTP experts.** Where the source stores the MTP experts fused (`[NE,2I,H]` / `[NE,H,I]`)
instead of per-expert, the patch splits them and RTN-quantises to INT4 — which is what makes
`--mtp-int4` work on those checkpoints.

**3.** `QE_NGRAM_EXTERNAL=1` indexes an externally built `ngram_int8.safetensors` instead of
symlinking the shards the FP8 n-gram table lives in. The producer for that is
`tools/convert_ngram_fp8.py`, which upstream does not ship — it is part of the same PR, since the
patch references it by name in a comment.

Small extra: `QE_CONV_THREADS` sets the torch thread count (default 6).

## Verifying a conversion

`scripts/convert-only.sh` prints an index sanity block: the number of index keys, how the
`ngram_q8` shards resolve, and whether MTP experts came out per-expert. Worth checking before you
point an engine at 121 GB of output.
