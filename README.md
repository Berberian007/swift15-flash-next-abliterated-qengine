# Swift-1.5 Flash-Next **Abliterated** on QEngine — abliteration + conversion kit

Everything needed to take **`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound`** (a W4A16
AutoRound export), remove its refusal direction by directional ablation, convert it to the layout
**QEngine** loads, and serve it on two consumer cards.

This is the pipeline behind the build measured in [club-3090 discussions
#1961](https://github.com/noonghunna/club-3090/discussions/1961) (quality) and
[issue #2000](https://github.com/noonghunna/club-3090/issues/2000) (NVLink, 2× 3090 Ti).

> **Abliterated weights have no reliable guardrails.** See the [Safety](#safety) section before
> you use or publish anything this kit produces.

## What's here

```
ablation/    the abliteration kit (recover the refusal direction, apply it, probe for refusals)
converter/   our patch to QEngine's tools/convert_flash_next.py  (+ offered upstream as PR #23)
scripts/     the pipeline: build-swift.sh (abliterate + convert), convert-only.sh (convert)
engine/      run.sh — the QEngine launcher we use, all knobs env-overridable
reports/     the measured write-up: NVLink vs PCIe vs host on this build
```

## What you have to supply

Nothing here ships weights. You need:

| | |
|---|---|
| **The W4A16 checkpoint** | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound` (~166 GB). The upstream repos are **gated** — accept their terms first. |
| **A refusal direction** | We do **not** ship ours (it is recovered from someone else's abliterated model). Build your own with `ablation/direction.py`, which takes a `(original, abliterated)` tensor pair and reports how cleanly the edit is rank-1. Alternatively use any published refusal direction you have the right to use. |
| **A QEngine model directory** | For two shared files the converter doesn't produce: `ngram_int8.safetensors` (the PLE n-gram table — or build it with `tools/convert_ngram_fp8.py`) and `expert_prior.bin` / `cold_experts.json`. |
| **A QEngine checkout** | `git clone https://github.com/bpx03/QEngine`, then apply `converter/convert_flash_next.patch` to `tools/convert_flash_next.py`. The patch is against `5028c50` (QEngine 1.3.5.7) and applies cleanly. |
| **An image with CUDA + torch** | The abliteration step runs on the GPU; the QEngine image works (`ghcr.io/bpx03/qengine:1.3.5.7-vision`). |
| **Disk + RAM** | ~166 GB source, ~166 GB abliterated copy, ~121 GB converted output. The conversion is CPU-bound and benefits from many cores. |

## Run it

Paths are environment variables; the defaults are relative to where you run it.

```bash
export SRC=./swift-w4a16                 # the W4A16 checkpoint
export ABL=./swift-abliterated           # where the abliterated copy goes
export DST=./swift-qengine               # where the QEngine checkpoint goes
export BASE_QENGINE=./qengine-base       # for the shared n-gram table + expert prior
export CONVERTER=../QEngine/tools/convert_flash_next.py
export DIRECTION=$PWD/abliteration/refusal_direction.safetensors

bash scripts/build-swift.sh
```

That runs four steps and logs to `swift-build.log`:

1. checks the checkpoint is complete (index vs shards on disk);
2. **abliterates on the GPU** (stops a running `qengine` container first — both want the cards);
3. **converts** with `QE_NGRAM_EXTERNAL=1 … --mtp-int4` (the W4A16 dense-dequant, fused-MTP and
   external-n-gram paths the patch adds);
4. copies the shared n-gram table + expert prior and sanity-checks the index.

Already have an abliterated checkpoint? `scripts/convert-only.sh` is step 3 alone.

Then probe it — worth doing, it is cheap and it is the point of the exercise:

```bash
python3 ablation/eval_refusal.py http://127.0.0.1:8198 <served-model-id> refusal.json 100
```

## Serve it

`engine/run.sh` is the launcher (QEngine on `:8198`, dashboard `:9393`). All knobs are env vars:
`QE_GPUS`, `FLASH_MAX_SEQS`, `FLASH_PAR`, `FLASH_KV`, `FLASH_COMM`, `FLASH_KV_ARCHIVE_GB`,
`FLASH_EC_MARGIN_MB`, `QE_REASONING_EFFORT`, `QE_THINKING_BUDGET`, `QE_SERVED_NAME`, `QE_VISION`,
`QE_DECODE_COMM`, `QE_COMM_CACHE`, `QE_P2P`.

```bash
MODEL_DIR=./swift-qengine QE_SERVED_NAME=my-swift-abl bash engine/run.sh
```

Two things we learned the hard way, both are in the launcher:

- **After any interconnect change, force the engine to re-measure.** QEngine picks its transport
  from an empirical probe whose result is cached in a Docker volume, and the cache key encodes the
  *PCIe* link, not NVLink — so a newly installed bridge goes unnoticed forever.
  `QE_COMM_CACHE=refresh` (prefill probe) and `QE_DECODE_COMM=remeasure` (decode path) re-measure.
- **Check what it picked**: `docker logs <container> | grep '\[comm\]'`. The peer bandwidth should
  read ~49–53 GB/s on a 3090-class NVLink pair and ~26 GB/s on PCIe 4.0 x16.

Measured on 2× RTX 3090 Ti (TP=2, 8 slots, 262,144 ctx, KV in VRAM): 8-pack **123/150** pass@1 ·
**136/150** pass@3 thinking-ON; KL **0.384** against the official BF16 base. The NVLink A/B
(+53% at c=8, prefill +11–21%, TTFT −10–18%) and the full caveats are in
[`reports/NVLINK-AB-20261111-EN.md`](reports/NVLINK-AB-20261111-EN.md).

## Safety

**This kit removes the refusal direction from the model.** The result answered 0 of 12 refusal
probes. That is the intent — and it means:

- there are **no reliable guardrails**; put your own moderation in front of anything user-facing;
- do not expose it to the public internet without that;
- think about whether publishing the weights is something you want to do at all. If you do, say
  what it is: state the abliteration, state the missing guardrails, and carry the notices.

We have not published our converted weights.

## Licence

Three different things, three different licences — `NOTICE` has the detail:

- **our code** (`ablation/`, `scripts/`, `engine/`, `reports/`) — MIT;
- **`converter/convert_flash_next.patch`** — a derivative of QEngine, therefore
  **PolyForm Noncommercial 1.0.0** (noncommercial only, notice required);
- **the checkpoints you build** — derivative works of the UkisAI / Qwen models, under the
  **Qwen Community License 1.0** (with its commercial Model-as-a-Service condition).

## Status

This is the pipeline we actually ran, with the paths lifted into environment variables and no
host-specific values. It has not been run end to end by anyone else. If you try it and something
in `scripts/` is wrong for your layout, that is a bug in the parameterisation, not in the pipeline —
open an issue.

The converter patch is offered to QEngine upstream as
[PR #23](https://github.com/bpx03/QEngine/pull/23); once/if that lands, prefer upstream and drop
`converter/`.
