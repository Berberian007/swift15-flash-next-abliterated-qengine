#!/usr/bin/env python3
"""Abliterate jessedye90/Swift-1.5-Qwen3.8-Flash-Next-W4A16-GB10 in place of its own quantisation.

usage: apply_flash.py SRC_DIR DST_DIR R_FLASH.pt
Edits the same 149 residual writers as orcarouter/Qwen3.8-Flash-Next-Uncensored (one global refusal
direction recovered from that model vs Qwen/Qwen3.8-Flash-Next), keeping every scale, zero-point, shard
name and untouched tensor byte-identical.
"""
import collections, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from ablate_quant import ablate_bf16_out_hidden, ablate_embedding, ablate_fp8_block, ablate_gptq

src, dst, r_path = sys.argv[1:4]
import ablate_quant
ablate_quant.ALPHA = float(os.environ.get("ABL_ALPHA", "1.0"))
print("alpha:", ablate_quant.ALPHA)
r = torch.load(r_path).cuda().float()
r = r / r.norm()
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]

GPTQ = re.compile(r"(?:^model\.language_model\.layers\.\d+\.(?:self_attn\.o_proj|linear_attn\.out_proj|mlp\.experts\.\d+\.down_proj)"
                  r"|^mtp\.layers\.\d+\.(?:self_attn\.o_proj|mlp\.shared_expert\.down_proj))\.qweight$")
FP8 = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert\.down_proj\.weight$")
BF16_OUT = re.compile(r"^(?:mtp\.layers\.\d+\.mlp\.experts\.\d+\.down_proj|model\.language_model\.layers\.\d+\.ple\.value_proj)\.weight$")
EMBED = "model.language_model.embed_tokens.weight"

plan = {n: kind for n in wm for kind, ok in (("gptq", GPTQ.search(n)), ("fp8", FP8.search(n)),
                                              ("bf16", BF16_OUT.search(n)), ("embed", n == EMBED)) if ok}
SKIP = os.environ.get("ABL_SKIP")  # regex of tensor names to leave untouched (variant builds)
if SKIP:
    plan = {n: k for n, k in plan.items() if not re.search(SKIP, n)}
    print("skip:", SKIP)
counts = collections.Counter(plan.values())
families = collections.Counter(re.sub(r"\.\d+\.", ".N.", re.sub(r"\.(qweight|weight)$", "", n)) for n in plan)
print("plan:", dict(counts))
for f, c in sorted(families.items()):
    print(f"  {c:6d} {f}")
# 13 o_proj + 36 out_proj + 48 layers of experts + 1 MTP experts + 48 FP8 shared + 1 MTP shared + value_proj + embed = 149
assert SKIP or counts["gptq"] == 13 + 36 + 48 * 512 + 1, counts
assert SKIP or counts["fp8"] == 48 and counts["bf16"] == 512 + 1 and counts["embed"] == 1, counts

os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith(".safetensors") or f.startswith("."):
        continue
    (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(dst, f), **({"dirs_exist_ok": True} if os.path.isdir(p) else {}))

readers = {}


def reader(shard):
    if shard not in readers:
        readers[shard] = safe_open(f"{src}/{shard}", "pt", device="cpu")
    return readers[shard]


def get(name, device="cuda"):
    return reader(wm[name]).get_tensor(name).to(device)


report = {}
by_shard = collections.defaultdict(list)
for n, kind in plan.items():
    by_shard[wm[n]].append(n)
shards = sorted(set(wm.values()))
t_start = time.time()
for si, shard in enumerate(shards):
    out_path = f"{dst}/{shard}"
    if os.path.exists(out_path):
        continue
    h = reader(shard)
    tensors = {k: h.get_tensor(k) for k in h.keys()}
    meta = h.metadata()
    for n in by_shard.get(shard, []):
        kind = plan[n]
        if kind == "gptq":
            p = n[: -len(".qweight")]
            new, st = ablate_gptq(get(n), get(p + ".scales"), get(p + ".qzeros"), r)
        elif kind == "fp8":
            new, st = ablate_fp8_block(get(n), get(n + "_scale_inv"), r)
        elif kind == "bf16":
            new, st = ablate_bf16_out_hidden(get(n), r)
        else:
            new, st = ablate_embedding(get(n), r)
        assert new.dtype == tensors[n].dtype and new.shape == tensors[n].shape, n
        tensors[n] = new.cpu()
        report[n] = st
    save_file(tensors, out_path + ".part", metadata=meta)
    os.replace(out_path + ".part", out_path)
    done = len(by_shard.get(shard, []))
    print(f"[{si + 1}/{len(shards)}] {shard}: {done} edited, {time.time() - t_start:.0f}s", flush=True)

json.dump(report, open(f"{dst}/abliteration-report.json", "w"))
worst = max(report.items(), key=lambda kv: kv[1]["resid"]) if report else None
print("edited tensors:", len(report), "worst residual:", worst)
