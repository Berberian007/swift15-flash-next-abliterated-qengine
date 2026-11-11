#!/usr/bin/env python3
"""Abliterate ukisai/Swift-1.5-...-W4A16-AutoRound using jessedye90's refusal direction.

Applies the Arditi projection (W <- W - r r^T W) inside ukisai's own quantisation grids,
exactly as jessedye90's ablate_quant.py does for the GB10 build. Residual writers here:
  INT4 (auto_gptq .qweight/.qzeros/.scales): layers.*.self_attn.o_proj,
        layers.*.linear_attn.out_proj, layers.*.mlp.experts.*.down_proj
  BF16 (.weight):  layers.*.mlp.shared_expert.down_proj,
        mtp.layers.*.self_attn.o_proj, mtp.layers.*.mlp.shared_expert.down_proj
Leaves embed_tokens and ple.value_proj untouched (jessedye90's deliberate choice: editing them
broke their strict coding task). Scales/zero-points and all other tensors stay byte-identical.

usage: ablate_swift.py SRC_DIR DST_DIR REFUSAL_DIRECTION.safetensors
"""
import collections, json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "abliteration"))
import ablate_quant
from ablate_quant import ablate_gptq, ablate_bf16_out_hidden

src, dst, r_file = sys.argv[1:4]
ablate_quant.ALPHA = float(os.environ.get("ABL_ALPHA", "1.0"))
DEV = os.environ.get("ABL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("ABL_THREADS", "60")))
print("alpha:", ablate_quant.ALPHA, "device:", DEV,
      "| cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count(), flush=True)

with safe_open(r_file, "pt") as h:
    r = h.get_tensor("refusal_direction").to(DEV).float()
r = r / r.norm()

wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]

GPTQ = re.compile(r"^model\.language_model\.layers\.\d+\.(?:self_attn\.o_proj|linear_attn\.out_proj|mlp\.experts\.\d+\.down_proj)\.qweight$")
BF16 = re.compile(r"^(?:model\.language_model\.layers\.\d+\.mlp\.shared_expert\.down_proj|mtp\.layers\.\d+\.(?:self_attn\.o_proj|mlp\.shared_expert\.down_proj))\.weight$")

plan = {}
for n in wm:
    if GPTQ.search(n):
        plan[n] = "gptq"
    elif BF16.search(n):
        plan[n] = "bf16"
counts = collections.Counter(plan.values())
print("plan:", dict(counts), flush=True)
fam = collections.Counter(re.sub(r"\.\d+\.", ".N.", n[: -len(".qweight")] if n.endswith(".qweight") else n[: -len(".weight")]) for n in plan)
for f, c in sorted(fam.items()):
    print(f"  {c:7d} {f}", flush=True)
assert counts["gptq"] == 12 + 36 + 48 * 512, counts
assert counts["bf16"] == 48 + 1 + 1, counts   # trunk shared (48) + mtp o_proj (1) + mtp shared (1)

os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith(".safetensors") or f.startswith("."):
        continue
    (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(dst, f),
                                                            **({"dirs_exist_ok": True} if os.path.isdir(p) else {}))

readers = {}
def reader(shard):
    if shard not in readers:
        readers[shard] = safe_open(f"{src}/{shard}", "pt", device="cpu")
    return readers[shard]

def get(name):
    return reader(wm[name]).get_tensor(name).to(DEV)

by_shard = collections.defaultdict(list)
for n in plan:
    by_shard[wm[n]].append(n)

report = {}
shards = sorted(set(wm.values()))
t0 = time.time()
for si, shard in enumerate(shards):
    out_path = f"{dst}/{shard}"
    if os.path.exists(out_path):
        continue
    h = reader(shard)
    tensors = {k: h.get_tensor(k) for k in h.keys()}
    meta = h.metadata()
    edited = 0
    for n in by_shard.get(shard, []):
        kind = plan[n]
        if kind == "gptq":
            p = n[: -len(".qweight")]
            new, st = ablate_gptq(get(n), get(p + ".scales"), get(p + ".qzeros"), r)
        else:
            new, st = ablate_bf16_out_hidden(get(n), r)
        assert new.dtype == tensors[n].dtype and new.shape == tensors[n].shape, n
        tensors[n] = new.cpu()
        report[n] = st
        edited += 1
    save_file(tensors, out_path + ".part", metadata=meta)
    os.replace(out_path + ".part", out_path)
    print(f"[{si+1}/{len(shards)}] {shard}: {edited} edited, {time.time()-t0:.0f}s", flush=True)

json.dump(report, open(f"{dst}/abliteration-report.json", "w"))
worst = max(report.items(), key=lambda kv: kv[1]["resid"]) if report else None
print("edited tensors:", len(report), "| worst residual:", worst, flush=True)
print("ABLATE_DONE")
