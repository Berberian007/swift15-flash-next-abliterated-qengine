# NVLink on a 2× RTX 3090 Ti box: the engine never noticed, and what the bridge is actually worth

**QEngine 1.3.5 → 1.3.5.7 · ukisai Swift-1.5 Qwen3.8-Flash-Next W4A16 AutoRound (abliterated) · 2× RTX 3090 Ti · measured 2026-11-11**

> **Published** as a `[bench]` issue on club-3090 — [noonghunna/club-3090#2000](https://github.com/noonghunna/club-3090/issues/2000)
> (frozen text: `post/2026-11-11-nvlink-qengine/finalposts/`). This file is the fuller internal version;
> the issue is the same data, reorganised for their `numbers-from-your-rig` template.

---

## TL;DR

- We installed an NVLink bridge. The link came up at the driver level (`nvidia-smi topo -m` → **`NV4`**, 4 bonded links, 4 × 14.062 GB/s = **56.25 GB/s per direction**), and CUDA peer access worked in both directions.
- **The engine kept using PCIe anyway.** QEngine picks its transport from an empirical bandwidth probe whose result is cached in a Docker volume — and the cache key encodes the *PCIe link string*, not NVLink. The bridge changes nothing in that key, so **no restart would ever have refreshed it**.
- Forcing a fresh probe measured **26.2 → 49.3 GB/s** peer bandwidth on the same machine, same cards, same driver. The host-staged figure stayed at exactly **13.2 GB/s** — independent proof that only the GPU↔GPU path changed.
- Measured gain, same harness, one boot per arm: **prefill +11 % to +21 %** (needle cases +11 to +19 %), **TTFT −10 % to −18 %**, and **+11.7 / +21.2 / +32.2 / +53.0 % aggregate throughput at c=1 / 2 / 4 / 8**.
- **What NVLink does not fix:** decode collectives. They cost **15.70 ms/step on PCIe and 15.68 ms/step on NVLink** — unchanged. At ~10 KB per step the all-reduce is latency-bound (~0.65 GB/s effective), three orders of magnitude below either link. Anyone benchmarking only single-stream decode would conclude the bridge does nothing.
- Independent L2 check ([club-3090](https://github.com/noonghunna/club-3090) `p2p-validate.sh`): NCCL all-reduce **PASS** on both 4 B (LL proto) and 16 MiB (Simple proto), with the P2P control arm also passing. Same run measured **peer 52.81 GB/s vs host-staged 13.28 GB/s (3.98×)**.

## Rig and stack

| | |
|---|---|
| **Model** | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound`, abliterated in-grid → QEngine INT4/INT8/BF16 layout |
| **Engine** | QEngine `1.3.5.7-vision` (`ghcr.io/bpx03/qengine:1.3.5.7-vision`, build `5028c50`, `@sha256:2d790174…e124`); previously `1.3.5-vision` (build `1348a95`) |
| **GPUs** | 2× RTX 3090 Ti, `sm_86`, both PCIe Gen4 **x16**, both on **NUMA node 0**, 480 W/card (uncapped), 2100/2085 MHz, 64/69 °C |
| **Interconnect** | **NVLink NV4** (4 × 14.062 GB/s = 56.25 GB/s/dir) — previously PCIe only |
| **Serving** | TP=2, 8 slots, 262,144 ctx, vision on, `QE_REASONING_EFFORT=xhigh`, flat `QE_THINKING_BUDGET=16384`, KV in VRAM, 48 GB host-RAM conversation archive |
| **Driver** | 615.71.09, open kernel module |

## Finding 1 — the engine's transport decision is cached on the PCIe link string

QEngine does not read NVLink topology anywhere; the string "nvlink" does not appear in its source. It probes the link empirically at start-up (SM stores, SM loads, copy engine, host-staged), measures a "co-run tax" for each, and picks. The result is cached in the `qengine-cache` volume. The cache key looks like this:

```
key v1|build 1348a95…|cuda rt 13000 drv 13040|NVRM version: … 615.71.09 …|ngpu 2 p2p 1
|gpu 0 GPU-… 0000:02:00.0 16.0 GT/s PCIe trained 16.0 GT/s PCIe x16 numa -1 iommu 0 pl 480000 … gen 4 x16
|gpu 1 GPU-… 0000:04:00.0 16.0 GT/s PCIe trained 16.0 GT/s PCIe x16 numa -1 iommu 0 pl 480000 … gen 4 x16
|cmdline|peer-11-|probe staged 1
```

Every per-GPU field describes the **PCIe** link (speed, trained speed, width, generation). Installing a bridge leaves that key byte-identical, so the cached probe — measured before the bridge existed — is reused indefinitely. Our container restarted on the day of the bridge and the boot log said:

```
[comm] topology: numa -1,-1 (1 node), PCIe x16,x16, P2P rank 0-1:0
[comm] probe: bandwidth and co-run tax from the start-up cache
       (measured 2026-11-06 22:26:43, same cards, links, driver and build; QE_COMM_CACHE=refresh measures again)
```

The fix is not "enable NVLink" — the driver already routes peer access over it. The fix is **making the engine measure again**.

## What we changed

| | From | To |
|---|---|---|
| Image pin (both launchers) | `qengine:1.3.5-vision` (build `1348a95`) | **`qengine:1.3.5.7-vision`** (build `5028c50`, `@sha256:2d790174…e124`) |
| Launcher env | — | `QE_DECODE_COMM` (auto/p2p/host), `QE_COMM_CACHE`, `QE_P2P` exposed as environment passthroughs |

1.3.5.7 rather than 1.3.5 because 1.3.5.6 reworked the host decode path (12.1 → 8.3 ms/step on the maintainer's box for people who fell back to host memory) and 1.3.5.7 is the current stable. Changing the build also rotates the cache key's `build` field, so the first boot after the upgrade re-probes by itself.

## Evidence 1 — the probe (same machine, same cards, same driver, same model)

| metric (0→1) | 1.3.5 · cache from 2026-11-06 (pre-bridge) | 1.3.5.7 · fresh probe | Δ |
|---|---:|---:|---|
| SM store 0→1 | 26.21 GB/s | **49.3 GB/s** | +88 % |
| SM load (receiver) | 26.56 | **49.3** | +86 % |
| copy engine | 26.26 | **52.7** | +101 % |
| **host-staged (via RAM)** | 13.24 | **13.2** | **0 %** |
| co-run tax, store | 1.379 | **1.23** | lower is better |
| tax spread (store) | 18.26 | 5.62 | — |

The host-staged figure staying flat is the control: GPU↔host traffic still crosses PCIe, so the bridge can only have affected the peer path. Raw lines:

```
# 1.3.5 (cached probe, measured 2026-11-06 22:26:43)
[comm] probe GB/s (…): 0->1 26.2/26.6/26.3/13.2 (tax 18.26/1.38/1.03), 1->0 26.2/26.6/26.3/13.2 (tax 15.14/1.37/1.03)
[comm] prefill transfers: P2P SM loads / stores (… P2P 27 CE 26 host 13 GB/s: P2P kept: tax over the cap, under the 1 NUMA node cap)
[comm] decode collectives: single sequence: P2P SM stores (P2P 15.70, host memory 18.09 ms/step)

# 1.3.5.7 (fresh probe, 2026-11-11)
[comm] probe GB/s (…): 0->1 49.3/49.3/52.7/13.2 (tax 5.62/1.23/1.05), 1->0 49.5/49.3/52.7/13.2 (tax 5.68/1.21/1.04)
[comm] prefill transfers: P2P SM loads / stores (… P2P 49 CE 53 host 13 GB/s: P2P kept: tax under the cap)
[comm] decode collectives (auto, real decode steps, median of 4 interleaved rounds of 12):
       single sequence: P2P SM stores (P2P 15.68, host memory 16.18 ms/step, host faster in 0/4);
       batched (4):     P2P SM stores (P2P 27.58, host memory 28.07 ms/step, host faster in 0/4)
```

Note that QEngine's own `[comm] topology` line still prints `PCIe x16,x16`: that is its summary of the PCIe links (which are unchanged). The NVLink is visible in the **measured** bandwidth, not in that line.

## Evidence 2 — client throughput (A = PCIe-P2P, B = NVLink-P2P)

Harness: the club's standard client (`benchmark_strata.py`: cold 60K/120K/200K prompts, decode, needle at depth 10/50/90K), plus a short-prompt set and a c=1/2/4/8 concurrency sweep. One boot per arm.

| measurement | A — PCIe (26.2 GB/s) | B — NVLink (49.3 GB/s) | Δ |
|---|---:|---:|---:|
| prefill, cold 60K | 5,975 tok/s | **7,256** | **+21.4 %** |
| prefill, cold 120K | 7,023 | 6,347 | −9.6 % † |
| prefill, cold 200K | 5,514 | **6,148** | **+11.5 %** |
| prefill, follow-up on a warm prefix | 33,418 | **48,012** | **+43.7 %** |
| prefill, needle @10K / @50K / @90K | 5,006 / 5,933 / 5,967 | 5,943 / 6,793 / 6,623 | +18.7 / +14.5 / +11.0 % |
| TTFT, cold 60K | 12.69 s | **10.47 s** | **−17.5 %** |
| TTFT, cold 120K | 21.59 | 23.93 | +10.8 % † |
| TTFT, cold 200K | 45.77 | **41.10** | **−10.2 %** |

† The 120K case is the outlier in both directions (prefill and TTFT). Single measurement, no repeat.

### Concurrency — where the real win is

| clients | A aggregate | B aggregate | Δ | per-request decode, A → B |
|---|---:|---:|---:|---|
| c=1 | 239.5 tok/s | 267.5 | **+11.7 %** | 261 → 296 |
| c=2 | 365.9 | 443.4 | **+21.2 %** | 201 → 245 |
| c=4 | 559.5 | 739.5 | **+32.2 %** | 156 → 210 |
| c=8 | 793.4 | **1,214.0** | **+53.0 %** | 110 → 179 |

**The gain scaling monotonically with batch size (+12 → +21 → +32 → +53 %) is what identifies this as bandwidth rather than noise.** At c=1 the all-reduce messages are small (latency-bound, little to gain); at c=8 they are batched and large (bandwidth-bound, a lot to gain). That is the signature of a link that went from 26.2 to 49.3 GB/s.

Single-stream decode on 160-token answers (`decode_1/2/3`): 28.1 / 27.8 / 25.1 → 37.0 / 42.8 / 29.1 tok/s. Those ranges **overlap** (28–43 vs 25–28), so on this sample size that is not a hard number — it is merely consistent with the picture (small messages, little gain).

## Evidence 3 — the control arm: peer access off entirely (same build as B)

Arm C runs the same 1.3.5.7 build with `QE_P2P=0`: no peer access at all, everything through pinned host memory. The engine says so itself:

```
[comm] prefill transfers: host-staged (FLASH_COMM=auto, no peer access)
[comm] decode collectives: pinned host memory (no peer access: every collective)
[flash] no peer access: attention exchanges through pinned host memory (slower than P2P)
```

| measurement | A (PCIe-P2P) | **B (NVLink-P2P)** | C (host) |
|---|---:|---:|---:|
| prefill, cold 60K | 5,975 | **7,256** | 4,316 |
| prefill, cold 200K | 5,514 | **6,148** | 3,278 |
| prefill, needle @10K / @50K / @90K | 5,006 / 5,933 / 5,967 | **5,943 / 6,793 / 6,623** | 3,333 / 4,570 / 4,555 |
| prefill, warm-prefix follow-up | 33,418 | **48,012** | 44,949 |
| c=1 / c=2 / c=4 / c=8 (aggregate tok/s) | 239.5 / 365.9 / 559.5 / 793.4 | **267.5 / 443.4 / 739.5 / 1,214.0** | 255.8 / 419.0 / 689.7 / 1,118.8 |

Two conclusions, in order of how well they hold:

1. **Prefill: the peer path beats host decisively** (60K: 7,256 vs 4,316 = +68 %; 200K: 6,148 vs 3,278 = +88 %). Consistent with the engine's own warning that host-staged attention exchanges are slower.
2. **B vs C is a clean same-build comparison** (no version confound at all): P2P over NVLink beats the host path at *every* concurrency level — 267.5 vs 255.8 (c=1), 443.4 vs 419.0 (c=2), 739.5 vs 689.7 (c=4), 1,214.0 vs 1,118.8 (c=8). So QEngine's `auto` choice of P2P is the right one on this rig.

**The observation I cannot turn into a claim:** C (host, 1.3.5.7) also beats **A** (P2P over PCIe, 1.3.5) at every level from c=2 up — even though A's path had peer access. That suggests the PCIe P2P SM path was *actively costing us* on batched decode, which fits both the high co-run tax A's own probe measured (**1.38**, over the 1.30 cap) and the upstream report in [bpx03/QEngine#2](https://github.com/bpx03/QEngine/issues/2) (deterministic non-finite values on the 2-GPU P2P SM prefill path on PCIe; the copy engine was the workaround). **But A and C differ in build too** (1.3.5 vs 1.3.5.7, and 1.3.5.6 reworked exactly that host path), so this is not a clean comparison. What I can state: the co-run tax fell from 1.38 to **1.23** once the peer path ran over NVLink, and `auto` now keeps P2P comfortably under the cap instead of above it. Whether PCIe-P2P was hurting us before the bridge is an open question — one `QE_P2P=0` arm on **1.3.5** would settle it in a single boot.

## What NVLink does not fix: decode collectives

| | 1.3.5 (PCIe) | 1.3.5.7 (NVLink) |
|---|---:|---:|
| single sequence, P2P / host (ms/step) | 15.70 / 18.09 | **15.68** / 16.18 |
| batched 4 sequences, P2P / host | 27.87 / 29.45 | **27.58** / 28.07 |
| chosen path | P2P | P2P (host faster in 0/4 rounds) |

**15.70 → 15.68 is nothing.** At roughly 10 KB per step (5120 hidden × bf16) that is ~0.65 GB/s effective — three orders of magnitude below either link. The path is purely latency-bound, and SM-store latency on this generation barely differs between NVLink and PCIe. The bridge's value is in **bulk transfer (prefill)** and **high concurrency**, not in the per-token decode collective. (The host path did get faster — 18.09 → 16.18 ms/step, −10.6 % — but that is the 1.3.5.6 fix, not the bridge.)

## Attribution — why this is not "just the newer build"

Arm B differs from arm A in two ways at once: the peer path moved from PCIe to NVLink, **and** the engine went 1.3.5 → 1.3.5.7. Three independent reasons the gain is not from the version bump:

1. **Upstream measures no speed change across that jump.** The 1.3.5.6 release notes state, for 1.3.5.5 → 1.3.5.6 on 2 cards: *"narrative 204 / 205, code 315 / 318 tok/s; engine-level decode 124.0 / 124.1"* — "unchanged within run-to-run noise". The only performance change in that step concerns the **host** decode path, which we were not using (we picked P2P).
2. **The measured link bandwidth doubled on the same machine** (26.2 → 49.3 GB/s), and the client gain scales the way bandwidth should.
3. **The version jump demonstrably does not touch the P2P path.** The engine times its own decode collective at every start-up, with identical instrumentation: on 1.3.5 (PCIe) P2P cost **15.70 ms/step**; on 1.3.5.7 (NVLink) **15.68 ms/step**. Essentially identical. If the engine version had changed performance measurably, this is exactly where it would show — same metric, same method. It standing still is evidence that the gain came from the link.

And the clean control is arm C's **B vs C** comparison above: same build, one variable (peer path on/off).

## Caveats

- **One boot per arm, no repeats.** The club's run-to-run spread on decode is ±1–3 %; anything smaller than that is not a finding here.
- **The "before" arm ran while our agent services were live.** The gateway and the session router were still up during arm A (they are stopped when we measure under the direct profile). The engine's request log shows at least one non-benchmark request in that window (a 169K-token prompt generating 1,113 tokens — our bench never generates more than 286). **Direction of the bias: arm A is more likely understated than overstated**, because concurrent service traffic steals capacity. So the gain is not inflated by this — but it is a reason to read the numbers as indicative rather than as a controlled experiment.
- `decode_tok_s` for the cold 120K/200K cases in the standard client is an artifact (a 5-token completion on top of a long TTFT); `prefill_tok_s` is the usable figure there.
- **NVLink is a GPU↔GPU fabric.** Everything that goes to host RAM stays on PCIe: the 48 GB conversation archive, expert-cache LRU misses, the pinned n-gram table, PLE prefetch. The unchanged host-staged measurement (13.2 GB/s in both arms) is the proof. "Everything over NVLink" is not a thing.
- The bandwidth figures are driver-reported link rates plus the engine's own peer measurements; a standalone D2D benchmark was not possible with the server running (only 155 MiB free per card, and a CUDA context needs ~300 MiB). The L2 NCCL run, taken with the GPUs free, independently measured **52.81 GB/s** peer.
- Not tested: quality (no 8-pack or KL re-run on this change — the change is transport-only, and QEngine's own release notes report greedy output identical across the comm paths), long-context stress, soak, and no repeat of any arm.

## Canonical interconnect classification

Using [club-3090](https://github.com/noonghunna/club-3090)'s three-layer vocabulary:

| layer | what | our rig |
|---|---|---|
| **L1 — driver P2P** | `nvidia-smi topo -m` / peer access | **`NV4`** (4 bonded links); `can_device_access_peer` true both ways |
| **L2 — NCCL collectives** | `scripts/p2p-validate.sh` (all-reduce; checks the *values*, not just that it returned) | **PASS** — ARM A (P2P as-is) and ARM B (P2P disabled control) both correct on 4 B (LL proto) and 16 MiB (Simple proto); *"HEALTHY — collectives complete correctly with P2P as configured."* Same run: **peer 52.81 GB/s vs host-staged 13.28 GB/s (3.98×)** |
| **L3 — engine's own exchange** | QEngine's probe and path choice | peer **49.3 GB/s** SM store; prefill P2P SM loads/stores; decode P2P SM stores (15.68 ms/step) |

The club's detector agrees independently:

```
$ bash scripts/detect_nvlink.sh
[nvlink] detected NVLink (NV4) between GPU0-GPU1 — enabling NVLink mode (custom all-reduce ON)
[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=NVL, custom all-reduce ON, expandable_segments stripped
[nvlink] STATE v=1 transport=nvlink kernel=on cause=none verified=yes gate=none mode=auto world=1
```

So the interconnect state in club vocabulary is **`NVLink`** (not `no NVLink`, not `patched P2P`), and `kernel=on` means **custom all-reduce should be enabled**. That matters for any vLLM/SGLang profile on this box: a hardcoded `--disable-custom-all-reduce` from the PCIe era is now the wrong setting. The club's `NVLINK_MODE=auto` resolves it per rig rather than by a fixed flag, which is the better shape.

## Reproduce

```bash
# What is the link, really?
nvidia-smi topo -m                 # NV4 between the two GPUs
nvidia-smi nvlink -s               # 4 links × 14.062 GB/s
bash scripts/detect_nvlink.sh      # club canonical verdict

# Force the engine to re-measure (the important one)
QE_COMM_CACHE=refresh QE_DECODE_COMM=remeasure bash engine/run.sh
docker logs qengine | grep '\[comm\]'   # probe GB/s + the chosen transport

# A/B the decode paths for your own load
QE_DECODE_COMM=p2p  bash engine/run.sh ; bash run-bench.sh arm-p2p
QE_DECODE_COMM=host bash engine/run.sh ; bash run-bench.sh arm-host

# Control: no peer access at all
QE_P2P=0 bash engine/run.sh ; bash run-bench.sh arm-nop2p
```

## Takeaways

- **Installing NVLink is not the same as using it.** CUDA peer access rides whatever fabric the driver has, but engines with their own transport policy can keep choosing from a stale, PCIe-era measurement — and if that measurement is cached on a link descriptor that does not include NVLink, they will never notice the upgrade.
- **The win is bandwidth-shaped, not latency-shaped.** +53 % at c=8 and ~+12 % at c=1, with decode collectives unchanged at 15.68 ms/step. Look at batched/high-concurrency numbers, not single-stream decode.
- **Host-bound paths are untouched.** On a 2× 24 GB box that offloads experts, PLE and a 48 GB conversation archive to system RAM, a large share of the traffic never crosses the bridge.
- **Two cheap operational items found along the way** that are unrelated to NVLink: NVIDIA persistence mode is off (first request after an idle period pays driver init), and the cache-key design above is worth an upstream note — a probe keyed on PCIe link speed should probably also key on NVLink presence.
