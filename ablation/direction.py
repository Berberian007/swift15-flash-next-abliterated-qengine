#!/usr/bin/env python3
"""Recover the abliteration edit from (original, abliterated) BF16 tensor pairs.

usage: direction.py ORIG_DIR ABL_DIR TARGETS HIDDEN OUT.pt

For every residual writer W the abliterated tensor should be W - a * r r^T W,
so D = W - W_abl is rank one along the hidden axis. Reports, per tensor:
  rank1   share of ||D||^2 carried by the top singular direction (1.0 = clean rank one)
  alpha   scale of the edit along r (1.0 = exact orthogonal projection)
  resid   ||r^T W_abl|| / ||r^T W||, how much of the refusal component the edit left
and saves every per-tensor direction plus their mean for comparison.
"""
import glob, json, sys, torch
from safetensors import safe_open

orig_dir, abl_dir, targets_file, hidden, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
targets = [t.strip() for t in open(targets_file) if t.strip()]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def index(d):
    m = {}
    for f in glob.glob(f"{d}/*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                m[k] = f
    return m


oi, ai = index(orig_dir), index(abl_dir)


def load(m, name):
    with safe_open(m[name], "pt") as h:
        return h.get_tensor(name)


def hidden_first(t):
    """Move the hidden-size axis to the front and flatten the rest: (hidden, N)."""
    axes = [i for i, s in enumerate(t.shape) if s == hidden]
    # Linear weights are (out, in); for residual writers the output axis is hidden. Prefer the
    # earliest hidden axis except for 3-D fused experts, whose layout is checked from the diff.
    return axes


results, dirs = {}, {}
for name in targets:
    w = load(oi, name)
    tdev = dev if w.numel() < 2e8 else "cpu"  # fused MTP experts / embeddings exceed the 4090 in fp32
    w = w.to(tdev, torch.float32)
    wa = load(ai, name).to(tdev, torch.float32)
    d = w - wa
    best = None
    for ax in hidden_first(w):
        m = d.movedim(ax, 0).reshape(hidden, -1)
        g = m @ m.T
        evals, evecs = torch.linalg.eigh(g)
        share = (evals[-1] / evals.sum()).item()
        if best is None or share > best[0]:
            best = (share, ax, evecs[:, -1])
    share, ax, r = best
    wm = w.movedim(ax, 0).reshape(hidden, -1)
    wam = wa.movedim(ax, 0).reshape(hidden, -1)
    dm = d.movedim(ax, 0).reshape(hidden, -1)
    proj = r @ wm
    alpha = ((r @ dm) @ proj / (proj @ proj)).item()
    resid = ((r @ wam).norm() / proj.norm()).item()
    rel = (dm.norm() / wm.norm()).item()
    results[name] = dict(axis=ax, shape=list(w.shape), rank1=share, alpha=alpha, resid=resid, rel_change=rel)
    dirs[name] = r.cpu()
    print(f"{name:70s} ax={ax} rank1={share:.6f} alpha={alpha:+.4f} resid={resid:.4f} |D|/|W|={rel:.2e}", flush=True)
    del w, wa, d, wm, wam, dm
    torch.cuda.empty_cache()

names = list(dirs)
R = torch.stack([dirs[n] for n in names])
# Directions are sign-ambiguous; align to the first before comparing.
R = R * torch.sign(R @ R[0]).unsqueeze(1)
cos = R @ R.T
print("\nmin |cos| between any two tensors' directions:", cos.abs().min().item())
print("mean |cos| to the first tensor:", cos[0].abs().mean().item())
torch.save({"names": names, "dirs": R, "results": results}, out)
json.dump(results, open(out + ".json", "w"), indent=1)
