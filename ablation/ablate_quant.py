"""Directional ablation inside already-quantised weights.

Every residual writer W (viewed as Wt: [rows, hidden]) gets T = Wt - (Wt r) r^T, the exact Arditi
projection. The tensor's existing quantisation grid (scales, zero-points, codebook) is kept, and each
element stays on one of the two grid points that bracket its target. Rounding to nearest erases most of
the edit (it is ~2% of the weight, well under one int4 step), so per row we start from round-to-nearest
and flip the cheapest near-tie elements toward the other bracket until the row's component along r is
cancelled: largest |r_i| * step per unit of added squared error first.
"""
import torch

INT4_CODEBOOK = torch.arange(-8, 8, dtype=torch.float32)  # GPTQ sym: value = (code - 8) * scale
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NVFP4_CODEBOOK = torch.unique(torch.cat([-E2M1, E2M1]))


def fp8_codebook():
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    vals = bits.view(torch.float8_e4m3fn).float()
    return torch.unique(vals[torch.isfinite(vals)])


FP8_CODEBOOK = fp8_codebook()


ALPHA = 1.0  # ablation strength: 1.0 = full Arditi projection (OrcaRouter), < 1 keeps part of the r component


def project_rows(wt, r):
    """Remove ALPHA of the r component from every row of wt [rows, hidden]."""
    return wt - ALPHA * torch.outer(wt @ r, r)


def constrained_round(x, scale, r, codebook, max_rows=8192):
    """x: target in grid units [rows, hidden]; scale: same shape (or broadcastable); r: [hidden].

    Returns grid values (in grid units) whose dequantised rows have ~zero component along r.
    """
    cb = codebook.to(x.device)
    out = torch.empty_like(x)
    for s0 in range(0, x.shape[0], max_rows):
        xs = x[s0:s0 + max_rows]
        ss = scale[s0:s0 + max_rows] if scale.dim() == 2 and scale.shape[0] == x.shape[0] else scale
        xs_c = xs.clamp(cb[0], cb[-1])
        hi_idx = torch.searchsorted(cb, xs_c).clamp(1, cb.numel() - 1)
        lo, hi = cb[hi_idx - 1], cb[hi_idx]
        near_hi = (hi - xs_c) < (xs_c - lo)
        near = torch.where(near_hi, hi, lo)
        alt = torch.where(near_hi, lo, hi)
        # Row error along r after round-to-nearest, against the target's own r-component (zero at full strength).
        e = ((near - xs) * ss) @ r
        delta_val = (alt - near) * ss                         # value change if flipped
        gain = -delta_val * r.unsqueeze(0) * torch.sign(e).unsqueeze(1)   # >0 means the flip reduces |e|
        cost = ((alt - xs) ** 2 - (near - xs) ** 2) * ss * ss
        score = torch.where(gain > 0, gain / (cost.clamp_min(0) + 1e-12), torch.zeros_like(gain))
        order = torch.argsort(score, dim=1, descending=True)
        g_sorted = torch.gather(torch.where(gain > 0, gain, torch.zeros_like(gain)), 1, order)
        cum = torch.cumsum(g_sorted, dim=1)
        target = e.abs().unsqueeze(1)
        # Choose k (flips) minimising |e| - cum_k, including k = 0.
        err = torch.cat([target, (target - cum).abs()], dim=1)
        k = torch.argmin(err, dim=1)
        flip_sorted = torch.arange(xs.shape[1], device=xs.device).unsqueeze(0) < k.unsqueeze(1)
        flip = torch.zeros_like(flip_sorted).scatter(1, order, flip_sorted)
        out[s0:s0 + max_rows] = torch.where(flip, alt, near)
    return out


def stats(orig_deq, new_deq, target, r):
    base = (orig_deq @ r).norm()
    return {
        "resid": ((new_deq @ r).norm() / base).item(),
        "target_resid": ((target @ r).norm() / base).item(),
        "rel_change": ((new_deq - orig_deq).norm() / orig_deq.norm()).item(),
        "err_vs_target": ((new_deq - target).norm() / target.norm()).item(),
    }


# ---- GPTQ int4 (sym, zero = 8) ----------------------------------------------------------------------------

def gptq_unpack(qweight):
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return q.reshape(-1, qweight.shape[1])


def gptq_pack(q):
    q = q.to(torch.int64).reshape(-1, 8, q.shape[1])
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64).view(1, 8, 1)
    packed = (q << shifts).sum(1) & 0xFFFFFFFF
    packed = torch.where(packed >= 2**31, packed - 2**32, packed)
    return packed.to(torch.int32)


def ablate_gptq(qweight, scales, qzeros, r):
    """Returns new qweight (same scales/qzeros) plus stats. Layout: [in/8, hidden]."""
    shifts = torch.arange(0, 32, 4, device=qzeros.device, dtype=torch.int32)
    z = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF).reshape(qzeros.shape[0], -1)
    assert torch.all(z == 7), "expected symmetric GPTQ v1 zero-points (stored 7 = zero 8)"
    q = gptq_unpack(qweight)
    g = q.shape[0] // scales.shape[0]
    s = scales.float().repeat_interleave(g, 0)
    wt = (q.float() - 8) * s
    t = project_rows(wt, r)
    grid = constrained_round(t / s, s, r, INT4_CODEBOOK)
    new_q = (grid + 8).round().clamp(0, 15).to(torch.int32)
    new_deq = (new_q.float() - 8) * s
    assert torch.equal(gptq_pack(q), qweight), "int4 pack/unpack round trip failed"
    return gptq_pack(new_q), stats(wt, new_deq, t, r)


# ---- FP8 E4M3 with 128x128 block scale_inv ([out=hidden, in]) ---------------------------------------------

def ablate_fp8_block(weight, scale_inv, r, block=128):
    w8 = weight.float()
    s = scale_inv.float().repeat_interleave(block, 0)[: w8.shape[0]].repeat_interleave(block, 1)[:, : w8.shape[1]]
    wt, st = (w8 * s).T, s.T                   # rows = input dim, columns = hidden
    t = project_rows(wt, r)
    grid = constrained_round(t / st, st, r, FP8_CODEBOOK)
    new_w = grid.T.contiguous().to(torch.float8_e4m3fn)
    assert torch.equal(new_w.float(), grid.T), "fp8 grid value not exactly representable"
    return new_w, stats(wt, (new_w.float() * s).T, t, r)


# ---- BF16 ---------------------------------------------------------------------------------------------------

def ablate_bf16_out_hidden(weight, r):
    """weight [hidden, in] (output axis is the residual)."""
    wt = weight.float().T
    t = project_rows(wt, r)
    new = t.T.contiguous().to(torch.bfloat16)
    return new, stats(wt, new.float().T, t, r)


def ablate_embedding(weight, r, chunk=32768):
    """embed_tokens [vocab, hidden]: rows are written straight into the residual."""
    out = torch.empty_like(weight)
    num = den = 0.0
    for i in range(0, weight.shape[0], chunk):
        e = weight[i:i + chunk].float()
        t = project_rows(e, r)
        out[i:i + chunk] = t.to(torch.bfloat16)
        num += ((out[i:i + chunk].float() @ r) ** 2).sum().item()
        den += ((e @ r) ** 2).sum().item()
    return out, {"resid": (num / den) ** 0.5}


# ---- FP8 E4M3 with one per-tensor scale ([out=hidden, in]) -----------------------------------------------

def ablate_fp8_tensor(weight, scale, r):
    s = scale.float()
    wt = (weight.float() * s).T
    t = project_rows(wt, r)
    grid = constrained_round(t / s, s, r, FP8_CODEBOOK)
    new_w = grid.T.contiguous().to(torch.float8_e4m3fn)
    assert torch.equal(new_w.float(), grid.T), "fp8 grid value not exactly representable"
    return new_w, stats(wt, (new_w.float() * s).T, t, r)


# ---- NVFP4 (modelopt): uint8 [out, in/2] two E2M1 codes, fp8 scale per 16 inputs, fp32 global scale -----

E2M1_MAG = E2M1


def nvfp4_unpack(packed, low_first=True):
    lo, hi = packed & 0xF, packed >> 4
    pair = (lo, hi) if low_first else (hi, lo)
    return torch.stack(pair, -1).reshape(packed.shape[0], -1)


def nvfp4_codes_to_values(codes):
    mag = E2M1_MAG.to(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def nvfp4_values_to_codes(vals):
    mag_idx = torch.searchsorted(E2M1_MAG.to(vals.device), vals.abs().contiguous())
    assert torch.equal(E2M1_MAG.to(vals.device)[mag_idx], vals.abs()), "value not on the E2M1 grid"
    return (mag_idx | ((vals < 0).to(torch.long) << 3)).to(torch.uint8)


def nvfp4_pack(codes, low_first=True):
    c = codes.reshape(codes.shape[0], -1, 2)
    a, b = (c[..., 0], c[..., 1]) if low_first else (c[..., 1], c[..., 0])
    return (a | (b << 4)).to(torch.uint8)


def nvfp4_scales(weight_scale, weight_scale_2, group=16):
    return weight_scale.float().repeat_interleave(group, 1) * weight_scale_2.float()


def ablate_nvfp4(packed, weight_scale, weight_scale_2, r, low_first=True, chunk_cols=None):
    codes = nvfp4_unpack(packed, low_first)
    assert torch.equal(nvfp4_pack(codes, low_first), packed), "nvfp4 pack/unpack round trip failed"
    s = nvfp4_scales(weight_scale, weight_scale_2)          # [out=hidden, in]
    w = nvfp4_codes_to_values(codes) * s
    wt, st = w.T.contiguous(), s.T.contiguous()             # rows = input dim, columns = hidden
    del w
    t = project_rows(wt, r)
    grid = constrained_round(t / st, st, r, NVFP4_CODEBOOK, max_rows=2048)
    new_codes = nvfp4_values_to_codes(grid.T.contiguous())
    new_packed = nvfp4_pack(new_codes, low_first)
    new_deq = (nvfp4_codes_to_values(new_codes) * s).T
    return new_packed, stats(wt, new_deq, t, r)
