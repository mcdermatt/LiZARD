"""
hash_sck_model.py
=================
Self-contained Instant-NGP Multi-Resolution 3D Hash Grid + Self-Conditioned Knot Network
(HashSCKNetwork) for fast Continuous LiDAR Hazard Fields & Return CDFs.

This module provides:
  1. Multi-Resolution 3D Hash Grid Encoding:
     - HashGrid3D: Fused Triton CUDA forward & backward kernel (computing exact
       gradients w.r.t. BOTH hash table weights and 3D input coordinates x_k),
       with automatic fallback to tinycudann (if installed) or pure PyTorch.
  2. Hash-Grid SCK Network:
     - HashSCKNetwork: Compact self-conditioned knot network that replaces the
       heavy 658k-param Fourier MLP trunk with O(1) 3D hash-grid lookups + a
       lightweight H=64 decoder. Features:
         * Origin-invariant Plücker ray context (d, m = o x d), projected once/ray
         * Gap-bounded residual knot flow (pinned endpoints, no sorting needed)
         * Shifted interval-midpoint CDF evaluation (so final round shifts are
           actually queried in 3D and receive direct spatial gradients)
         * Optional 3D-local CDF head (local_cdf=True) for cross-ray consistency
         * Interval-width-aware softmax CDF (with log(delta_s) bias + escape bin)
           or Beer-Lambert hazard quadrature
         * 3D-grounded ray-drop head (pools refined 3D knot features + ray context)
  3. Fast Sync-Free Loss & Training Loop:
     - compute_hash_sck_loss: Exact grid-free Root-CRPS on hitting rays + optional
       sky/non-return empty-space carving loss + ray-drop BCE, with zero per-batch
       CPU-GPU .item() synchronization stalls.
     - train_hash_sck_model: High-throughput training loop with split hash/MLP
       learning rates, fused Adam, optional CosineAnnealingLR, and timing stats.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False

from sck_model import (
    PositionalEncoding,
    exact_continuous_cdf_loss,
    vectorized_gpu_origin_slide,
    GPURayBatcher,
    generate_point_cloud_from_network,
)


# ==============================================================================
# 1. Fused Triton Multi-Resolution 3D Hash Grid Kernels
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _hash_grid_3d_fwd_kernel(
        x_ptr,          # (M, 3) float32 normalized in [0, 1)
        table_ptr,      # (L, T, 2) float32
        res_ptr,        # (L,) float32
        level_w_ptr,    # (L,) float32 per-level CTF weight in [0, 1]
        out_ptr,        # (M, L * 2) float32
        M,
        L: tl.constexpr,
        T_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_l = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        lw = tl.load(level_w_ptr + pid_l)
        out_base = offs_m.to(tl.int64) * (L * 2) + pid_l * 2
        if lw == 0.0:
            zeros = tl.zeros([BLOCK_M], dtype=tl.float32)
            tl.store(out_ptr + out_base + 0, zeros, mask=mask_m)
            tl.store(out_ptr + out_base + 1, zeros, mask=mask_m)
            return

        # Load normalized 3D coordinates
        x = tl.load(x_ptr + offs_m * 3 + 0, mask=mask_m, other=0.0)
        y = tl.load(x_ptr + offs_m * 3 + 1, mask=mask_m, other=0.0)
        z = tl.load(x_ptr + offs_m * 3 + 2, mask=mask_m, other=0.0)

        res = tl.load(res_ptr + pid_l)
        sx = x * res
        sy = y * res
        sz = z * res

        fx = tl.floor(sx)
        fy = tl.floor(sy)
        fz = tl.floor(sz)

        tx = sx - fx
        ty = sy - fy
        tz = sz - fz
        ox = 1.0 - tx
        oy = 1.0 - ty
        oz = 1.0 - tz

        ix0 = fx.to(tl.uint32)
        iy0 = fy.to(tl.uint32)
        iz0 = fz.to(tl.uint32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        # Spatial hash primes (Instant-NGP / Teschner et al.)
        P2: tl.constexpr = 2654435761
        P3: tl.constexpr = 805459861

        hy0 = iy0 * P2
        hy1 = iy1 * P2
        hz0 = iz0 * P3
        hz1 = iz1 * P3

        level_base = pid_l * (T_MASK + 1) * 2

        acc0 = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Unrolled trilinear lookup across all 8 voxel corners
        for cx in tl.static_range(2):
            ix = ix0 if cx == 0 else ix1
            wx = ox if cx == 0 else tx
            for cy in tl.static_range(2):
                hy = hy0 if cy == 0 else hy1
                wxy = wx * (oy if cy == 0 else ty)
                hxy = ix ^ hy
                for cz in tl.static_range(2):
                    hz = hz0 if cz == 0 else hz1
                    w = wxy * (oz if cz == 0 else tz)
                    h = (hxy ^ hz) & T_MASK
                    addr = level_base + h.to(tl.int64) * 2
                    f0 = tl.load(table_ptr + addr + 0, mask=mask_m, other=0.0)
                    f1 = tl.load(table_ptr + addr + 1, mask=mask_m, other=0.0)
                    acc0 += w * f0
                    acc1 += w * f1

        tl.store(out_ptr + out_base + 0, acc0 * lw, mask=mask_m)
        tl.store(out_ptr + out_base + 1, acc1 * lw, mask=mask_m)

    @triton.jit
    def _hash_grid_3d_bwd_kernel(
        x_ptr,              # (M, 3) float32 normalized in [0, 1)
        table_ptr,          # (L, T, 2) float32
        res_ptr,            # (L,) float32
        level_w_ptr,        # (L,) float32 per-level CTF weight in [0, 1]
        grad_out_ptr,       # (M, L * 2) float32
        grad_table_ptr,     # (L, T, 2) float32 (accumulated into in-place)
        grad_x_lvl_ptr,     # (M, L, 3) float32 (used only if COMPUTE_DX)
        M,
        L: tl.constexpr,
        T_MASK: tl.constexpr,
        COMPUTE_DX: tl.constexpr,
        COMPUTE_DT: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_l = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        lw = tl.load(level_w_ptr + pid_l)
        if lw == 0.0:
            if COMPUTE_DX:
                dx_base = (offs_m.to(tl.int64) * L + pid_l) * 3
                zeros = tl.zeros([BLOCK_M], dtype=tl.float32)
                tl.store(grad_x_lvl_ptr + dx_base + 0, zeros, mask=mask_m)
                tl.store(grad_x_lvl_ptr + dx_base + 1, zeros, mask=mask_m)
                tl.store(grad_x_lvl_ptr + dx_base + 2, zeros, mask=mask_m)
            return

        g_base = offs_m.to(tl.int64) * (L * 2) + pid_l * 2
        g0 = tl.load(grad_out_ptr + g_base + 0, mask=mask_m, other=0.0) * lw
        g1 = tl.load(grad_out_ptr + g_base + 1, mask=mask_m, other=0.0) * lw
        active_m = mask_m & ((g0 != 0.0) | (g1 != 0.0))

        x = tl.load(x_ptr + offs_m * 3 + 0, mask=mask_m, other=0.0)
        y = tl.load(x_ptr + offs_m * 3 + 1, mask=mask_m, other=0.0)
        z = tl.load(x_ptr + offs_m * 3 + 2, mask=mask_m, other=0.0)

        res = tl.load(res_ptr + pid_l)
        sx = x * res
        sy = y * res
        sz = z * res

        fx = tl.floor(sx)
        fy = tl.floor(sy)
        fz = tl.floor(sz)

        tx = sx - fx
        ty = sy - fy
        tz = sz - fz
        ox = 1.0 - tx
        oy = 1.0 - ty
        oz = 1.0 - tz

        ix0 = fx.to(tl.uint32)
        iy0 = fy.to(tl.uint32)
        iz0 = fz.to(tl.uint32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        P2: tl.constexpr = 2654435761
        P3: tl.constexpr = 805459861

        hy0 = iy0 * P2
        hy1 = iy1 * P2
        hz0 = iz0 * P3
        hz1 = iz1 * P3

        level_base = pid_l * (T_MASK + 1) * 2

        gx = tl.zeros([BLOCK_M], dtype=tl.float32)
        gy = tl.zeros([BLOCK_M], dtype=tl.float32)
        gz = tl.zeros([BLOCK_M], dtype=tl.float32)

        for cx in tl.static_range(2):
            ix = ix0 if cx == 0 else ix1
            wx = ox if cx == 0 else tx
            dwx = -1.0 if cx == 0 else 1.0
            for cy in tl.static_range(2):
                hy = hy0 if cy == 0 else hy1
                wy = oy if cy == 0 else ty
                dwy = -1.0 if cy == 0 else 1.0
                hxy = ix ^ hy
                for cz in tl.static_range(2):
                    hz = hz0 if cz == 0 else hz1
                    wz = oz if cz == 0 else tz
                    dwz = -1.0 if cz == 0 else 1.0

                    w = wx * wy * wz
                    h = (hxy ^ hz) & T_MASK
                    addr = level_base + h.to(tl.int64) * 2

                    if COMPUTE_DT:
                        tl.atomic_add(grad_table_ptr + addr + 0, w * g0, mask=active_m)
                        tl.atomic_add(grad_table_ptr + addr + 1, w * g1, mask=active_m)

                    if COMPUTE_DX:
                        f0 = tl.load(table_ptr + addr + 0, mask=active_m, other=0.0)
                        f1 = tl.load(table_ptr + addr + 1, mask=active_m, other=0.0)
                        dot = f0 * g0 + f1 * g1
                        gx += (dwx * wy * wz) * dot
                        gy += (wx * dwy * wz) * dot
                        gz += (wx * wy * dwz) * dot

        if COMPUTE_DX:
            dx_base = (offs_m.to(tl.int64) * L + pid_l) * 3
            tl.store(grad_x_lvl_ptr + dx_base + 0, gx * res, mask=mask_m)
            tl.store(grad_x_lvl_ptr + dx_base + 1, gy * res, mask=mask_m)
            tl.store(grad_x_lvl_ptr + dx_base + 2, gz * res, mask=mask_m)


class _TritonHashGrid3DFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_norm: torch.Tensor,
        tables: torch.Tensor,
        resolutions: torch.Tensor,
        level_weights: torch.Tensor,
        direct_grad_accum: bool,
    ):
        # x_norm: (M, 3) contiguous float32 in [0, 1)
        # tables: (L, T, 2) contiguous float32
        x_norm = x_norm.contiguous()
        tables = tables.contiguous()
        M = x_norm.shape[0]
        L, T, F_dim = tables.shape
        assert F_dim == 2, "Triton HashGrid3D kernel currently supports n_features=2."
        assert (T & (T - 1)) == 0, "Hash table size T must be a power of two."

        out = torch.empty((M, L * 2), device=x_norm.device, dtype=torch.float32)
        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M), L)
        _hash_grid_3d_fwd_kernel[grid](
            x_norm, tables, resolutions, level_weights, out,
            M, L=L, T_MASK=T - 1, BLOCK_M=BLOCK_M,
        )
        ctx.save_for_backward(x_norm, tables, resolutions, level_weights)
        ctx.direct_grad_accum = bool(direct_grad_accum)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_norm, tables, resolutions, level_weights = ctx.saved_tensors
        grad_out = grad_out.contiguous().float()
        M = x_norm.shape[0]
        L, T, _ = tables.shape
        compute_dx = ctx.needs_input_grad[0]
        compute_dt = ctx.needs_input_grad[1]

        if compute_dt:
            if ctx.direct_grad_accum:
                if tables.grad is None:
                    tables.grad = torch.zeros_like(tables)
                grad_tables_buf = tables.grad
                ret_grad_tables = None
            else:
                grad_tables_buf = torch.zeros_like(tables)
                ret_grad_tables = grad_tables_buf
        else:
            grad_tables_buf = torch.empty((1,), device=x_norm.device, dtype=torch.float32)
            ret_grad_tables = None

        if compute_dx:
            grad_x_lvl = torch.empty((M, L, 3), device=x_norm.device, dtype=torch.float32)
        else:
            grad_x_lvl = torch.empty((1,), device=x_norm.device, dtype=torch.float32)

        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M), L)
        _hash_grid_3d_bwd_kernel[grid](
            x_norm, tables, resolutions, level_weights, grad_out, grad_tables_buf, grad_x_lvl,
            M, L=L, T_MASK=T - 1, COMPUTE_DX=compute_dx, COMPUTE_DT=compute_dt, BLOCK_M=BLOCK_M,
        )

        grad_x = grad_x_lvl.sum(dim=1) if compute_dx else None
        return grad_x, ret_grad_tables, None, None, None


class HashGrid3D(nn.Module):
    """Multi-Resolution 3D Spatial Hash Grid (Instant-NGP style).

    Automatically selects the fastest available backend:
      - 'triton': Fused Triton CUDA forward+backward kernel (supports exact dL/dx
        so gradients flow into self-conditioned knot shifts).
      - 'tcnn': tiny-cuda-nn HashGrid (if installed and n_features <= 8).
      - 'pytorch': Pure PyTorch fallback (CPU or CUDA).

    Args:
        n_levels: Number of multi-resolution grid levels (default 12).
        n_features: Feature channels per level (default 2; required for Triton kernel).
        log2_hashmap_size: Log2 of entries per level table (default 16 -> 65,536).
        base_resolution: Coarsest grid resolution per axis (default 16).
        max_resolution: Finest grid resolution per axis (default 512).
        bound: Half-extent of cubic scene AABB [-bound, +bound]^3 in meters (default 15.0).
        backend: 'auto' (default), 'triton', 'tcnn', or 'pytorch'.
    """

    def __init__(
        self,
        n_levels: int = 12,
        n_features: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 512,
        bound: float = 15.0,
        backend: str = "auto",
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.table_size = 1 << log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.bound = float(bound)
        self.out_dim = n_levels * n_features

        # Geometric progression of level resolutions
        if n_levels > 1:
            growth = (max_resolution / base_resolution) ** (1.0 / (n_levels - 1))
        else:
            growth = 1.0
        resolutions = [int(round(base_resolution * (growth ** i))) for i in range(n_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.float32))
        # Progressive coarse-to-fine level weights (all 1.0 = full resolution by default)
        self.register_buffer("level_weights", torch.ones((n_levels,), dtype=torch.float32))
        self.register_buffer("level_mask", torch.ones((1, self.out_dim), dtype=torch.float32))

        if backend == "auto":
            if HAS_TCNN:
                self.backend = "tcnn"
            elif HAS_TRITON and n_features == 2:
                self.backend = "triton"
            else:
                self.backend = "pytorch"
        else:
            self.backend = backend

        if self.backend == "tcnn":
            if not HAS_TCNN:
                raise ImportError("tinycudann is not installed; use backend='triton' or 'auto'.")
            self.tcnn_enc = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": growth,
                },
                dtype=torch.float32,
            )
            self.tables = None
        else:
            # Instant-NGP uniform [-1e-4, 1e-4] initialization
            self.tables = nn.Parameter(
                torch.empty(n_levels, self.table_size, n_features).uniform_(-1e-4, 1e-4)
            )
            self.tcnn_enc = None
            # Buffers for pure-PyTorch fallback
            self.register_buffer("primes", torch.tensor([1, 2654435761, 805459861], dtype=torch.int64))
            self.register_buffer(
                "offsets",
                torch.tensor(
                    [
                        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
                    ],
                    dtype=torch.int64,
                ),
            )
            self.register_buffer("level_idx", torch.arange(n_levels, dtype=torch.int64))

    def set_active_levels(self, active_levels: float) -> None:
        """Sets smooth per-level weights in-place for progressive coarse-to-fine training.

        Levels l < floor(active_levels) receive weight 1.0, the transitional level ramps
        smoothly from 0.0 to 1.0, and finer levels are masked to 0.0 (no torch.compile break).
        """
        with torch.no_grad():
            lvls = torch.arange(self.n_levels, device=self.level_mask.device, dtype=torch.float32)
            w = torch.clamp(float(active_levels) - lvls, 0.0, 1.0)
            self.level_weights.copy_(w)
            self.level_mask.copy_(w.repeat_interleave(self.n_features).unsqueeze(0))

    def reset_parameters(self) -> None:
        """Re-initializes hash table entries in-place (preserves torch.compile cache)."""
        with torch.no_grad():
            if self.tables is not None:
                self.tables.uniform_(-1e-4, 1e-4)
            elif self.tcnn_enc is not None:
                for p in self.tcnn_enc.parameters():
                    p.uniform_(-1e-4, 1e-4)
            self.level_weights.fill_(1.0)
            self.level_mask.fill_(1.0)

    def _forward_pytorch(self, x_norm: torch.Tensor) -> torch.Tensor:
        M = x_norm.shape[0]
        scaled = x_norm.unsqueeze(1) * self.resolutions.view(1, -1, 1)  # (M, L, 3)
        floor = torch.floor(scaled).to(torch.int64)
        frac = scaled - floor.to(scaled.dtype)

        corners = floor.unsqueeze(2) + self.offsets.view(1, 1, 8, 3)  # (M, L, 8, 3)
        h = (
            (corners[..., 0] * self.primes[0])
            ^ (corners[..., 1] * self.primes[1])
            ^ (corners[..., 2] * self.primes[2])
        ) & (self.table_size - 1)

        feats = self.tables[self.level_idx.view(1, -1, 1), h]  # (M, L, 8, F)

        wx = torch.stack([1.0 - frac[..., 0], frac[..., 0]], dim=-1)
        wy = torch.stack([1.0 - frac[..., 1], frac[..., 1]], dim=-1)
        wz = torch.stack([1.0 - frac[..., 2], frac[..., 2]], dim=-1)
        w = (
            wx.unsqueeze(-1).unsqueeze(-1)
            * wy.unsqueeze(-2).unsqueeze(-1)
            * wz.unsqueeze(-2).unsqueeze(-2)
        ).reshape(M, self.n_levels, 8, 1)

        return (feats * w).sum(dim=2).reshape(M, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes world-space 3D coordinates x of shape (..., 3) -> (..., out_dim)."""
        orig_shape = x.shape[:-1]
        x_norm = ((x.reshape(-1, 3) + self.bound) / (2.0 * self.bound)).clamp(0.0, 1.0 - 1e-6)

        if self.backend == "tcnn" and x_norm.is_cuda:
            out = self.tcnn_enc(x_norm).float() * self.level_mask
        elif self.backend == "triton" and x_norm.is_cuda and HAS_TRITON and self.n_features == 2:
            # Direct in-place grad accumulation into tables.grad avoids repeated
            # full-table zeros_like() allocations across self-conditioning rounds
            # (falls back to functional grad_tables under torch.compile).
            is_compiling = hasattr(torch, "compiler") and torch.compiler.is_compiling()
            direct_accum = bool(
                self.tables is not None
                and self.tables.is_leaf
                and self.tables.requires_grad
                and not is_compiling
            )
            out = _TritonHashGrid3DFn.apply(
                x_norm.float(), self.tables, self.resolutions, self.level_weights, direct_accum
            )
        else:
            out = self._forward_pytorch(x_norm) * self.level_mask

        return out.reshape(*orig_shape, self.out_dim)


# ==============================================================================
# 1b. Coarse Unsigned Distance Field (UDF) Grid & Multi-Band Knot Placer
# ==============================================================================

if HAS_TRITON:

    @triton.jit
    def _edt_1d_last_dim_kernel(
        in_ptr,       # (L_lines, N) float32 contiguous
        out_ptr,      # (L_lines, N) float32 contiguous
        h_sq,         # float32: voxel spacing squared along this axis
        L_lines,
        N,
        BLOCK_N: tl.constexpr,
    ):
        pid_l = tl.program_id(0)
        pid_q = tl.program_id(1)
        if pid_l >= L_lines:
            return

        base = pid_l.to(tl.int64) * N
        q = pid_q * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_q = q < N
        qf = q.to(tl.float32)

        best = tl.full([BLOCK_N], 1e20, dtype=tl.float32)
        for p in range(0, N):
            fp = tl.load(in_ptr + base + p)
            if fp < 1e18:
                diff = qf - p
                cand = fp + h_sq * diff * diff
                best = tl.minimum(best, cand)

        tl.store(out_ptr + base + q, best, mask=mask_q)

    @triton.jit
    def _udf_trace_bands_kernel(
        ray_o_ptr,        # (B, 3) float32 contiguous
        ray_d_ptr,        # (B, 3) float32 contiguous
        udf_ptr,          # (nx, ny, nz) float32 contiguous
        band_a_ptr,       # (B, M_MAX) float32 pre-filled with s_max
        band_b_ptr,       # (B, M_MAX) float32 pre-filled with s_max
        B,
        ox_min, oy_min, oz_min,
        ox_max, oy_max, oz_max,
        inv_dx, inv_dy, inv_dz,
        nx, ny, nz,
        s_min, s_max,
        band_radius,
        safe_margin,
        fine_step,
        M_MAX: tl.constexpr,
        MAX_STEPS: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B
        b64 = offs_b.to(tl.int64)

        ox = tl.load(ray_o_ptr + b64 * 3 + 0, mask=mask_b, other=0.0)
        oy = tl.load(ray_o_ptr + b64 * 3 + 1, mask=mask_b, other=0.0)
        oz = tl.load(ray_o_ptr + b64 * 3 + 2, mask=mask_b, other=0.0)

        dx = tl.load(ray_d_ptr + b64 * 3 + 0, mask=mask_b, other=1.0)
        dy = tl.load(ray_d_ptr + b64 * 3 + 1, mask=mask_b, other=0.0)
        dz = tl.load(ray_d_ptr + b64 * 3 + 2, mask=mask_b, other=0.0)

        # Ray-AABB slab intersection
        eps = 1e-8
        sdx = tl.where(tl.abs(dx) > eps, dx, tl.where(dx >= 0.0, eps, -eps))
        sdy = tl.where(tl.abs(dy) > eps, dy, tl.where(dy >= 0.0, eps, -eps))
        sdz = tl.where(tl.abs(dz) > eps, dz, tl.where(dz >= 0.0, eps, -eps))

        t0x = (ox_min - ox) / sdx
        t1x = (ox_max - ox) / sdx
        t0y = (oy_min - oy) / sdy
        t1y = (oy_max - oy) / sdy
        t0z = (oz_min - oz) / sdz
        t1z = (oz_max - oz) / sdz

        tmin_x = tl.minimum(t0x, t1x)
        tmax_x = tl.maximum(t0x, t1x)
        tmin_y = tl.minimum(t0y, t1y)
        tmax_y = tl.maximum(t0y, t1y)
        tmin_z = tl.minimum(t0z, t1z)
        tmax_z = tl.maximum(t0z, t1z)

        t_enter = tl.maximum(s_min, tl.maximum(tmin_x, tl.maximum(tmin_y, tmin_z)))
        t_exit = tl.minimum(s_max, tl.minimum(tmax_x, tl.minimum(tmax_y, tmax_z)))

        active_ray = mask_b & (t_enter < t_exit)
        t = t_enter
        in_band = tl.zeros([BLOCK_B], dtype=tl.int1)
        band_start = t_enter
        m_cnt = tl.zeros([BLOCK_B], dtype=tl.int32)

        nx_m1 = nx - 1
        ny_m1 = ny - 1
        nz_m1 = nz - 1
        ny64 = tl.full([1], ny, dtype=tl.int64)
        nz64 = tl.full([1], nz, dtype=tl.int64)

        step_idx = 0
        alive = active_ray & (t < t_exit)
        while (step_idx < MAX_STEPS) and (tl.sum(alive.to(tl.int32), axis=0) > 0):
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz

            ix = tl.minimum(tl.maximum(tl.floor((px - ox_min) * inv_dx + 0.5).to(tl.int32), 0), nx_m1)
            iy = tl.minimum(tl.maximum(tl.floor((py - oy_min) * inv_dy + 0.5).to(tl.int32), 0), ny_m1)
            iz = tl.minimum(tl.maximum(tl.floor((pz - oz_min) * inv_dz + 0.5).to(tl.int32), 0), nz_m1)

            flat = (ix.to(tl.int64) * ny64 + iy.to(tl.int64)) * nz64 + iz.to(tl.int64)
            d = tl.load(udf_ptr + flat, mask=alive, other=1e6)

            hit_now = alive & (d <= band_radius)
            entering = hit_now & (~in_band)
            exiting = alive & (~hit_now) & in_band

            band_start = tl.where(entering & (m_cnt < M_MAX), t, band_start)
            in_band = tl.where(entering, True, in_band)

            slot = tl.minimum(m_cnt, M_MAX - 1)
            is_new = exiting & (m_cnt < M_MAX)
            out_off = b64 * M_MAX + slot.to(tl.int64)
            tl.store(band_a_ptr + out_off, band_start, mask=is_new)
            tl.store(band_b_ptr + out_off, t, mask=exiting)

            m_cnt = tl.where(is_new, m_cnt + 1, m_cnt)
            in_band = tl.where(exiting, False, in_band)

            step = tl.where(hit_now, fine_step, tl.maximum(d - safe_margin, fine_step))
            t = tl.where(alive, t + step, t)
            alive = active_ray & (t < t_exit)
            step_idx += 1

        # Close any band still open when leaving the AABB / hitting s_max
        still_open = active_ray & in_band
        slot = tl.minimum(m_cnt, M_MAX - 1)
        is_new = still_open & (m_cnt < M_MAX)
        t_end = tl.minimum(t, t_exit)
        out_off = b64 * M_MAX + slot.to(tl.int64)
        tl.store(band_a_ptr + out_off, band_start, mask=is_new)
        tl.store(band_b_ptr + out_off, t_end, mask=still_open)

    @triton.jit
    def _bands_to_knots_kernel(
        band_a_ptr,       # (B, M_MAX) float32
        band_b_ptr,       # (B, M_MAX) float32
        rand_ptr,         # (B, K) float32 (only read when USE_JITTER)
        out_ptr,          # (B, K) float32
        B,
        K,
        s_min,
        s_max,
        active_frac,
        M_MAX: tl.constexpr,
        USE_JITTER: tl.constexpr,
        BLOCK_B: tl.constexpr,
        K_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B
        b64 = offs_b.to(tl.int64)

        k = tl.arange(0, K_PAD)
        mask_k = k < K
        mask_bk = mask_b[:, None] & mask_k[None, :]

        # Stratified quantile targets u in (0, 1) for interior knots k in [1, K-2]
        if USE_JITTER:
            r_off = b64[:, None] * K + k[None, :].to(tl.int64)
            r_val = tl.load(rand_ptr + r_off, mask=mask_bk, other=0.5)
            u = ((k[None, :].to(tl.float32) - 1.0) + (0.05 + 0.90 * r_val)) / tl.maximum(K - 2, 1)
        else:
            u = k[None, :].to(tl.float32) / tl.maximum(K - 1, 1)

        # Pass 1: total active weight (sqrt(len)) and total free weight (len)
        sum_act = tl.zeros([BLOCK_B], dtype=tl.float32)
        sum_free = tl.zeros([BLOCK_B], dtype=tl.float32)
        prev_b = tl.full([BLOCK_B], s_min, dtype=tl.float32)

        for m in tl.static_range(M_MAX):
            off_m = b64 * M_MAX + m
            am = tl.load(band_a_ptr + off_m, mask=mask_b, other=s_max)
            bm = tl.load(band_b_ptr + off_m, mask=mask_b, other=s_max)
            gf = tl.maximum(am - prev_b, 0.0)
            ga = tl.maximum(bm - am, 0.0)
            sum_free += tl.where(gf > 1e-5, gf, 0.0)
            sum_act += tl.where(ga > 1e-5, tl.sqrt(ga + 1e-8), 0.0)
            prev_b = bm

        gf_last = tl.maximum(s_max - prev_b, 0.0)
        sum_free += tl.where(gf_last > 1e-5, gf_last, 0.0)

        has_act = sum_act > 1e-7
        has_free = sum_free > 1e-7
        p_act = tl.where(has_act & has_free, active_frac, tl.where(has_act, 1.0, 0.0))
        scale_act = p_act / tl.maximum(sum_act, 1e-8)
        scale_free = (1.0 - p_act) / tl.maximum(sum_free, 1e-8)

        # Pass 2: invert piecewise-linear segment CDF in strictly increasing t order
        s_out = tl.full([BLOCK_B, K_PAD], s_max, dtype=tl.float32)
        assigned = tl.zeros([BLOCK_B, K_PAD], dtype=tl.int1)
        w_cum = tl.zeros([BLOCK_B], dtype=tl.float32)
        prev_b = tl.full([BLOCK_B], s_min, dtype=tl.float32)

        for m in tl.static_range(M_MAX):
            off_m = b64 * M_MAX + m
            am = tl.load(band_a_ptr + off_m, mask=mask_b, other=s_max)
            bm = tl.load(band_b_ptr + off_m, mask=mask_b, other=s_max)

            # Free gap [prev_b, am]
            gf = tl.maximum(am - prev_b, 0.0)
            dw_f = tl.where(gf > 1e-5, gf * scale_free, 0.0)
            w_next = w_cum + dw_f
            hit_f = (~assigned) & (dw_f[:, None] > 1e-7) & (u <= w_next[:, None] + 1e-6)
            alpha_f = tl.minimum(tl.maximum((u - w_cum[:, None]) / tl.maximum(dw_f[:, None], 1e-7), 0.0), 1.0)
            s_out = tl.where(hit_f, prev_b[:, None] + alpha_f * gf[:, None], s_out)
            assigned = assigned | hit_f
            w_cum = w_next

            # Active band [am, bm]
            ga = tl.maximum(bm - am, 0.0)
            dw_a = tl.where(ga > 1e-5, tl.sqrt(ga + 1e-8) * scale_act, 0.0)
            w_next = w_cum + dw_a
            hit_a = (~assigned) & (dw_a[:, None] > 1e-7) & (u <= w_next[:, None] + 1e-6)
            alpha_a = tl.minimum(tl.maximum((u - w_cum[:, None]) / tl.maximum(dw_a[:, None], 1e-7), 0.0), 1.0)
            s_out = tl.where(hit_a, am[:, None] + alpha_a * ga[:, None], s_out)
            assigned = assigned | hit_a
            w_cum = w_next

            prev_b = bm

        # Final free gap [prev_b, s_max]
        gf_last = tl.maximum(s_max - prev_b, 0.0)
        dw_l = tl.where(gf_last > 1e-5, gf_last * scale_free, 0.0)
        w_next = w_cum + dw_l
        hit_l = (~assigned) & (dw_l[:, None] > 1e-7) & (u <= w_next[:, None] + 1e-5)
        alpha_l = tl.minimum(tl.maximum((u - w_cum[:, None]) / tl.maximum(dw_l[:, None], 1e-7), 0.0), 1.0)
        s_out = tl.where(hit_l, prev_b[:, None] + alpha_l * gf_last[:, None], s_out)

        # Pin endpoints exactly
        s_out = tl.where(k[None, :] == 0, s_min, s_out)
        s_out = tl.where(k[None, :] == (K - 1), s_max, s_out)

        out_off = b64[:, None] * K + k[None, :].to(tl.int64)
        tl.store(out_ptr + out_off, s_out, mask=mask_bk)


class CoarseUDFGrid(nn.Module):
    """Porous-safe Coarse 3D Unsigned Distance Field (UDF) for multi-band knot placement.

    Replaces the learned Stage A shift hash grid + MLP with an explicit coarse GPU UDF
    built directly from LiDAR hit envelopes:
      1. Scatters hit points into a tight-AABB coarse voxel grid (preserving porous
         foliage, glass windows, and thin fences without TSDF pass-through carving).
      2. Computes the exact 3D Euclidean Distance Transform (UDF) on GPU in ~1-2 ms via
         3 separable 1D Triton passes (or PyTorch fallback).
      3. For each ray (both training and novel-view rendering), sphere-traces the UDF
         in a single fused Triton kernel to discover ALL active bands [a_m, b_m] along
         the ray, then allocates knots via piecewise-linear importance quantiles
         (~75% in active bands with sqrt(length) weighting, ~25% in free space).

    All grid tensors are registered as PyTorch buffers so checkpoints remain 100%
    self-contained for novel-view rendering.
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float],
        bbox_max: Tuple[float, float, float],
        grid_res: int = 128,
        min_hits: int = 1,
        band_radius: Optional[float] = None,
        max_bands: int = 6,
    ):
        super().__init__()
        x0, y0, z0 = (float(v) for v in bbox_min)
        x1, y1, z1 = (float(v) for v in bbox_max)
        sx = max(x1 - x0, 1e-3)
        sy = max(y1 - y0, 1e-3)
        sz = max(z1 - z0, 1e-3)
        max_span = max(sx, sy, sz)

        self.nx = max(16, int(round(grid_res * (sx / max_span))))
        self.ny = max(16, int(round(grid_res * (sy / max_span))))
        self.nz = max(16, int(round(grid_res * (sz / max_span))))
        self.bbox_min = (x0, y0, z0)
        self.bbox_max = (x1, y1, z1)
        self.spacing = (
            sx / max(1, self.nx - 1),
            sy / max(1, self.ny - 1),
            sz / max(1, self.nz - 1),
        )
        self.min_vox = min(self.spacing)
        self.max_vox = max(self.spacing)
        self.vox_diag = math.sqrt(sum(d * d for d in self.spacing))
        self.min_hits = max(1, int(min_hits))
        self.max_bands = max(1, int(max_bands))
        # Default band_radius ~ 1.05 * vox_diag (~0.25-0.35m) brackets thin fences/glass
        # and bridges minor scan-line gaps without overly wide bands.
        self.band_radius = float(band_radius) if band_radius is not None else float(1.05 * self.vox_diag)
        # Conservative sphere-tracing margin guarantees we never jump over an occupied voxel
        self.safe_margin = float(self.band_radius + 0.25 * self.vox_diag)
        self.fine_step = float(0.75 * self.min_vox)

        self.register_buffer("bbox_min_t", torch.tensor([x0, y0, z0], dtype=torch.float32))
        self.register_buffer("bbox_max_t", torch.tensor([x1, y1, z1], dtype=torch.float32))
        self._inv_spacing = (
            (self.nx - 1) / sx,
            (self.ny - 1) / sy,
            (self.nz - 1) / sz,
        )
        self.register_buffer("inv_spacing", torch.tensor(self._inv_spacing, dtype=torch.float32))
        self.register_buffer("hit_counts", torch.zeros((self.nx, self.ny, self.nz), dtype=torch.int32))
        self.register_buffer("udf_grid", torch.full((self.nx, self.ny, self.nz), 1e6, dtype=torch.float32))
        self.register_buffer("has_data", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_is_act_mask", ((torch.arange(2 * self.max_bands + 1) & 1) == 1).unsqueeze(0))
        self._has_data: bool = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._has_data = bool(self.has_data.item())

    @torch.no_grad()
    def clear(self) -> None:
        self.hit_counts.zero_()
        self.udf_grid.fill_(1e6)
        self.has_data.fill_(False)
        self._has_data = False

    @torch.no_grad()
    def update_from_batch(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        hit_range: torch.Tensor,
        hit_mask: Optional[torch.Tensor] = None,
        rebuild: bool = False,
    ) -> None:
        """Accumulates 3D hit points into the coarse voxel grid (zero CPU-GPU sync)."""
        r = hit_range.view(-1)
        pts = ray_o + r.unsqueeze(-1) * ray_d  # (B, 3)
        ijk = torch.floor((pts - self.bbox_min_t) * self.inv_spacing + 0.5).long()
        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < self.nx)
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < self.ny)
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < self.nz)
        )
        if hit_mask is not None:
            valid = valid & (hit_mask.view(-1) > 0.5)

        flat = (ijk[:, 0] * self.ny + ijk[:, 1]) * self.nz + ijk[:, 2]
        flat = flat[valid]
        if flat.numel() > 0:
            ones = torch.ones_like(flat, dtype=torch.int32)
            self.hit_counts.view(-1).scatter_add_(0, flat, ones)
            if rebuild:
                self.rebuild_udf()

    @staticmethod
    def _edt_1d_axis(d2: torch.Tensor, spacing: float) -> torch.Tensor:
        """Exact 1D squared Euclidean distance transform along the last dimension."""
        orig_shape = d2.shape
        N = orig_shape[-1]
        L_lines = d2.numel() // N
        inp = d2.reshape(L_lines, N).contiguous()
        h_sq = float(spacing * spacing)

        if HAS_TRITON and inp.is_cuda:
            out = torch.empty_like(inp)
            BLOCK_N = 128
            grid = (L_lines, triton.cdiv(N, BLOCK_N))
            _edt_1d_last_dim_kernel[grid](inp, out, h_sq, L_lines, N, BLOCK_N=BLOCK_N)
            return out.reshape(orig_shape)

        # Pure PyTorch fallback (chunked over lines to keep memory bounded)
        out = torch.empty_like(inp)
        coords = torch.arange(N, device=inp.device, dtype=torch.float32)
        dist_sq = h_sq * (coords.unsqueeze(1) - coords.unsqueeze(0)) ** 2  # (N, N)
        chunk = max(1, min(L_lines, 4096))
        for start in range(0, L_lines, chunk):
            end = min(start + chunk, L_lines)
            # (c, N, 1) + (1, N, N) -> min over p (dim=1) -> (c, N)
            out[start:end] = (inp[start:end].unsqueeze(2) + dist_sq.unsqueeze(0)).min(dim=1).values
        return out.reshape(orig_shape)

    @torch.no_grad()
    def rebuild_udf(self) -> float:
        """Computes the exact 3D Euclidean UDF from accumulated hit_counts.

        Returns:
            Fraction of occupied voxels in [0, 1].
        """
        occ = self.hit_counts >= self.min_hits
        occ_count = occ.sum()
        if not bool(occ_count.item() > 0):
            self.udf_grid.fill_(1e6)
            self.has_data.fill_(False)
            self._has_data = False
            return 0.0

        dx, dy, dz = self.spacing
        d2 = torch.where(occ, torch.zeros((), device=occ.device, dtype=torch.float32), torch.full((), 1e20, device=occ.device, dtype=torch.float32))
        # Separable exact 3D squared EDT: Z -> Y -> X
        d2 = self._edt_1d_axis(d2, dz)
        d2 = self._edt_1d_axis(d2.permute(0, 2, 1), dy).permute(0, 2, 1)
        d2 = self._edt_1d_axis(d2.permute(1, 2, 0), dx).permute(2, 0, 1)
        self.udf_grid.copy_(torch.sqrt(d2.clamp(min=0.0)))
        self.has_data.fill_(True)
        self._has_data = True
        return float(occ_count.item()) / float(self.nx * self.ny * self.nz)

    @torch.no_grad()
    def build_from_rays(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        hit_range: torch.Tensor,
        hit_mask: Optional[torch.Tensor] = None,
        clear_first: bool = True,
    ) -> float:
        """One-shot GPU build of the coarse UDF from a full ray dataset."""
        if clear_first:
            self.clear()
        self.update_from_batch(ray_o, ray_d, hit_range, hit_mask, rebuild=False)
        return self.rebuild_udf()

    @torch.no_grad()
    def get_occupied_world_points(self, radius: Optional[float] = None) -> torch.Tensor:
        """Returns world-space (N, 3) centers of voxels within `radius` of occupied geometry."""
        r_thr = self.band_radius if radius is None else float(radius)
        idx = torch.nonzero(self.udf_grid <= r_thr, as_tuple=False)
        if idx.numel() == 0:
            return torch.empty((0, 3), device=self.udf_grid.device, dtype=torch.float32)
        dx, dy, dz = self.spacing
        step_t = torch.tensor([dx, dy, dz], device=self.udf_grid.device, dtype=torch.float32)
        return self.bbox_min_t.unsqueeze(0) + idx.float() * step_t.unsqueeze(0)

    def _trace_bands_pytorch(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        s_min: float,
        s_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure-PyTorch fallback for multi-band UDF sphere tracing (CPU or non-Triton)."""
        B = ray_o.shape[0]
        device = ray_o.device
        M = self.max_bands
        band_a = torch.full((B, M), s_max, device=device, dtype=torch.float32)
        band_b = torch.full((B, M), s_max, device=device, dtype=torch.float32)

        eps = 1e-8
        sd = torch.where(ray_d.abs() > eps, ray_d, torch.where(ray_d >= 0, eps, -eps))
        t0 = (self.bbox_min_t - ray_o) / sd
        t1 = (self.bbox_max_t - ray_o) / sd
        t_enter = torch.maximum(torch.full((B,), s_min, device=device), torch.minimum(t0, t1).max(dim=-1).values)
        t_exit = torch.minimum(torch.full((B,), s_max, device=device), torch.maximum(t0, t1).min(dim=-1).values)

        active_ray = t_enter < t_exit
        t = t_enter.clone()
        in_band = torch.zeros(B, device=device, dtype=torch.bool)
        band_start = t_enter.clone()
        m_cnt = torch.zeros(B, device=device, dtype=torch.long)
        ray_idx = torch.arange(B, device=device)

        for _ in range(192):
            alive = active_ray & (t < t_exit)
            if not bool(alive.any()):
                break
            pts = ray_o + t.unsqueeze(-1) * ray_d
            ijk = torch.floor((pts - self.bbox_min_t) * self.inv_spacing + 0.5).long()
            ix = ijk[:, 0].clamp(0, self.nx - 1)
            iy = ijk[:, 1].clamp(0, self.ny - 1)
            iz = ijk[:, 2].clamp(0, self.nz - 1)
            d = self.udf_grid[ix, iy, iz]

            hit_now = alive & (d <= self.band_radius)
            entering = hit_now & (~in_band)
            exiting = alive & (~hit_now) & in_band

            band_start = torch.where(entering & (m_cnt < M), t, band_start)
            in_band = torch.where(entering, True, in_band)

            if bool(exiting.any()):
                slot = m_cnt.clamp(max=M - 1)
                is_new = exiting & (m_cnt < M)
                band_a[ray_idx[is_new], slot[is_new]] = band_start[is_new]
                band_b[ray_idx[exiting], slot[exiting]] = t[exiting]
                m_cnt = torch.where(is_new, m_cnt + 1, m_cnt)
                in_band = torch.where(exiting, False, in_band)

            step = torch.where(hit_now, self.fine_step, torch.clamp(d - self.safe_margin, min=self.fine_step))
            t = torch.where(alive, t + step, t)

        still_open = active_ray & in_band
        if bool(still_open.any()):
            slot = m_cnt.clamp(max=M - 1)
            is_new = still_open & (m_cnt < M)
            t_end = torch.minimum(t, t_exit)
            band_a[ray_idx[is_new], slot[is_new]] = band_start[is_new]
            band_b[ray_idx[still_open], slot[still_open]] = t_end[still_open]

        return band_a, band_b

    @torch.compiler.disable
    @torch.no_grad()
    def place_knots(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        num_knots: int,
        s_min: float,
        s_max: float,
        active_frac: float = 0.75,
        jitter: bool = False,
    ) -> torch.Tensor:
        """Places K strictly ordered knots along each ray guided by the coarse 3D UDF.

        Args:
            ray_o: Ray origins, shape (B, 3).
            ray_d: Unit ray directions, shape (B, 3).
            num_knots: Total knots K per ray (endpoints pinned at s_min and s_max).
            s_min: Near ray bound in meters.
            s_max: Far ray bound in meters.
            active_frac: Fraction of interior knots allocated to active surface/volume bands.
            jitter: If True (training), adds stratified jitter within each quantile stratum.

        Returns:
            s_knots: Strictly monotonic knot distances, shape (B, K).
        """
        B = ray_o.shape[0]
        K = int(num_knots)
        device = ray_o.device
        K_int = K - 2

        if K <= 2 or not self._has_data:
            if jitter and K_int > 0:
                base = torch.arange(K_int, device=device, dtype=torch.float32).unsqueeze(0)
                u = (base + (0.05 + 0.90 * torch.rand((B, K_int), device=device))) / float(K_int)
                s_int = s_min + u * (s_max - s_min)
                return torch.cat(
                    [
                        torch.full((B, 1), s_min, device=device, dtype=torch.float32),
                        s_int,
                        torch.full((B, 1), s_max, device=device, dtype=torch.float32),
                    ],
                    dim=-1,
                )
            return torch.linspace(s_min, s_max, K, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1).contiguous()

        M = self.max_bands
        if HAS_TRITON and ray_o.is_cuda:
            ro_c = ray_o.float().contiguous()
            rd_c = ray_d.float().contiguous()
            band_a = torch.full((B, M), float(s_max), device=device, dtype=torch.float32)
            band_b = torch.full((B, M), float(s_max), device=device, dtype=torch.float32)
            BLOCK_B = 128
            grid = (triton.cdiv(B, BLOCK_B),)
            _udf_trace_bands_kernel[grid](
                ro_c,
                rd_c,
                self.udf_grid,
                band_a,
                band_b,
                B,
                self.bbox_min[0],
                self.bbox_min[1],
                self.bbox_min[2],
                self.bbox_max[0],
                self.bbox_max[1],
                self.bbox_max[2],
                self._inv_spacing[0],
                self._inv_spacing[1],
                self._inv_spacing[2],
                self.nx,
                self.ny,
                self.nz,
                float(s_min),
                float(s_max),
                self.band_radius,
                self.safe_margin,
                self.fine_step,
                M_MAX=M,
                MAX_STEPS=192,
                BLOCK_B=BLOCK_B,
            )
            if K <= 128:
                K_PAD = 1 << (K - 1).bit_length()
                K_PAD = max(16, K_PAD)
                BLOCK_K = 16 if K_PAD >= 64 else 32
                s_knots = torch.empty((B, K), device=device, dtype=torch.float32)
                rand_buf = torch.rand((B, K), device=device, dtype=torch.float32) if jitter else s_knots
                _bands_to_knots_kernel[(triton.cdiv(B, BLOCK_K),)](
                    band_a,
                    band_b,
                    rand_buf,
                    s_knots,
                    B,
                    K,
                    float(s_min),
                    float(s_max),
                    float(active_frac),
                    M_MAX=M,
                    USE_JITTER=bool(jitter),
                    BLOCK_B=BLOCK_K,
                    K_PAD=K_PAD,
                )
                return s_knots
        else:
            band_a, band_b = self._trace_bands_pytorch(ray_o.float(), ray_d.float(), s_min, s_max)

        # Build sorted segment boundaries T: [s_min, a_0, b_0, a_1, b_1, ..., a_{M-1}, b_{M-1}, s_max]
        # Even segments (0, 2, ..., 2M) = free-space gaps; Odd segments (1, 3, ..., 2M-1) = active bands.
        ab = torch.stack([band_a, band_b], dim=-1).reshape(B, 2 * M)  # (B, 2M)
        s0_col = torch.full((B, 1), float(s_min), device=device, dtype=torch.float32)
        s1_col = torch.full((B, 1), float(s_max), device=device, dtype=torch.float32)
        T = torch.cat([s0_col, ab, s1_col], dim=-1)  # (B, 2M + 2)
        S = 2 * M + 1

        seg_len = (T[:, 1:] - T[:, :-1]).clamp(min=0.0)  # (B, S)
        is_act = self._is_act_mask  # (1, S)

        # sqrt(length) weighting on active bands keeps thin solid walls dense alongside wide bushes
        w_act_raw = torch.where(is_act & (seg_len > 1e-5), torch.sqrt(seg_len + 1e-8), torch.zeros_like(seg_len))
        w_free_raw = torch.where((~is_act) & (seg_len > 1e-5), seg_len, torch.zeros_like(seg_len))

        sum_act = w_act_raw.sum(dim=-1, keepdim=True)
        sum_free = w_free_raw.sum(dim=-1, keepdim=True)

        has_act = sum_act > 1e-7
        has_free = sum_free > 1e-7
        p_act = torch.where(
            has_act & has_free,
            torch.full_like(sum_act, float(active_frac)),
            torch.where(has_act, torch.ones_like(sum_act), torch.zeros_like(sum_act)),
        )

        w_seg = (
            p_act * (w_act_raw / sum_act.clamp(min=1e-8))
            + (1.0 - p_act) * (w_free_raw / sum_free.clamp(min=1e-8))
        )  # (B, S), rows sum to 1.0

        W = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.float32), torch.cumsum(w_seg, dim=-1)], dim=-1)
        W[:, -1] = 1.0  # Guard against fp32 cumsum drift

        # Stratified quantile targets u in (0, 1)
        if jitter:
            base = torch.arange(K_int, device=device, dtype=torch.float32).unsqueeze(0)
            u = (base + (0.05 + 0.90 * torch.rand((B, K_int), device=device))) / float(K_int)
        else:
            u = torch.linspace(1.0 / (K - 1), (K - 2) / (K - 1), K_int, device=device, dtype=torch.float32)
            u = u.unsqueeze(0).expand(B, -1).contiguous()

        # Piecewise-linear inverse-CDF lookup (right=True skips zero-weight segments cleanly)
        idx = torch.searchsorted(W.contiguous(), u.contiguous(), right=True).clamp(1, S)
        idx_l = idx - 1
        W_l = torch.gather(W, 1, idx_l)
        W_r = torch.gather(W, 1, idx)
        T_l = torch.gather(T, 1, idx_l)
        T_r = torch.gather(T, 1, idx)

        alpha = ((u - W_l) / (W_r - W_l).clamp(min=1e-7)).clamp(0.0, 1.0)
        s_int = T_l + alpha * (T_r - T_l)

        return torch.cat([s0_col, s_int, s1_col], dim=-1)


# ==============================================================================
# 2. Hash-Grid Self-Conditioned Knot Network (HashSCKNetwork)
# ==============================================================================

class HashSCKNetwork(nn.Module):
    """Instant-NGP Hash-Grid Self-Conditioned Knot Network for LiDAR Return CDFs.

    Moves scene geometry into a multi-resolution 3D HashGrid3D so the per-knot
    decoder shrinks to a lightweight H=64 MLP (~25k MLP params), dramatically
    increasing throughput and convergence speed while improving 3D multi-view
    consistency.

    Key architectural upgrades over the Fourier baseline:
      1. Multi-resolution 3D HashGrid3D encoding for knot world coordinates x_k.
      2. Origin-invariant Plücker ray context (d, m = o x d), projected ONCE per ray.
      3. Two-stage refinement: predicts gap-bounded knot shifts at current knots,
         then queries HashGrid3D at the SHIFTED interval midpoints to compute the
         monotonic CDF (so final-round knot shifts are physically evaluated in 3D).
      4. Interval-width-aware softmax CDF (adds log(delta_s) bias + escape bin) or
         physical Beer-Lambert hazard quadrature ('hazard').
      5. 3D-grounded ray-drop head that pools refined 3D midpoint features + c_max
         + ray context instead of classifying from ray coordinates alone.

    Args:
        num_knots: Number of candidate knots K placed along each ray (default 32).
        s_min: Near bound distance along ray in meters (default 0.1).
        s_max: Far bound distance along ray in meters (default 15.0).
        hidden_dim: Hidden dimension of compact MLP decoders (default 64).
        num_hidden_blocks: Residual blocks in knot shift decoder (default 1).
        num_rounds: Number of self-conditioning refinement rounds R (default 2).
        max_displacement: Maximum interior knot shift per round in meters (default 1.0).
        cdf_mode: 'softmax' (default, interval softmax + log(delta_s) + escape bin)
            or 'hazard' (Beer-Lambert 1 - exp(-cumsum(lambda * delta_s))).
        local_cdf: If True (default), CDF head reads purely from 3D hash features
            at interval midpoints (+ direction d if use_viewdir) without ray_feat,
            maximizing cross-ray 3D consistency (README Rung 5a).
        use_viewdir: If True, passes unit ray direction d to the CDF head.
        n_levels: HashGrid3D resolution levels (default 16).
        n_features: HashGrid3D features per level (default 2).
        log2_hashmap_size: Log2 hash table entries per level (default 18).
        base_resolution: HashGrid3D coarsest resolution (default 16).
        max_resolution: HashGrid3D finest resolution (default 2048).
        scene_bound: Half-extent of cubic 3D hash grid [-bound, +bound]^3 (default s_max).
        hash_backend: 'auto' (default), 'triton', 'tcnn', or 'pytorch'.
        coarse_to_fine: Enable progressive coarse-to-fine hash level masking (default True).
        ctf_start_levels: Initial coarsest hash levels active at Epoch 0 (default 4).
        ctf_end_frac: Training fraction at which all hash levels unlock (default 0.6).
        lambda_curv: RMS 3D curvature (2nd-order Laplacian) smoothness loss weight (default 0.4).
        curv_samples: 3D points per batch for curvature loss (default 2048).
        curv_eps_start: Initial finite-difference step in meters for curvature loss (default 0.12).
        curv_eps_end: Final finite-difference step in meters for curvature loss (default 0.03).
        lambda_hash_l2: Fine-level-only L2 decay weight on hash tables (default 5e-3; 0 disables).
        reg_every: Regularization cadence for 3D curvature + fine hash L2 (default 1 = every step).
        separate_shift_grid: If True (default), uses a dedicated coarse HashGrid3D for
            knot-shift prediction so empty-space attraction gradients never clash with
            the fine scene HashGrid3D.
        shift_n_levels: Resolution levels for shift_hash_grid (default 8).
        shift_log2_hashmap_size: Log2 entries per level for shift_hash_grid (default 14).
        shift_base_resolution: Coarsest grid resolution for shift_hash_grid (default 8).
        shift_max_resolution: Finest grid resolution for shift_hash_grid (default 128).
    """

    def __init__(
        self,
        num_knots: int = 16,                  # [Fidelity, Speed] Spline knots K/ray. More = finer depth/multi-surface resolution; scales hash lookups (2K-1/round) & MLP compute O(K).
        s_min: float = 0.1,                   # [Fidelity] Near pinned ray bound (m). Keep tight to closest valid return to avoid wasting knot spacing.
        s_max: float = 15.0,                  # [Fidelity] Far pinned ray bound (m) & default scene_bound. Tightening to max hit range boosts knot density for free.
        hidden_dim: int = 64,                 # [Fidelity, Speed] Width H of shift/CDF/ray-drop MLPs. Helps resolve hash collisions; scales MLP compute ~O(H^2 * K * R) (main MLP bottleneck).
        num_hidden_blocks: int = 1,           # [Fidelity, Speed] Residual blocks in knot-shift MLP only (not cdf_mlp). More = smarter knot placement; 0 is fastest.
        num_rounds: int = 3,                  # [Fidelity, Speed] Self-conditioning passes R. More = tighter surface bracketing, but scales total forward/backward/loss time ~O(R).
        max_displacement: float = 1.0,        # [Fidelity] Per-round knot shift cap (m), paired with 0.45*gap. Rarely active at K=32, s_max=8 unless K is small or s_max is large.
        cdf_mode: str = "softmax",            # [Fidelity] 'softmax' (width-aware + escape bin, smoother grads) vs 'hazard' (Beer-Lambert quadrature, stricter volumetric additivity).
        local_cdf: bool = True,               # [Fidelity, Speed] True = 3D hash(x_mid) only -> best cross-ray consistency & novel views; False = adds ray_feat (+H inputs, slightly slower).
        use_viewdir: bool = False,            # [Fidelity] Appends unit ray_d (+3 dims) to cdf_mlp. Enable only for anisotropic/view-dependent returns; False preserves isotropic 3D consistency.
        n_levels: int = 16,                   # [Fidelity, Speed, Hashmap] Multi-res grid scales (base -> max). More = smoother multi-scale detail; scales hashmap VRAM & kernel work O(L).
        n_features: int = 2,                  # [Fidelity, Speed, Hashmap] Channels/entry; scales hashmap VRAM O(F). Note: fused Triton kernel requires n_features=2 (else falls back to slow PyTorch without tcnn).
        log2_hashmap_size: int = 18,          # [Fidelity, Hashmap] Entries/level T=2^P (18 -> 262,144). Top fidelity lever for fewer fine-level collisions at ~O(1) compute; doubles hashmap VRAM per +1 (~4x incl. Adam+grad).
        base_resolution: int = 16,            # [Fidelity] Coarsest grid resolution across [-bound, +bound]. Lower smooths sparse views; higher shifts levels toward finer detail.
        max_resolution: int = 2048,            # [Fidelity] Finest grid resolution across 2*bound (voxel = 2*bound/max_res). Raise (1024-2048) with larger log2_hashmap_size for sharper thin edges.
        scene_bound: Optional[float] = None,  # [Fidelity] Half-extent of cubic 3D hash grid [-bound, +bound]^3 (default s_max). Free fidelity boost when tightened to actual scene AABB.
        hash_backend: str = "auto",           # [Speed, Hashmap] 'tcnn' (fastest + dense coarse levels, if installed) > 'triton' (fast fused CUDA, n_features=2) >> 'pytorch' (slow fallback).
        # --- Knot Placement Mode ('shift' = learned SCK flow, 'udf' = Coarse 3D UDF multi-band walker) ---
        knot_mode: str = "shift",             # [Speed, Fidelity] 'shift' = learned self-conditioned knot flow (R rounds); 'udf' = non-neural coarse 3D UDF sphere-tracer (R=1, no shift MLP/grid, ~5-10x faster).
        udf_grid_res: int = 128,              # [Fidelity, Speed] Coarse UDF resolution along longest AABB axis (knot_mode='udf').
        udf_band_radius: Optional[float] = None,  # [Fidelity] Active band half-width around hit voxels in meters (None = 1.05 * voxel_diag).
        udf_active_frac: float = 0.75,        # [Fidelity] Fraction of interior knots placed inside active surface/volume bands (remainder in free space).
        udf_max_bands: int = 6,               # [Fidelity] Max distinct active bands tracked per ray (multi-surface glass/foliage/walls).
        udf_min_hits: int = 1,                # [Fidelity] Min LiDAR hits per coarse voxel to mark occupied.
        udf_bbox_min: Optional[Tuple[float, float, float]] = None,  # [Fidelity] Optional tight AABB min for coarse UDF (default [-bound]^3).
        udf_bbox_max: Optional[Tuple[float, float, float]] = None,  # [Fidelity] Optional tight AABB max for coarse UDF (default [+bound]^3).
        # --- Separate Knot-Shift Hash Grid ("where to look" vs. "what is there") ---
        separate_shift_grid: bool = True,     # [Fidelity, Speed, Hashmap] True = dedicated coarse HashGrid3D for knot shifts (eliminates empty-space vs. sharp-surface gradient tension & speeds up Stage A); False = share scene hash_grid.
        shift_n_levels: int = 16,              # [Fidelity, Speed, Hashmap] Coarse resolution levels for shift_hash_grid (only used when separate_shift_grid=True).
        shift_log2_hashmap_size: int = 14,    # [Fidelity, Hashmap] Entries/level T=2^P for shift_hash_grid (14 -> 16,384, ~1.05 MB total, L2-cache hot & low-collision).
        shift_base_resolution: int = 14,       # [Fidelity] Coarsest grid resolution for shift_hash_grid (wide, smooth attraction basins in empty space).
        shift_max_resolution: int = 1024,      # [Fidelity] Finest grid resolution for shift_hash_grid (smooth medium-scale pull without fine-level collision jitter).
        # --- 3D Smoothness & Regularization ---
        coarse_to_fine: bool = True,          # [Fidelity] Progressive coarse-to-fine hash masking: locks in smooth coarse walls/ground before fine levels unlock (~0% cost).
        ctf_start_levels: int = 4,            # [Fidelity] Number of coarsest collision-free hash levels active at Epoch 0 when coarse_to_fine=True.
        ctf_end_frac: float = 0.6,            # [Fidelity] Training fraction at which all hash levels finish unlocking (0.6 = full fine resolution for final 40% of training).
        lambda_curv: float = 0.4,             # [Fidelity, Speed] RMS 3D curvature (2nd-order Laplacian) smoothness weight; flattens bumpy ripples while preserving sharp planar walls (~5% cost; 0 disables).
        curv_samples: int = 2048,             # [Fidelity, Speed] Random 3D points/batch for curvature loss (75% near hit surfaces, 25% along free-space ray interiors).
        curv_eps_start: float = 0.12,         # [Fidelity] Initial finite-difference step size (m) for 3D curvature at Epoch 0 (smooths broad waves).
        curv_eps_end: float = 0.03,           # [Fidelity] Final finite-difference step size (m) for 3D curvature once fine levels unlock (smooths cm-scale ripples).
        lambda_hash_l2: float = 5e-3,         # [Fidelity] Fine-level-only L2 decay on hash tables (0 on coarse levels, linearly ramped to 1 on finest level). Scaled for 16x2^18 table; kills collision speckles without softening thin branches.
        reg_every: int = 1,                   # [Speed] Regularization cadence for 3D curvature + fine hash L2 (1 = every step; >1 weakens regularization under Adam).
    ):
        super().__init__()
        if knot_mode not in ("shift", "udf"):
            raise ValueError(f"Unsupported knot_mode={knot_mode!r}; expected 'shift' or 'udf'.")
        self.knot_mode = knot_mode
        self.num_knots = num_knots
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.num_rounds = 1 if knot_mode == "udf" else num_rounds
        self.max_displacement = float(max_displacement)
        self.cdf_mode = cdf_mode
        self.local_cdf = True if knot_mode == "udf" else local_cdf
        self.use_viewdir = use_viewdir
        self.udf_active_frac = float(udf_active_frac)
        self.separate_shift_grid = bool(separate_shift_grid) and (knot_mode == "shift")
        self.coarse_to_fine = bool(coarse_to_fine)
        self.ctf_start_levels = int(ctf_start_levels)
        self.ctf_end_frac = float(ctf_end_frac)
        self.lambda_curv = float(lambda_curv)
        self.curv_samples = int(curv_samples)
        self.curv_eps_start = float(curv_eps_start)
        self.curv_eps_end = float(curv_eps_end)
        self.lambda_hash_l2 = float(lambda_hash_l2)
        self.reg_every = max(1, int(reg_every))
        bound = float(scene_bound) if scene_bound is not None else float(s_max)

        # Per-level weights for fine-only L2 decay: 0 on coarse levels (< ctf_start_levels),
        # ramping linearly to 1.0 on the finest level where hash collisions concentrate.
        lvl_idx = torch.arange(n_levels, dtype=torch.float32)
        denom = max(1.0, float(n_levels - self.ctf_start_levels))
        fine_w = torch.clamp((lvl_idx - self.ctf_start_levels + 1.0) / denom, 0.0, 1.0)
        self.register_buffer("fine_l2_weights", fine_w.view(n_levels, 1, 1))

        # 1a. High-Resolution 3D Scene Spatial Hash Grid ("what is there")
        self.hash_grid = HashGrid3D(
            n_levels=n_levels,
            n_features=n_features,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            bound=bound,
            backend=hash_backend,
        )
        hash_dim = self.hash_grid.out_dim

        # 1b. Stage A ("where to look"): either Coarse 3D UDF Grid OR Learned Shift HashGrid + MLP
        if self.knot_mode == "udf":
            bmin = udf_bbox_min if udf_bbox_min is not None else (-bound, -bound, -bound)
            bmax = udf_bbox_max if udf_bbox_max is not None else (bound, bound, bound)
            self.udf_grid: Optional[CoarseUDFGrid] = CoarseUDFGrid(
                bbox_min=bmin,
                bbox_max=bmax,
                grid_res=udf_grid_res,
                min_hits=udf_min_hits,
                band_radius=udf_band_radius,
                max_bands=udf_max_bands,
            )
            self.shift_hash_grid = None
            self.pos_enc_ray = None
            self.ray_in = None
            self.ray_proj = None
            self.knot_proj = None
            self.shift_blocks = None
            self.shift_head = None
        else:
            self.udf_grid = None
            if self.separate_shift_grid:
                self.shift_hash_grid: Optional[HashGrid3D] = HashGrid3D(
                    n_levels=shift_n_levels,
                    n_features=n_features,
                    log2_hashmap_size=shift_log2_hashmap_size,
                    base_resolution=shift_base_resolution,
                    max_resolution=shift_max_resolution,
                    bound=bound,
                    backend=hash_backend,
                )
                shift_hash_dim = self.shift_hash_grid.out_dim
            else:
                self.shift_hash_grid = None
                shift_hash_dim = hash_dim

            # 2. Compact Origin-Invariant Plücker Ray Encoder: (d, m = o x d) in R^6
            self.pos_enc_ray = PositionalEncoding(in_dim=6, num_freqs=4, max_coord=bound)
            self.ray_in = nn.Sequential(
                nn.Linear(self.pos_enc_ray.out_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # Split joint projection so ray_feat is projected ONCE per ray (not K*R times!)
            self.ray_proj = nn.Linear(hidden_dim, hidden_dim)
            self.knot_proj = nn.Linear(shift_hash_dim + 3, hidden_dim, bias=False)

            # 3. Lightweight Residual Shift Decoder (where to look)
            self.shift_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_hidden_blocks)
            ])
            self.shift_head = nn.Linear(hidden_dim, 1)

        # 4. Compact CDF / Hazard Head evaluated at interval midpoints (what is there)
        cdf_in_dim = hash_dim + 3  # hash(x_mid) + normalized x_mid
        if use_viewdir:
            cdf_in_dim += 3
        if not self.local_cdf:
            cdf_in_dim += hidden_dim

        self.cdf_mlp = nn.Sequential(
            nn.Linear(cdf_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cdf_out = nn.Linear(hidden_dim, 1)

        # 5. 3D-Grounded Ray-Drop Head: pools refined 3D midpoint features + c_max (+ ray_feat in shift mode)
        drop_in_dim = (hidden_dim + 1) if self.knot_mode == "udf" else (hidden_dim * 2 + 1)
        self.ray_drop_head = nn.Sequential(
            nn.Linear(drop_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Clean initial biases
        self._init_custom_biases()

        scene_hash_params = sum(p.numel() for p in self.hash_grid.parameters())
        shift_hash_params = (
            sum(p.numel() for p in self.shift_hash_grid.parameters())
            if self.shift_hash_grid is not None
            else 0
        )
        hash_params = scene_hash_params + shift_hash_params
        mlp_params = sum(p.numel() for p in self.parameters()) - hash_params
        if self.udf_grid is not None:
            udf_mb = (self.udf_grid.udf_grid.numel() * 4 + self.udf_grid.hit_counts.numel() * 4) / 1e6
            print(
                f"HashSCKNetwork [{self.hash_grid.backend} | knot_mode=udf] | "
                f"MLP params: {mlp_params:,} | "
                f"Scene hash: {scene_hash_params:,} ({scene_hash_params * 4 / 1e6:.2f} MB) | "
                f"Coarse UDF: {self.udf_grid.nx}x{self.udf_grid.ny}x{self.udf_grid.nz} "
                f"(vox={self.udf_grid.max_vox:.2f}m, band={self.udf_grid.band_radius:.2f}m, {udf_mb:.2f} MB)"
            )
        elif self.shift_hash_grid is not None:
            print(
                f"HashSCKNetwork [{self.hash_grid.backend} | knot_mode=shift] | "
                f"MLP params: {mlp_params:,} | "
                f"Scene hash: {scene_hash_params:,} ({scene_hash_params * 4 / 1e6:.2f} MB) | "
                f"Shift hash: {shift_hash_params:,} ({shift_hash_params * 4 / 1e6:.2f} MB)"
            )
        else:
            print(
                f"HashSCKNetwork [{self.hash_grid.backend} | knot_mode=shift] | "
                f"MLP params: {mlp_params:,} | Shared hash: {scene_hash_params:,} "
                f"({scene_hash_params * 4 / 1e6:.2f} MB)"
            )

    def _init_custom_biases(self) -> None:
        with torch.no_grad():
            if self.shift_head is not None:
                self.shift_head.bias.fill_(0.0)
            if self.cdf_mode == "softmax":
                self.cdf_out.bias.fill_(0.0)
            else:
                self.cdf_out.bias.fill_(-4.0)
            self.ray_drop_head[-1].bias.fill_(2.0)  # ~88% initial hit prior

    def reset_weights(self, seed: Optional[int] = None) -> None:
        """Resets all MLP and hash-grid parameters in-place without triggering torch.compile recompilation.

        Works seamlessly whether called directly on HashSCKNetwork or on a torch.compile()-wrapped instance.

        Args:
            seed: Optional random seed for reproducible re-initialization.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            for module in self.modules():
                if module is not self and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self._init_custom_biases()

    def parameter_groups(
        self,
        lr_mlp: float = 1e-3,
        lr_hash: float = 1e-2,
        weight_decay_mlp: float = 1e-6,
        lr_shift_hash: Optional[float] = 2e-3,
    ) -> List[Dict[str, Any]]:
        """Returns Instant-NGP style optimizer param groups (split scene hash, shift hash, and MLP LRs)."""
        scene_hash_ids = {id(p) for p in self.hash_grid.parameters()}
        shift_hash_ids = (
            {id(p) for p in self.shift_hash_grid.parameters()}
            if self.shift_hash_grid is not None
            else set()
        )
        all_hash_ids = scene_hash_ids | shift_hash_ids
        mlp_params = [p for p in self.parameters() if id(p) not in all_hash_ids]

        if lr_shift_hash is not None and self.shift_hash_grid is not None:
            scene_hash_params = [p for p in self.parameters() if id(p) in scene_hash_ids]
            shift_hash_params = [p for p in self.parameters() if id(p) in shift_hash_ids]
            return [
                {"params": scene_hash_params, "lr": lr_hash, "weight_decay": 0.0},
                {"params": shift_hash_params, "lr": lr_shift_hash, "weight_decay": 0.0},
                {"params": mlp_params, "lr": lr_mlp, "weight_decay": weight_decay_mlp},
            ]

        hash_params = [p for p in self.parameters() if id(p) in all_hash_ids]
        return [
            {"params": hash_params, "lr": lr_hash, "weight_decay": 0.0},
            {"params": mlp_params, "lr": lr_mlp, "weight_decay": weight_decay_mlp},
        ]

    def set_coarse_to_fine_progress(self, progress: float, start_levels: Optional[int] = None) -> float:
        """Updates progressive coarse-to-fine hash level mask in-place.

        Args:
            progress: Training progress in [0.0, 1.0] across the coarse-to-fine schedule.
            start_levels: Number of coarsest hash levels active at progress=0.0
                (defaults to self.ctf_start_levels).

        Returns:
            Current floating-point active level count.
        """
        p = max(0.0, min(1.0, float(progress)))
        lo_val = self.ctf_start_levels if start_levels is None else int(start_levels)
        lo = max(1, min(lo_val, self.hash_grid.n_levels))
        active = lo + p * (self.hash_grid.n_levels - lo)
        self.hash_grid.set_active_levels(active)
        return active

    def fine_hash_l2_loss(self) -> torch.Tensor:
        """Computes level-weighted L2 penalty on fine hash table entries only.

        Coarse levels (< ctf_start_levels) receive zero weight so broad surfaces stay
        untouched, while finer levels ramp linearly to 1.0 to suppress collision speckles.
        """
        if self.hash_grid.tables is not None:
            return ((self.hash_grid.tables ** 2) * self.fine_l2_weights).mean()
        if self.hash_grid.tcnn_enc is not None:
            params = list(self.hash_grid.tcnn_enc.parameters())
            if params:
                return sum((p ** 2).mean() for p in params) / len(params)
        return torch.zeros((), device=self.fine_l2_weights.device, dtype=torch.float32)

    def eval_local_3d_field(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable local 3D occupancy/solidity scalar in (0, 1) at arbitrary 3D points.

        Used by 3D curvature regularization and direct volumetric queries; matches the exact
        scalar mapping extracted by Marching Cubes.
        """
        orig_shape = x.shape[:-1]
        p = x.reshape(-1, 3)
        h = self.hash_grid(p)
        parts = [h, p / self.hash_grid.bound]
        if self.use_viewdir:
            parts.append(torch.zeros_like(p))
        if not self.local_cdf:
            hidden_dim = self.ray_proj.out_features
            parts.append(torch.zeros((p.shape[0], hidden_dim), device=p.device, dtype=p.dtype))
        feat = self.cdf_mlp(torch.cat(parts, dim=-1))
        raw = self.cdf_out(feat).squeeze(-1)
        if self.cdf_mode == "softmax":
            val = torch.sigmoid((raw - 3.5) / 2.5)
        else:
            val = 1.0 - torch.exp(-F.softplus(raw))
        return val.reshape(*orig_shape)

    def _eval_stage_b(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        s_knots: torch.Tensor,
        ray_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage B: Evaluates the 3D scene HashGrid + CDF head at knot interval midpoints."""
        B, K = s_knots.shape
        s_mid = 0.5 * (s_knots[:, :-1] + s_knots[:, 1:])  # (B, K-1)
        delta_s = torch.clamp(s_knots[:, 1:] - s_knots[:, :-1], min=1e-4)  # (B, K-1)

        x_mid = ray_o.unsqueeze(1) + s_mid.unsqueeze(-1) * ray_d.unsqueeze(1)  # (B, K-1, 3)
        h_mid = self.hash_grid(x_mid)

        cdf_parts = [h_mid, x_mid / self.hash_grid.bound]
        if self.use_viewdir:
            cdf_parts.append(ray_d.unsqueeze(1).expand(-1, K - 1, -1))
        if not self.local_cdf and ray_feat is not None:
            cdf_parts.append(ray_feat.unsqueeze(1).expand(-1, K - 1, -1))

        cdf_feat = self.cdf_mlp(torch.cat(cdf_parts, dim=-1))  # (B, K-1, H)
        raw_out = self.cdf_out(cdf_feat).squeeze(-1)  # (B, K-1)

        if self.cdf_mode == "softmax":
            # Interval-width-aware softmax: adding log(delta_s) gives physical density
            # scaling across narrow vs. wide segments while avoiding exp saturation.
            logits = raw_out + torch.log(delta_s)
            escape = torch.zeros((B, 1), device=logits.device, dtype=logits.dtype)
            w = torch.softmax(torch.cat([logits, escape], dim=-1), dim=-1)[:, :-1]  # (B, K-1)
            c_knots = torch.cat(
                [torch.zeros((B, 1), device=w.device, dtype=w.dtype), torch.cumsum(w, dim=-1)],
                dim=-1,
            )
        else:
            # Physical Beer-Lambert quadrature: c = 1 - exp(-cumsum(lambda * delta_s))
            density = F.softplus(raw_out)
            cum_hazard = torch.cumsum(density * delta_s, dim=-1)
            c_intervals = 1.0 - torch.exp(-cum_hazard)
            c_knots = torch.cat(
                [torch.zeros((B, 1), device=c_intervals.device, dtype=c_intervals.dtype), c_intervals],
                dim=-1,
            )

        pooled_3d = cdf_feat.max(dim=1).values  # (B, H)
        return c_knots, pooled_3d

    def forward_single_pass(
        self,
        ray_feat: torch.Tensor,
        ray_proj_feat: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        s_knots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes one round of gap-bounded knot refinement + shifted midpoint CDF query.

        Returns:
            new_s_knots: (B, K) refined knot distances.
            c_knots: (B, K) monotonic return CDF values at new_s_knots.
            shift: (B, K) applied displacements.
            pooled_3d: (B, hidden_dim) max-pooled 3D midpoint feature for ray-drop head.
        """
        # --- Stage A: Predict interior knot shifts (endpoints are pinned at shift == 0) ---
        shift_grid = self.shift_hash_grid if self.shift_hash_grid is not None else self.hash_grid
        s_interior = s_knots[:, 1:-1]  # (B, K-2)
        x_knots = ray_o.unsqueeze(1) + s_interior.unsqueeze(-1) * ray_d.unsqueeze(1)  # (B, K-2, 3)
        h_knots = shift_grid(x_knots)
        knot_in = torch.cat([h_knots, x_knots / shift_grid.bound], dim=-1)

        # Add pre-projected ray feature (computed once per ray!)
        h = F.relu(ray_proj_feat.unsqueeze(1) + self.knot_proj(knot_in))
        for block in self.shift_blocks:
            h = F.relu(h + block(h))
        shift_logits = self.shift_head(h).squeeze(-1)  # (B, K-2)

        # Pinned endpoints + 45% nearest-neighbor gap bound (strictly monotone, no sort)
        gaps = s_knots[:, 1:] - s_knots[:, :-1]  # (B, K-1)
        room = 0.45 * torch.minimum(gaps[:, :-1], gaps[:, 1:])  # (B, K-2)
        shift = torch.zeros_like(s_knots)
        shift[:, 1:-1] = torch.tanh(shift_logits) * torch.clamp(
            room, max=self.max_displacement
        )
        new_s_knots = s_knots + shift

        # --- Stage B: Evaluate 3D HashGrid at SHIFTED interval midpoints ---
        c_knots, pooled_3d = self._eval_stage_b(ray_o, ray_d, new_s_knots, ray_feat=ray_feat)
        return new_s_knots, c_knots, shift, pooled_3d

    def forward(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Knot placement (UDF-guided or multi-round self-conditioned) + 3D-grounded ray-drop."""
        if self.knot_mode == "udf":
            assert self.udf_grid is not None
            s_knots = self.udf_grid.place_knots(
                ray_o=ray_o,
                ray_d=ray_d,
                num_knots=self.num_knots,
                s_min=self.s_min,
                s_max=self.s_max,
                active_frac=self.udf_active_frac,
                jitter=self.training,
            )
            c_knots, pooled_3d = self._eval_stage_b(ray_o, ray_d, s_knots)
            c_max = c_knots[:, -1:]
            p_hit = torch.sigmoid(self.ray_drop_head(torch.cat([pooled_3d, c_max], dim=-1)))
            return s_knots, c_knots, [(s_knots, c_knots)], p_hit

        B = ray_o.shape[0]

        # 1. Origin-invariant Plücker ray encoding (d, m = o x d), computed ONCE per ray
        ray_m = torch.cross(ray_o, ray_d, dim=-1)
        ray_feat = F.relu(self.ray_in(self.pos_enc_ray(torch.cat([ray_d, ray_m], dim=-1))))
        ray_proj_feat = self.ray_proj(ray_feat)  # Projected ONCE, not K*R times

        # 2. Initialize uniform knots across [s_min, s_max]
        s_knots = torch.linspace(self.s_min, self.s_max, self.num_knots, device=ray_o.device)
        s_knots = s_knots.unsqueeze(0).expand(B, -1)

        outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pooled_3d = None
        for _ in range(self.num_rounds):
            s_knots, c_knots, _, pooled_3d = self.forward_single_pass(
                ray_feat, ray_proj_feat, ray_o, ray_d, s_knots
            )
            outputs.append((s_knots, c_knots))
            s_knots = s_knots.detach()

        # 3. 3D-grounded ray-drop prediction using refined 3D features + c_max + ray_feat
        c_max = c_knots[:, -1:]
        drop_in = torch.cat([pooled_3d, ray_feat, c_max], dim=-1)
        p_hit = torch.sigmoid(self.ray_drop_head(drop_in))  # (B, 1)

        return s_knots, c_knots, outputs, p_hit


# ==============================================================================
# 3. Fast Sync-Free Loss & Training Loop
# ==============================================================================

def compute_hash_sck_loss(
    round_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    hit_range: torch.Tensor,
    p_hit: Optional[torch.Tensor] = None,
    hit_mask: Optional[torch.Tensor] = None,
    weights: Optional[List[float]] = None,
    lambda_raydrop: float = 0.1,
    lambda_sky_cdf: float = 0.1,
    s_max: float = 15.0,
) -> torch.Tensor:
    """Zero-sync exact grid-free Root-CRPS loss + sky empty-space carving + ray-drop BCE.

    Eliminates all per-batch .item() CPU-GPU synchronization stalls.
    In addition to supervising hitting rays with exact closed-form Root-CRPS,
    optionally supervises non-returning (sky) rays toward C(t) = 0 everywhere
    (lambda_sky_cdf > 0) for direct 3D empty-space carving.
    """
    num_r = len(round_outputs)
    if weights is None:
        weights = [0.5 ** (num_r - 1 - i) for i in range(num_r)]

    device = round_outputs[0][0].device
    r = hit_range.view(-1)

    if hit_mask is not None:
        mask = hit_mask.view(-1)
        n_hits = torch.clamp(mask.sum(), min=1.0)
        sky_mask = 1.0 - mask
        n_sky = torch.clamp(sky_mask.sum(), min=1.0)
    else:
        mask = None
        n_hits = None
        sky_mask = None
        n_sky = None

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    eps = 1e-6

    for (s_round, c_round), w in zip(round_outputs, weights):
        _, l2_per_ray = exact_continuous_cdf_loss(s_round, c_round, r, s_max=s_max)
        if mask is not None:
            l2_round = (l2_per_ray * mask).sum() / n_hits
        else:
            l2_round = l2_per_ray.mean()

        total_loss = total_loss + w * torch.sqrt(l2_round + eps)

        # Free-space carving on non-returning (sky) rays: C(t) should remain 0 along ray
        if sky_mask is not None and lambda_sky_cdf > 0.0:
            # Exact L2 integral of piecewise-linear C(t)^2 over [s_0, s_{K-1}]
            ds = s_round[:, 1:] - s_round[:, :-1]
            c_l = c_round[:, :-1]
            c_r = c_round[:, 1:]
            sky_l2_per_ray = torch.sum((ds / 3.0) * (c_l * c_l + c_l * c_r + c_r * c_r), dim=-1)
            sky_l2 = (sky_l2_per_ray * sky_mask).sum() / n_sky
            total_loss = total_loss + w * lambda_sky_cdf * torch.sqrt(sky_l2 + eps)

    if p_hit is not None and mask is not None and lambda_raydrop > 0.0:
        loss_raydrop = F.binary_cross_entropy(p_hit.view(-1), mask)
        total_loss = total_loss + lambda_raydrop * loss_raydrop

    return total_loss


def compute_3d_curvature_loss(
    model: torch.nn.Module,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    hit_range: torch.Tensor,
    hit_mask: Optional[torch.Tensor] = None,
    num_samples: int = 2048,
    eps: float = 0.05,
    s_max: float = 15.0,
) -> torch.Tensor:
    """Neuralangelo-style RMS 2nd-order finite-difference 3D curvature (Laplacian) regularizer.

    Samples 3D points near observed hit surfaces (75%) and along observed ray interiors (25%),
    evaluates f(x + δ), f(x), f(x - δ) along random unit 3D directions in one batched pass,
    and returns sqrt(mean((f(x + δ) - 2 f(x) + f(x - δ))^2) + 1e-6) so its scale matches
    Root-CRPS instead of vanishing quadratically on small ripples. Because the second derivative
    vanishes on planar ramps, it flattens noisy hash-grid bumps/cauliflower ripples while
    preserving sharp planar walls, roofs, and trunks. Zero CPU-GPU syncs.
    """
    orig = getattr(model, "_orig_mod", model)
    B = ray_o.shape[0]
    device = ray_o.device
    n = min(int(num_samples), B)
    if n <= 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    idx = torch.randint(0, B, (n,), device=device)
    o_s = ray_o[idx]
    d_s = ray_d[idx]
    r_s = hit_range.view(-1)[idx].clamp(0.1, s_max)
    m_s = hit_mask.view(-1)[idx] if hit_mask is not None else torch.ones((n,), device=device)

    n_surf = int(0.75 * n)
    # 75% near surfaces (jittered around hit points on hitting rays; random along ray on sky rays)
    u_sky = torch.rand((n_surf,), device=device)
    t_surf = torch.where(m_s[:n_surf] > 0.5, r_s[:n_surf], 0.1 + u_sky * (s_max - 0.1))
    x_surf = o_s[:n_surf] + t_surf.unsqueeze(-1) * d_s[:n_surf]
    x_surf = x_surf + torch.randn_like(x_surf) * (2.5 * eps)

    # 25% along observed ray interiors [0.1, t_end] to suppress free-space speckles
    n_free = n - n_surf
    t_end = torch.where(m_s[n_surf:] > 0.5, r_s[n_surf:], torch.full((n_free,), s_max, device=device))
    t_free = 0.1 + torch.rand((n_free,), device=device) * torch.clamp(t_end - 0.1, min=0.05)
    x_free = o_s[n_surf:] + t_free.unsqueeze(-1) * d_s[n_surf:]

    x_mid = torch.cat([x_surf, x_free], dim=0)  # (n, 3)

    # Random unit 3D perturbation direction
    u_dir = torch.randn_like(x_mid)
    u_dir = u_dir / torch.clamp(torch.linalg.norm(u_dir, dim=-1, keepdim=True), min=1e-6)
    delta = float(eps) * u_dir

    pts = torch.cat([x_mid, x_mid + delta, x_mid - delta], dim=0)  # (3n, 3)
    vals = orig.eval_local_3d_field(pts).view(3, n)
    lap = vals[1] - 2.0 * vals[0] + vals[2]
    return torch.sqrt((lap * lap).mean() + 1e-6)


def compute_fine_hash_l2_loss(model: torch.nn.Module) -> torch.Tensor:
    """Wrapper calling HashSCKNetwork.fine_hash_l2_loss() safely across torch.compile."""
    orig = getattr(model, "_orig_mod", model)
    return orig.fine_hash_l2_loss()


def train_hash_sck_model(
    model: HashSCKNetwork,
    dataloader: GPURayBatcher,
    num_epochs: int = 20,
    lr_mlp: float = 1e-3,
    lr_hash: float = 1e-2,
    lr_shift_hash: Optional[float] = 2e-3,
    use_scheduler: bool = True,
    augment_origin_slide: bool = True,
    lambda_raydrop: float = 0.1,
    lambda_sky_cdf: float = 0.1,
    lambda_curv: Optional[float] = None,
    curv_samples: Optional[int] = None,
    curv_eps_start: Optional[float] = None,
    curv_eps_end: Optional[float] = None,
    coarse_to_fine: Optional[bool] = None,
    ctf_start_levels: Optional[int] = None,
    ctf_end_frac: Optional[float] = None,
    lambda_hash_l2: Optional[float] = None,
    reg_every: Optional[int] = None,
    s_max: Optional[float] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, Any]:
    """High-throughput sync-free training loop for HashSCKNetwork.

    Uses separate hash-table and MLP learning rates, fused Adam (when on CUDA),
    optional proportional cosine LR decay, progressive coarse-to-fine hash level
    masking, 3D curvature (Laplacian) smoothness regularization, fine-level hash
    L2 decay, and GPU-resident loss accumulation (only a single CPU-GPU sync per
    epoch). Regularization arguments default to the attributes configured on `model`.
    """
    orig = getattr(model, "_orig_mod", model)
    if device is None:
        device = next(orig.parameters()).device
    if s_max is None:
        s_max = orig.s_max
    if lambda_curv is None:
        lambda_curv = orig.lambda_curv
    if curv_samples is None:
        curv_samples = orig.curv_samples
    if curv_eps_start is None:
        curv_eps_start = orig.curv_eps_start
    if curv_eps_end is None:
        curv_eps_end = orig.curv_eps_end
    if coarse_to_fine is None:
        coarse_to_fine = orig.coarse_to_fine
    if ctf_start_levels is None:
        ctf_start_levels = orig.ctf_start_levels
    if ctf_end_frac is None:
        ctf_end_frac = orig.ctf_end_frac
    if lambda_hash_l2 is None:
        lambda_hash_l2 = orig.lambda_hash_l2
    reg_every_n = max(1, int(orig.reg_every if reg_every is None else reg_every))

    fused_ok = device.type == "cuda"
    optimizer = torch.optim.Adam(
        orig.parameter_groups(lr_mlp=lr_mlp, lr_hash=lr_hash, lr_shift_hash=lr_shift_hash),
        eps=1e-15,
        fused=fused_ok,
    )
    t_max = max(1, num_epochs)
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, ep / t_max))),
        )
        if use_scheduler
        else None
    )

    loss_history: List[float] = []
    epoch_times: List[float] = []
    dataset_len = dataloader.N if hasattr(dataloader, "N") else len(dataloader)
    steps_per_epoch = max(1, len(dataloader))
    total_steps = max(1, num_epochs * steps_per_epoch)
    ctf_steps = max(1.0, float(ctf_end_frac) * total_steps)

    if not coarse_to_fine:
        orig.hash_grid.set_active_levels(orig.hash_grid.n_levels)

    # Auto-build coarse UDF from GPU dataset rays if not already populated
    if orig.knot_mode == "udf" and orig.udf_grid is not None and not orig.udf_grid._has_data:
        if hasattr(dataloader, "rays_o") and hasattr(dataloader, "hit_ranges") and dataloader.hit_ranges is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_udf = time.perf_counter()
            occ_frac = orig.udf_grid.build_from_rays(
                dataloader.rays_o, dataloader.rays_d, dataloader.hit_ranges, dataloader.hit_masks
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            if verbose:
                print(
                    f"Built CoarseUDFGrid in {(time.perf_counter() - t_udf) * 1000.0:.2f} ms "
                    f"({occ_frac * 100.0:.2f}% occupied voxels)"
                )

    if verbose:
        print(
            f"Training HashSCKNetwork on {device} "
            f"({dataset_len:,} rays, {steps_per_epoch} batches/epoch, "
            f"origin_slide={augment_origin_slide}, sky_carving={lambda_sky_cdf}, "
            f"coarse_to_fine={coarse_to_fine}, lambda_curv={lambda_curv}, "
            f"lambda_hash_l2={lambda_hash_l2}, reg_every={reg_every_n})..."
        )

    global_step = 0
    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        epoch_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
        num_batches = len(dataloader)

        for batch in dataloader:
            ctf_prog = min(1.0, global_step / ctf_steps)
            if coarse_to_fine:
                orig.set_coarse_to_fine_progress(ctf_prog, start_levels=ctf_start_levels)

            optimizer.zero_grad(set_to_none=True)
            ray_o = batch["ray_o"]
            ray_d = batch["ray_d"]
            hit_range = batch["hit_range"]
            hit_mask = batch.get("hit_mask", None)

            if augment_origin_slide:
                ray_o, hit_range, _ = vectorized_gpu_origin_slide(
                    ray_o=ray_o,
                    ray_d=ray_d,
                    hit_ranges=hit_range,
                    hit_masks=hit_mask,
                )

            _, _, round_outputs, p_hit = model(ray_o, ray_d)
            loss = compute_hash_sck_loss(
                round_outputs=round_outputs,
                hit_range=hit_range,
                p_hit=p_hit,
                hit_mask=hit_mask,
                lambda_raydrop=lambda_raydrop,
                lambda_sky_cdf=lambda_sky_cdf,
                s_max=s_max,
            )
            if (global_step % reg_every_n) == 0:
                reg_scale = float(reg_every_n)
                if lambda_curv > 0.0:
                    curv_eps = curv_eps_start + ctf_prog * (curv_eps_end - curv_eps_start)
                    loss_curv = compute_3d_curvature_loss(
                        model=orig,
                        ray_o=ray_o,
                        ray_d=ray_d,
                        hit_range=hit_range,
                        hit_mask=hit_mask,
                        num_samples=curv_samples,
                        eps=curv_eps,
                        s_max=s_max,
                    )
                    loss = loss + (reg_scale * lambda_curv) * loss_curv
                if lambda_hash_l2 > 0.0:
                    loss = loss + (reg_scale * lambda_hash_l2) * orig.fine_hash_l2_loss()

            loss.backward()
            optimizer.step()
            epoch_loss_gpu += loss.detach()
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_loss = (epoch_loss_gpu / max(1, num_batches)).item()
        loss_history.append(avg_loss)
        epoch_times.append(elapsed)

        if verbose:
            rays_per_sec = dataset_len / max(elapsed, 1e-6)
            print(
                f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - "
                f"Loss: {avg_loss:.6f} | {elapsed:.2f}s ({rays_per_sec:,.0f} rays/s)"
            )

        if callback is not None:
            callback(epoch, avg_loss)

    return {
        "loss_history": loss_history,
        "epoch_times": epoch_times,
        "optimizer": optimizer,
    }
