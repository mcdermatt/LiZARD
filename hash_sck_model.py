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

        out_base = offs_m.to(tl.int64) * (L * 2) + pid_l * 2
        tl.store(out_ptr + out_base + 0, acc0, mask=mask_m)
        tl.store(out_ptr + out_base + 1, acc1, mask=mask_m)

    @triton.jit
    def _hash_grid_3d_bwd_kernel(
        x_ptr,              # (M, 3) float32 normalized in [0, 1)
        table_ptr,          # (L, T, 2) float32
        res_ptr,            # (L,) float32
        grad_out_ptr,       # (M, L * 2) float32
        grad_table_ptr,     # (L, T, 2) float32 (zero-initialized)
        grad_x_lvl_ptr,     # (M, L, 3) float32 (used only if COMPUTE_DX)
        M,
        L: tl.constexpr,
        T_MASK: tl.constexpr,
        COMPUTE_DX: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_l = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

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

        g_base = offs_m.to(tl.int64) * (L * 2) + pid_l * 2
        g0 = tl.load(grad_out_ptr + g_base + 0, mask=mask_m, other=0.0)
        g1 = tl.load(grad_out_ptr + g_base + 1, mask=mask_m, other=0.0)

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

                    tl.atomic_add(grad_table_ptr + addr + 0, w * g0, mask=mask_m)
                    tl.atomic_add(grad_table_ptr + addr + 1, w * g1, mask=mask_m)

                    if COMPUTE_DX:
                        f0 = tl.load(table_ptr + addr + 0, mask=mask_m, other=0.0)
                        f1 = tl.load(table_ptr + addr + 1, mask=mask_m, other=0.0)
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
    def forward(ctx, x_norm: torch.Tensor, tables: torch.Tensor, resolutions: torch.Tensor):
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
            x_norm, tables, resolutions, out,
            M, L=L, T_MASK=T - 1, BLOCK_M=BLOCK_M,
        )
        ctx.save_for_backward(x_norm, tables, resolutions)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_norm, tables, resolutions = ctx.saved_tensors
        grad_out = grad_out.contiguous().float()
        M = x_norm.shape[0]
        L, T, _ = tables.shape
        compute_dx = ctx.needs_input_grad[0]

        grad_tables = torch.zeros_like(tables)
        if compute_dx:
            grad_x_lvl = torch.empty((M, L, 3), device=x_norm.device, dtype=torch.float32)
        else:
            grad_x_lvl = torch.empty((1,), device=x_norm.device, dtype=torch.float32)

        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M), L)
        _hash_grid_3d_bwd_kernel[grid](
            x_norm, tables, resolutions, grad_out, grad_tables, grad_x_lvl,
            M, L=L, T_MASK=T - 1, COMPUTE_DX=compute_dx, BLOCK_M=BLOCK_M,
        )

        grad_x = grad_x_lvl.sum(dim=1) if compute_dx else None
        return grad_x, grad_tables, None


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
        bound: Half-extent of cubic scene AABB [-bound, +bound]^3 in meters (default 8.0).
        backend: 'auto' (default), 'triton', 'tcnn', or 'pytorch'.
    """

    def __init__(
        self,
        n_levels: int = 12,
        n_features: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 512,
        bound: float = 8.0,
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

    def reset_parameters(self) -> None:
        """Re-initializes hash table entries in-place (preserves torch.compile cache)."""
        with torch.no_grad():
            if self.tables is not None:
                self.tables.uniform_(-1e-4, 1e-4)
            elif self.tcnn_enc is not None:
                for p in self.tcnn_enc.parameters():
                    p.uniform_(-1e-4, 1e-4)

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
            out = self.tcnn_enc(x_norm).float()
        elif self.backend == "triton" and x_norm.is_cuda and HAS_TRITON and self.n_features == 2:
            out = _TritonHashGrid3DFn.apply(x_norm.float(), self.tables, self.resolutions)
        else:
            out = self._forward_pytorch(x_norm)

        return out.reshape(*orig_shape, self.out_dim)


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
        s_max: Far bound distance along ray in meters (default 8.0).
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
        n_levels: HashGrid3D resolution levels (default 12).
        n_features: HashGrid3D features per level (default 2).
        log2_hashmap_size: Log2 hash table entries per level (default 16).
        base_resolution: HashGrid3D coarsest resolution (default 16).
        max_resolution: HashGrid3D finest resolution (default 512).
        scene_bound: Half-extent of cubic 3D hash grid [-bound, +bound]^3 (default s_max).
        hash_backend: 'auto' (default), 'triton', 'tcnn', or 'pytorch'.
    """

    def __init__(
        self,
        num_knots: int = 32,
        s_min: float = 0.1,
        s_max: float = 8.0,
        hidden_dim: int = 64,
        num_hidden_blocks: int = 1,
        num_rounds: int = 2,
        max_displacement: float = 1.0,
        cdf_mode: str = "softmax",
        local_cdf: bool = True,
        use_viewdir: bool = False,
        n_levels: int = 12,
        n_features: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 512,
        scene_bound: Optional[float] = None,
        hash_backend: str = "auto",
    ):
        super().__init__()
        self.num_knots = num_knots
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.num_rounds = num_rounds
        self.max_displacement = float(max_displacement)
        self.cdf_mode = cdf_mode
        self.local_cdf = local_cdf
        self.use_viewdir = use_viewdir
        bound = float(scene_bound) if scene_bound is not None else float(s_max)

        # 1. Shared Multi-Resolution 3D Spatial Hash Grid
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

        # 2. Compact Origin-Invariant Plücker Ray Encoder: (d, m = o x d) in R^6
        self.pos_enc_ray = PositionalEncoding(in_dim=6, num_freqs=4, max_coord=bound)
        self.ray_in = nn.Sequential(
            nn.Linear(self.pos_enc_ray.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Split joint projection so ray_feat is projected ONCE per ray (not K*R times!)
        self.ray_proj = nn.Linear(hidden_dim, hidden_dim)
        self.knot_proj = nn.Linear(hash_dim + 3, hidden_dim, bias=False)

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

        # 4. Compact CDF / Hazard Head evaluated at shifted interval midpoints (what is there)
        cdf_in_dim = hash_dim + 3  # hash(x_mid) + normalized x_mid
        if use_viewdir:
            cdf_in_dim += 3
        if not local_cdf:
            cdf_in_dim += hidden_dim

        self.cdf_mlp = nn.Sequential(
            nn.Linear(cdf_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cdf_out = nn.Linear(hidden_dim, 1)

        # 5. 3D-Grounded Ray-Drop Head: pools refined 3D midpoint features + c_max + ray_feat
        self.ray_drop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Clean initial biases
        self._init_custom_biases()

        hash_params = sum(p.numel() for p in self.hash_grid.parameters())
        mlp_params = sum(p.numel() for p in self.parameters()) - hash_params
        print(
            f"HashSCKNetwork [{self.hash_grid.backend}] | "
            f"MLP params: {mlp_params:,} | Hash table params: {hash_params:,} "
            f"({hash_params * 4 / 1e6:.2f} MB)"
        )

    def _init_custom_biases(self) -> None:
        with torch.no_grad():
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
        self, lr_mlp: float = 1e-3, lr_hash: float = 1e-2, weight_decay_mlp: float = 1e-6
    ) -> List[Dict[str, Any]]:
        """Returns Instant-NGP style optimizer param groups (higher LR for hash table)."""
        hash_param_ids = {id(p) for p in self.hash_grid.parameters()}
        hash_params = [p for p in self.parameters() if id(p) in hash_param_ids]
        mlp_params = [p for p in self.parameters() if id(p) not in hash_param_ids]
        return [
            {"params": hash_params, "lr": lr_hash, "weight_decay": 0.0},
            {"params": mlp_params, "lr": lr_mlp, "weight_decay": weight_decay_mlp},
        ]

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
        B, K = s_knots.shape

        # --- Stage A: Predict interior knot shifts at current knot positions ---
        x_knots = ray_o.unsqueeze(1) + s_knots.unsqueeze(-1) * ray_d.unsqueeze(1)  # (B, K, 3)
        h_knots = self.hash_grid(x_knots)
        knot_in = torch.cat([h_knots, x_knots / self.hash_grid.bound], dim=-1)

        # Add pre-projected ray feature (computed once per ray!)
        h = F.relu(ray_proj_feat.unsqueeze(1) + self.knot_proj(knot_in))
        for block in self.shift_blocks:
            h = F.relu(h + block(h))
        shift_logits = self.shift_head(h).squeeze(-1)  # (B, K)

        # Pinned endpoints + 45% nearest-neighbor gap bound (strictly monotone, no sort)
        gaps = s_knots[:, 1:] - s_knots[:, :-1]  # (B, K-1)
        room = 0.45 * torch.minimum(gaps[:, :-1], gaps[:, 1:])  # (B, K-2)
        shift = torch.zeros_like(s_knots)
        shift[:, 1:-1] = torch.tanh(shift_logits[:, 1:-1]) * torch.clamp(
            room, max=self.max_displacement
        )
        new_s_knots = s_knots + shift

        # --- Stage B: Evaluate 3D HashGrid at SHIFTED interval midpoints ---
        s_mid = 0.5 * (new_s_knots[:, :-1] + new_s_knots[:, 1:])  # (B, K-1)
        delta_s = torch.clamp(new_s_knots[:, 1:] - new_s_knots[:, :-1], min=1e-4)  # (B, K-1)

        x_mid = ray_o.unsqueeze(1) + s_mid.unsqueeze(-1) * ray_d.unsqueeze(1)  # (B, K-1, 3)
        h_mid = self.hash_grid(x_mid)

        cdf_parts = [h_mid, x_mid / self.hash_grid.bound]
        if self.use_viewdir:
            cdf_parts.append(ray_d.unsqueeze(1).expand(-1, K - 1, -1))
        if not self.local_cdf:
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
        return new_s_knots, c_knots, shift, pooled_3d

    def forward(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Multi-round self-conditioned knot estimation + 3D-grounded ray-drop prediction."""
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
    s_max: float = 8.0,
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


def train_hash_sck_model(
    model: HashSCKNetwork,
    dataloader: GPURayBatcher,
    num_epochs: int = 20,
    lr_mlp: float = 1e-3,
    lr_hash: float = 1e-2,
    use_scheduler: bool = True,
    augment_origin_slide: bool = True,
    lambda_raydrop: float = 0.1,
    lambda_sky_cdf: float = 0.1,
    s_max: Optional[float] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> Dict[str, Any]:
    """High-throughput sync-free training loop for HashSCKNetwork.

    Uses separate hash-table and MLP learning rates, fused Adam (when on CUDA),
    optional CosineAnnealingLR, and GPU-resident loss accumulation (only a single
    CPU-GPU sync per epoch).
    """
    if device is None:
        device = next(model.parameters()).device
    if s_max is None:
        s_max = model.s_max

    fused_ok = device.type == "cuda"
    optimizer = torch.optim.Adam(
        model.parameter_groups(lr_mlp=lr_mlp, lr_hash=lr_hash),
        eps=1e-15,
        fused=fused_ok,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_mlp * 0.05)
        if use_scheduler
        else None
    )

    loss_history: List[float] = []
    epoch_times: List[float] = []
    dataset_len = dataloader.N if hasattr(dataloader, "N") else len(dataloader)

    if verbose:
        print(
            f"Training HashSCKNetwork on {device} "
            f"({dataset_len:,} rays, {len(dataloader)} batches/epoch, "
            f"origin_slide={augment_origin_slide}, sky_carving={lambda_sky_cdf})..."
        )

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        epoch_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
        num_batches = len(dataloader)

        for batch in dataloader:
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
            loss.backward()
            optimizer.step()
            epoch_loss_gpu += loss.detach()

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
