"""
sck_model.py
============
Self-Conditioned Knot Network (SCK) for Continuous LiDAR Hazard Fields & Return CDFs.

This module provides:
  1. Network Structure:
     - PositionalEncoding: Scene-scaled vectorized sinusoidal coordinate encoder.
     - DeeperSCKNetwork (alias SCKNetwork): Recurrent / self-conditioned knot network
       with residual knot flow displacements, volumetric density heads, and a dedicated
       ray-drop classification head (PLiNK architecture).
  2. Loss Calculation:
     - interpolate_cdf_to_grid: Piecewise linear CDF evaluation on a fixed query grid (F.pad).
     - compute_sck_loss / SCKLoss: PLiNK-style masked continuous CDF loss (evaluated strictly
       on hitting rays) + binary cross-entropy ray-drop loss + optional L1 Wasserstein term.
  3. Dataset & Fast GPU Batcher:
     - LidarRayDataset: PyTorch Dataset for rays with GPU support and hit_mask tracking.
     - GPURayBatcher: Ultra-fast zero-copy batcher for rays stored directly in GPU memory.
     - vectorized_gpu_origin_slide: Fully vectorized GPU origin-sliding augmentation.
  4. Training Loop:
     - train_step: Single forward-backward optimization step.
     - train_epoch: Full pass over a DataLoader or GPURayBatcher.
     - train_model: High-level multi-epoch training loop with loss tracking.
  5. Inference:
     - generate_point_cloud_from_network: Inverts predicted continuous CDF to extract 3D points
       with PLiNK ray-drop filtering to cleanly suppress sky and non-return floaters.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from synthetic_data_generation_utils import PointCloud
except ImportError:
    try:
        from .synthetic_data_generation_utils import PointCloud
    except Exception:
        PointCloud = None  # Handled dynamically in generate_point_cloud_from_network


# ==============================================================================
# 1. Network Structure
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Scene-scaled sinusoidal positional encoding.

    Vectorized across all frequency bands in a single tensor operation to eliminate
    Python loops and minimize CUDA kernel launch overhead.

    Args:
        in_dim: Input dimension of coordinate tensor (e.g., 3 for 3D points, 9 for ray + Plücker).
        num_freqs: Number of octave frequency bands.
        max_coord: Maximum scene coordinate extent for normalization.
    """

    def __init__(self, in_dim: int, num_freqs: int = 8, max_coord: float = 15.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_coord = max_coord

        # Frequencies from 2^0 to 2^(num_freqs - 1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim + in_dim * 2 * num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize into [-1, 1] relative to scene scale
        x_norm = x / self.max_coord

        # Vectorized angles across all frequencies simultaneously: shape (*x.shape, num_freqs)
        angles = x_norm.unsqueeze(-1) * (self.freqs * math.pi)
        sins = torch.sin(angles)
        coss = torch.cos(angles)

        # Stack sin and cos and flatten to match frequency-interleaved layout
        # Shape: (*x.shape, num_freqs, 2, in_dim) -> flatten to (*x.shape, num_freqs * 2 * in_dim)
        sc = torch.stack([sins, coss], dim=-1)
        sc_flat = sc.permute(*range(x.ndim - 1), -2, -1, -3).flatten(-3)

        return torch.cat([x_norm, sc_flat], dim=-1)


class DeeperSCKNetwork(nn.Module):
    """Self-Conditioned Knot Network with Residual Knot Flow and Ray-Drop Head.

    Combines:
      - Continuous self-conditioned residual knot flow for surface distance estimation
      - PLiNK-style dedicated ray-drop head p_hit in [0, 1] to cleanly classify and
        prune rays that escape into open sky without hitting obstacles.

    Args:
        num_knots: Number of candidate knots K placed along each ray.
        s_min: Near bound distance along the ray (meters).
        s_max: Far bound distance along the ray (meters).
        hidden_dim: Feature dimension of internal layers.
        num_hidden_blocks: Number of residual MLP blocks in decoder trunk.
        num_rounds: Number of self-conditioning refinement iterations R.
        max_displacement: Maximum knot shift distance per round (meters).
    """

    def __init__(
        self,
        num_knots: int = 32,
        s_min: float = 0.1,
        s_max: float = 15.0,
        hidden_dim: int = 256,
        num_hidden_blocks: int = 3,
        num_rounds: int = 2,
        max_displacement: float = 1.0,
    ):
        super().__init__()
        self.num_knots = num_knots
        self.s_min = s_min
        self.s_max = s_max
        self.num_rounds = num_rounds
        self.max_displacement = max_displacement

        # 1. Positional encoders for high-frequency 3D coordinates and rays
        self.pos_enc_x = PositionalEncoding(in_dim=3, num_freqs=10, max_coord=s_max)
        self.pos_enc_ray = PositionalEncoding(in_dim=9, num_freqs=10, max_coord=s_max)

        self.ray_in = nn.Sequential(
            nn.Linear(self.pos_enc_ray.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.knot_in = nn.Sequential(
            nn.Linear(self.pos_enc_x.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Deeper residual decoder trunk
        self.joint_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_hidden_blocks)
        ])

        # Channel 0: Knot displacement shift (residual flow)
        # Channel 1: Local volumetric density (hazard)
        self.out_head = nn.Linear(hidden_dim, 2)

        # 3. Dedicated Ray-Drop Classification Head (PLiNK architecture)
        # Evaluates global ray feature to predict p_hit in [0, 1]
        self.ray_drop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initial biases
        with torch.no_grad():
            self.out_head.bias[0].fill_(0.0)
            self.out_head.bias[1].fill_(-4.0)
            self.ray_drop_head[-1].bias.fill_(2.0)  # Initial prior: ~88% hit rate

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total model parameters: {total_params:,}")

    def forward_single_pass(
        self,
        ray_feat: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        s_knots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Executes a single round of knot refinement and hazard integration.

        Args:
            ray_feat: Precomputed ray features, shape (B, hidden_dim).
            ray_o: Ray origins, shape (B, 3).
            ray_d: Ray directions, shape (B, 3).
            s_knots: Current knot distances along rays, shape (B, K).

        Returns:
            new_s_knots: Sorted refined knot distances, shape (B, K).
            c_knots: Monotonic cumulative return CDF values at knots, shape (B, K).
            shift: Predicted local displacements, shape (B, K).
            density: Local volumetric densities, shape (B, K).
        """
        B, K = s_knots.shape

        # 3D Knot position encoding evaluated at CURRENT knot coordinates
        x_knots = ray_o.unsqueeze(1) + s_knots.unsqueeze(-1) * ray_d.unsqueeze(1)
        knot_feat = F.relu(self.knot_in(self.pos_enc_x(x_knots)))

        # Joint features through residual trunk
        ray_feat_expanded = ray_feat.unsqueeze(1).expand(-1, K, -1)
        h = F.relu(self.joint_in(torch.cat([ray_feat_expanded, knot_feat], dim=-1)))

        for block in self.res_blocks:
            h = F.relu(h + block(h))

        out = self.out_head(h)

        # 1. Residual Knot Displacement (Flow): nudge candidate knots locally
        shift = torch.tanh(out[..., 0]) * self.max_displacement
        shifted_s = s_knots + shift
        shifted_s = torch.clamp(shifted_s, self.s_min, self.s_max)

        # Maintain monotonic ordering along ray
        new_s_knots, sort_indices = torch.sort(shifted_s, dim=-1)

        # 2. Compute hazard intervals between sorted knots
        # Permute density with sort_indices so densities track knots if crossings occur
        raw_density = F.softplus(out[..., 1])
        density = torch.gather(raw_density, -1, sort_indices)

        s_boundary = torch.full((B, 1), self.s_min, device=ray_o.device)
        delta_s = new_s_knots - torch.cat([s_boundary, new_s_knots[:, :-1]], dim=-1)
        delta_s = torch.clamp(delta_s, min=1e-4)

        hazard_increments = density * delta_s
        cum_hazard = torch.cumsum(hazard_increments, dim=-1)
        c_knots = 1.0 - torch.exp(-cum_hazard)

        return new_s_knots, c_knots, shift, density

    def forward(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Performs iterative multi-round self-conditioned knot estimation + ray-drop prediction.

        Args:
            ray_o: Ray origins, shape (B, 3).
            ray_d: Ray directions, shape (B, 3).

        Returns:
            s_final: Final refined knot distances, shape (B, K).
            c_final: Final cumulative return CDF values at knots, shape (B, K).
            round_outputs: List of (s_knots, c_knots) tuples for each refinement round.
            p_hit: Predicted hit probability in [0, 1], shape (B, 1).
        """
        B = ray_o.shape[0]

        # 1. Global Ray encoding (ray origin, direction, and Plücker moment m = o x d)
        ray_m = torch.cross(ray_o, ray_d, dim=-1)
        ray_input = torch.cat([ray_o, ray_d, ray_m], dim=-1)
        ray_feat = F.relu(self.ray_in(self.pos_enc_ray(ray_input)))

        # 2. Ray-Drop prediction (PLiNK style)
        p_hit = torch.sigmoid(self.ray_drop_head(ray_feat))  # (B, 1)

        # 3. Start Round 0 with clean, uniformly spaced candidate knots across [s_min, s_max]
        s_knots = torch.linspace(self.s_min, self.s_max, self.num_knots, device=ray_o.device)
        s_knots = s_knots.unsqueeze(0).expand(B, -1)

        outputs = []
        for r in range(self.num_rounds):
            s_knots, c_knots, shift, density = self.forward_single_pass(
                ray_feat, ray_o, ray_d, s_knots
            )
            outputs.append((s_knots, c_knots))
            # Feed refined knot positions into next round of self-conditioning (stop gradient)
            s_knots = s_knots.detach()

        return s_knots, c_knots, outputs, p_hit


# Convenient class alias
SCKNetwork = DeeperSCKNetwork


# ==============================================================================
# 2. Loss Calculation
# ==============================================================================

def interpolate_cdf_to_grid(
    s_knots: torch.Tensor,
    c_knots: torch.Tensor,
    eval_grid: torch.Tensor,
) -> torch.Tensor:
    """Interpolates predicted piecewise-linear CDF (s_knots, c_knots) onto a fixed query grid.

    Optimized using F.pad for zero-allocation boundary padding.

    Args:
        s_knots: Predicted knot range distances along ray, shape (B, K).
        c_knots: Cumulative return probabilities at knots, shape (B, K).
        eval_grid: 1D tensor of query distances, shape (M,).

    Returns:
        (B, M) interpolated CDF values evaluated at eval_grid.
    """
    B, K = s_knots.shape
    query = eval_grid.unsqueeze(0).expand(B, -1)

    # Prepend boundary knot (s=0, C=0) using F.pad
    s_ext = F.pad(s_knots, (1, 0), value=0.0)  # (B, K+1)
    c_ext = F.pad(c_knots, (1, 0), value=0.0)  # (B, K+1)

    # Find bin indices using searchsorted
    idx = torch.searchsorted(s_ext.contiguous(), query.contiguous(), right=True)
    idx = torch.clamp(idx, 1, K)

    # Gather adjacent control points
    idx_left = idx - 1
    s_0 = torch.gather(s_ext, 1, idx_left)
    s_1 = torch.gather(s_ext, 1, idx)
    c_0 = torch.gather(c_ext, 1, idx_left)
    c_1 = torch.gather(c_ext, 1, idx)

    # Linear interpolation weight
    denom = torch.clamp(s_1 - s_0, min=1e-5)
    alpha = torch.clamp((query - s_0) / denom, 0.0, 1.0)

    c_interp = c_0 + alpha * (c_1 - c_0)
    return c_interp


def exact_continuous_cdf_loss(
    s_knots: torch.Tensor,
    c_knots: torch.Tensor,
    hit_range: torch.Tensor,
    s_max: float = 15.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes exact, closed-form continuous L1 (Wasserstein-1) and L2 (CRPS) loss.

    Evaluates the continuous integral against the exact step-function target 1_{t >= r*}:
        L_1 = \\int_0^{s_max} |F_spline(t) - 1_{t >= r*}| dt
        L_2 = \\int_0^{s_max} (F_spline(t) - 1_{t >= r*})^2 dt
    analytically across trapezoids without any discrete evaluation grid or discretization error.

    Args:
        s_knots: Refined knot distances along rays, shape (B, K).
        c_knots: Monotonic return CDF values at knots, shape (B, K).
        hit_range: Ground truth return range r*, shape (B,) or (B, 1).
        s_max: Maximum scene range.

    Returns:
        loss_l1: Exact continuous L1 error per ray, shape (B,).
        loss_l2: Exact continuous L2 error per ray, shape (B,).
    """
    B, K = s_knots.shape
    device = s_knots.device

    # Boundary knots: (0, 0) and (s_max, c_max)
    s_0 = torch.zeros(B, 1, device=device)
    c_0 = torch.zeros(B, 1, device=device)
    s_end = torch.full((B, 1), s_max, device=device)
    c_end = c_knots[:, -1:]

    s_all = torch.cat([s_0, s_knots, s_end], dim=-1)  # (B, K+2)
    c_all = torch.cat([c_0, c_knots, c_end], dim=-1)  # (B, K+2)

    r = hit_range.view(B, 1).clamp(0.0, s_max)

    # Insert r* into the knot sequence
    idx = torch.searchsorted(s_all.contiguous(), r.contiguous(), right=True)
    idx = torch.clamp(idx, 1, K + 1)

    s_left = torch.gather(s_all, 1, idx - 1)
    s_right = torch.gather(s_all, 1, idx)
    c_left = torch.gather(c_all, 1, idx - 1)
    c_right = torch.gather(c_all, 1, idx)

    denom = torch.clamp(s_right - s_left, min=1e-5)
    alpha = torch.clamp((r - s_left) / denom, 0.0, 1.0)
    c_r = c_left + alpha * (c_right - c_left)

    s_aug = torch.cat([s_all, r], dim=-1)
    c_aug = torch.cat([c_all, c_r], dim=-1)

    s_sorted, sort_idx = torch.sort(s_aug, dim=-1)
    c_sorted = torch.gather(c_aug, -1, sort_idx)

    s_l = s_sorted[:, :-1]
    s_r = s_sorted[:, 1:]
    c_l = c_sorted[:, :-1]
    c_r = c_sorted[:, 1:]

    ds = s_r - s_l
    mid_s = 0.5 * (s_l + s_r)
    is_after = (mid_s >= r).float()

    # Exact L1 area (Wasserstein-1 / Earth Mover's Distance)
    area_before = ds * 0.5 * (c_l + c_r)
    area_after = ds * 0.5 * ((1.0 - c_l) + (1.0 - c_r))
    loss_l1 = torch.sum((1.0 - is_after) * area_before + is_after * area_after, dim=-1)

    # Exact L2 integral (Continuous Ranked Probability Score / CRPS)
    e_l = 1.0 - c_l
    e_r = 1.0 - c_r
    l2_before = (ds / 3.0) * (c_l**2 + c_l * c_r + c_r**2)
    l2_after = (ds / 3.0) * (e_l**2 + e_l * e_r + e_r**2)
    loss_l2 = torch.sum((1.0 - is_after) * l2_before + is_after * l2_after, dim=-1)

    return loss_l1, loss_l2


def compute_sck_loss(
    round_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    hit_range: Optional[torch.Tensor] = None,
    gt_cdf: Optional[torch.Tensor] = None,
    eval_grid: Optional[torch.Tensor] = None,
    p_hit: Optional[torch.Tensor] = None,
    hit_mask: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
    weights: Optional[List[float]] = None,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    lambda_raydrop: float = 1.0,
    s_max: float = 15.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes continuous CDF loss with PLiNK-style ray-drop masking and BCE classification.

    Supports two modes:
      1. Exact Grid-Free Mode (Recommended):
         When hit_range (r*) is provided, computes exact closed-form Wasserstein-1 and
         CRPS integrals via trapezoids with ZERO discretization error. No eval_grid needed!
      2. Discrete Grid Mode (Fallback):
         When gt_cdf and eval_grid are provided without hit_range, resamples predicted
         knots onto eval_grid.

    Args:
        round_outputs: List of (s_round, c_round) tuples from model forward pass.
        hit_range: Ground truth return range r*, shape (B,) or (B, 1).
        gt_cdf: Optional ground truth return CDF on eval_grid, shape (B, M).
        eval_grid: Optional evaluation distances, shape (M,).
        p_hit: Predicted hit probability from ray_drop_head, shape (B, 1) or (B,).
        hit_mask: Ground truth hit mask (1.0 = hit, 0.0 = ray-drop/sky), shape (B,) or (B, 1).
        dt: Grid step size (only used in discrete grid fallback mode).
        weights: Optional custom list of weights for each self-conditioning round.
        l1_weight: Weight for Wasserstein-1 L1 term.
        l2_weight: Weight for continuous L2 / CRPS term.
        lambda_raydrop: Weight for the binary cross-entropy ray-drop loss.
        s_max: Far bound for scene ray distances (meters).

    Returns:
        total_loss: Differentiable scalar loss tensor.
        metrics: Dictionary with individual round losses, ray-drop loss, and total loss.
    """
    num_r = len(round_outputs)
    if weights is None:
        weights = [0.5 ** (num_r - 1 - i) for i in range(num_r)]

    B = round_outputs[0][0].shape[0]
    device = round_outputs[0][0].device

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    metrics: Dict[str, float] = {}

    # PLiNK target_drop_mask preparation
    mask = None
    n_hits = float(B)
    if hit_mask is not None:
        mask = hit_mask.view(-1).to(device)
        n_hits = float(torch.clamp(mask.sum(), min=1.0).item())

    # Mode 1: Exact Closed-Form Grid-Free Analytic Loss
    if hit_range is not None:
        r = hit_range.view(-1).to(device)
        for i, ((s_round, c_round), w) in enumerate(zip(round_outputs, weights)):
            l1_per_ray, l2_per_ray = exact_continuous_cdf_loss(
                s_round, c_round, r, s_max=s_max
            )
            if mask is not None:
                l1_round = (l1_per_ray * mask).sum() / n_hits
                l2_round = (l2_per_ray * mask).sum() / n_hits
            else:
                l1_round = l1_per_ray.mean()
                l2_round = l2_per_ray.mean()

            round_loss = l2_weight * l2_round + l1_weight * l1_round
            total_loss = total_loss + w * round_loss
            metrics[f"l1_round_{i}"] = l1_round.item()
            metrics[f"l2_round_{i}"] = l2_round.item()
            metrics[f"loss_round_{i}"] = round_loss.item()

    # Mode 2: Discrete Grid Fallback Mode
    elif gt_cdf is not None and eval_grid is not None:
        eval_grid = eval_grid.to(device)
        if dt is None:
            dt = float((eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1))

        mask_2d = mask.view(-1, 1) if mask is not None else None
        for i, ((s_round, c_round), w) in enumerate(zip(round_outputs, weights)):
            pred_grid = interpolate_cdf_to_grid(s_round, c_round, eval_grid)
            diff = pred_grid - gt_cdf

            if mask_2d is not None:
                diff = diff * mask_2d
                round_l2 = dt * (diff ** 2).sum() / n_hits
                round_l1 = dt * (torch.abs(diff)).sum() / n_hits
            else:
                round_l2 = dt * torch.sum(diff ** 2, dim=-1).mean()
                round_l1 = dt * torch.sum(torch.abs(diff), dim=-1).mean()

            round_loss = l2_weight * round_l2 + l1_weight * round_l1
            total_loss = total_loss + w * round_loss
            metrics[f"l1_round_{i}"] = round_l1.item()
            metrics[f"l2_round_{i}"] = round_l2.item()
            metrics[f"loss_round_{i}"] = round_loss.item()
    else:
        raise ValueError("Either hit_range (recommended) or (gt_cdf, eval_grid) must be provided to compute_sck_loss.")

    # PLiNK ray-drop classification loss (BCE)
    if p_hit is not None and hit_mask is not None:
        p_hit_flat = p_hit.view(-1)
        hit_mask_flat = hit_mask.view(-1).float().to(p_hit.device)
        loss_raydrop = F.binary_cross_entropy(p_hit_flat, hit_mask_flat)
        total_loss = total_loss + lambda_raydrop * loss_raydrop
        metrics["loss_raydrop"] = loss_raydrop.item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


class SCKLoss(nn.Module):
    """PyTorch Module wrapper for the multi-round Continuous CDF loss and Ray-Drop BCE.
    
    Supports both exact grid-free analytic loss (passing hit_range) and discrete grid loss.
    """

    def __init__(
        self,
        eval_grid: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        lambda_raydrop: float = 1.0,
        s_max: float = 15.0,
    ):
        super().__init__()
        if eval_grid is not None:
            self.register_buffer("eval_grid", eval_grid)
            self.dt = dt if dt is not None else float((eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1))
        else:
            self.eval_grid = None
            self.dt = None
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.lambda_raydrop = lambda_raydrop
        self.s_max = s_max

    def forward(
        self,
        round_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        hit_range: Optional[torch.Tensor] = None,
        gt_cdf: Optional[torch.Tensor] = None,
        p_hit: Optional[torch.Tensor] = None,
        hit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss, _ = compute_sck_loss(
            round_outputs=round_outputs,
            hit_range=hit_range,
            gt_cdf=gt_cdf,
            eval_grid=self.eval_grid,
            p_hit=p_hit,
            hit_mask=hit_mask,
            dt=self.dt,
            l1_weight=self.l1_weight,
            l2_weight=self.l2_weight,
            lambda_raydrop=self.lambda_raydrop,
            s_max=self.s_max,
        )
        return loss


# ==============================================================================
# 3. Dataset & GPU Batching
# ==============================================================================

def vectorized_gpu_origin_slide(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    hit_ranges: torch.Tensor,
    hit_masks: Optional[torch.Tensor] = None,
    gt_cdf: Optional[torch.Tensor] = None,
    eval_grid: Optional[torch.Tensor] = None,
    prob: float = 0.5,
    max_slide: float = 0.7,
    s_min: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fully vectorized GPU origin-sliding augmentation for entire ray batches.

    Slides ray origins forward into empty space [0, max_slide * hit_range).
    Directly updates hit_ranges (r* -> r* - delta) in closed form, and optionally
    shifts dense gt_cdf if an evaluation grid is provided. Only slides rays that
    actually hit obstacles (hit_mask > 0).

    Args:
        ray_o: Ray origins on GPU, shape (B, 3).
        ray_d: Ray directions on GPU, shape (B, 3).
        hit_ranges: Hit range distances on GPU, shape (B,) or (B, 1).
        hit_masks: Optional hit masks (1.0 = hit, 0.0 = sky), shape (B,) or (B, 1).
        gt_cdf: Optional ground truth CDF on GPU, shape (B, M).
        eval_grid: Optional uniform grid points on GPU, shape (M,).
        prob: Probability of sliding each valid ray.
        max_slide: Maximum fraction of hit_range to slide forward.
        s_min: Minimum near bound distance.

    Returns:
        ray_o_aug: Augmented ray origins, shape (B, 3).
        hit_ranges_aug: Augmented hit ranges, shape matching hit_ranges.
        cdf_aug: Augmented ground-truth CDFs (or None if gt_cdf was not provided).
    """
    B = ray_o.shape[0]
    device = ray_o.device
    r_flat = hit_ranges.view(-1)
    min_dist = eval_grid[0].item() if eval_grid is not None else s_min

    # 1. Select rays with obstacles ahead to slide (exclude sky/non-returns)
    valid_rays = (r_flat > min_dist)
    if hit_masks is not None:
        valid_rays = valid_rays & (hit_masks.view(-1) > 0.0)

    do_slide = (torch.rand(B, device=device) < prob) & valid_rays
    u = torch.rand(B, device=device) * max_slide
    delta = torch.where(do_slide, u * r_flat, torch.zeros_like(r_flat))  # (B,)

    # 2. Shift ray origin forward and adjust hit ranges
    ray_o_aug = ray_o + delta.unsqueeze(-1) * ray_d
    hit_ranges_aug = (r_flat - delta).view_as(hit_ranges)

    # 3. Optional dense grid shift if gt_cdf is provided
    cdf_final = None
    if gt_cdf is not None and eval_grid is not None:
        M = gt_cdf.shape[1]
        query_s = eval_grid.unsqueeze(0) + delta.unsqueeze(-1)  # (B, M)
        step = (eval_grid[-1] - eval_grid[0]) / (M - 1)
        coords = (query_s - min_dist) / step

        idx_left = coords.floor().long().clamp(0, M - 2)
        alpha = (coords - idx_left.float()).clamp(0.0, 1.0)

        c_left = torch.gather(gt_cdf, 1, idx_left)
        c_right = torch.gather(gt_cdf, 1, idx_left + 1)
        cdf_interp = c_left + alpha * (c_right - c_left)

        c_rightmost = gt_cdf[:, -1:].expand(-1, M)
        cdf_shifted = torch.where(coords >= (M - 1), c_rightmost, cdf_interp)
        cdf_final = torch.where(do_slide.unsqueeze(-1), cdf_shifted, gt_cdf)

    return ray_o_aug, hit_ranges_aug, cdf_final


class GPURayBatcher:
    """Zero-copy, high-throughput ray batcher residing directly in GPU VRAM.

    Bypasses PyTorch DataLoader multiprocessing serialization, CPU worker threads,
    and host-to-device transfers, providing up to 25x faster batch iteration.
    Supports both exact grid-free training (using hit_ranges) and legacy grid training.

    Args:
        rays_o: Ray origins, shape (N, 3).
        rays_d: Ray directions, shape (N, 3).
        hit_ranges: Optional hit ranges for exact grid-free training and origin sliding, shape (N,).
        hit_masks: Optional hit masks for ray drop supervision, shape (N,).
        gt_cdfs: Optional ground truth CDFs, shape (N, M).
        batch_size: Batch size for iteration.
        shuffle: If True, randomly permutes rays every epoch.
        device: Torch device (defaults to 'cuda' if available).
    """

    def __init__(
        self,
        rays_o: Union[np.ndarray, torch.Tensor],
        rays_d: Union[np.ndarray, torch.Tensor],
        hit_ranges: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hit_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gt_cdfs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 1024,
        shuffle: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.rays_o = torch.as_tensor(rays_o, dtype=torch.float32, device=self.device)
        self.rays_d = torch.as_tensor(rays_d, dtype=torch.float32, device=self.device)
        self.hit_ranges = (
            torch.as_tensor(hit_ranges, dtype=torch.float32, device=self.device)
            if hit_ranges is not None
            else None
        )
        self.hit_masks = (
            torch.as_tensor(hit_masks, dtype=torch.float32, device=self.device)
            if hit_masks is not None
            else None
        )
        self.gt_cdfs = (
            torch.as_tensor(gt_cdfs, dtype=torch.float32, device=self.device)
            if gt_cdfs is not None
            else None
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = self.rays_o.shape[0]

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.N, device=self.device)
            rays_o = self.rays_o[perm]
            rays_d = self.rays_d[perm]
            hit_ranges = (
                self.hit_ranges[perm] if self.hit_ranges is not None else None
            )
            hit_masks = (
                self.hit_masks[perm] if self.hit_masks is not None else None
            )
            gt_cdfs = (
                self.gt_cdfs[perm] if self.gt_cdfs is not None else None
            )
        else:
            rays_o = self.rays_o
            rays_d = self.rays_d
            hit_ranges = self.hit_ranges
            hit_masks = self.hit_masks
            gt_cdfs = self.gt_cdfs

        for i in range(0, self.N, self.batch_size):
            batch = {
                "ray_o": rays_o[i : i + self.batch_size],
                "ray_d": rays_d[i : i + self.batch_size],
            }
            if hit_ranges is not None:
                batch["hit_range"] = hit_ranges[i : i + self.batch_size]
            if hit_masks is not None:
                batch["hit_mask"] = hit_masks[i : i + self.batch_size]
            if gt_cdfs is not None:
                batch["gt_cdf"] = gt_cdfs[i : i + self.batch_size]
            yield batch


class LidarRayDataset(Dataset):
    """PyTorch Dataset for LiDAR ray hazard fields and return CDFs.

    Optimized to eliminate dead calculations. Supports direct GPU storage, hit_mask tracking,
    and exact grid-free training using hit_ranges.
    """

    def __init__(
        self,
        rays_o: Union[np.ndarray, torch.Tensor],
        rays_d: Union[np.ndarray, torch.Tensor],
        hit_ranges: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hit_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gt_cdfs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_knots: int = 32,
        s_min: float = 0.1,
        s_max: float = 15.0,
        augment_origin_slide: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        dev = torch.device(device) if device is not None else None
        self.rays_o = torch.as_tensor(rays_o, dtype=torch.float32, device=dev)
        self.rays_d = torch.as_tensor(rays_d, dtype=torch.float32, device=dev)
        self.hit_ranges = (
            torch.as_tensor(hit_ranges, dtype=torch.float32, device=dev)
            if hit_ranges is not None
            else None
        )
        self.hit_masks = (
            torch.as_tensor(hit_masks, dtype=torch.float32, device=dev)
            if hit_masks is not None
            else None
        )
        self.gt_cdfs = (
            torch.as_tensor(gt_cdfs, dtype=torch.float32, device=dev)
            if gt_cdfs is not None
            else None
        )

        self.num_knots = num_knots
        self.s_min = s_min
        self.s_max = s_max
        self.augment_origin_slide = augment_origin_slide

    def __len__(self) -> int:
        return len(self.rays_o)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "ray_o": self.rays_o[idx],
            "ray_d": self.rays_d[idx],
        }
        if self.hit_ranges is not None:
            item["hit_range"] = self.hit_ranges[idx]
        if self.hit_masks is not None:
            item["hit_mask"] = self.hit_masks[idx]
        if self.gt_cdfs is not None:
            item["gt_cdf"] = self.gt_cdfs[idx]
        return item


# ==============================================================================
# 4. Training Loop & Step Utilities
# ==============================================================================

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    eval_grid: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    lambda_raydrop: float = 1.0,
    s_max: float = 15.0,
) -> float:
    """Executes a single forward and backward optimization step on a training batch.

    Supports exact grid-free analytic loss when 'hit_range' is present in batch,
    as well as discrete grid fallback mode when 'gt_cdf' and 'eval_grid' are provided.

    Args:
        model: DeeperSCKNetwork model instance.
        optimizer: Torch optimizer (e.g. Adam).
        batch: Batch dictionary containing 'ray_o', 'ray_d', optional 'hit_range', 'hit_mask', 'gt_cdf'.
        eval_grid: Optional 1D grid distances tensor for discrete CDF loss evaluation.
        dt: Step size of eval_grid (only for discrete grid mode).
        device: Torch device. If None, derived from model parameter device.
        augment_origin_slide: If True, applies vectorized GPU origin sliding.
        l1_weight: Weight for linear Wasserstein loss term.
        l2_weight: Weight for continuous L2 / CRPS loss term.
        lambda_raydrop: Weight for ray-drop classification loss.
        s_max: Maximum scene range.

    Returns:
        float: Training loss for this batch.
    """
    model.train()
    optimizer.zero_grad()

    if device is None:
        device = next(model.parameters()).device

    ray_o = batch["ray_o"].to(device)
    ray_d = batch["ray_d"].to(device)
    hit_range = batch.get("hit_range", None)
    if hit_range is not None:
        hit_range = hit_range.to(device)

    gt_cdf = batch.get("gt_cdf", None)
    if gt_cdf is not None:
        gt_cdf = gt_cdf.to(device)

    hit_mask = batch.get("hit_mask", None)
    if hit_mask is not None:
        hit_mask = hit_mask.to(device)

    if eval_grid is not None:
        eval_grid = eval_grid.to(device)

    # Optional fast GPU vectorized origin sliding
    if augment_origin_slide and hit_range is not None:
        ray_o, hit_range, gt_cdf = vectorized_gpu_origin_slide(
            ray_o=ray_o,
            ray_d=ray_d,
            hit_ranges=hit_range,
            hit_masks=hit_mask,
            gt_cdf=gt_cdf,
            eval_grid=eval_grid,
        )

    # Forward pass through self-conditioning rounds + ray-drop head
    s_final, c_final, round_outputs, p_hit = model(ray_o, ray_d)

    # Compute continuous multi-round loss (Exact Grid-Free or Grid-based) + Ray-Drop BCE
    total_loss, _ = compute_sck_loss(
        round_outputs=round_outputs,
        hit_range=hit_range,
        gt_cdf=gt_cdf,
        eval_grid=eval_grid,
        p_hit=p_hit,
        hit_mask=hit_mask,
        dt=dt,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        lambda_raydrop=lambda_raydrop,
        s_max=s_max,
    )

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_epoch(
    model: nn.Module,
    dataloader: Union[DataLoader, GPURayBatcher, Any],
    optimizer: torch.optim.Optimizer,
    eval_grid: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    lambda_raydrop: float = 1.0,
    s_max: float = 15.0,
) -> float:
    """Runs one full training epoch over the dataloader or GPURayBatcher.

    Args:
        model: DeeperSCKNetwork instance.
        dataloader: DataLoader or GPURayBatcher yielding ray batches.
        optimizer: Torch optimizer.
        eval_grid: Optional 1D grid distances tensor for discrete grid mode.
        dt: Optional step size of eval_grid.
        device: Torch device.
        augment_origin_slide: Whether to apply vectorized GPU origin sliding.
        l1_weight: Weight for Wasserstein-1 L1 term.
        l2_weight: Weight for continuous L2 / CRPS term.
        lambda_raydrop: Weight for ray-drop classification loss.
        s_max: Maximum scene range.

    Returns:
        float: Average loss across all batches in the epoch.
    """
    epoch_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            eval_grid=eval_grid,
            dt=dt,
            device=device,
            augment_origin_slide=augment_origin_slide,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            lambda_raydrop=lambda_raydrop,
            s_max=s_max,
        )
        epoch_loss += loss

    return epoch_loss / max(1, num_batches)


def train_model(
    model: nn.Module,
    dataloader: Union[DataLoader, GPURayBatcher, Any],
    optimizer: torch.optim.Optimizer,
    eval_grid: Optional[torch.Tensor] = None,
    num_epochs: int = 32,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 1.0,
    l2_weight: float = 1.0,
    lambda_raydrop: float = 1.0,
    s_max: float = 15.0,
) -> Dict[str, Any]:
    """High-level training loop for Self-Conditioned Knot CDF estimation and Ray Drop.

    Operates in exact closed-form grid-free mode whenever the dataloader yields 'hit_range'.
    No eval_grid required!

    Args:
        model: DeeperSCKNetwork instance.
        dataloader: DataLoader or GPURayBatcher for training rays.
        optimizer: Torch optimizer.
        eval_grid: Optional 1D grid distances tensor for discrete grid mode.
        num_epochs: Number of epochs to train.
        dt: Optional grid step size (for discrete grid mode).
        device: Torch device to train on. If None, derived from model parameters.
        verbose: If True, prints epoch progress.
        callback: Optional callback fn(epoch, avg_loss) called after each epoch.
        augment_origin_slide: If True, applies fast vectorized GPU origin sliding.
        l1_weight: Weight for linear Wasserstein-1 L1 loss term.
        l2_weight: Weight for continuous L2 / CRPS term.
        lambda_raydrop: Weight for ray-drop classification loss.
        s_max: Maximum scene range.

    Returns:
        Dict with 'loss_history': list of average losses per epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    if eval_grid is not None:
        eval_grid = eval_grid.to(device)
        if dt is None:
            dt = float((eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1))

    loss_history: List[float] = []

    if verbose:
        dataset_len = (
            len(dataloader.dataset)
            if hasattr(dataloader, "dataset")
            else (dataloader.N if hasattr(dataloader, "N") else len(dataloader))
        )
        mode_str = "Exact Grid-Free Analytic Loss" if eval_grid is None else "Discrete Grid Loss"
        print(f"Training on {device} ({dataset_len} rays, {len(dataloader)} batches/epoch) [{mode_str}]...")

    for epoch in range(num_epochs):
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            eval_grid=eval_grid,
            dt=dt,
            device=device,
            augment_origin_slide=augment_origin_slide,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            lambda_raydrop=lambda_raydrop,
            s_max=s_max,
        )
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Loss: {avg_loss:.6f}")

        if callback is not None:
            callback(epoch, avg_loss)

    return {"loss_history": loss_history}


# ==============================================================================
# 5. Point Cloud Inference
# ==============================================================================

def generate_point_cloud_from_network(
    model: nn.Module,
    sensor: Any,
    sensor_position: Union[List[float], np.ndarray] = [0.0, 0.0, 1.5],
    sensor_rotation: Optional[Any] = None,
    method: str = "threshold",
    threshold: float = 0.5,
    non_return_threshold: float = 0.5,
    ray_drop_threshold: Optional[float] = None,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Any:
    """Renders a 3D PointCloud with PLiNK ray-drop filtering to cleanly suppress sky floaters.

    Args:
        model: Trained DeeperSCKNetwork model.
        sensor: LidarProjectionModel instance with get_rays(...) method.
        sensor_position: 3D coordinates [x, y, z] of sensor.
        sensor_rotation: Rotation object or matrix (scipy Rotation or None).
        method: "threshold" for surface hits, or "sample" for stochastic returns.
        threshold: CDF threshold for hit detection (0.5 = median return distance).
        non_return_threshold: Maximum allowable likelihood of non-return in [0, 1].
            Rays with P(non-return) = (1 - p_hit) > non_return_threshold will NOT
            generate a measurement (pruned as sky / non-returns). Default: 0.5.
        ray_drop_threshold: Legacy alias for hit probability threshold.
            If provided, effective_non_return_threshold = 1.0 - ray_drop_threshold.
        batch_size: Ray batch size for inference.
        device: Torch device for model execution.
        verbose: If True, prints rendering summary statistics.

    Returns:
        PointCloud instance containing points, probabilities, ranges, and metadata.
    """
    if PointCloud is None:
        raise ImportError(
            "PointCloud class could not be imported from synthetic_data_generation_utils. "
            "Ensure synthetic_data_generation_utils.py is available in the Python path."
        )

    if device is None:
        device = next(model.parameters()).device

    # Determine effective maximum non-return probability threshold
    if ray_drop_threshold is not None:
        effective_non_return_threshold = 1.0 - ray_drop_threshold
    else:
        effective_non_return_threshold = non_return_threshold

    model.eval()

    # 1. Generate full ray grid from the LiDAR projection model
    rays_o_np, rays_d_np = sensor.get_rays(
        sensor_position=sensor_position,
        sensor_rotation=sensor_rotation,
        flatten=True,
    )
    total_rays = len(rays_o_np)

    all_points: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_ranges: List[torch.Tensor] = []
    all_rays_d: List[torch.Tensor] = []
    all_rays_o: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, total_rays, batch_size):
            batch_o = torch.from_numpy(rays_o_np[i : i + batch_size]).float().to(device)
            batch_d = torch.from_numpy(rays_d_np[i : i + batch_size]).float().to(device)
            B = batch_o.shape[0]

            # 2. Predict knot distances, CDF values, and Ray-Drop hit probability
            s_knots, c_knots, _, p_hit = model(batch_o, batch_d)  # (B, K), p_hit is (B, 1)
            K = s_knots.shape[1]

            # 3. Prepend boundary knot (s=0, C=0) using F.pad
            s_ext = F.pad(s_knots, (1, 0), value=0.0)  # (B, K+1)
            c_ext = F.pad(c_knots, (1, 0), value=0.0)  # (B, K+1)
            c_max = c_ext[:, -1]

            # 4. PLiNK Ray-Drop Filter:
            # Likelihood of non-return is p_non_return = 1.0 - p_hit
            # If likelihood of non-return is higher than non_return_threshold, DO NOT generate a measurement
            p_non_return = (1.0 - p_hit).squeeze(-1)
            ray_hit_mask = (p_non_return <= effective_non_return_threshold)

            if method == "threshold":
                target = torch.full((B, 1), threshold, device=device)
                valid_mask = ray_hit_mask & (c_max >= threshold)
            elif method == "sample":
                u = torch.rand(B, 1, device=device)
                valid_mask = ray_hit_mask & (u.squeeze(-1) <= c_max)
                target = u
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'threshold' or 'sample'.")

            # 5. Invert piecewise linear CDF to solve for exact return range s*
            idx = torch.searchsorted(c_ext.contiguous(), target.contiguous(), right=True)
            idx = torch.clamp(idx, 1, K)

            idx_left = idx - 1
            s_0 = torch.gather(s_ext, 1, idx_left).squeeze(-1)
            s_1 = torch.gather(s_ext, 1, idx).squeeze(-1)
            c_0 = torch.gather(c_ext, 1, idx_left).squeeze(-1)
            c_1 = torch.gather(c_ext, 1, idx).squeeze(-1)

            denom = torch.clamp(c_1 - c_0, min=1e-5)
            alpha = torch.clamp((target.squeeze(-1) - c_0) / denom, 0.0, 1.0)
            pred_ranges = s_0 + alpha * (s_1 - s_0)

            # 6. Compute 3D Cartesian coordinates: p = o + s* * d
            pts = batch_o + pred_ranges.unsqueeze(-1) * batch_d

            # Accumulate directly on GPU without stalling for CPU synchronization
            if valid_mask.any():
                all_points.append(pts[valid_mask])
                all_probs.append(c_max[valid_mask])
                all_ranges.append(pred_ranges[valid_mask])
                all_rays_d.append(batch_d[valid_mask])
                all_rays_o.append(batch_o[valid_mask])

    metadata = {
        "sensor_position": sensor_position,
        "method": method,
        "threshold": threshold,
        "non_return_threshold": effective_non_return_threshold,
        "total_rays": total_rays,
        "returned_points": 0,
        "dropped_rays": total_rays,
    }

    if len(all_points) == 0:
        if verbose:
            print("Warning: No returns met the threshold!")
        return PointCloud(np.empty((0, 3)), np.empty(0), np.empty(0), metadata=metadata)

    # Single asynchronous concatenation on GPU followed by one PCIe transfer to CPU
    points = torch.cat(all_points, dim=0).cpu().numpy()
    probs = torch.cat(all_probs, dim=0).cpu().numpy()
    ranges = torch.cat(all_ranges, dim=0).cpu().numpy()
    rays_d = torch.cat(all_rays_d, dim=0).cpu().numpy()
    rays_o = torch.cat(all_rays_o, dim=0).cpu().numpy()

    metadata["returned_points"] = len(points)
    metadata["dropped_rays"] = total_rays - len(points)

    if verbose:
        print(
            f"Rendered point cloud: {len(points):,} points ({total_rays - len(points):,} rays dropped as non-returns / below threshold)."
        )

    return PointCloud(
        points=points,
        probabilities=probs,
        ranges=ranges,
        ray_directions=rays_d,
        rays_o=rays_o,
        metadata=metadata,
    )


# Convenient aliases for point cloud rendering
render_novel_point_cloud = generate_point_cloud_from_network
render_point_cloud = generate_point_cloud_from_network
