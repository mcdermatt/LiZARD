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
        max_displacement: float = 0.6,
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


def compute_sck_loss(
    round_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    gt_cdf: torch.Tensor,
    eval_grid: torch.Tensor,
    p_hit: Optional[torch.Tensor] = None,
    hit_mask: Optional[torch.Tensor] = None,
    dt: Optional[float] = None,
    weights: Optional[List[float]] = None,
    l1_weight: float = 0.0,
    lambda_raydrop: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes continuous CDF loss with PLiNK-style ray-drop masking and BCE classification.

    When hit_mask is provided:
      1. CDF loss is MASKED by hit_mask so rays that escape into open sky do NOT force
         the continuous spline network to fight trying to fit all-zero curves.
      2. The ray-drop head p_hit is supervised with binary cross-entropy against hit_mask.

    Args:
        round_outputs: List of (s_round, c_round) tuples from model forward pass.
        gt_cdf: Ground truth return CDF on eval_grid, shape (B, M).
        eval_grid: Evaluation distances, shape (M,).
        p_hit: Predicted hit probability from ray_drop_head, shape (B, 1) or (B,).
        hit_mask: Ground truth hit mask (1.0 = hit, 0.0 = ray-drop/sky), shape (B,) or (B, 1).
        dt: Grid step size. If None, inferred as (eval_grid[-1] - eval_grid[0]) / (M - 1).
        weights: Optional custom list of weights for each self-conditioning round.
        l1_weight: Weight for the linear L1 term (0.0 = pure L2/CRPS, >0 = elastic loss).
        lambda_raydrop: Weight for the binary cross-entropy ray-drop loss.

    Returns:
        total_loss: Differentiable scalar loss tensor.
        metrics: Dictionary with individual round losses, ray-drop loss, and total loss.
    """
    num_r = len(round_outputs)
    if dt is None:
        dt = (eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1)

    if weights is None:
        weights = [0.5 ** (num_r - 1 - i) for i in range(num_r)]

    total_loss = torch.tensor(0.0, device=gt_cdf.device, dtype=gt_cdf.dtype)
    metrics: Dict[str, float] = {}

    # PLiNK target_drop_mask preparation
    mask = None
    n_hits = float(gt_cdf.shape[0])
    if hit_mask is not None:
        mask = hit_mask.view(-1, 1).to(gt_cdf.device)
        n_hits = float(torch.clamp(mask.sum(), min=1.0).item())

    # Multi-round continuous CDF loss (masked by hit_mask)
    for i, ((s_round, c_round), w) in enumerate(zip(round_outputs, weights)):
        pred_grid = interpolate_cdf_to_grid(s_round, c_round, eval_grid)
        diff = pred_grid - gt_cdf

        if mask is not None:
            # PLiNK formulation: mask out ray drops from the continuous CDF loss
            diff = diff * mask
            round_l2 = dt * (diff ** 2).sum() / n_hits
            if l1_weight > 0.0:
                round_l1 = dt * (torch.abs(diff)).sum() / n_hits
                round_loss = round_l2 + l1_weight * round_l1
                metrics[f"l1_round_{i}"] = round_l1.item()
            else:
                round_loss = round_l2
        else:
            round_l2 = dt * torch.sum(diff ** 2, dim=-1).mean()
            if l1_weight > 0.0:
                round_l1 = dt * torch.sum(torch.abs(diff), dim=-1).mean()
                round_loss = round_l2 + l1_weight * round_l1
                metrics[f"l1_round_{i}"] = round_l1.item()
            else:
                round_loss = round_l2

        total_loss = total_loss + w * round_loss
        metrics[f"loss_round_{i}"] = round_loss.item()

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
    """PyTorch Module wrapper for the multi-round Continuous CDF loss and Ray-Drop BCE."""

    def __init__(
        self,
        eval_grid: torch.Tensor,
        dt: Optional[float] = None,
        l1_weight: float = 0.0,
        lambda_raydrop: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("eval_grid", eval_grid)
        self.dt = dt if dt is not None else float((eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1))
        self.l1_weight = l1_weight
        self.lambda_raydrop = lambda_raydrop

    def forward(
        self,
        round_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
        gt_cdf: torch.Tensor,
        p_hit: Optional[torch.Tensor] = None,
        hit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss, _ = compute_sck_loss(
            round_outputs,
            gt_cdf,
            self.eval_grid,
            p_hit=p_hit,
            hit_mask=hit_mask,
            dt=self.dt,
            l1_weight=self.l1_weight,
            lambda_raydrop=self.lambda_raydrop,
        )
        return loss


# ==============================================================================
# 3. Dataset & GPU Batching
# ==============================================================================

def vectorized_gpu_origin_slide(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    gt_cdf: torch.Tensor,
    hit_ranges: torch.Tensor,
    eval_grid: torch.Tensor,
    prob: float = 0.5,
    max_slide: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully vectorized GPU origin-sliding augmentation for entire ray batches.

    Slides ray origins forward into empty space [0, max_slide * hit_range)
    and shifts the ground-truth CDF using 1D GPU linear interpolation in microseconds.

    Args:
        ray_o: Ray origins on GPU, shape (B, 3).
        ray_d: Ray directions on GPU, shape (B, 3).
        gt_cdf: Ground truth CDF on GPU, shape (B, M).
        hit_ranges: Hit range distances on GPU, shape (B,).
        eval_grid: Uniform grid points on GPU, shape (M,).
        prob: Probability of sliding each valid ray.
        max_slide: Maximum fraction of hit_range to slide forward.

    Returns:
        ray_o_aug: Augmented ray origins, shape (B, 3).
        cdf_aug: Augmented ground-truth CDFs, shape (B, M).
    """
    B, M = gt_cdf.shape
    device = ray_o.device
    s_min = eval_grid[0]

    # 1. Select rays with obstacles ahead to slide
    do_slide = (torch.rand(B, device=device) < prob) & (hit_ranges > s_min)
    u = torch.rand(B, device=device) * max_slide
    delta = torch.where(do_slide, u * hit_ranges, torch.zeros_like(hit_ranges)).unsqueeze(-1)  # (B, 1)

    # 2. Shift ray origin forward
    ray_o_aug = ray_o + delta * ray_d

    # 3. 1D vectorized linear interpolation of gt_cdf at query_s = (eval_grid + delta)
    query_s = eval_grid.unsqueeze(0) + delta  # (B, M)
    step = (eval_grid[-1] - eval_grid[0]) / (M - 1)
    coords = (query_s - s_min) / step

    idx_left = coords.floor().long().clamp(0, M - 2)
    alpha = (coords - idx_left.float()).clamp(0.0, 1.0)

    c_left = torch.gather(gt_cdf, 1, idx_left)
    c_right = torch.gather(gt_cdf, 1, idx_left + 1)
    cdf_interp = c_left + alpha * (c_right - c_left)

    # Beyond boundary takes the rightmost CDF value
    c_rightmost = gt_cdf[:, -1:].expand(-1, M)
    cdf_shifted = torch.where(coords >= (M - 1), c_rightmost, cdf_interp)

    cdf_final = torch.where(do_slide.unsqueeze(-1), cdf_shifted, gt_cdf)
    return ray_o_aug, cdf_final


class GPURayBatcher:
    """Zero-copy, high-throughput ray batcher residing directly in GPU VRAM.

    Bypasses PyTorch DataLoader multiprocessing serialization, CPU worker threads,
    and host-to-device transfers, providing up to 25x faster batch iteration.

    Args:
        rays_o: Ray origins, shape (N, 3).
        rays_d: Ray directions, shape (N, 3).
        gt_cdfs: Ground truth CDFs, shape (N, M).
        hit_ranges: Optional hit ranges for origin sliding, shape (N,).
        hit_masks: Optional hit masks for ray drop supervision, shape (N,).
        batch_size: Batch size for iteration.
        shuffle: If True, randomly permutes rays every epoch.
        device: Torch device (defaults to 'cuda' if available).
    """

    def __init__(
        self,
        rays_o: Union[np.ndarray, torch.Tensor],
        rays_d: Union[np.ndarray, torch.Tensor],
        gt_cdfs: Union[np.ndarray, torch.Tensor],
        hit_ranges: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hit_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 1024,
        shuffle: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.rays_o = torch.as_tensor(rays_o, dtype=torch.float32, device=self.device)
        self.rays_d = torch.as_tensor(rays_d, dtype=torch.float32, device=self.device)
        self.gt_cdfs = torch.as_tensor(gt_cdfs, dtype=torch.float32, device=self.device)
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

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.N = self.rays_o.shape[0]

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.N, device=self.device)
        else:
            indices = torch.arange(self.N, device=self.device)

        for i in range(0, self.N, self.batch_size):
            idx = indices[i : i + self.batch_size]
            batch = {
                "ray_o": self.rays_o[idx],
                "ray_d": self.rays_d[idx],
                "gt_cdf": self.gt_cdfs[idx],
            }
            if self.hit_ranges is not None:
                batch["hit_range"] = self.hit_ranges[idx]
            if self.hit_masks is not None:
                batch["hit_mask"] = self.hit_masks[idx]
            yield batch


class LidarRayDataset(Dataset):
    """PyTorch Dataset for LiDAR ray hazard fields and return CDFs.

    Optimized to eliminate dead calculations. Supports direct GPU storage and hit_mask tracking.
    """

    def __init__(
        self,
        rays_o: Union[np.ndarray, torch.Tensor],
        rays_d: Union[np.ndarray, torch.Tensor],
        hit_ranges: Union[np.ndarray, torch.Tensor],
        hit_masks: Union[np.ndarray, torch.Tensor],
        gt_cdfs: Union[np.ndarray, torch.Tensor],
        num_knots: int = 32,
        s_min: float = 0.1,
        s_max: float = 15.0,
        augment_origin_slide: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        dev = torch.device(device) if device is not None else None
        self.rays_o = torch.as_tensor(rays_o, dtype=torch.float32, device=dev)
        self.rays_d = torch.as_tensor(rays_d, dtype=torch.float32, device=dev)
        self.hit_ranges = torch.as_tensor(hit_ranges, dtype=torch.float32, device=dev)
        self.hit_masks = torch.as_tensor(hit_masks, dtype=torch.float32, device=dev)
        self.gt_cdfs = torch.as_tensor(gt_cdfs, dtype=torch.float32, device=dev)

        self.num_knots = num_knots
        self.s_min = s_min
        self.s_max = s_max
        self.augment_origin_slide = augment_origin_slide
        self.eval_distances = np.linspace(s_min, s_max, self.gt_cdfs.shape[-1])

    def __len__(self) -> int:
        return len(self.rays_o)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ray_o": self.rays_o[idx],
            "ray_d": self.rays_d[idx],
            "gt_cdf": self.gt_cdfs[idx],
            "hit_range": self.hit_ranges[idx],
            "hit_mask": self.hit_masks[idx],
        }


# ==============================================================================
# 4. Training Loop & Step Utilities
# ==============================================================================

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    eval_grid: torch.Tensor,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 0.0,
    lambda_raydrop: float = 1.0,
) -> float:
    """Executes a single forward and backward optimization step on a training batch.

    Args:
        model: DeeperSCKNetwork model instance.
        optimizer: Torch optimizer (e.g. Adam).
        batch: Batch dictionary containing 'ray_o', 'ray_d', 'gt_cdf', and 'hit_mask'.
        eval_grid: 1D grid distances tensor for CDF loss evaluation.
        dt: Step size of eval_grid. If None, computed from eval_grid.
        device: Torch device. If None, derived from model parameter device.
        augment_origin_slide: If True, applies vectorized GPU origin sliding.
        l1_weight: Weight for linear Wasserstein loss term.
        lambda_raydrop: Weight for ray-drop classification loss.

    Returns:
        float: Training loss for this batch.
    """
    model.train()
    optimizer.zero_grad()

    if device is None:
        device = next(model.parameters()).device

    ray_o = batch["ray_o"].to(device)
    ray_d = batch["ray_d"].to(device)
    gt_cdf = batch["gt_cdf"].to(device)
    eval_grid = eval_grid.to(device)
    hit_mask = batch.get("hit_mask", None)
    if hit_mask is not None:
        hit_mask = hit_mask.to(device)

    # Optional fast GPU vectorized origin sliding
    if augment_origin_slide and "hit_range" in batch:
        hit_ranges = batch["hit_range"].to(device)
        ray_o, gt_cdf = vectorized_gpu_origin_slide(ray_o, ray_d, gt_cdf, hit_ranges, eval_grid)

    # Forward pass through self-conditioning rounds + ray-drop head
    s_final, c_final, round_outputs, p_hit = model(ray_o, ray_d)

    # Compute PLiNK-style masked continuous multi-round CDF loss + Ray-Drop BCE
    total_loss, _ = compute_sck_loss(
        round_outputs=round_outputs,
        gt_cdf=gt_cdf,
        eval_grid=eval_grid,
        p_hit=p_hit,
        hit_mask=hit_mask,
        dt=dt,
        l1_weight=l1_weight,
        lambda_raydrop=lambda_raydrop,
    )

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_epoch(
    model: nn.Module,
    dataloader: Union[DataLoader, GPURayBatcher, Any],
    optimizer: torch.optim.Optimizer,
    eval_grid: torch.Tensor,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 0.0,
    lambda_raydrop: float = 1.0,
) -> float:
    """Runs one full training epoch over the dataloader or GPURayBatcher.

    Args:
        model: DeeperSCKNetwork instance.
        dataloader: DataLoader or GPURayBatcher yielding ray batches.
        optimizer: Torch optimizer.
        eval_grid: 1D grid distances tensor for CDF loss evaluation.
        dt: Optional step size of eval_grid.
        device: Torch device.
        augment_origin_slide: Whether to apply vectorized GPU origin sliding.
        l1_weight: Optional linear L1 loss weight.
        lambda_raydrop: Weight for ray-drop classification loss.

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
            lambda_raydrop=lambda_raydrop,
        )
        epoch_loss += loss

    return epoch_loss / max(1, num_batches)


def train_model(
    model: nn.Module,
    dataloader: Union[DataLoader, GPURayBatcher, Any],
    optimizer: torch.optim.Optimizer,
    eval_grid: torch.Tensor,
    num_epochs: int = 32,
    dt: Optional[float] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
    augment_origin_slide: bool = False,
    l1_weight: float = 0.0,
    lambda_raydrop: float = 1.0,
) -> Dict[str, Any]:
    """High-level training loop for Self-Conditioned Knot CDF estimation and Ray Drop.

    Args:
        model: DeeperSCKNetwork instance.
        dataloader: DataLoader or GPURayBatcher for training rays.
        optimizer: Torch optimizer.
        eval_grid: 1D grid distances tensor for CDF loss evaluation.
        num_epochs: Number of epochs to train.
        dt: Grid step size. If None, computed from eval_grid.
        device: Torch device to train on. If None, derived from model parameters.
        verbose: If True, prints epoch progress.
        callback: Optional callback fn(epoch, avg_loss) called after each epoch.
        augment_origin_slide: If True, applies fast vectorized GPU origin sliding.
        l1_weight: Weight for linear L1 loss term to prevent plateauing.
        lambda_raydrop: Weight for ray-drop classification loss.

    Returns:
        Dict with 'loss_history': list of average losses per epoch.
    """
    if device is None:
        device = next(model.parameters()).device

    eval_grid = eval_grid.to(device)
    if dt is None:
        dt = (eval_grid[-1] - eval_grid[0]).item() / (eval_grid.shape[0] - 1)

    loss_history: List[float] = []

    if verbose:
        dataset_len = (
            len(dataloader.dataset)
            if hasattr(dataloader, "dataset")
            else (dataloader.N if hasattr(dataloader, "N") else len(dataloader))
        )
        print(f"Training on {device} ({dataset_len} rays, {len(dataloader)} batches/epoch)...")

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
            lambda_raydrop=lambda_raydrop,
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
