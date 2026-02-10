"""
Seam consistency losses for panorama generation.

This module provides loss functions that encourage seamless 360° panoramas:
- Wrap seam loss: ensures left-right edge continuity in ERP images
- Pole consistency loss: ensures consistent appearance in pole regions
- Weighted per-pixel losses for border emphasis
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def wrap_seam_loss(images: Tensor, reduction: str = "mean") -> Tensor:
    """
    Compute wrap seam loss between left and right edges of ERP images.
    
    For a seamless 360° panorama, the left edge (x=0) should match the right
    edge (x=W-1) since they represent the same longitude in world space.
    
    Args:
        images: Batch of ERP images, shape (B, C, H, W)
        reduction: "mean", "sum", or "none"
    
    Returns:
        Scalar loss (if reduction="mean" or "sum") or per-sample loss (B,)
    
    Formula:
        L_wrap = mean_y || images[:, :, y, 0] - images[:, :, y, W-1] ||_1
    """
    # Extract left and right edges
    left_edge = images[:, :, :, 0]       # (B, C, H)
    right_edge = images[:, :, :, -1]     # (B, C, H)
    
    # Compute L1 distance
    edge_diff = torch.abs(left_edge - right_edge)  # (B, C, H)
    
    # Reduce over channels and height
    if reduction == "none":
        return edge_diff.mean(dim=(1, 2))  # (B,)
    elif reduction == "sum":
        return edge_diff.sum()
    else:  # mean
        return edge_diff.mean()


def pole_consistency_loss(
    images: Tensor,
    pole_band_deg: float = 20.0,
    sigma_deg: float = 10.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute pole consistency loss for ERP images.
    
    Pixels near the poles (north/south) should have consistent values since
    they represent a small area that's stretched across the entire width.
    
    Args:
        images: Batch of ERP images, shape (B, C, H, W)
        pole_band_deg: Degrees from pole to consider as pole region (default 20°)
        sigma_deg: Sigma for Gaussian weighting of neighbor distances
        reduction: "mean", "sum", or "none"
    
    Returns:
        Scalar loss encouraging pole region consistency
    """
    B, C, H, W = images.shape
    device = images.device
    dtype = images.dtype
    
    # Convert degrees to radians
    pole_band_rad = pole_band_deg * math.pi / 180.0
    
    # Compute latitude for each row
    y_coords = torch.arange(H, device=device, dtype=dtype)
    phi = math.pi / 2.0 - math.pi * ((y_coords + 0.5) / H)  # (H,)
    
    # Identify pole region rows
    north_pole_threshold = math.pi / 2.0 - pole_band_rad
    south_pole_threshold = -math.pi / 2.0 + pole_band_rad
    
    is_north_pole = phi > north_pole_threshold  # (H,)
    is_south_pole = phi < south_pole_threshold  # (H,)
    is_pole = is_north_pole | is_south_pole     # (H,)
    
    pole_row_indices = torch.where(is_pole)[0]
    
    if len(pole_row_indices) == 0:
        # No pole region, return zero loss
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    # For pole rows, compute variance across width (low variance = consistent)
    pole_pixels = images[:, :, pole_row_indices, :]  # (B, C, num_pole_rows, W)
    
    # Compute per-row variance across width
    row_mean = pole_pixels.mean(dim=-1, keepdim=True)  # (B, C, num_pole_rows, 1)
    row_var = ((pole_pixels - row_mean) ** 2).mean(dim=-1)  # (B, C, num_pole_rows)
    
    # Weight by distance to pole (closer = higher weight)
    pole_phis = phi[pole_row_indices]  # (num_pole_rows,)
    dist_from_pole = torch.minimum(
        torch.abs(pole_phis - math.pi / 2),   # distance from north
        torch.abs(pole_phis + math.pi / 2)    # distance from south
    )  # (num_pole_rows,)
    
    sigma_rad = sigma_deg * math.pi / 180.0
    weights = torch.exp(-dist_from_pole ** 2 / (2 * sigma_rad ** 2))  # (num_pole_rows,)
    weights = weights / weights.sum()  # normalize
    
    # Weighted variance
    weighted_var = (row_var * weights.view(1, 1, -1)).sum(dim=-1)  # (B, C)
    
    if reduction == "none":
        return weighted_var.mean(dim=1)  # (B,)
    elif reduction == "sum":
        return weighted_var.sum()
    else:  # mean
        return weighted_var.mean()


def compute_seam_weights(
    H: int,
    W: int,
    border_width: int = 2,
    seam_boost: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute per-pixel weights for seam-aware reconstruction loss.
    
    Pixels at the left and right borders receive higher weights to emphasize
    wrap seam consistency during training.
    
    Args:
        H: Image height
        W: Image width
        border_width: Width of border region in pixels
        seam_boost: Weight multiplier for border pixels
        device: Device for tensor
        dtype: Data type for tensor
    
    Returns:
        Weight tensor of shape (H, W), values in [1.0, seam_boost]
    """
    weights = torch.ones(H, W, device=device, dtype=dtype)
    
    # Boost left border
    weights[:, :border_width] = seam_boost
    
    # Boost right border
    weights[:, -border_width:] = seam_boost
    
    return weights


def get_pole_neighbor_weights(
    H: int,
    W: int,
    pole_band_deg: float = 20.0,
    sigma_deg: float = 10.0,
    max_neighbors: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Pre-compute neighbor indices and weights for pole consistency loss.
    
    This function computes which pixels are in the pole region and their
    weighted neighbor relationships for efficient loss computation.
    
    Args:
        H: Image height
        W: Image width
        pole_band_deg: Degrees from pole for pole region
        sigma_deg: Sigma for Gaussian neighbor weighting
        max_neighbors: Maximum neighbors per pixel (4 or 8)
        device: Device for tensors
        dtype: Data type
    
    Returns:
        pole_mask: Boolean mask (H, W), True for pole region pixels
        neighbor_indices: Flat indices of neighbors for pole pixels
        neighbor_weights: Gaussian weights for each neighbor relationship
    """
    # Compute latitude for each row
    y_coords = torch.arange(H, device=device, dtype=dtype)
    phi = math.pi / 2.0 - math.pi * ((y_coords + 0.5) / H)
    
    # Identify pole regions
    pole_band_rad = pole_band_deg * math.pi / 180.0
    is_north = phi > (math.pi / 2.0 - pole_band_rad)
    is_south = phi < (-math.pi / 2.0 + pole_band_rad)
    
    # Create full mask
    pole_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    pole_rows = is_north | is_south
    pole_mask[pole_rows, :] = True
    
    # For each pole pixel, compute neighbor indices and weights
    pole_indices = torch.where(pole_mask.flatten())[0]
    num_pole_pixels = len(pole_indices)
    
    if num_pole_pixels == 0:
        return (
            pole_mask,
            torch.empty(0, max_neighbors, dtype=torch.long, device=device),
            torch.empty(0, max_neighbors, dtype=dtype, device=device),
        )
    
    # Define neighbor offsets
    if max_neighbors == 4:
        offsets = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=device)
    else:
        offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],          [0, 1],
            [1, -1], [1, 0], [1, 1]
        ], device=device)
    
    # Convert flat indices to (y, x) coordinates
    pole_y = pole_indices // W
    pole_x = pole_indices % W
    
    # Compute neighbor coordinates
    neighbor_y = pole_y.unsqueeze(1) + offsets[:, 0].unsqueeze(0)  # (num_pole, num_neighbors)
    neighbor_x = pole_x.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
    
    # Clamp y (no vertical wrap)
    neighbor_y = neighbor_y.clamp(0, H - 1)
    
    # Wrap x (horizontal wrap for 360°)
    neighbor_x = neighbor_x % W
    
    # Convert to flat indices
    neighbor_flat = neighbor_y * W + neighbor_x
    
    # Compute Gaussian weights based on spherical distance
    # For simplicity, use uniform weights (proper spherical distance adds complexity)
    sigma_rad = sigma_deg * math.pi / 180.0
    
    # Approximate: weight = 1.0 for all neighbors (could be refined with true geodesic distance)
    neighbor_weights = torch.ones(num_pole_pixels, max_neighbors, device=device, dtype=dtype)
    neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True)
    
    return pole_mask, neighbor_flat, neighbor_weights


class PanoSeamLoss(nn.Module):
    """
    Combined seam loss for panorama training.
    
    Combines wrap seam loss and pole consistency loss with configurable
    weights and texture/structure step differentiation.
    
    Args:
        wrap_weight: Weight for wrap seam loss
        pole_weight: Weight for pole consistency loss
        pole_band_deg: Degrees from pole for pole region
        pole_tau_deg: Temperature for pole loss weighting
        pole_sigma_deg: Sigma for neighbor Gaussian weights
        texture_step_boost: Multiplier for texture (late) scale steps
        structure_step_weight: Multiplier for structure (early) scale steps
    """
    
    def __init__(
        self,
        wrap_weight: float = 1.0,
        pole_weight: float = 1.0,
        pole_band_deg: float = 20.0,
        pole_tau_deg: float = 10.0,
        pole_sigma_deg: float = 10.0,
        texture_step_boost: float = 1.0,
        structure_step_weight: float = 0.1,
    ):
        super().__init__()
        self.wrap_weight = wrap_weight
        self.pole_weight = pole_weight
        self.pole_band_deg = pole_band_deg
        self.pole_tau_deg = pole_tau_deg
        self.pole_sigma_deg = pole_sigma_deg
        self.texture_step_boost = texture_step_boost
        self.structure_step_weight = structure_step_weight
    
    def forward(
        self,
        pred: Tensor,
        target: Optional[Tensor] = None,
        is_texture_step: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Compute panorama seam losses.
        
        Args:
            pred: Predicted images, shape (B, C, H, W)
            target: Target images (optional, for reconstruction-based losses)
            is_texture_step: Whether this is a texture (late) scale step
        
        Returns:
            Dict with keys: "wrap_loss", "pole_loss", "total_loss"
        """
        # Determine step weight
        step_weight = self.texture_step_boost if is_texture_step else self.structure_step_weight
        
        # Compute wrap seam loss
        l_wrap = wrap_seam_loss(pred, reduction="mean")
        
        # Compute pole consistency loss
        l_pole = pole_consistency_loss(
            pred,
            pole_band_deg=self.pole_band_deg,
            sigma_deg=self.pole_sigma_deg,
            reduction="mean",
        )
        
        # Apply weights
        weighted_wrap = self.wrap_weight * l_wrap * step_weight
        weighted_pole = self.pole_weight * l_pole * step_weight
        
        total = weighted_wrap + weighted_pole
        
        return {
            "wrap_loss": weighted_wrap,
            "pole_loss": weighted_pole,
            "total_loss": total,
        }
    
    def extra_repr(self) -> str:
        return (
            f"wrap_weight={self.wrap_weight}, pole_weight={self.pole_weight}, "
            f"pole_band_deg={self.pole_band_deg}, texture_boost={self.texture_step_boost}"
        )


class WeightedReconstructionLoss(nn.Module):
    """
    Weighted reconstruction loss with border emphasis.
    
    Applies higher weights to border pixels for seam-aware training.
    
    Args:
        loss_type: "l1" or "l2"
        border_width: Width of border region
        seam_boost: Weight multiplier for border pixels
    """
    
    def __init__(
        self,
        loss_type: str = "l1",
        border_width: int = 2,
        seam_boost: float = 1.5,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.border_width = border_width
        self.seam_boost = seam_boost
        self._cached_weights: Optional[Tensor] = None
        self._cached_hw: Optional[Tuple[int, int]] = None
    
    def _get_weights(self, H: int, W: int, device, dtype) -> Tensor:
        """Get or compute cached weights."""
        if self._cached_weights is None or self._cached_hw != (H, W):
            self._cached_weights = compute_seam_weights(
                H, W, self.border_width, self.seam_boost, device, dtype
            )
            self._cached_hw = (H, W)
        return self._cached_weights.to(device=device, dtype=dtype)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute weighted reconstruction loss.
        
        Args:
            pred: Predicted images, shape (B, C, H, W)
            target: Target images, shape (B, C, H, W)
        
        Returns:
            Scalar weighted loss
        """
        B, C, H, W = pred.shape
        
        # Compute per-pixel loss
        if self.loss_type == "l1":
            pixel_loss = torch.abs(pred - target)
        else:  # l2
            pixel_loss = (pred - target) ** 2
        
        # Get weights
        weights = self._get_weights(H, W, pred.device, pred.dtype)
        
        # Apply weights (broadcast over B and C)
        weighted_loss = pixel_loss * weights.view(1, 1, H, W)
        
        # Normalize by total weight to keep scale consistent
        total_weight = weights.sum()
        normalized_loss = weighted_loss.sum() / (B * C * total_weight)
        
        return normalized_loss
    
    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, border_width={self.border_width}, seam_boost={self.seam_boost}"


def cubemap_seam_loss(faces: Tensor, reduction: str = "mean") -> Tensor:
    """
    Compute seam loss for cubemap 6-face images.
    
    Measures consistency at all 12 edges between adjacent cubemap faces.
    
    Args:
        faces: Batch of cubemap faces, shape (B, 6, C, H, W)
        reduction: "mean", "sum", or "none"
    
    Returns:
        Scalar loss measuring edge consistency across all adjacent face pairs
    """
    B, num_faces, C, H, W = faces.shape
    assert num_faces == 6, f"Expected 6 faces, got {num_faces}"
    
    device = faces.device
    dtype = faces.dtype
    
    # Define adjacency: (face_a, edge_a, face_b, edge_b, needs_flip)
    # Edges: "top"=row 0, "bottom"=row -1, "left"=col 0, "right"=col -1
    adjacencies = [
        # Face 0 (front) adjacencies
        (0, "top", 4, "bottom", False),
        (0, "bottom", 5, "top", False),
        (0, "left", 3, "right", False),
        (0, "right", 1, "left", False),
        # Face 1 (right) adjacencies
        (1, "top", 4, "right", True),
        (1, "bottom", 5, "right", True),
        (1, "right", 2, "left", False),
        # Face 2 (back) adjacencies
        (2, "top", 4, "top", True),
        (2, "bottom", 5, "bottom", True),
        (2, "right", 3, "left", False),
        # Face 4 (up) and Face 5 (down) remaining - already covered
        # Face 3 (left) adjacencies with up/down
        (3, "top", 4, "left", True),
        (3, "bottom", 5, "left", True),
    ]
    
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    num_edges = 0
    
    for face_a, edge_a, face_b, edge_b, needs_flip in adjacencies:
        # Extract edges
        edge_a_vals = _extract_edge(faces[:, face_a], edge_a)  # (B, C, edge_len)
        edge_b_vals = _extract_edge(faces[:, face_b], edge_b)  # (B, C, edge_len)
        
        # Flip if needed for alignment
        if needs_flip:
            edge_b_vals = edge_b_vals.flip(-1)
        
        # Compute L1 difference
        edge_diff = torch.abs(edge_a_vals - edge_b_vals).mean()
        total_loss = total_loss + edge_diff
        num_edges += 1
    
    if reduction == "mean":
        return total_loss / num_edges
    elif reduction == "sum":
        return total_loss
    else:
        return total_loss / num_edges


def _extract_edge(face: Tensor, edge: str) -> Tensor:
    """
    Extract edge pixels from a face image.
    
    Args:
        face: Face image, shape (B, C, H, W)
        edge: "top", "bottom", "left", or "right"
    
    Returns:
        Edge values, shape (B, C, edge_length)
    """
    if edge == "top":
        return face[:, :, 0, :]       # (B, C, W)
    elif edge == "bottom":
        return face[:, :, -1, :]      # (B, C, W)
    elif edge == "left":
        return face[:, :, :, 0]       # (B, C, H)
    elif edge == "right":
        return face[:, :, :, -1]      # (B, C, H)
    else:
        raise ValueError(f"Unknown edge: {edge}")
