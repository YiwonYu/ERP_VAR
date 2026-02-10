"""
Evaluation metrics for panorama seam quality.

This module provides metric functions to measure the quality of panorama seams:
- Wrap seam metrics: measure continuity between left-right edges in ERP images
- Pole consistency metrics: measure uniformity in polar regions
- Cubemap seam metrics: measure edge alignment across cubemap faces

All metrics are differentiable and support GPU tensors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def wrap_seam_mse(images: Tensor) -> Tensor:
    """
    Compute Mean Squared Error between left and right edges of ERP images.

    For a seamless 360° panorama, the left edge (x=0) should match the right
    edge (x=W-1) since they represent the same longitude in world space.

    Args:
        images: Batch of ERP images, shape (B, C, H, W)

    Returns:
        Per-batch MSE, shape (B,)
    """
    # Extract left and right edges
    left_edge = images[:, :, :, 0]       # (B, C, H)
    right_edge = images[:, :, :, -1]     # (B, C, H)

    # Compute squared difference
    edge_diff = (left_edge - right_edge) ** 2  # (B, C, H)

    # Reduce over channels and height
    mse = edge_diff.mean(dim=(1, 2))  # (B,)

    return mse


def wrap_seam_mae(images: Tensor) -> Tensor:
    """
    Compute Mean Absolute Error between left and right edges of ERP images.

    For a seamless 360° panorama, the left edge (x=0) should match the right
    edge (x=W-1) since they represent the same longitude in world space.

    Args:
        images: Batch of ERP images, shape (B, C, H, W)

    Returns:
        Per-batch MAE, shape (B,)
    """
    # Extract left and right edges
    left_edge = images[:, :, :, 0]       # (B, C, H)
    right_edge = images[:, :, :, -1]     # (B, C, H)

    # Compute absolute difference
    edge_diff = torch.abs(left_edge - right_edge)  # (B, C, H)

    # Reduce over channels and height
    mae = edge_diff.mean(dim=(1, 2))  # (B,)

    return mae


def pole_consistency_score(images: Tensor, pole_band_deg: float = 20.0) -> Tensor:
    """
    Measure variance within polar regions of ERP images.

    Pixels near the poles (north/south) should have consistent values since
    they represent a small area that's stretched across the entire width.
    Lower variance = more consistent pole rendering.

    Args:
        images: Batch of ERP images, shape (B, C, H, W)
        pole_band_deg: Degrees from pole to consider as pole region (default 20°)

    Returns:
        Per-batch consistency score (inverse of variance), shape (B,).
        Higher score = better consistency.
    """
    B, _, H, _ = images.shape
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
        # No pole region, return high score (perfect consistency)
        return torch.ones(B, device=device, dtype=dtype)

    # For pole rows, compute variance across width
    pole_pixels = images[:, :, pole_row_indices, :]  # (B, C, num_pole_rows, W)

    # Compute per-row variance across width
    row_mean = pole_pixels.mean(dim=-1, keepdim=True)  # (B, C, num_pole_rows, 1)
    row_var = ((pole_pixels - row_mean) ** 2).mean(dim=-1)  # (B, C, num_pole_rows)

    # Average variance across pole region
    avg_var = row_var.mean(dim=(1, 2))  # (B,)

    # Convert variance to score: lower variance = higher score
    # Use inverse: score = 1 / (1 + variance)
    score = 1.0 / (1.0 + avg_var)

    return score


def cubemap_seam_mse(faces: Tensor) -> Tensor:
    """
    Compute MSE across all shared edges between adjacent cubemap faces.

    Measures seam quality across the 12 edges in a cubemap.

    Args:
        faces: Batch of cubemap faces, shape (B, 6, C, H, W) or (6, B, C, H, W)

    Returns:
        Per-batch average seam MSE, shape (B,)
    """
    # Handle both (B, 6, C, H, W) and (6, B, C, H, W) formats
    if faces.shape[1] == 6:
        # Format: (B, 6, C, H, W)
        b, num_faces, _, _, _ = faces.shape
    elif faces.shape[0] == 6:
        # Format: (6, B, C, H, W) - transpose to (B, 6, C, H, W)
        num_faces, b, _, _, _ = faces.shape
        faces = faces.transpose(0, 1)
    else:
        raise ValueError(f"Expected shape (B, 6, C, H, W) or (6, B, C, H, W), got {faces.shape}")

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
        # Face 3 (left) adjacencies with up/down
        (3, "top", 4, "left", True),
        (3, "bottom", 5, "left", True),
    ]

    total_mse = torch.zeros(b, device=device, dtype=dtype)
    num_edges = 0

    for face_a, edge_a, face_b, edge_b, needs_flip in adjacencies:
        # Extract edges
        edge_a_vals = _extract_cubemap_edge(faces[:, face_a], edge_a)  # (B, C, edge_len)
        edge_b_vals = _extract_cubemap_edge(faces[:, face_b], edge_b)  # (B, C, edge_len)

        # Flip if needed for alignment
        if needs_flip:
            edge_b_vals = edge_b_vals.flip(-1)

        # Compute MSE per batch
        edge_diff = (edge_a_vals - edge_b_vals) ** 2  # (B, C, edge_len)
        edge_mse = edge_diff.mean(dim=(1, 2))  # (B,)

        total_mse = total_mse + edge_mse
        num_edges += 1

    # Average MSE across all edges
    avg_mse = total_mse / num_edges

    return avg_mse


def cubemap_seam_mae(faces: Tensor) -> Tensor:
    """
    Compute MAE across all shared edges between adjacent cubemap faces.

    Measures seam quality across the 12 edges in a cubemap.

    Args:
        faces: Batch of cubemap faces, shape (B, 6, C, H, W) or (6, B, C, H, W)

    Returns:
        Per-batch average seam MAE, shape (B,)
    """
    # Handle both (B, 6, C, H, W) and (6, B, C, H, W) formats
    if faces.shape[1] == 6:
        # Format: (B, 6, C, H, W)
        b, num_faces, _, _, _ = faces.shape
    elif faces.shape[0] == 6:
        # Format: (6, B, C, H, W) - transpose to (B, 6, C, H, W)
        num_faces, b, _, _, _ = faces.shape
        faces = faces.transpose(0, 1)
    else:
        raise ValueError(f"Expected shape (B, 6, C, H, W) or (6, B, C, H, W), got {faces.shape}")

    assert num_faces == 6, f"Expected 6 faces, got {num_faces}"

    device = faces.device
    dtype = faces.dtype

    # Define adjacency: (face_a, edge_a, face_b, edge_b, needs_flip)
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
        # Face 3 (left) adjacencies with up/down
        (3, "top", 4, "left", True),
        (3, "bottom", 5, "left", True),
    ]

    total_mae = torch.zeros(b, device=device, dtype=dtype)
    num_edges = 0

    for face_a, edge_a, face_b, edge_b, needs_flip in adjacencies:
        # Extract edges
        edge_a_vals = _extract_cubemap_edge(faces[:, face_a], edge_a)  # (B, C, edge_len)
        edge_b_vals = _extract_cubemap_edge(faces[:, face_b], edge_b)  # (B, C, edge_len)

        # Flip if needed for alignment
        if needs_flip:
            edge_b_vals = edge_b_vals.flip(-1)

        # Compute MAE per batch
        edge_diff = torch.abs(edge_a_vals - edge_b_vals)  # (B, C, edge_len)
        edge_mae = edge_diff.mean(dim=(1, 2))  # (B,)

        total_mae = total_mae + edge_mae
        num_edges += 1

    # Average MAE across all edges
    avg_mae = total_mae / num_edges

    return avg_mae


def compute_all_pano_metrics(images: Tensor, mode: str = "erp") -> dict[str, Tensor]:
    """
    Compute all panorama evaluation metrics.

    Args:
        images: Input images
                - For mode="erp": shape (B, C, H, W)
                - For mode="cubemap": shape (B, 6, C, H, W) or (6, B, C, H, W)
        mode: "erp" for equirectangular or "cubemap" for cubemap format

    Returns:
        Dictionary with metrics:
        - "wrap_seam_mse": Per-batch wrap seam MSE (for ERP)
        - "wrap_seam_mae": Per-batch wrap seam MAE (for ERP)
        - "pole_consistency_score": Per-batch pole consistency (for ERP)
        - "cubemap_seam_mse": Per-batch seam MSE (for cubemap)
        - "cubemap_seam_mae": Per-batch seam MAE (for cubemap)
    """
    metrics = {}

    if mode == "erp":
        assert images.dim() == 4, f"ERP mode expects (B, C, H, W), got shape {images.shape}"
        metrics["wrap_seam_mse"] = wrap_seam_mse(images)
        metrics["wrap_seam_mae"] = wrap_seam_mae(images)
        metrics["pole_consistency_score"] = pole_consistency_score(images)

    elif mode == "cubemap":
        assert images.dim() in (4, 5), f"Cubemap mode expects (B, 6, C, H, W) or (6, B, C, H, W), got shape {images.shape}"
        metrics["cubemap_seam_mse"] = cubemap_seam_mse(images)
        metrics["cubemap_seam_mae"] = cubemap_seam_mae(images)

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'erp' or 'cubemap'.")

    return metrics


def _extract_cubemap_edge(face: Tensor, edge: str) -> Tensor:
    """
    Extract edge pixels from a cubemap face image.

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
