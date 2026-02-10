"""
Spherical geometry utilities for Equirectangular Panorama (ERP) processing.

Coordinate Conventions:
- ERP image has shape (H, W) where W = 2*H (2:1 aspect ratio)
- x: horizontal pixel coordinate [0, W-1], corresponds to longitude
- y: vertical pixel coordinate [0, H-1], corresponds to latitude
- theta (longitude): [-pi, pi], where x=0 -> theta=-pi, x=W-1 -> theta=+pi
- phi (latitude): [-pi/2, pi/2], where y=0 -> phi=+pi/2 (north pole), y=H-1 -> phi=-pi/2 (south pole)
- 3D direction vector: [x, y, z] = [cos(phi)*cos(theta), sin(phi), cos(phi)*sin(theta)]
  - This follows a right-handed coordinate system where:
    - +Y is up (north pole)
    - +Z is forward at theta=0
    - +X is right at theta=pi/2
"""

import math
from typing import Tuple

import torch
from torch import Tensor


def pixel_to_lonlat(x: Tensor, y: Tensor, W: int, H: int) -> Tuple[Tensor, Tensor]:
    """
    Convert ERP pixel coordinates to longitude/latitude angles.
    
    Args:
        x: Horizontal pixel coordinates, any shape
        y: Vertical pixel coordinates, same shape as x
        W: Image width
        H: Image height
    
    Returns:
        theta: Longitude in radians [-pi, pi], same shape as x
        phi: Latitude in radians [-pi/2, pi/2], same shape as y
    """
    # theta = 2*pi*((x+0.5)/W) - pi
    theta = 2.0 * math.pi * ((x + 0.5) / W) - math.pi
    
    # phi = pi/2 - pi*((y+0.5)/H)
    phi = math.pi / 2.0 - math.pi * ((y + 0.5) / H)
    
    return theta, phi


def lonlat_to_direction(theta: Tensor, phi: Tensor) -> Tensor:
    """
    Convert longitude/latitude angles to 3D unit direction vectors.
    
    Args:
        theta: Longitude in radians [-pi, pi], any shape
        phi: Latitude in radians [-pi/2, pi/2], same shape as theta
    
    Returns:
        directions: Unit direction vectors, shape (*input_shape, 3)
    """
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # d = [cos(phi)*cos(theta), sin(phi), cos(phi)*sin(theta)]
    dx = cos_phi * cos_theta
    dy = sin_phi
    dz = cos_phi * sin_theta
    
    # Stack along last dimension
    directions = torch.stack([dx, dy, dz], dim=-1)
    
    return directions


def pixel_to_direction(x: Tensor, y: Tensor, W: int, H: int) -> Tensor:
    """
    Convert ERP pixel coordinates directly to 3D unit direction vectors.
    
    Args:
        x: Horizontal pixel coordinates, any shape
        y: Vertical pixel coordinates, same shape as x
        W: Image width
        H: Image height
    
    Returns:
        directions: Unit direction vectors, shape (*input_shape, 3)
    """
    theta, phi = pixel_to_lonlat(x, y, W, H)
    return lonlat_to_direction(theta, phi)


def erp_direction_grid(
    H: int,
    W: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Pre-compute direction vectors for all pixels in an ERP image.
    
    Args:
        H: Image height
        W: Image width
        device: Device to create tensor on
        dtype: Data type for the tensor
    
    Returns:
        directions: Direction vectors, shape (H, W, 3)
    """
    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=dtype)
    x_coords = torch.arange(W, device=device, dtype=dtype)
    
    # Create meshgrid (y, x ordering for image coordinates)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    # Convert to directions
    directions = pixel_to_direction(xx, yy, W, H)
    
    return directions


def spherical_distance_matrix(dirs1: Tensor, dirs2: Tensor) -> Tensor:
    """
    Compute pairwise geodesic (great-circle) distances between direction vectors.
    
    Args:
        dirs1: Direction vectors, shape (N, 3)
        dirs2: Direction vectors, shape (M, 3)
    
    Returns:
        distances: Pairwise distances in radians, shape (N, M)
    """
    # Compute dot products: (N, 3) @ (3, M) -> (N, M)
    cos_gamma = torch.mm(dirs1, dirs2.t())
    
    # Clamp to avoid numerical issues with arccos
    cos_gamma = cos_gamma.clamp(-1.0, 1.0)
    
    # gamma = arccos(cos_gamma)
    distances = torch.acos(cos_gamma)
    
    return distances


def spherical_distance_approx(dirs1: Tensor, dirs2: Tensor) -> Tensor:
    """
    Fast approximation of squared geodesic distance for attention bias.
    
    Uses the approximation: gamma^2 ≈ 2*(1 - cos_gamma) where cos_gamma = dot(d1, d2)
    This is accurate for small angles and avoids the expensive arccos operation.
    
    Args:
        dirs1: Direction vectors, shape (N, 3) or (..., N, 3)
        dirs2: Direction vectors, shape (M, 3) or (..., M, 3)
    
    Returns:
        sq_distances: Approximate squared distances, shape (N, M) or (..., N, M)
    """
    # Handle different input shapes
    if dirs1.dim() == 2 and dirs2.dim() == 2:
        # Simple case: (N, 3) and (M, 3)
        cos_gamma = torch.mm(dirs1, dirs2.t())
    else:
        # Batched case using einsum
        cos_gamma = torch.einsum("...id,...jd->...ij", dirs1, dirs2)
    
    # gamma^2 ≈ 2*(1 - cos_gamma)
    sq_distances = 2.0 * (1.0 - cos_gamma)
    
    return sq_distances


def get_pole_mask(
    H: int,
    W: int,
    pole_band_deg: float = 20.0,
    device: str = "cpu",
) -> Tensor:
    """
    Create a boolean mask marking pixels within pole_band_deg of the poles.
    
    Args:
        H: Image height
        W: Image width
        pole_band_deg: Angular distance from poles in degrees
        device: Device to create tensor on
    
    Returns:
        mask: Boolean mask, shape (H, W), True for pole region pixels
    """
    pole_band_rad = pole_band_deg * math.pi / 180.0
    
    # Create y coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    
    # Convert to latitude
    phi = math.pi / 2.0 - math.pi * ((y_coords + 0.5) / H)
    
    # Check if within pole band
    # North pole: phi > pi/2 - pole_band_rad
    # South pole: phi < -pi/2 + pole_band_rad
    north_pole_threshold = math.pi / 2.0 - pole_band_rad
    south_pole_threshold = -math.pi / 2.0 + pole_band_rad
    
    is_pole = (phi > north_pole_threshold) | (phi < south_pole_threshold)
    
    # Expand to (H, W)
    mask = is_pole.unsqueeze(1).expand(H, W)
    
    return mask


def get_wrap_seam_indices(H: int, W: int) -> Tuple[Tensor, Tensor]:
    """
    Get pixel indices for the left and right edges (wrap seam).
    
    The wrap seam is where x=0 (left edge) should connect to x=W-1 (right edge)
    for a seamless 360° panorama.
    
    Args:
        H: Image height
        W: Image width
    
    Returns:
        left_indices: Indices for left edge, shape (H,)
        right_indices: Indices for right edge, shape (H,)
    """
    # Flat indices for left column (x=0)
    left_indices = torch.arange(H) * W  # [0, W, 2W, 3W, ...]
    
    # Flat indices for right column (x=W-1)
    right_indices = torch.arange(H) * W + (W - 1)  # [W-1, 2W-1, 3W-1, ...]
    
    return left_indices, right_indices


def get_border_mask(
    H: int,
    W: int,
    border_width: int = 2,
    device: str = "cpu",
) -> Tensor:
    """
    Create a boolean mask marking border pixels (left and right edges).
    
    Args:
        H: Image height
        W: Image width
        border_width: Width of border in pixels
        device: Device to create tensor on
    
    Returns:
        mask: Boolean mask, shape (H, W), True for border pixels
    """
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    
    # Left border
    mask[:, :border_width] = True
    
    # Right border
    mask[:, -border_width:] = True
    
    return mask


def lonlat_to_pixel(
    theta: Tensor,
    phi: Tensor,
    W: int,
    H: int,
) -> Tuple[Tensor, Tensor]:
    """
    Convert longitude/latitude angles to ERP pixel coordinates.
    
    Inverse of pixel_to_lonlat.
    
    Args:
        theta: Longitude in radians [-pi, pi]
        phi: Latitude in radians [-pi/2, pi/2]
        W: Image width
        H: Image height
    
    Returns:
        x: Horizontal pixel coordinates (float)
        y: Vertical pixel coordinates (float)
    """
    # Inverse of theta = 2*pi*((x+0.5)/W) - pi
    # x = (theta + pi) * W / (2*pi) - 0.5
    x = (theta + math.pi) * W / (2.0 * math.pi) - 0.5
    
    # Inverse of phi = pi/2 - pi*((y+0.5)/H)
    # y = (pi/2 - phi) * H / pi - 0.5
    y = (math.pi / 2.0 - phi) * H / math.pi - 0.5
    
    return x, y


def direction_to_lonlat(directions: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert 3D direction vectors to longitude/latitude angles.
    
    Args:
        directions: Unit direction vectors, shape (..., 3)
    
    Returns:
        theta: Longitude in radians [-pi, pi]
        phi: Latitude in radians [-pi/2, pi/2]
    """
    dx = directions[..., 0]
    dy = directions[..., 1]
    dz = directions[..., 2]
    
    # phi = arcsin(dy)
    phi = torch.asin(dy.clamp(-1.0, 1.0))
    
    # theta = atan2(dz, dx)
    theta = torch.atan2(dz, dx)
    
    return theta, phi


def direction_to_pixel(
    directions: Tensor,
    W: int,
    H: int,
) -> Tuple[Tensor, Tensor]:
    """
    Convert 3D direction vectors to ERP pixel coordinates.
    
    Args:
        directions: Unit direction vectors, shape (..., 3)
        W: Image width
        H: Image height
    
    Returns:
        x: Horizontal pixel coordinates (float)
        y: Vertical pixel coordinates (float)
    """
    theta, phi = direction_to_lonlat(directions)
    return lonlat_to_pixel(theta, phi, W, H)


def spherical_neighbor_weights(
    H: int,
    W: int,
    sigma_deg: float = 10.0,
    max_neighbors: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Compute neighbor weights based on spherical distance for pole consistency.
    
    For each pixel, computes weights for its immediate neighbors based on
    geodesic distance, with Gaussian weighting.
    
    Args:
        H: Image height
        W: Image width
        sigma_deg: Sigma for Gaussian weighting in degrees
        max_neighbors: Maximum number of neighbors per pixel (4 or 8)
        device: Device to create tensors on
        dtype: Data type for tensors
    
    Returns:
        neighbor_offsets: Relative positions of neighbors, shape (num_neighbors, 2)
        weights_grid: Weights for each pixel's neighbors, shape (H, W, num_neighbors)
    """
    sigma_rad = sigma_deg * math.pi / 180.0
    
    # Define neighbor offsets (dy, dx)
    if max_neighbors == 4:
        offsets = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=device)
    else:
        offsets = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],          [0, 1],
            [1, -1], [1, 0], [1, 1]
        ], device=device)
    
    num_neighbors = offsets.shape[0]
    
    # Get direction grid
    directions = erp_direction_grid(H, W, device=device, dtype=dtype)
    
    # Initialize weights
    weights = torch.zeros(H, W, num_neighbors, device=device, dtype=dtype)
    
    for n, (dy, dx) in enumerate(offsets):
        # Compute neighbor coordinates with wrapping
        y_neighbor = torch.arange(H, device=device) + dy
        x_neighbor = torch.arange(W, device=device) + dx
        
        # Clamp y (no vertical wrapping)
        y_neighbor = y_neighbor.clamp(0, H - 1)
        
        # Wrap x (horizontal wrapping)
        x_neighbor = x_neighbor % W
        
        # Get directions for all pixels and their neighbors
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        yy_n, xx_n = torch.meshgrid(y_neighbor, x_neighbor, indexing="ij")
        
        dirs_center = directions[yy, xx]  # (H, W, 3)
        dirs_neighbor = directions[yy_n, xx_n]  # (H, W, 3)
        
        # Compute dot product
        cos_gamma = (dirs_center * dirs_neighbor).sum(dim=-1)
        cos_gamma = cos_gamma.clamp(-1.0, 1.0)
        
        # Compute weight: exp(-gamma^2 / sigma^2)
        gamma_sq = 2.0 * (1.0 - cos_gamma)  # approximation
        weight = torch.exp(-gamma_sq / (sigma_rad ** 2))
        
        weights[:, :, n] = weight
    
    return offsets, weights
