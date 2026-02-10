"""
Cubemap geometry utilities for 360° panorama processing.

A cubemap is an environment map represented as 6 square faces arranged in a cube.
Each face is indexed by a unique ID (0-5) and is aligned with a canonical coordinate
system where +X is right, +Y is up, and +Z is forward (following right-handed convention).

Face Layout and IDs:
- FACE_FRONT=0:  Looking down +Z axis (forward)
- FACE_RIGHT=1:  Looking down +X axis (right)
- FACE_BACK=2:   Looking down -Z axis (backward)
- FACE_LEFT=3:   Looking down -X axis (left)
- FACE_TOP=4:    Looking down +Y axis (top)
- FACE_BOTTOM=5: Looking down -Y axis (bottom)

Coordinate Convention for Each Face:
For a face looking along a primary axis, we define (u, v) coordinates on the face:
- (u, v) both range [0, 1] on a unit square face
- The center of the face is at (0.5, 0.5)
- For pixel indices (x, y) in [0, H-1] and [0, W-1]:
  u = (x + 0.5) / W,  v = (y + 0.5) / H
"""

import math
from typing import Dict, Tuple

import torch
from torch import Tensor


# Face ID constants
FACE_FRONT = 0
FACE_RIGHT = 1
FACE_BACK = 2
FACE_LEFT = 3
FACE_TOP = 4
FACE_BOTTOM = 5

# Cubemap adjacency: for each face, map neighbor name to (neighbor_face_id, shared_edge)
# Shared edge values: 'left', 'right', 'top', 'bottom'
CUBEMAP_ADJACENCY: Dict[int, Dict[str, Tuple[int, str]]] = {
    FACE_FRONT: {
        'left': (FACE_LEFT, 'right'),
        'right': (FACE_RIGHT, 'left'),
        'top': (FACE_TOP, 'bottom'),
        'bottom': (FACE_BOTTOM, 'top'),
    },
    FACE_RIGHT: {
        'left': (FACE_FRONT, 'right'),
        'right': (FACE_BACK, 'left'),
        'top': (FACE_TOP, 'right'),
        'bottom': (FACE_BOTTOM, 'right'),
    },
    FACE_BACK: {
        'left': (FACE_RIGHT, 'right'),
        'right': (FACE_LEFT, 'left'),
        'top': (FACE_TOP, 'top'),
        'bottom': (FACE_BOTTOM, 'bottom'),
    },
    FACE_LEFT: {
        'left': (FACE_BACK, 'right'),
        'right': (FACE_FRONT, 'left'),
        'top': (FACE_TOP, 'left'),
        'bottom': (FACE_BOTTOM, 'left'),
    },
    FACE_TOP: {
        'left': (FACE_LEFT, 'top'),
        'right': (FACE_RIGHT, 'top'),
        'top': (FACE_BACK, 'top'),
        'bottom': (FACE_FRONT, 'top'),
    },
    FACE_BOTTOM: {
        'left': (FACE_LEFT, 'bottom'),
        'right': (FACE_RIGHT, 'bottom'),
        'top': (FACE_FRONT, 'bottom'),
        'bottom': (FACE_BACK, 'bottom'),
    },
}


def get_face_direction_vectors(
    face_id: int,
    H: int,
    W: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute 3D direction vectors for all pixel positions on a cubemap face.

    For each pixel (x, y) on the face, returns a unit direction vector pointing
    from the origin (center of the cube) through that pixel.

    Args:
        face_id: Face ID (0-5), one of FACE_* constants
        H: Face height in pixels
        W: Face width in pixels
        device: Device to create tensor on
        dtype: Data type for the tensor

    Returns:
        directions: Unit direction vectors, shape (H, W, 3)
                    directions[y, x] is the direction for pixel at (x, y)
    """
    # Create normalized coordinates in [0, 1] range
    y_coords = torch.arange(H, device=device, dtype=dtype)
    x_coords = torch.arange(W, device=device, dtype=dtype)

    # Create 2D grids
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    u_grid = (xx + 0.5) / W
    v_grid = (yy + 0.5) / H

    # Map from [0, 1] to [-1, 1] (face-local coordinates)
    u_local = 2.0 * u_grid - 1.0  # [-1, 1]
    v_local = 2.0 * v_grid - 1.0  # [-1, 1]

    # Compute 3D direction vectors based on face orientation
    # Each face is a unit square at distance 1 from the origin along a primary axis
    if face_id == FACE_FRONT:
        # Looking down +Z: right=+X, up=+Y
        dx = u_local
        dy = -v_local  # v=0 is top (positive Y in 3D), so negate for proper mapping
        dz = torch.ones_like(u_local)
    elif face_id == FACE_RIGHT:
        # Looking down +X: right=-Z, up=+Y
        dx = torch.ones_like(u_local)
        dy = -v_local
        dz = -u_local
    elif face_id == FACE_BACK:
        # Looking down -Z: right=-X, up=+Y
        dx = -u_local
        dy = -v_local
        dz = -torch.ones_like(u_local)
    elif face_id == FACE_LEFT:
        # Looking down -X: right=+Z, up=+Y
        dx = -torch.ones_like(u_local)
        dy = -v_local
        dz = u_local
    elif face_id == FACE_TOP:
        # Looking down +Y: right=+X, down=+Z (away from camera)
        dx = u_local
        dy = torch.ones_like(u_local)
        dz = v_local
    elif face_id == FACE_BOTTOM:
        # Looking down -Y: right=+X, down=-Z (away from camera)
        dx = u_local
        dy = -torch.ones_like(u_local)
        dz = -v_local
    else:
        raise ValueError(f"Invalid face_id: {face_id}")

    # Stack and normalize to unit vectors
    directions = torch.stack([dx, dy, dz], dim=-1)
    directions = torch.nn.functional.normalize(directions, p=2, dim=-1)

    return directions


def get_cubemap_all_directions(
    H: int,
    W: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute 3D direction vectors for all 6 cubemap faces.

    Args:
        H: Face height in pixels
        W: Face width in pixels
        device: Device to create tensor on
        dtype: Data type for the tensor

    Returns:
        directions: Direction vectors for all faces, shape (6, H, W, 3)
                    directions[face_id, y, x] is the direction for that position
    """
    directions_list = []
    for face_id in range(6):
        face_dirs = get_face_direction_vectors(face_id, H, W, device=device, dtype=dtype)
        directions_list.append(face_dirs)

    directions = torch.stack(directions_list, dim=0)
    return directions


def get_shared_border_mask(
    face_id: int,
    neighbor_face: int,
    H: int,
    W: int,
    border_width: int = 1,
    device: str = "cpu",
) -> Tensor:
    """
    Create a boolean mask for tokens on the shared edge between two cubemap faces.

    Args:
        face_id: ID of the primary face
        neighbor_face: ID of the neighboring face
        H: Face height in pixels
        W: Face width in pixels
        border_width: Width of the border region in pixels
        device: Device to create tensor on

    Returns:
        mask: Boolean mask, shape (H, W), True for border tokens
    """
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)

    # Determine which edge of face_id is shared with neighbor_face
    adjacency = CUBEMAP_ADJACENCY[face_id]
    shared_edge = None
    for direction, (neighbor_id, _) in adjacency.items():
        if neighbor_id == neighbor_face:
            shared_edge = direction
            break

    if shared_edge is None:
        return mask

    # Mark the appropriate border region based on shared edge
    if shared_edge == 'left':
        mask[:, :border_width] = True
    elif shared_edge == 'right':
        mask[:, -border_width:] = True
    elif shared_edge == 'top':
        mask[:border_width, :] = True
    elif shared_edge == 'bottom':
        mask[-border_width:, :] = True

    return mask


def cubemap_to_erp(
    cubemap: Tensor,
    erp_height: int,
) -> Tensor:
    """
    Convert a cubemap (6 faces) to Equirectangular Panorama (ERP) format.

    Args:
        cubemap: Cubemap tensor with shape (6, C, H, W) where:
                 - First dimension indexes the 6 faces in order [front, right, back, left, top, bottom]
                 - C is the number of channels
                 - H, W are face dimensions
        erp_height: Height of the output ERP image

    Returns:
        erp: ERP image with shape (C, erp_height, erp_height*2)
    """
    _, C, face_size, _ = cubemap.shape
    device = cubemap.device
    dtype = cubemap.dtype

    erp_width = erp_height * 2

    # Create ERP coordinate grids
    y_coords = torch.arange(erp_height, device=device, dtype=dtype)
    x_coords = torch.arange(erp_width, device=device, dtype=dtype)
    yy_erp, xx_erp = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Convert ERP coordinates to longitude/latitude
    theta = 2.0 * math.pi * ((xx_erp + 0.5) / erp_width) - math.pi
    phi = math.pi / 2.0 - math.pi * ((yy_erp + 0.5) / erp_height)

    # Convert lon/lat to 3D direction vectors
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    dx = cos_phi * cos_theta
    dy = sin_phi
    dz = cos_phi * sin_theta

    # Determine which face each direction vector belongs to
    abs_dx = torch.abs(dx)
    abs_dy = torch.abs(dy)
    abs_dz = torch.abs(dz)

    # Find dominant axis for each direction
    is_front = (dz >= abs_dx) & (dz >= abs_dy)
    is_right = (dx >= abs_dy) & (dx >= abs_dz)
    is_back = (-dz >= abs_dx) & (-dz >= abs_dy)
    is_left = (-dx >= abs_dy) & (-dx >= abs_dz)
    is_top = (dy >= abs_dx) & (dy >= abs_dz)
    is_bottom = (-dy >= abs_dx) & (-dy >= abs_dz)

    # Initialize output
    erp = torch.zeros(C, erp_height, erp_width, device=device, dtype=dtype)

    # For each cubemap face, sample the corresponding pixels
    for face_id in range(6):
        if face_id == FACE_FRONT:
            mask = is_front
            u = dx / (dz + 1e-8)
            v = -dy / (dz + 1e-8)
        elif face_id == FACE_RIGHT:
            mask = is_right
            u = -dz / (dx + 1e-8)
            v = -dy / (dx + 1e-8)
        elif face_id == FACE_BACK:
            mask = is_back
            u = -dx / (-dz + 1e-8)
            v = -dy / (-dz + 1e-8)
        elif face_id == FACE_LEFT:
            mask = is_left
            u = dz / (-dx + 1e-8)
            v = -dy / (-dx + 1e-8)
        elif face_id == FACE_TOP:
            mask = is_top
            u = dx / (dy + 1e-8)
            v = dz / (dy + 1e-8)
        else:  # FACE_BOTTOM
            mask = is_bottom
            u = dx / (-dy + 1e-8)
            v = -dz / (-dy + 1e-8)

        # Convert from [-1, 1] to face pixel coordinates
        x_face = (u + 1.0) / 2.0 * face_size - 0.5
        y_face = (v + 1.0) / 2.0 * face_size - 0.5

        # Clamp to valid face coordinates
        x_face = torch.clamp(x_face, 0, face_size - 1)
        y_face = torch.clamp(y_face, 0, face_size - 1)

        # Bilinear sampling
        x_floor = torch.floor(x_face).long()
        y_floor = torch.floor(y_face).long()
        x_ceil = torch.clamp(x_floor + 1, max=face_size - 1)
        y_ceil = torch.clamp(y_floor + 1, max=face_size - 1)

        wx = x_face - x_floor.float()
        wy = y_face - y_floor.float()

        # Gather 4 corners
        v00 = cubemap[face_id, :, y_floor, x_floor]  # (C, H, W)
        v01 = cubemap[face_id, :, y_floor, x_ceil]
        v10 = cubemap[face_id, :, y_ceil, x_floor]
        v11 = cubemap[face_id, :, y_ceil, x_ceil]

        # Bilinear interpolation
        v0 = v00 * (1 - wx) + v01 * wx  # (C, H, W)
        v1 = v10 * (1 - wx) + v11 * wx
        v_interp = v0 * (1 - wy) + v1 * wy  # (C, H, W)

        # Apply mask and accumulate
        mask_expanded = mask.unsqueeze(0)  # (1, H, W)
        erp = torch.where(mask_expanded, v_interp, erp)

    return erp


def erp_to_cubemap(
    erp: Tensor,
    face_size: int,
) -> Tensor:
    """
    Convert an Equirectangular Panorama (ERP) image to cubemap (6 faces) format.

    Args:
        erp: ERP image with shape (C, H, W) where H/W = 1/2
        face_size: Size of each cubemap face (output will be 6, C, face_size, face_size)

    Returns:
        cubemap: Cubemap tensor with shape (6, C, face_size, face_size)
    """
    C, erp_height, erp_width = erp.shape
    device = str(erp.device)
    dtype = erp.dtype

    # Create cubemap
    cubemap = torch.zeros(6, C, face_size, face_size, device=erp.device, dtype=dtype)

    # For each face, compute directions and sample from ERP
    for face_id in range(6):
        # Get direction vectors for this face
        face_dirs = get_face_direction_vectors(face_id, face_size, face_size, device=device, dtype=dtype)

        # Extract direction components
        dx = face_dirs[..., 0]
        dy = face_dirs[..., 1]
        dz = face_dirs[..., 2]

        # Convert 3D direction to longitude/latitude
        theta = torch.atan2(dz, dx)
        phi = torch.asin(torch.clamp(dy, -1.0, 1.0))

        # Convert lon/lat to ERP pixel coordinates
        x_erp = (theta + math.pi) * erp_width / (2.0 * math.pi) - 0.5
        y_erp = (math.pi / 2.0 - phi) * erp_height / math.pi - 0.5

        # Handle wraparound in x direction
        x_erp = x_erp % erp_width

        # Clamp y to valid range
        y_erp = torch.clamp(y_erp, 0, erp_height - 1)

        # Bilinear sampling
        x_floor = torch.floor(x_erp).long()
        y_floor = torch.floor(y_erp).long()
        x_ceil = (x_floor + 1) % erp_width
        y_ceil = torch.clamp(y_floor + 1, max=erp_height - 1)

        wx = x_erp - x_floor.float()
        wy = y_erp - y_floor.float()

        # Gather 4 corners
        v00 = erp[:, y_floor, x_floor]  # (C, face_size, face_size)
        v01 = erp[:, y_floor, x_ceil]
        v10 = erp[:, y_ceil, x_floor]
        v11 = erp[:, y_ceil, x_ceil]

        # Bilinear interpolation
        v0 = v00 * (1 - wx.unsqueeze(0)) + v01 * wx.unsqueeze(0)
        v1 = v10 * (1 - wx.unsqueeze(0)) + v11 * wx.unsqueeze(0)
        v_interp = v0 * (1 - wy.unsqueeze(0)) + v1 * wy.unsqueeze(0)

        cubemap[face_id] = v_interp

    return cubemap
