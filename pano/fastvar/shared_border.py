"""
Shared border latent synchronization for cubemap 6-face generation.

This module implements cross-face border synchronization for seamless cubemap 
generation in text-to-image models. It supports two modes:
- "avg": Average overlapping latents from adjacent faces
- "copy_owner": One face owns each edge, propagating its border to neighbors
"""

import torch
from torch import Tensor


# Cubemap face adjacency mapping
# Maps each face ID to its neighbors with the edge relationship
# Face IDs: 0=right, 1=left, 2=top, 3=bottom, 4=front, 5=back
CUBEMAP_ADJACENCY: dict[int, dict[str, tuple[int, str]]] = {
    # Right face (0)
    0: {
        "left": (4, "right"),      # connects to front's right edge
        "right": (5, "left"),      # connects to back's left edge
        "top": (2, "right"),       # connects to top's right edge
        "bottom": (3, "right"),    # connects to bottom's right edge
    },
    # Left face (1)
    1: {
        "left": (5, "right"),      # connects to back's right edge
        "right": (4, "left"),      # connects to front's left edge
        "top": (2, "left"),        # connects to top's left edge
        "bottom": (3, "left"),     # connects to bottom's left edge
    },
    # Top face (2)
    2: {
        "left": (1, "top"),        # connects to left's top edge
        "right": (0, "top"),       # connects to right's top edge
        "top": (5, "top"),         # connects to back's top edge
        "bottom": (4, "top"),      # connects to front's top edge
    },
    # Bottom face (3)
    3: {
        "left": (1, "bottom"),     # connects to left's bottom edge
        "right": (0, "bottom"),    # connects to right's bottom edge
        "top": (4, "bottom"),      # connects to front's bottom edge
        "bottom": (5, "bottom"),   # connects to back's bottom edge
    },
    # Front face (4)
    4: {
        "left": (1, "right"),      # connects to left's right edge
        "right": (0, "left"),      # connects to right's left edge
        "top": (2, "bottom"),      # connects to top's bottom edge
        "bottom": (3, "top"),      # connects to bottom's top edge
    },
    # Back face (5)
    5: {
        "left": (0, "right"),      # connects to right's right edge
        "right": (1, "left"),      # connects to left's left edge
        "top": (2, "top"),         # connects to top's top edge
        "bottom": (3, "bottom"),   # connects to bottom's bottom edge
    },
}


class SharedBorderLatent:
    """
    Synchronizes latent borders across cubemap faces for seamless generation.
    
    Supports two modes:
    - "avg": Average latents from adjacent faces at shared borders
    - "copy_owner": One face owns each edge, propagating to neighbors
    """

    def __init__(self, mode: str = "avg", border_width: int = 2):
        """
        Initialize shared border synchronizer.
        
        Args:
            mode: "avg" or "copy_owner" for border synchronization
            border_width: Number of tokens at each border to synchronize
        """
        if mode not in ("avg", "copy_owner"):
            raise ValueError(f"mode must be 'avg' or 'copy_owner', got {mode}")
        
        self.mode: str = mode
        self.border_width: int = border_width

    def get_border_indices(
        self, edge: str, H: int, W: int
    ) -> Tensor:
        """
        Get indices of border tokens for a specified edge.
        
        Args:
            edge: Edge name ("left", "right", "top", "bottom")
            H: Spatial height of face latent
            W: Spatial width of face latent
        
        Returns:
            Boolean mask of shape (H, W), True for border tokens
        """
        if edge not in ("left", "right", "top", "bottom"):
            raise ValueError(f"edge must be left/right/top/bottom, got {edge}")
        
        mask = torch.zeros(H, W, dtype=torch.bool)
        
        if edge == "left":
            mask[:, :self.border_width] = True
        elif edge == "right":
            mask[:, -self.border_width:] = True
        elif edge == "top":
            mask[:self.border_width, :] = True
        elif edge == "bottom":
            mask[-self.border_width:, :] = True
        
        return mask

    def synchronize(self, face_latents: Tensor) -> Tensor:
        """
        Synchronize borders across all cubemap faces.
        
        Args:
            face_latents: Cubemap face latents of shape (6, B, C, H, W) or (B, 6, C, H, W)
        
        Returns:
            Synchronized latents with same shape as input
        """
        # Detect input format and normalize to (6, B, C, H, W)
        if face_latents.shape[0] == 6:
            # Already (6, B, C, H, W)
            latents = face_latents
            shape_order = "faces_first"
        elif face_latents.shape[1] == 6:
            # (B, 6, C, H, W) -> transpose to (6, B, C, H, W)
            latents = face_latents.transpose(0, 1)
            shape_order = "batch_first"
        else:
            raise ValueError(
                f"Input must have 6 faces in dim 0 or 1, got shape {face_latents.shape}"
            )
        
        num_faces, B, C, H, W = latents.shape
        
        # Create output tensor as copy to avoid in-place modification issues
        synchronized = latents.clone()
        
        # Process each face and its borders
        for face_id in range(num_faces):
            neighbors = CUBEMAP_ADJACENCY[face_id]
            
            for edge, (neighbor_id, neighbor_edge) in neighbors.items():
                # Get border masks
                face_mask = self.get_border_indices(edge, H, W)
                neighbor_mask = self.get_border_indices(neighbor_edge, H, W)
                
                # Extract border latents (B, C, H, W) -> (B, C, border_tokens)
                face_border = latents[face_id, :, :, face_mask].view(B, C, -1)
                neighbor_border = latents[neighbor_id, :, :, neighbor_mask].view(B, C, -1)
                
                if self.mode == "avg":
                    # Average the two borders
                    synchronized_border = (face_border + neighbor_border) / 2.0
                    
                    # Update both faces with synchronized border
                    synchronized[face_id, :, :, face_mask] = synchronized_border
                    synchronized[neighbor_id, :, :, neighbor_mask] = synchronized_border
                
                elif self.mode == "copy_owner":
                    # Face with lower ID owns the border
                    owner_id = min(face_id, neighbor_id)
                    
                    if owner_id == face_id:
                        # Current face owns the border, copy to neighbor
                        synchronized[neighbor_id, :, :, neighbor_mask] = face_border
                    else:
                        # Neighbor owns the border, copy to current face
                        synchronized[face_id, :, :, face_mask] = neighbor_border
        
        # Restore original shape if needed
        if shape_order == "batch_first":
            synchronized = synchronized.transpose(0, 1)  # (B, 6, C, H, W)
        
        return synchronized


def synchronize_cubemap_borders(
    face_latents: Tensor,
    mode: str = "avg",
    border_width: int = 2,
) -> Tensor:
    """
    Functional API for cubemap border synchronization.
    
    Args:
        face_latents: Cubemap face latents of shape (6, B, C, H, W) or (B, 6, C, H, W)
        mode: "avg" or "copy_owner" for border synchronization
        border_width: Number of tokens at each border to synchronize
    
    Returns:
        Synchronized latents with same shape as input
    """
    synchronizer = SharedBorderLatent(mode=mode, border_width=border_width)
    return synchronizer.synchronize(face_latents)
