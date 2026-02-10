"""
3D Direction Embedding module for panorama-aware position encoding.

This module provides embeddings based on 3D spherical directions, which can be
added to existing position embeddings to inject spherical geometry awareness
into transformers for panorama generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def fourier_encode(x: Tensor, num_bands: int) -> Tensor:
    """
    Encode input with sinusoidal positional encoding.
    
    Args:
        x: Input tensor of shape (..., D)
        num_bands: Number of frequency bands
    
    Returns:
        Encoded tensor of shape (..., D * (1 + 2*num_bands))
        Contains: [x, sin(f_1*x), cos(f_1*x), sin(f_2*x), cos(f_2*x), ...]
        where f_i = 2^(i-1)
    """
    if num_bands == 0:
        return x
    
    # Frequencies: 2^0, 2^1, 2^2, ..., 2^(num_bands-1)
    freqs = 2.0 ** torch.linspace(
        0, num_bands - 1, num_bands,
        device=x.device, dtype=x.dtype
    )
    
    # Expand x for broadcasting: (..., D, 1) * (num_bands,) -> (..., D, num_bands)
    x_expanded = x.unsqueeze(-1) * freqs
    
    # Compute sin and cos
    sin_features = x_expanded.sin()  # (..., D, num_bands)
    cos_features = x_expanded.cos()  # (..., D, num_bands)
    
    # Flatten and concatenate: (..., D * num_bands) each
    sin_flat = sin_features.flatten(start_dim=-2)
    cos_flat = cos_features.flatten(start_dim=-2)
    
    # Concatenate: [x, sin, cos]
    encoded = torch.cat([x, sin_flat, cos_flat], dim=-1)
    
    return encoded


class Direction3DEmbedding(nn.Module):
    """
    Converts 3D unit direction vectors to learned embeddings.
    
    For each token position, the 3D direction on the sphere is passed through
    an MLP to produce an additive embedding that can be combined with existing
    position embeddings.
    
    Args:
        embed_dim: Output embedding dimension (should match model hidden dim)
        hidden_dim: Intermediate MLP dimension
        input_dim: Input dimension (3 for 3D direction vectors)
        use_fourier_features: Whether to use Fourier positional encoding on directions
        num_fourier_bands: Number of frequency bands for Fourier encoding
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        input_dim: int = 3,
        use_fourier_features: bool = True,
        num_fourier_bands: int = 4,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.use_fourier_features = use_fourier_features
        self.num_fourier_bands = num_fourier_bands
        
        # Compute actual input dimension after Fourier encoding
        if use_fourier_features:
            # Original + sin + cos for each band
            actual_input_dim = input_dim * (1 + 2 * num_fourier_bands)
        else:
            actual_input_dim = input_dim
        
        # MLP: input -> hidden -> output
        self.mlp = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # Initialize with small weights for additive embedding
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights so embedding starts near zero."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Make final layer even smaller
        final_layer = self.mlp[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, std=0.01)
    
    def forward(self, directions: Tensor) -> Tensor:
        """
        Compute embeddings from direction vectors.
        
        Args:
            directions: Unit direction vectors, shape (B, N, 3) or (N, 3)
        
        Returns:
            embeddings: Shape (B, N, embed_dim) or (N, embed_dim)
        """
        # Apply Fourier encoding if enabled
        if self.use_fourier_features:
            x = fourier_encode(directions, self.num_fourier_bands)
        else:
            x = directions
        
        # Pass through MLP
        embeddings = self.mlp(x)
        
        return embeddings


def compute_erp_directions(
    H: int,
    W: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute 3D direction vectors for all pixels in an ERP image.
    
    Args:
        H: Image height
        W: Image width
        device: Device to create tensor on
        dtype: Data type
    
    Returns:
        directions: Shape (H, W, 3) unit direction vectors
    """
    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=dtype)
    x_coords = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    # Convert to longitude/latitude
    # theta = 2*pi*((x+0.5)/W) - pi
    theta = 2.0 * math.pi * ((xx + 0.5) / W) - math.pi
    
    # phi = pi/2 - pi*((y+0.5)/H)
    phi = math.pi / 2.0 - math.pi * ((yy + 0.5) / H)
    
    # Convert to 3D direction
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    dx = cos_phi * cos_theta
    dy = sin_phi
    dz = cos_phi * sin_theta
    
    directions = torch.stack([dx, dy, dz], dim=-1)
    
    return directions


def compute_cubemap_face_directions(
    face_id: int,
    face_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute 3D direction vectors for a single cubemap face.
    
    Face conventions:
        0: front  (+Z)
        1: right  (+X)
        2: back   (-Z)
        3: left   (-X)
        4: up     (+Y)
        5: down   (-Y)
    
    Args:
        face_id: Face index (0-5)
        face_size: Size of each face (square)
        device: Device to create tensor on
        dtype: Data type
    
    Returns:
        directions: Shape (face_size, face_size, 3) unit direction vectors
    """
    # Create coordinate grids in [-1, 1]
    coords = torch.linspace(-1, 1, face_size, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(coords, coords, indexing="ij")
    
    # Create direction vectors based on face
    if face_id == 0:  # front (+Z)
        dx = uu
        dy = -vv
        dz = torch.ones_like(uu)
    elif face_id == 1:  # right (+X)
        dx = torch.ones_like(uu)
        dy = -vv
        dz = -uu
    elif face_id == 2:  # back (-Z)
        dx = -uu
        dy = -vv
        dz = -torch.ones_like(uu)
    elif face_id == 3:  # left (-X)
        dx = -torch.ones_like(uu)
        dy = -vv
        dz = uu
    elif face_id == 4:  # up (+Y)
        dx = uu
        dy = torch.ones_like(uu)
        dz = vv
    elif face_id == 5:  # down (-Y)
        dx = uu
        dy = -torch.ones_like(uu)
        dz = -vv
    else:
        raise ValueError(f"Invalid face_id: {face_id}. Must be 0-5.")
    
    # Stack and normalize
    directions = torch.stack([dx, dy, dz], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    return directions


def compute_direction_embeddings(
    embed_module: Direction3DEmbedding,
    H: int,
    W: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    mode: str = "erp",
    face_id: Optional[int] = None,
) -> Tensor:
    """
    Compute direction embeddings for all positions in an image grid.
    
    Args:
        embed_module: Direction3DEmbedding instance
        H: Image height (or face_size for cubemap)
        W: Image width (or face_size for cubemap)
        device: Device
        dtype: Data type
        mode: "erp" for equirectangular, "cubemap" for cubemap face
        face_id: Face index (0-5) when mode="cubemap"
    
    Returns:
        embeddings: Shape (H, W, embed_dim) or (H*W, embed_dim)
    """
    if mode == "erp":
        directions = compute_erp_directions(H, W, device=device, dtype=dtype)
    elif mode == "cubemap":
        if face_id is None:
            raise ValueError("face_id must be provided for cubemap mode")
        directions = compute_cubemap_face_directions(
            face_id, H, device=device, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'erp' or 'cubemap'.")
    
    # Flatten to (H*W, 3)
    directions_flat = directions.view(-1, 3)
    
    # Compute embeddings
    embeddings_flat = embed_module(directions_flat)
    
    # Reshape back to (H, W, embed_dim)
    embeddings = embeddings_flat.view(H, W, -1)
    
    return embeddings


class CachedDirection3DEmbedding(nn.Module):
    """
    Direction3DEmbedding with caching for fixed grid sizes.
    
    Caches computed embeddings for common (H, W) combinations to avoid
    recomputation during inference.
    
    Args:
        embed_dim: Output embedding dimension
        hidden_dim: Intermediate MLP dimension
        use_fourier_features: Whether to use Fourier encoding
        num_fourier_bands: Number of frequency bands
        max_cache_size: Maximum number of cached grids
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        use_fourier_features: bool = True,
        num_fourier_bands: int = 4,
        max_cache_size: int = 10,
    ):
        super().__init__()
        
        self.embed = Direction3DEmbedding(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_fourier_features=use_fourier_features,
            num_fourier_bands=num_fourier_bands,
        )
        
        self.max_cache_size = max_cache_size
        self._cache: dict = {}
        self._cache_order: list = []
    
    def _get_cache_key(
        self,
        H: int,
        W: int,
        mode: str,
        face_id: Optional[int],
        device: str,
        dtype: torch.dtype,
    ) -> Tuple:
        """Generate cache key."""
        return (H, W, mode, face_id, device, str(dtype))
    
    def _evict_oldest(self):
        """Remove oldest cache entry if at capacity."""
        if len(self._cache) >= self.max_cache_size and self._cache_order:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
    
    def get_embeddings(
        self,
        H: int,
        W: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        mode: str = "erp",
        face_id: Optional[int] = None,
    ) -> Tensor:
        """
        Get embeddings for grid, using cache if available.
        
        Args:
            H: Height
            W: Width
            device: Device
            dtype: Data type
            mode: "erp" or "cubemap"
            face_id: Face ID for cubemap mode
        
        Returns:
            embeddings: Shape (H, W, embed_dim)
        """
        cache_key = self._get_cache_key(H, W, mode, face_id, device, dtype)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = compute_direction_embeddings(
                self.embed, H, W,
                device=device, dtype=dtype,
                mode=mode, face_id=face_id,
            )
        
        # Cache result
        self._evict_oldest()
        self._cache[cache_key] = embeddings
        self._cache_order.append(cache_key)
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_order.clear()
    
    def forward(self, directions: Tensor) -> Tensor:
        """Pass-through to underlying embedding module."""
        return self.embed(directions)
