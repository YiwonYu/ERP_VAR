"""
Spherical attention bias for panorama-aware transformers.

This module provides attention mechanisms that are aware of spherical geometry,
using geodesic distance-based biases to improve panorama generation quality.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_spherical_attention_bias(
    dirs_q: Tensor,
    dirs_k: Tensor,
    lambda_: float = 1.0,
    tau_deg: Optional[float] = None,
) -> Tensor:
    """
    Compute spherical distance-based attention bias.
    
    The bias is based on geodesic distance between query and key positions
    on the sphere. Closer tokens (smaller geodesic distance) get higher
    attention scores.
    
    Formula: bias = -lambda * 2 * (1 - dot(d_i, d_j))
    This approximates -lambda * gamma^2 where gamma is the geodesic distance.
    
    Args:
        dirs_q: Query direction vectors, shape (N_q, 3) or (B, N_q, 3)
        dirs_k: Key direction vectors, shape (N_k, 3) or (B, N_k, 3)
        lambda_: Bias strength (higher = stronger locality preference)
        tau_deg: Optional temperature in degrees (scales bias by 1/tau_rad^2)
    
    Returns:
        Attention bias matrix, shape (N_q, N_k) or (B, N_q, N_k)
    """
    # Handle batched and non-batched inputs
    if dirs_q.dim() == 2 and dirs_k.dim() == 2:
        # Non-batched: (N_q, 3) x (3, N_k) -> (N_q, N_k)
        cos_gamma = torch.mm(dirs_q, dirs_k.t())
    elif dirs_q.dim() == 3 and dirs_k.dim() == 3:
        # Batched: (B, N_q, 3) x (B, 3, N_k) -> (B, N_q, N_k)
        cos_gamma = torch.bmm(dirs_q, dirs_k.transpose(-2, -1))
    else:
        # Mixed or general case using einsum
        cos_gamma = torch.einsum("...id,...jd->...ij", dirs_q, dirs_k)
    
    # Clamp for numerical stability
    cos_gamma = cos_gamma.clamp(-1.0, 1.0)
    
    # Compute bias: -lambda * 2 * (1 - cos_gamma) ≈ -lambda * gamma^2
    bias = -lambda_ * 2.0 * (1.0 - cos_gamma)
    
    # Optional temperature scaling
    if tau_deg is not None:
        tau_rad = tau_deg * math.pi / 180.0
        bias = bias / (tau_rad ** 2)
    
    return bias


def get_erp_directions_for_tokens(
    H_tokens: int,
    W_tokens: int,
    H_pixels: int,
    W_pixels: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute direction vectors for token grid positions.
    
    Each token represents a patch of pixels. The direction is computed
    at the center of each patch.
    
    Args:
        H_tokens: Number of token rows
        W_tokens: Number of token columns
        H_pixels: Image height in pixels
        W_pixels: Image width in pixels
        device: Device for tensor
        dtype: Data type for tensor
    
    Returns:
        Direction vectors, shape (H_tokens * W_tokens, 3) or (H_tokens, W_tokens, 3)
    """
    # Compute patch size
    patch_h = H_pixels / H_tokens
    patch_w = W_pixels / W_tokens
    
    # Token center coordinates (in pixel space)
    y_centers = (torch.arange(H_tokens, device=device, dtype=dtype) + 0.5) * patch_h
    x_centers = (torch.arange(W_tokens, device=device, dtype=dtype) + 0.5) * patch_w
    
    # Create grid
    yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")
    
    # Convert to longitude/latitude
    # theta = 2*pi*((x+0.5)/W) - pi  (but we already have center coords)
    theta = 2.0 * math.pi * (xx / W_pixels) - math.pi
    phi = math.pi / 2.0 - math.pi * (yy / H_pixels)
    
    # Convert to 3D directions
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    dx = cos_phi * cos_theta
    dy = sin_phi
    dz = cos_phi * sin_theta
    
    directions = torch.stack([dx, dy, dz], dim=-1)  # (H_tokens, W_tokens, 3)
    
    return directions


def create_spherical_neighbor_mask(
    directions: Tensor,
    max_angle_deg: float = 45.0,
) -> Tensor:
    """
    Create a sparse attention mask based on spherical distance.
    
    Only tokens within max_angle_deg of each other can attend to each other.
    This creates a locality-aware sparse attention pattern.
    
    Args:
        directions: Direction vectors, shape (N, 3) or (H, W, 3)
        max_angle_deg: Maximum angle (in degrees) for attention
    
    Returns:
        Boolean mask, shape (N, N), True where attention is allowed
    """
    # Flatten if needed
    if directions.dim() == 3:
        N = directions.shape[0] * directions.shape[1]
        directions = directions.view(N, 3)
    else:
        N = directions.shape[0]
    
    # Compute cosine of max angle (threshold)
    max_angle_rad = max_angle_deg * math.pi / 180.0
    cos_threshold = math.cos(max_angle_rad)
    
    # Compute pairwise dot products
    cos_angles = torch.mm(directions, directions.t())  # (N, N)
    
    # Create mask: True where angle < max_angle (i.e., cos > threshold)
    mask = cos_angles >= cos_threshold
    
    return mask


class SphericalAttentionBias(nn.Module):
    """
    Spherical geometry-aware attention bias module.
    
    Adds geodesic distance-based bias to attention logits to encourage
    locality on the sphere.
    
    Args:
        lambda_: Bias strength
        tau_deg: Optional temperature in degrees
        fallback_mode: "standard" for full attention with bias, "sparse_neighbor" for sparse
    """
    
    def __init__(
        self,
        lambda_: float = 1.0,
        tau_deg: Optional[float] = None,
        fallback_mode: str = "standard",
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.tau_deg = tau_deg
        self.fallback_mode = fallback_mode
        
        # Cache for directions and bias
        self._cached_directions: Optional[Tensor] = None
        self._cached_bias: Optional[Tensor] = None
        self._cached_shape: Optional[Tuple[int, int]] = None
    
    def get_bias(
        self,
        H_tokens: int,
        W_tokens: int,
        H_pixels: int,
        W_pixels: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Get or compute cached attention bias."""
        cache_key = (H_tokens, W_tokens)
        
        if self._cached_shape != cache_key or self._cached_bias is None:
            # Compute directions
            directions = get_erp_directions_for_tokens(
                H_tokens, W_tokens, H_pixels, W_pixels, device, dtype
            )
            directions_flat = directions.view(-1, 3)
            
            # Compute bias
            bias = compute_spherical_attention_bias(
                directions_flat, directions_flat, self.lambda_, self.tau_deg
            )
            
            self._cached_directions = directions_flat
            self._cached_bias = bias
            self._cached_shape = cache_key
        
        return self._cached_bias.to(device=device, dtype=dtype)
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        directions: Tensor,
        base_attn_fn: Optional[Callable] = None,
    ) -> Tensor:
        """
        Apply spherical-biased attention.
        
        Args:
            q: Queries, shape (B, N, H, D) or (B, H, N, D)
            k: Keys, shape (B, N, H, D) or (B, H, N, D)
            v: Values, shape (B, N, H, D) or (B, H, N, D)
            directions: Direction vectors, shape (N, 3) or (H_t, W_t, 3)
            base_attn_fn: Optional base attention function to wrap
        
        Returns:
            Attention output, same shape as v
        """
        # Flatten directions if needed
        if directions.dim() == 3:
            dirs_flat = directions.view(-1, 3)
        else:
            dirs_flat = directions
        
        # Compute bias
        bias = compute_spherical_attention_bias(
            dirs_flat, dirs_flat, self.lambda_, self.tau_deg
        )  # (N, N)
        
        if self.fallback_mode == "sparse_neighbor":
            # Create sparse mask
            mask = create_spherical_neighbor_mask(dirs_flat, max_angle_deg=45.0)
            return self._sparse_attention(q, k, v, mask, bias)
        else:
            # Standard attention with bias
            return self._biased_attention(q, k, v, bias)
    
    def _biased_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """Standard scaled dot-product attention with additive bias."""
        # Determine input format and reshape if needed
        # Assume q, k, v are (B, N, H, D) - FlashAttn format
        B, N, H, D = q.shape
        
        # Reshape to (B, H, N, D) for attention computation
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)
        
        # Add spherical bias (broadcast over batch and heads)
        attn = attn + bias.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        
        # Softmax and apply to values
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, N, D)
        
        # Reshape back to (B, N, H, D)
        out = out.transpose(1, 2)
        
        return out
    
    def _sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        bias: Tensor,
    ) -> Tensor:
        """Sparse attention with neighbor mask."""
        B, N, H, D = q.shape
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = 1.0 / math.sqrt(D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add bias
        attn = attn + bias.unsqueeze(0).unsqueeze(0)
        
        # Apply sparse mask (set non-neighbors to -inf)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn = attn.masked_fill(~mask_expanded, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Handle NaN from all-masked rows
        attn = torch.nan_to_num(attn, nan=0.0)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        
        return out


class SphericalBiasedFlashAttnWrapper(nn.Module):
    """
    Wrapper to add spherical bias to FlashAttention.
    
    Since FlashAttention doesn't easily support arbitrary additive bias,
    this wrapper provides fallback modes:
    - If bias can be injected via attn_mask, use FlashAttn
    - Otherwise, fall back to standard attention with bias
    
    Args:
        lambda_: Spherical bias strength
        tau_deg: Optional temperature
        use_flash_if_possible: Try to use FlashAttn when possible
    """
    
    def __init__(
        self,
        lambda_: float = 1.0,
        tau_deg: Optional[float] = None,
        use_flash_if_possible: bool = True,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.tau_deg = tau_deg
        self.use_flash_if_possible = use_flash_if_possible
        self.spherical_bias = SphericalAttentionBias(lambda_, tau_deg, "standard")
        
        # Check if flash_attn is available
        self._has_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self._flash_attn_func = flash_attn_func
            self._has_flash_attn = True
        except ImportError:
            self._flash_attn_func = None
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        H_tokens: int,
        W_tokens: int,
        H_pixels: int,
        W_pixels: int,
        softmax_scale: Optional[float] = None,
    ) -> Tensor:
        """
        Apply attention with spherical bias.
        
        Args:
            q: Queries (B, N, H, D)
            k: Keys (B, N, H, D)
            v: Values (B, N, H, D)
            H_tokens, W_tokens: Token grid dimensions
            H_pixels, W_pixels: Pixel dimensions
            softmax_scale: Optional attention scale
        
        Returns:
            Attention output (B, N, H, D)
        """
        # Get directions
        directions = get_erp_directions_for_tokens(
            H_tokens, W_tokens, H_pixels, W_pixels,
            device=q.device, dtype=q.dtype
        )
        
        # Use spherical biased attention (standard fallback)
        # FlashAttn v2 doesn't support arbitrary additive bias well
        return self.spherical_bias(q, k, v, directions)


def apply_spherical_bias_to_logits(
    logits: Tensor,
    H_tokens: int,
    W_tokens: int,
    H_pixels: int,
    W_pixels: int,
    lambda_: float = 1.0,
    tau_deg: Optional[float] = None,
) -> Tensor:
    """
    Apply spherical distance bias directly to attention logits.
    
    Utility function for injecting bias into existing attention implementations.
    
    Args:
        logits: Attention logits, shape (B, H, N, N) or (B, N, N)
        H_tokens, W_tokens: Token grid dimensions
        H_pixels, W_pixels: Pixel dimensions
        lambda_: Bias strength
        tau_deg: Optional temperature
    
    Returns:
        Biased logits, same shape as input
    """
    device = logits.device
    dtype = logits.dtype
    
    # Compute directions and bias
    directions = get_erp_directions_for_tokens(
        H_tokens, W_tokens, H_pixels, W_pixels, device, dtype
    )
    dirs_flat = directions.view(-1, 3)
    
    bias = compute_spherical_attention_bias(dirs_flat, dirs_flat, lambda_, tau_deg)
    
    # Add bias (handle different logit shapes)
    if logits.dim() == 4:  # (B, H, N, N)
        bias = bias.unsqueeze(0).unsqueeze(0)
    elif logits.dim() == 3:  # (B, N, N)
        bias = bias.unsqueeze(0)
    
    return logits + bias
