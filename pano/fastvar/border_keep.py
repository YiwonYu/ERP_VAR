"""
FastVAR border token preservation for panorama seam consistency.

This module extends FastVAR's cached token pruning to always keep border tokens
(left and right edges) that are critical for wrap seam consistency in 360° panoramas.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.types import Device


def do_nothing(x: Tensor, *args, **kwargs) -> Tensor:
    """Identity function for when no pruning is applied."""
    return x


def get_border_token_mask(
    H_tokens: int,
    W_tokens: int,
    border_width: int = 2,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    """
    Create a boolean mask marking border tokens (left and right edges).
    
    Border tokens are within border_width of the left or right edge.
    These tokens are critical for wrap seam consistency in ERP panoramas.
    
    Args:
        H_tokens: Number of token rows
        W_tokens: Number of token columns
        border_width: Width of border in tokens
        device: Device for tensor
    
    Returns:
        Boolean mask of shape (H_tokens, W_tokens), True for border tokens
    """
    mask = torch.zeros(H_tokens, W_tokens, dtype=torch.bool, device=device)
    
    # Left border: x < border_width
    if border_width > 0:
        mask[:, :border_width] = True
    
    # Right border: x >= W_tokens - border_width
    if border_width > 0 and W_tokens > border_width:
        mask[:, -border_width:] = True
    
    return mask


def compute_border_boosted_importance(
    x: Tensor,
    cur_shape: Tuple[int, int, int],
    border_width: int = 2,
    seam_boost: float = 1.5,
) -> Tensor:
    """
    Compute token importance scores with boosted values for border tokens.
    
    Uses the same MSE-from-mean importance as FastVAR, but multiplies
    border token scores by seam_boost to prioritize their retention.
    
    Args:
        x: Token features, shape (B, L, C) where L = H * W
        cur_shape: Tuple of (batch, H_tokens, W_tokens)
        border_width: Width of border in tokens
        seam_boost: Multiplier for border token importance
    
    Returns:
        Importance scores of shape (B, L, 1)
    """
    B, L, C = x.shape
    _, H, W = cur_shape
    
    # Compute mean feature (global average)
    mean_x = x.view(B, H, W, C).permute(0, 3, 1, 2)
    mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x, (1, 1))
    mean_x = mean_x.permute(0, 2, 3, 1).view(B, 1, C)
    
    # Compute MSE difference from mean (base importance)
    mse_difference = torch.sum((x - mean_x) ** 2, dim=-1, keepdim=True)  # (B, L, 1)
    
    # Create border mask and flatten
    border_mask = get_border_token_mask(H, W, border_width, device=x.device)
    border_mask_flat = border_mask.flatten().unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    
    # Boost border token importance
    boosted_importance = mse_difference.clone()
    boosted_importance = torch.where(
        border_mask_flat.expand(B, -1, -1),
        mse_difference * seam_boost,
        mse_difference
    )
    
    return boosted_importance


def ensure_border_tokens_kept(
    select_indices: Tensor,
    cur_shape: Tuple[int, int, int],
    num_remain: int,
    border_width: int,
) -> Tensor:
    """
    Modify selection indices to ensure all border tokens are included.
    
    If num_remain is less than the number of border tokens, all border tokens
    are kept and the remaining slots are filled from the highest-importance
    non-border tokens.
    
    Args:
        select_indices: Original indices sorted by importance, shape (B, L, 1)
        cur_shape: Tuple of (batch, H_tokens, W_tokens)
        num_remain: Number of tokens to keep
        border_width: Width of border in tokens
    
    Returns:
        Modified indices with border tokens guaranteed, shape (B, num_remain, 1)
    """
    B, L, _ = select_indices.shape
    _, H, W = cur_shape
    device = select_indices.device
    
    # Get border token indices
    border_mask = get_border_token_mask(H, W, border_width, device=device)
    border_indices = torch.where(border_mask.flatten())[0]  # (num_border,)
    num_border = len(border_indices)
    
    if num_border == 0:
        # No border tokens, return original selection
        return select_indices[:, :num_remain, :]
    
    # If we can keep more than border tokens, add high-importance non-border tokens
    if num_remain >= num_border:
        # Start with all border indices
        result = torch.zeros(B, num_remain, 1, dtype=torch.long, device=device)
        
        # For each batch, fill in border tokens first, then top non-border tokens
        for b in range(B):
            batch_indices = select_indices[b, :, 0]  # (L,)
            
            # Separate border and non-border indices by checking against border_indices
            is_border = torch.isin(batch_indices, border_indices)
            border_selected = batch_indices[is_border]
            non_border_selected = batch_indices[~is_border]
            
            # Take all border tokens
            n_border_to_use = min(len(border_selected), num_border)
            result[b, :n_border_to_use, 0] = border_selected[:n_border_to_use]
            
            # Fill remaining with top non-border tokens
            n_non_border = num_remain - n_border_to_use
            if n_non_border > 0 and len(non_border_selected) > 0:
                result[b, n_border_to_use:n_border_to_use + min(n_non_border, len(non_border_selected)), 0] = \
                    non_border_selected[:n_non_border]
        
        return result
    else:
        # Can't keep all border tokens, just keep as many as possible
        # Prioritize corners and edges
        result = torch.zeros(B, num_remain, 1, dtype=torch.long, device=device)
        for b in range(B):
            result[b, :, 0] = border_indices[:num_remain]
        return result


def masked_previous_scale_cache_with_border_keep(
    cur_x: Tensor,
    num_remain: int,
    cur_shape: Tuple[int, int, int],
    border_width: int = 2,
    seam_boost: float = 1.5,
) -> Tuple[Callable[..., Tensor], Callable[..., Tensor], Callable[..., Tensor]]:
    """
    Token pruning with guaranteed border token preservation.
    
    Modified version of fastvar_utils.masked_previous_scale_cache that ensures
    border tokens (left and right edges) are always kept for wrap seam consistency.
    
    Args:
        cur_x: Current token features, shape (B, L, C)
        num_remain: Number of tokens to keep after pruning
        cur_shape: Tuple of (batch, H_tokens, W_tokens)
        border_width: Width of border in tokens to always keep
        seam_boost: Importance multiplier for border tokens
    
    Returns:
        merge: Function to select and gather kept tokens
        unmerge: Function to restore pruned tokens from cache
        get_src_tgt_idx: Function to get selection indices
    """
    B, L, C = cur_x.shape
    _, H, W = cur_shape
    device = cur_x.device
    
    # Compute boosted importance scores
    importance = compute_border_boosted_importance(cur_x, cur_shape, border_width, seam_boost)
    
    # Sort by importance (descending)
    select_indices = torch.argsort(importance, dim=1, descending=True)  # (B, L, 1)
    
    # Ensure border tokens are included
    filtered_indices = ensure_border_tokens_kept(select_indices, cur_shape, num_remain, border_width)
    
    def merge(merged_cur_x: Tensor) -> Tensor:
        """Select and gather kept tokens."""
        return torch.gather(
            merged_cur_x,
            dim=1,
            index=filtered_indices.expand(-1, -1, merged_cur_x.shape[-1])
        )
    
    def unmerge(unmerged_cur_x: Tensor, unmerged_cache_x: Tensor, cached_hw: Optional[Tuple[int, int]] = None) -> Tensor:
        """Restore pruned tokens from previous scale cache."""
        if cached_hw is None:
            cached_hw = (H, W)
        
        # Upsample cache to current resolution
        cache_upsampled = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
        cache_upsampled = torch.nn.functional.interpolate(
            cache_upsampled,
            size=(H, W),
            mode='area'
        ).permute(0, 2, 3, 1).view(B, L, -1)
        
        # Scatter kept tokens back
        cache_upsampled.scatter_(
            dim=1,
            index=filtered_indices.expand(-1, -1, unmerged_cur_x.shape[-1]),
            src=unmerged_cur_x
        )
        
        return cache_upsampled
    
    def get_src_tgt_idx() -> Tensor:
        """Get the indices of kept tokens."""
        return filtered_indices
    
    return merge, unmerge, get_src_tgt_idx


def compute_merge_with_border_keep(
    x: Tensor,
    prune_scale_list: Optional[List[int]] = None,
    is_later_layer: bool = False,
    x_shape: Optional[Tuple[int, int, int]] = None,
    border_width: int = 2,
    seam_boost: float = 1.5,
) -> Tuple[Callable[..., Tensor], Callable[..., Tensor], Callable[..., Tensor]]:
    """
    Drop-in replacement for fastvar_utils.compute_merge with border preservation.
    
    Uses the same interface as the original compute_merge but applies
    border-aware token selection.
    
    Args:
        x: Token features, shape (B, L, C)
        prune_scale_list: List of scale widths where pruning is applied
        is_later_layer: Whether this is a later transformer layer (enables pruning)
        x_shape: Tuple of (batch, H_tokens, W_tokens)
        border_width: Width of border in tokens to always keep
        seam_boost: Importance multiplier for border tokens
    
    Returns:
        merge: Token selection function
        unmerge: Token restoration function
        get_idx: Index retrieval function
    """
    if x_shape is None:
        return do_nothing, do_nothing, do_nothing
    
    if prune_scale_list is None:
        prune_scale_list = [32, 40]
    
    _, original_h, original_w = x_shape
    
    if original_w in prune_scale_list and is_later_layer:
        # Hardcoded ratios from original FastVAR
        ratio_hard_code = {32: 0.4, 40: 0.5}
        ratio = ratio_hard_code.get(original_w, 0.5)
        r = int(x.shape[1] * ratio)
        num_remain = x.shape[1] - r
        
        m, u, id_fn = masked_previous_scale_cache_with_border_keep(
            x, num_remain, x_shape, border_width, seam_boost
        )
    else:
        m, u, id_fn = do_nothing, do_nothing, do_nothing
    
    return m, u, id_fn


def get_num_border_tokens(H_tokens: int, W_tokens: int, border_width: int) -> int:
    """
    Compute the number of border tokens.
    
    Args:
        H_tokens: Token grid height
        W_tokens: Token grid width
        border_width: Border width in tokens
    
    Returns:
        Total number of border tokens
    """
    if border_width == 0:
        return 0
    
    # Left border: border_width columns
    # Right border: border_width columns
    # Avoid double counting if width is small
    if W_tokens <= 2 * border_width:
        return H_tokens * W_tokens  # All tokens are border tokens
    
    return H_tokens * border_width * 2
