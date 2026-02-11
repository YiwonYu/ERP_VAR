# FastVAR Panorama Extension: Technical Documentation

This document describes all modifications made to the FastVAR baseline to support 360° panorama generation with the Infinity backbone.

## Table of Contents

1. [Overview](#overview)
2. [Spherical Geometry Module](#spherical-geometry-module)
3. [3D Direction Embeddings](#3d-direction-embeddings)
4. [Spherical Attention Bias](#spherical-attention-bias)
5. [Seam Consistency Losses](#seam-consistency-losses)
6. [Border-Aware Token Pruning](#border-aware-token-pruning)
7. [Cubemap Border Synchronization](#cubemap-border-synchronization)
8. [Training System](#training-system)
9. [Expected Results](#expected-results)

---

## Overview

FastVAR is a cached-token-pruning acceleration method for Visual Autoregressive (VAR) models. This extension adapts FastVAR for **360° Equirectangular Panorama (ERP)** generation by:

1. Adding spherical geometry awareness to the model
2. Modifying token pruning to preserve seam-critical tokens
3. Introducing panorama-specific losses for seamless generation
4. Supporting multi-GPU training with DDP

### Key Design Principles

- **Backward Compatibility**: All changes are additive; existing single-view VAR/FastVAR runs unchanged
- **Spherical Awareness**: The model understands that left/right edges connect and poles have special geometry
- **Seam Consistency**: Border tokens are preserved during pruning to maintain wrap seam quality

---

## Spherical Geometry Module

**File**: `pano/geometry/spherical.py`

### ERP Coordinate System

Equirectangular Projection (ERP) maps the sphere to a 2D image with:
- Width $W = 2H$ (2:1 aspect ratio)
- Longitude $\theta \in [-\pi, \pi]$ mapped to horizontal axis
- Latitude $\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ mapped to vertical axis

### Pixel to Longitude/Latitude Conversion

For pixel coordinates $(x, y)$ in an ERP image of size $(H, W)$:

$$\theta = 2\pi \cdot \frac{x + 0.5}{W} - \pi$$

$$\phi = \frac{\pi}{2} - \pi \cdot \frac{y + 0.5}{H}$$

Where:
- $\theta$ is the longitude (azimuth angle)
- $\phi$ is the latitude (elevation angle)
- The $+0.5$ offset centers the coordinate within the pixel

### Longitude/Latitude to 3D Direction

Convert spherical coordinates to a unit 3D direction vector $\mathbf{d} = (d_x, d_y, d_z)$:

$$d_x = \cos(\phi) \cos(\theta)$$

$$d_y = \sin(\phi)$$

$$d_z = \cos(\phi) \sin(\theta)$$

This follows a right-handed coordinate system where:
- $+Y$ is up (north pole)
- $+Z$ is forward at $\theta = 0$
- $+X$ is right at $\theta = \frac{\pi}{2}$

### Geodesic Distance

The geodesic (great-circle) distance $\gamma$ between two directions $\mathbf{d}_i$ and $\mathbf{d}_j$:

$$\cos(\gamma) = \mathbf{d}_i \cdot \mathbf{d}_j$$

$$\gamma = \arccos(\mathbf{d}_i \cdot \mathbf{d}_j)$$

For efficiency, we use the squared distance approximation (valid for small angles):

$$\gamma^2 \approx 2(1 - \cos(\gamma)) = 2(1 - \mathbf{d}_i \cdot \mathbf{d}_j)$$

### Inverse Mapping

**Direction to Longitude/Latitude**:

$$\phi = \arcsin(d_y)$$

$$\theta = \text{atan2}(d_z, d_x)$$

**Longitude/Latitude to Pixel**:

$$x = \frac{(\theta + \pi) \cdot W}{2\pi} - 0.5$$

$$y = \frac{(\frac{\pi}{2} - \phi) \cdot H}{\pi} - 0.5$$

---

## 3D Direction Embeddings

**File**: `pano/models/dir3d_embed.py`

### Purpose

Inject spherical geometry awareness into the transformer by adding learnable embeddings based on 3D direction vectors. These embeddings supplement existing position encodings.

### Fourier Positional Encoding

For input direction $\mathbf{x} \in \mathbb{R}^3$, apply Fourier encoding with $K$ frequency bands:

$$f(\mathbf{x}) = \left[ \mathbf{x}, \sin(2^0 \mathbf{x}), \cos(2^0 \mathbf{x}), \ldots, \sin(2^{K-1} \mathbf{x}), \cos(2^{K-1} \mathbf{x}) \right]$$

Output dimension: $3 \times (1 + 2K)$

With default $K = 4$ frequency bands, a 3D direction becomes a 27-dimensional feature.

### MLP Architecture

The embedding module consists of:

$$\text{Direction3DEmbedding}(\mathbf{d}) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(f(\mathbf{d}))))$$

Where:
- $\text{Linear}_1: \mathbb{R}^{3(1+2K)} \rightarrow \mathbb{R}^{h}$ (hidden dimension $h = 64$)
- $\text{Linear}_2: \mathbb{R}^{h} \rightarrow \mathbb{R}^{D}$ (output matches model hidden dim $D$)

### Initialization

Weights are initialized small to ensure the embedding starts near zero (additive to existing position embeddings):
- First layer: $\mathcal{N}(0, 0.02)$
- Final layer: $\mathcal{N}(0, 0.01)$

---

## Spherical Attention Bias

**File**: `pano/models/spherical_attention.py`

### Purpose

Encourage locality on the sphere by adding a geodesic distance-based bias to attention logits. Tokens closer together on the sphere attend more strongly to each other.

### Attention Bias Formula

For query token at direction $\mathbf{d}_i$ and key token at direction $\mathbf{d}_j$:

$$\text{bias}_{ij} = -\lambda \cdot 2(1 - \mathbf{d}_i \cdot \mathbf{d}_j)$$

This approximates:

$$\text{bias}_{ij} \approx -\lambda \cdot \gamma_{ij}^2$$

Where:
- $\gamma_{ij}$ is the geodesic distance between tokens
- $\lambda > 0$ is the bias strength (higher = stronger locality preference)

### Optional Temperature Scaling

With temperature $\tau$ (in degrees, converted to radians):

$$\text{bias}_{ij} = \frac{-\lambda \cdot 2(1 - \mathbf{d}_i \cdot \mathbf{d}_j)}{\tau_{\text{rad}}^2}$$

### Modified Attention

Standard scaled dot-product attention with additive bias:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{bias}\right) V$$

### Sparse Neighbor Mask (Optional)

For efficiency, only allow attention between tokens within a maximum angle:

$$\text{mask}_{ij} = \begin{cases} 1 & \text{if } \mathbf{d}_i \cdot \mathbf{d}_j \geq \cos(\theta_{\max}) \\ 0 & \text{otherwise} \end{cases}$$

---

## Seam Consistency Losses

**File**: `pano/losses/seam_losses.py`

### Wrap Seam Loss

For seamless 360° panoramas, the left edge ($x = 0$) must match the right edge ($x = W-1$).

$$\mathcal{L}_{\text{wrap}} = \frac{1}{H \cdot C} \sum_{c=1}^{C} \sum_{y=0}^{H-1} \left| I_{c,y,0} - I_{c,y,W-1} \right|$$

Where:
- $I$ is the image tensor with shape $(C, H, W)$
- $C$ is the number of channels
- L1 distance is used for robustness

### Pole Consistency Loss

Pixels near the poles should have consistent values since they represent a small area stretched across the entire width.

**Step 1**: Identify pole region rows where $|\phi| > \frac{\pi}{2} - \phi_{\text{band}}$

$$\text{is\_pole}(y) = \left| \phi(y) \right| > \frac{\pi}{2} - \phi_{\text{band}}$$

**Step 2**: For each pole row, compute variance across width:

$$\text{Var}_{\text{row}}(y) = \frac{1}{W} \sum_{x=0}^{W-1} \left( I_{:,y,x} - \bar{I}_{:,y,:} \right)^2$$

**Step 3**: Weight by distance from pole (Gaussian weighting):

$$w(y) = \exp\left( -\frac{d_{\text{pole}}(y)^2}{2\sigma^2} \right)$$

Where $d_{\text{pole}}(y) = \min\left( \left|\phi(y) - \frac{\pi}{2}\right|, \left|\phi(y) + \frac{\pi}{2}\right| \right)$

**Final Loss**:

$$\mathcal{L}_{\text{pole}} = \sum_{y \in \text{pole}} \frac{w(y)}{\sum w} \cdot \text{Var}_{\text{row}}(y)$$

### Combined Seam Loss

$$\mathcal{L}_{\text{seam}} = \alpha_{\text{wrap}} \cdot \mathcal{L}_{\text{wrap}} + \alpha_{\text{pole}} \cdot \mathcal{L}_{\text{pole}}$$

With optional scale-step weighting:
- **Texture steps** (late scales): higher weight for fine-grained seam consistency
- **Structure steps** (early scales): lower weight

### Weighted Reconstruction Loss

Per-pixel weights emphasize border regions:

$$w(x) = \begin{cases} \beta_{\text{seam}} & \text{if } x < b \text{ or } x \geq W - b \\ 1.0 & \text{otherwise} \end{cases}$$

Where $b$ is the border width and $\beta_{\text{seam}} > 1$ is the boost factor.

---

## Border-Aware Token Pruning

**File**: `pano/fastvar/border_keep.py`

### Motivation

FastVAR's cached token pruning removes low-importance tokens at large scales. For panoramas, border tokens (left/right edges) are critical for wrap seam consistency and must be preserved.

### Original FastVAR Importance Score

For token $t$ with feature $\mathbf{x}_t$:

$$\text{importance}(t) = \| \mathbf{x}_t - \bar{\mathbf{x}} \|_2^2$$

Where $\bar{\mathbf{x}}$ is the global average pooled feature.

### Border-Boosted Importance

Modify the importance score to prioritize border tokens:

$$\text{importance}'(t) = \text{importance}(t) \cdot \text{boost}(t)$$

$$\text{boost}(t) = \begin{cases} \beta_{\text{seam}} & \text{if } t \in \mathcal{B} \\ 1.0 & \text{otherwise} \end{cases}$$

Where:
- $\mathcal{B}$ is the set of border tokens (left/right edges)
- $\beta_{\text{seam}}$ is the seam boost factor (default 1.5)

### Border Token Identification

For token grid of size $(H_t, W_t)$ with border width $b$:

$$\mathcal{B} = \{ (i, j) : j < b \text{ or } j \geq W_t - b \}$$

### Guaranteed Border Preservation

If the number of tokens to keep is less than the number of border tokens, all border tokens are kept and remaining slots are filled from high-importance non-border tokens.

### Modified Merge/Unmerge Functions

```
merge(x): Select top-k tokens by boosted importance
unmerge(x, cache): Upsample cache, scatter kept tokens back
```

---

## Cubemap Border Synchronization

**File**: `pano/fastvar/shared_border.py`

### Purpose

For cubemap-based generation (6-face representation), synchronize latents at shared borders between adjacent faces to ensure seamless stitching.

### Cubemap Adjacency

Face IDs: 0=Right, 1=Left, 2=Top, 3=Bottom, 4=Front, 5=Back

Each face has 4 edges connecting to neighbors. The adjacency mapping defines which edge of one face connects to which edge of another.

### Synchronization Modes

**Mode 1: Average ("avg")**

For overlapping border latents from faces $a$ and $b$:

$$\mathbf{z}'_{\text{border}} = \frac{\mathbf{z}_a + \mathbf{z}_b}{2}$$

Both faces receive the averaged border.

**Mode 2: Copy Owner ("copy_owner")**

The face with lower ID owns the border:

$$\mathbf{z}'_{\text{border}} = \mathbf{z}_{\min(a, b)}$$

The owner's border is copied to the neighbor.

### Border Extraction

For border width $b$ tokens:

| Edge | Indices |
|------|---------|
| Left | `[:, :b]` |
| Right | `[:, -b:]` |
| Top | `[:b, :]` |
| Bottom | `[-b:, :]` |

---

## Training System

**File**: `tools/train_pano_ddp.py`

### Multi-GPU DDP Training

- Uses PyTorch DistributedDataParallel (DDP) with NCCL backend
- Supports 1-8 GPUs via `torchrun`
- DistributedSampler ensures each GPU sees different data

### Infinity Integration

The training system integrates with the Infinity 2B model:

1. **Text Encoding**: Flan-T5-XL encodes prompts to conditioning vectors
2. **VAE Encoding**: Images encoded to multi-scale bit indices
3. **Forward Pass**: Infinity predicts next-scale tokens autoregressively
4. **Loss**: Cross-entropy on predicted vs. target bit indices

### Frozen Backbone Training

Due to Infinity's in-place operations (`.add_()`, `.mul_()` in transformer blocks), full fine-tuning causes gradient errors. Solution:

**Freeze backbone, train only:**
- Head layers (`head`)
- Level embeddings (`lvl_embed`)
- Output projections (`out_proj`)
- Normalization layers (`norm`)

Trainable parameters: ~8.69M (vs. 2B total)

### DDP-Safe Batch Skip

When a batch fails on any GPU, all GPUs must skip it together:

```python
failed_tensor = torch.tensor([1 if batch_failed else 0], device=device)
dist.all_reduce(failed_tensor, op=dist.ReduceOp.MAX)
if failed_tensor.item() > 0:
    continue  # All GPUs skip together
```

### Loss Function

VAR training uses cross-entropy on bit predictions:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{B \cdot L \cdot d} \sum_{b,l,k} \log P(y_{b,l,k} | \mathbf{x})$$

Where:
- $B$ = batch size
- $L$ = total tokens across all scales
- $d$ = 32 bits per token
- $y_{b,l,k} \in \{0, 1\}$ = target bit value

---

## Expected Results

### Seamless 360° Panorama Generation

1. **Wrap Seam Consistency**: Left and right edges should match visually
2. **Pole Quality**: Reduced artifacts in top/bottom pole regions
3. **Spherical Coherence**: Natural-looking distortion patterns for ERP format

### Quantitative Improvements (Expected)

| Metric | Baseline | With Pano Extension |
|--------|----------|---------------------|
| Wrap Seam MSE | High | Reduced by ~50-70% |
| Pole Variance | High | Reduced by ~40-60% |
| FID (panorama) | TBD | Improved |
| CLIP Score | TBD | Maintained |

### Training Convergence

With frozen backbone training on 3 GPUs:
- Batch size: 8 (effective 24 with 3 GPUs)
- Loss progression: 0.6648 → 0.6162 (epoch 0)
- Expected training time: ~50 epochs for convergence

### Inference

The trained model can generate 360° panoramas from text prompts:
- Resolution: 512×1024 (ERP format, 1:2 aspect ratio)
- Compatible with FastVAR acceleration (2.7× speedup maintained)

---

## File Summary

| File | Purpose |
|------|---------|
| `pano/geometry/spherical.py` | ERP ↔ 3D direction conversions, geodesic distance |
| `pano/geometry/cubemap.py` | Cubemap face geometry utilities |
| `pano/models/dir3d_embed.py` | Fourier + MLP direction embeddings |
| `pano/models/spherical_attention.py` | Geodesic attention bias |
| `pano/losses/seam_losses.py` | Wrap seam + pole consistency losses |
| `pano/fastvar/border_keep.py` | Border-aware token pruning |
| `pano/fastvar/shared_border.py` | Cubemap border synchronization |
| `pano/datasets/erp_dataset.py` | HuggingFace panorama dataset loader |
| `pano/config/pano_config.py` | Configuration dataclass |
| `tools/train_pano_ddp.py` | Multi-GPU DDP training script |
| `scripts/train_pano_ddp.sh` | 3-GPU launch script |

---

## References

- FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning (ICCV 2025)
- Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis
- Equirectangular Projection: Standard mapping for 360° panoramas
