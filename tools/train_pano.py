#!/usr/bin/env python3
"""
Panorama training entrypoint for FastVAR.

Supports both ERP (equirectangular) and cubemap 6-face training modes
with spherical-aware attention and seam consistency losses.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from pano.config.pano_config import PanoConfig
from pano.datasets.erp_dataset import ERPDataset, get_erp_transform, create_erp_dataloader
from pano.datasets.cubemap_dataset import CubemapDataset, create_cubemap_dataloader
from pano.losses.seam_losses import PanoSeamLoss
from pano.models.dir3d_embed import Direction3DEmbedding, compute_direction_embeddings
from pano.models.spherical_attention import SphericalAttentionBias
from pano.fastvar.border_keep import compute_merge_with_border_keep
from pano.fastvar.shared_border import synchronize_cubemap_borders
from pano.metrics.seam_metrics import compute_all_pano_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train panorama VAR/FastVAR model")
    
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument("--mode", type=str, default="erp", choices=["erp", "cubemap"], help="Training mode")
    parser.add_argument("--output_dir", type=str, default="./outputs/pano_train", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of training data")
    
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    
    parser.add_argument("--spherical_bias_lambda", type=float, default=None)
    parser.add_argument("--spherical_bias_tau_deg", type=float, default=None)
    parser.add_argument("--pole_band_deg", type=float, default=None)
    parser.add_argument("--pole_tau_deg", type=float, default=None)
    parser.add_argument("--pole_sigma_deg", type=float, default=None)
    parser.add_argument("--border_keep_w", type=int, default=None)
    parser.add_argument("--seam_boost", type=float, default=None)
    parser.add_argument("--shared_border_mode", type=str, default=None, choices=["avg", "copy_owner"])
    parser.add_argument("--shared_border_width_tokens", type=int, default=None)
    parser.add_argument("--wrap_seam_weight", type=float, default=None)
    parser.add_argument("--pole_loss_weight", type=float, default=None)
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> PanoConfig:
    """Load config from file and merge CLI overrides."""
    path = Path(config_path)
    
    if path.suffix in [".yaml", ".yml"]:
        config = PanoConfig.from_yaml(config_path)
    elif path.suffix == ".json":
        config = PanoConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    config.mode = args.mode
    config = config.merge_cli_args(args)
    config.validate()
    
    return config


class DummyVARModel(nn.Module):
    """
    Placeholder VAR model for demonstration.
    Replace with actual Infinity/HART model integration.
    """
    
    def __init__(self, hidden_dim: int = 768, num_layers: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, 3, 4, 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        features = self.encoder(x)
        _, c, h, w = features.shape
        features_flat = features.view(B, c, -1).permute(0, 2, 1)
        features_transformed = self.transformer(features_flat)
        features_2d = features_transformed.permute(0, 2, 1).view(B, c, h, w)
        output = self.decoder(features_2d)
        output = nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        return output


class PanoTrainer:
    """Panorama training wrapper with seam-aware components."""
    
    def __init__(
        self,
        model: nn.Module,
        config: PanoConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        if config.use_dir3d_embed:
            self.dir_embed = Direction3DEmbedding(
                embed_dim=config.dir3d_embed_dim
            ).to(device)
        else:
            self.dir_embed = None
        
        if config.use_spherical_attn_bias:
            self.spherical_bias = SphericalAttentionBias(
                lambda_=config.spherical_bias_lambda,
                tau_deg=config.spherical_bias_tau_deg,
                fallback_mode=config.spherical_attn_fallback,
            )
        else:
            self.spherical_bias = None
        
        if config.use_seam_losses:
            self.seam_loss = PanoSeamLoss(
                wrap_weight=config.wrap_seam_weight,
                pole_weight=config.pole_consistency_weight,
                pole_band_deg=config.pole_band_deg,
                pole_tau_deg=config.pole_tau_deg,
                pole_sigma_deg=config.pole_sigma_deg,
                texture_step_boost=config.texture_step_boost,
                structure_step_weight=config.structure_step_weight,
            )
        else:
            self.seam_loss = None
        
        self.recon_loss = nn.L1Loss()
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        step_type: str = "texture",
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with seam consistency terms."""
        losses = {}
        
        losses["recon"] = self.recon_loss(pred, target)
        
        if self.seam_loss is not None:
            is_texture_step = step_type == "texture"
            seam_losses = self.seam_loss(pred, target=target, is_texture_step=is_texture_step)
            
            losses["wrap_seam"] = seam_losses.get("wrap_loss", torch.tensor(0.0, device=self.device))
            losses["pole"] = seam_losses.get("pole_loss", torch.tensor(0.0, device=self.device))
        
        total = losses["recon"]
        if "wrap_seam" in losses:
            total = total + losses["wrap_seam"]
        if "pole" in losses:
            total = total + losses["pole"]
        
        losses["total"] = total
        return losses
    
    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        optimizer.zero_grad()
        
        if self.config.mode == "erp":
            images = batch["image"].to(self.device)
            pred = self.model(images)
            target = images
        else:
            # Cubemap mode: process 6 faces
            faces = batch["faces"].to(self.device)
            B, num_faces, C, H, W = faces.shape
            images = faces.view(B * num_faces, C, H, W)
            
            pred = self.model(images)
            pred = pred.view(B, num_faces, C, H, W)
            
            if self.config.use_shared_border_latent:
                pred = synchronize_cubemap_borders(
                    pred,
                    mode=self.config.shared_border_mode,
                    border_width=self.config.shared_border_width_tokens,
                )
            target = faces
        
        losses = self.compute_loss(pred, target)
        
        losses["total"].backward()
        optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}


def train_epoch(
    trainer: PanoTrainer,
    dataloader: "DataLoader[Any]",
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    epoch_losses: Dict[str, float] = {}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        losses = trainer.train_step(batch, optimizer)
        
        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = epoch_losses["total"] / num_batches
            logger.info(f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] Loss: {avg_loss:.4f}")
    
    return {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: PanoConfig,
    output_dir: Path,
    losses: Dict[str, float],
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.to_dict(),
        "losses": losses,
    }
    
    path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def main() -> None:
    args = parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(args.config, args)
    logger.info(f"Loaded config:\n{config}")
    
    config.save_yaml(str(output_dir / "config.yaml"))
    
    if config.mode == "erp":
        transform = get_erp_transform(config.erp_height)
        dataset = ERPDataset(args.data_root, transform=transform)
        dataloader = create_erp_dataloader(dataset, args.batch_size, args.num_workers)
    else:
        dataset = CubemapDataset(args.data_root, face_size=config.cubemap_face_size)
        dataloader = create_cubemap_dataloader(dataset, args.batch_size, args.num_workers)
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    model = DummyVARModel()
    trainer = PanoTrainer(model, config, device=args.device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        losses = train_epoch(trainer, dataloader, optimizer, epoch, args.log_interval)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.1f}s - "
            f"Loss: {losses['total']:.4f}, "
            f"Wrap Seam: {losses.get('wrap_seam', 0):.4f}, "
            f"Pole: {losses.get('pole', 0):.4f}"
        )
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, config, output_dir, losses)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
