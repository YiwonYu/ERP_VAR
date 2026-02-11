#!/usr/bin/env python3
"""
Panorama training entrypoint for FastVAR with Infinity backend.

Supports both ERP (equirectangular) and cubemap 6-face training modes
with spherical-aware attention and seam consistency losses.

Training flow:
1. Load image from dataset
2. Encode image through VAE -> get bit_indices (ground truth tokens)
3. Encode text prompt -> get text conditioning
4. Forward through Infinity transformer -> get logits
5. Compute cross-entropy loss on bit predictions + panorama seam losses
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Infinity"))

from pano.config.pano_config import PanoConfig
from pano.datasets.erp_dataset import ERPDataset, get_erp_transform, create_erp_dataloader
from pano.datasets.cubemap_dataset import CubemapDataset, create_cubemap_dataloader
from pano.losses.seam_losses import PanoSeamLoss
from pano.models.dir3d_embed import Direction3DEmbedding
from pano.models.spherical_attention import SphericalAttentionBias
from pano.fastvar.border_keep import compute_merge_with_border_keep
from pano.fastvar.shared_border import synchronize_cubemap_borders
from pano.metrics.seam_metrics import compute_all_pano_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

H_DIV_W_ERP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train panorama VAR/FastVAR model with Infinity backend")
    
    # Config and data
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument("--mode", type=str, default="erp", choices=["erp", "cubemap"], help="Training mode")
    parser.add_argument("--output_dir", type=str, default="./outputs/pano_train", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of training data")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    
    # Infinity model arguments
    parser.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "1M"], help="Model resolution preset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Infinity model weights")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE weights")
    parser.add_argument("--text_encoder_ckpt", type=str, required=True, help="Path to T5 text encoder")
    parser.add_argument("--model_type", type=str, default="infinity_2b", 
                        choices=["infinity_2b", "infinity_layer12", "infinity_layer16", 
                                 "infinity_layer24", "infinity_layer32", "infinity_layer40", "infinity_layer48"])
    parser.add_argument("--vae_type", type=int, default=32, help="VAE codebook dimension")
    parser.add_argument("--rope2d_each_sa_layer", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rope2d_normalized_by_hw", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--use_scale_schedule_embedding", type=int, default=0, choices=[0, 1])
    parser.add_argument("--add_lvl_embeding_only_first_block", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_bit_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--text_channels", type=int, default=2048)
    parser.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cache_dir", type=str, default="/dev/shm")
    parser.add_argument("--checkpoint_type", type=str, default="torch")
    
    # Panorama-specific parameters
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
    
    # Training options
    parser.add_argument("--freeze_vae", action="store_true", default=True, help="Freeze VAE weights")
    parser.add_argument("--freeze_text_encoder", action="store_true", default=True, help="Freeze text encoder")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch (don't load pretrained)")
    
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
    config.pn = args.pn
    config = config.merge_cli_args(args)
    config.validate()
    
    return config


class InfinityPanoTrainer:
    """Panorama training wrapper with Infinity model and seam-aware components."""
    
    def __init__(
        self,
        args: argparse.Namespace,
        config: PanoConfig,
        device: str = "cuda",
    ):
        self.args = args
        self.config = config
        self.device = device
        
        self._load_models()
        
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
    
    def _load_models(self) -> None:
        """Load Infinity model, VAE, and text encoder."""
        from tools.run_infinity import (
            load_tokenizer, load_visual_tokenizer,
        )
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        
        logger.info("Loading text encoder...")
        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=self.args.text_encoder_ckpt
        )
        if self.args.freeze_text_encoder:
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)
        
        logger.info("Loading VAE...")
        self.vae = load_visual_tokenizer(self.args)
        if self.args.freeze_vae:
            self.vae.eval()
            self.vae.requires_grad_(False)
        
        logger.info("Loading Infinity transformer...")
        self._load_infinity_for_training()
        
        self.dynamic_resolution_h_w = dynamic_resolution_h_w
        self.h_div_w_templates = h_div_w_templates
        
        logger.info("All models loaded successfully")
    
    def _load_infinity_for_training(self) -> None:
        """Load Infinity transformer in training mode."""
        from infinity.models.infinity import Infinity
        
        model_kwargs = self._get_model_kwargs()
        h_div_w = H_DIV_W_ERP if self.config.mode == "erp" else 1.0
        scale_schedule = self._get_scale_schedule(h_div_w)
        
        self.infinity = Infinity(
            vae_local=self.vae,
            text_channels=self.args.text_channels,
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=self.args.use_flex_attn == 1,
            add_lvl_embeding_only_first_block=self.args.add_lvl_embeding_only_first_block == 1,
            use_bit_label=self.args.use_bit_label == 1,
            rope2d_each_sa_layer=self.args.rope2d_each_sa_layer == 1,
            rope2d_normalized_by_hw=self.args.rope2d_normalized_by_hw,
            pn=self.args.pn,
            apply_spatial_patchify=self.args.apply_spatial_patchify == 1,
            inference_mode=False,
            train_h_div_w_list=[h_div_w],
            **model_kwargs,
        ).to(self.device)
        
        if not self.args.train_from_scratch:
            model_path = self.args.model_path
            slim_model_path = model_path.replace('ar-', 'slim-')
            load_path = slim_model_path if os.path.exists(slim_model_path) else model_path
            
            logger.info(f"Loading pretrained weights from {load_path}")
            state_dict = torch.load(load_path, map_location=self.device)
            missing, unexpected = self.infinity.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")
        
        self.infinity.train()
        self.infinity.rng = torch.Generator(device=self.device)
        
        param_count = sum(p.numel() for p in self.infinity.parameters()) / 1e9
        logger.info(f"Infinity model size: {param_count:.2f}B parameters")
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model architecture kwargs based on model_type."""
        model_configs = {
            'infinity_2b': dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
            'infinity_layer12': dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer16': dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer24': dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer32': dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer40': dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer48': dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        }
        if self.args.model_type not in model_configs:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
        return model_configs[self.args.model_type]
    
    def _get_scale_schedule(self, h_div_w: float) -> List[Tuple[int, int, int]]:
        """Get scale schedule for given aspect ratio."""
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        
        h_div_w_template = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.args.pn]['scales']
        return [(1, h, w) for (_, h, w) in scale_schedule]
    
    def encode_text(self, prompt: str) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
        """Encode text prompt to conditioning tuple."""
        tokens = self.text_tokenizer(
            text=[prompt], 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=mask
            )['last_hidden_state'].float()
        
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        max_len = max(lens)
        
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        
        return (kv_compact, lens, cu_seqlens_k, max_len)
    
    def encode_image(
        self, 
        images: torch.Tensor, 
        scale_schedule: List[Tuple[int, int, int]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Encode images through VAE to get bit indices for training."""
        with torch.no_grad():
            h, z, all_indices, all_bit_indices, _, var_input = self.vae.encode(
                images, 
                scale_schedule=scale_schedule
            )
        return h, all_bit_indices, var_input
    
    def compute_loss(
        self,
        logits_BLV: torch.Tensor,
        target_indices: List[torch.Tensor],
        images: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
        step_type: str = "texture",
    ) -> Dict[str, torch.Tensor]:
        """Compute total training loss with CE + seam losses."""
        losses = {}
        
        target_flat = torch.cat([idx.view(idx.shape[0], -1) for idx in target_indices], dim=1)
        
        ce_loss = F.cross_entropy(
            logits_BLV.view(-1, logits_BLV.shape[-1]),
            target_flat.view(-1),
            reduction='mean'
        )
        losses["ce"] = ce_loss
        
        if self.seam_loss is not None and self.config.mode == "erp":
            try:
                pred_indices = logits_BLV.argmax(dim=-1)
                ptr = 0
                all_pred_indices = []
                for (t, h, w) in scale_schedule:
                    scale_len = t * h * w
                    scale_indices = pred_indices[:, ptr:ptr+scale_len]
                    all_pred_indices.append(scale_indices)
                    ptr += scale_len
                
                with torch.no_grad():
                    label_type = 'bit' if self.args.use_bit_label else 'index'
                    _, recon_images = self.vae.decode_from_indices(
                        all_pred_indices, 
                        scale_schedule, 
                        label_type
                    )
                
                is_texture_step = step_type == "texture"
                seam_losses = self.seam_loss(recon_images, target=images, is_texture_step=is_texture_step)
                
                losses["wrap_seam"] = seam_losses.get("wrap_loss", torch.tensor(0.0, device=self.device))
                losses["pole"] = seam_losses.get("pole_loss", torch.tensor(0.0, device=self.device))
            except Exception as e:
                logger.warning(f"Seam loss computation failed: {e}")
                losses["wrap_seam"] = torch.tensor(0.0, device=self.device)
                losses["pole"] = torch.tensor(0.0, device=self.device)
        
        total = losses["ce"]
        if "wrap_seam" in losses:
            total = total + losses["wrap_seam"]
        if "pole" in losses:
            total = total + losses["pole"]
        
        losses["total"] = total
        return losses
    
    def train_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, float], torch.Tensor]:
        """Single training step returning (loss_dict, loss_tensor for backward)."""
        self.infinity.train()
        
        if self.config.mode == "erp":
            images = batch["image"].to(self.device)
            prompts = batch.get("prompt", ["A 360 degree panoramic image"] * images.shape[0])
        else:
            faces = batch["faces"].to(self.device)
            B, num_faces, C, H, W = faces.shape
            images = faces.view(B * num_faces, C, H, W)
            prompts = batch.get("prompt", ["A panoramic view"] * B)
            prompts = [p for p in prompts for _ in range(num_faces)]
        
        h_div_w = H_DIV_W_ERP if self.config.mode == "erp" else 1.0
        scale_schedule = self._get_scale_schedule(h_div_w)
        
        if images.max() > 1.0:
            images = images / 127.5 - 1.0
        
        _, all_bit_indices, var_input = self.encode_image(images, scale_schedule)
        text_cond = self.encode_text(prompts[0] if isinstance(prompts, list) else prompts)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            logits_BLV = self.infinity.forward(
                label_B_or_BLT=text_cond,
                x_BLC_wo_prefix=var_input,
                scale_schedule=scale_schedule,
                cfg_infer=False,
            )
        
        losses = self.compute_loss(
            logits_BLV=logits_BLV,
            target_indices=all_bit_indices,
            images=images,
            scale_schedule=scale_schedule,
        )
        
        return {k: v.item() for k, v in losses.items()}, losses["total"]
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = list(self.infinity.parameters())
        if self.dir_embed is not None:
            params.extend(self.dir_embed.parameters())
        return params


def train_epoch(
    trainer: InfinityPanoTrainer,
    dataloader: "DataLoader[Any]",
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Train for one epoch."""
    epoch_losses: Dict[str, float] = {}
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        loss_dict, loss_tensor = trainer.train_step(batch)
        
        scaled_loss = loss_tensor / args.gradient_accumulation_steps
        scaled_loss.backward()
        
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v
        num_batches += 1
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.get_trainable_parameters(), 
                    args.max_grad_norm
                )
            optimizer.step()
            optimizer.zero_grad()
        
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = epoch_losses["total"] / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} "
                f"CE: {epoch_losses.get('ce', 0)/num_batches:.4f}"
            )
    
    if num_batches % args.gradient_accumulation_steps != 0:
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainer.get_trainable_parameters(), 
                args.max_grad_norm
            )
        optimizer.step()
        optimizer.zero_grad()
    
    return {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}


def save_checkpoint(
    trainer: InfinityPanoTrainer,
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
        "infinity_state_dict": trainer.infinity.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.to_dict(),
        "losses": losses,
    }
    
    if trainer.dir_embed is not None:
        checkpoint["dir_embed_state_dict"] = trainer.dir_embed.state_dict()
    
    path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def main() -> None:
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    
    trainer = InfinityPanoTrainer(args, config, device=args.device)
    
    trainable_params = trainer.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.infinity.load_state_dict(checkpoint["infinity_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("dir_embed_state_dict") and trainer.dir_embed is not None:
            trainer.dir_embed.load_state_dict(checkpoint["dir_embed_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        losses = train_epoch(trainer, dataloader, optimizer, epoch, args)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.1f}s - "
            f"Loss: {losses['total']:.4f}, "
            f"CE: {losses.get('ce', 0):.4f}, "
            f"Wrap Seam: {losses.get('wrap_seam', 0):.4f}, "
            f"Pole: {losses.get('pole', 0):.4f}"
        )
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(trainer, optimizer, scheduler, epoch, config, output_dir, losses)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
