#!/usr/bin/env python3
"""
Multi-GPU panorama training script with DDP support for FastVAR + Infinity.

Launch with: torchrun --nproc_per_node=3 tools/train_pano_ddp.py [args]
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Infinity"))

from pano.config.pano_config import PanoConfig
from pano.datasets.erp_dataset import HFPanoramaDataset, create_hf_pano_dataloader
from pano.losses.seam_losses import PanoSeamLoss
from pano.models.dir3d_embed import Direction3DEmbedding
from pano.models.spherical_attention import SphericalAttentionBias

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

H_DIV_W_ERP = 0.5


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU panorama training with DDP")
    
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/pano_ddp")
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_height", type=int, default=512)
    
    parser.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "1M"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--text_encoder_ckpt", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="infinity_2b")
    parser.add_argument("--vae_type", type=int, default=32)
    parser.add_argument("--rope2d_each_sa_layer", type=int, default=1)
    parser.add_argument("--rope2d_normalized_by_hw", type=int, default=2)
    parser.add_argument("--use_scale_schedule_embedding", type=int, default=0)
    parser.add_argument("--add_lvl_embeding_only_first_block", type=int, default=1)
    parser.add_argument("--use_bit_label", type=int, default=1)
    parser.add_argument("--text_channels", type=int, default=2048)
    parser.add_argument("--apply_spatial_patchify", type=int, default=0)
    parser.add_argument("--use_flex_attn", type=int, default=0)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default="/dev/shm")
    parser.add_argument("--checkpoint_type", type=str, default="torch")
    
    parser.add_argument("--use_seam_loss", type=int, default=1)
    parser.add_argument("--wrap_seam_weight", type=float, default=0.1)
    parser.add_argument("--pole_loss_weight", type=float, default=0.05)
    
    parser.add_argument("--freeze_backbone", type=int, default=0,
                        help="Freeze backbone, only train head/embeddings")
    parser.add_argument("--train_last_n_blocks", type=int, default=-1,
                        help="Only train last N blocks (-1 = all)")
    
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="fastvar-pano")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace) -> PanoConfig:
    path = Path(config_path)
    if path.suffix in [".yaml", ".yml"]:
        config = PanoConfig.from_yaml(config_path)
    elif path.suffix == ".json":
        config = PanoConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    config.mode = "erp"
    config.pn = args.pn
    config.use_seam_losses = args.use_seam_loss == 1
    config.wrap_seam_weight = args.wrap_seam_weight
    config.pole_consistency_weight = args.pole_loss_weight
    config.validate()
    
    return config


class InfinityDDPTrainer:
    
    def __init__(
        self,
        args: argparse.Namespace,
        config: PanoConfig,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        self.args = args
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"
        
        self._load_models()
        
        if config.use_seam_losses:
            self.seam_loss = PanoSeamLoss(
                wrap_weight=config.wrap_seam_weight,
                pole_weight=config.pole_consistency_weight,
            )
        else:
            self.seam_loss = None
    
    def _load_models(self) -> None:
        from tools.run_infinity import load_tokenizer, load_visual_tokenizer
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        from infinity.models.infinity import Infinity
        
        if is_main_process():
            logger.info("Loading text encoder...")
        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=self.args.text_encoder_ckpt
        )
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        if is_main_process():
            logger.info("Loading VAE...")
        self.vae = load_visual_tokenizer(self.args)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        if is_main_process():
            logger.info("Loading Infinity transformer...")
        
        model_kwargs = self._get_model_kwargs()
        scale_schedule = self._get_scale_schedule(H_DIV_W_ERP)
        
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
            train_h_div_w_list=[H_DIV_W_ERP],
            **model_kwargs,
        ).to(self.device)
        
        model_path = self.args.model_path
        slim_model_path = model_path.replace('ar-', 'slim-')
        load_path = slim_model_path if os.path.exists(slim_model_path) else model_path
        
        if is_main_process():
            logger.info(f"Loading pretrained weights from {load_path}")
        state_dict = torch.load(load_path, map_location=self.device)
        self.infinity.load_state_dict(state_dict, strict=False)
        
        self.infinity.train()
        self.infinity.rng = torch.Generator(device=self.device)
        
        # Apply freezing if requested (before wrapping with DDP)
        if self.args.freeze_backbone:
            trainable_count = 0
            frozen_count = 0
            for name, param in self.infinity.named_parameters():
                # Keep head, level embeddings, and output projections trainable
                if any(kw in name.lower() for kw in ['head', 'lvl_embed', 'out_proj', 'norm']):
                    param.requires_grad = True
                    trainable_count += param.numel()
                else:
                    param.requires_grad = False
                    frozen_count += param.numel()
            if is_main_process():
                logger.info(f"Frozen backbone mode enabled")
                logger.info(f"  Trainable params: {trainable_count/1e6:.2f}M")
                logger.info(f"  Frozen params: {frozen_count/1e6:.2f}M")
        elif self.args.train_last_n_blocks > 0:
            # Alternative: only train last N transformer blocks
            trainable_count = 0
            frozen_count = 0
            for name, param in self.infinity.named_parameters():
                should_train = False
                # Always train head and embeddings
                if any(kw in name.lower() for kw in ['head', 'lvl_embed', 'out_proj']):
                    should_train = True
                # Train last N blocks
                elif 'blocks.' in name:
                    try:
                        block_idx = int(name.split('blocks.')[1].split('.')[0])
                        total_blocks = 32  # For infinity_2b
                        if block_idx >= total_blocks - self.args.train_last_n_blocks:
                            should_train = True
                    except (IndexError, ValueError):
                        pass
                
                param.requires_grad = should_train
                if should_train:
                    trainable_count += param.numel()
                else:
                    frozen_count += param.numel()
            if is_main_process():
                logger.info(f"Training last {self.args.train_last_n_blocks} blocks")
                logger.info(f"  Trainable params: {trainable_count/1e6:.2f}M")
                logger.info(f"  Frozen params: {frozen_count/1e6:.2f}M")
        
        if self.world_size > 1:
            self.infinity = DDP(
                self.infinity, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        
        self.dynamic_resolution_h_w = dynamic_resolution_h_w
        self.h_div_w_templates = h_div_w_templates
        
        if is_main_process():
            model = self.infinity.module if hasattr(self.infinity, 'module') else self.infinity
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            logger.info(f"Infinity model size: {param_count:.2f}B parameters")
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        model_configs = {
            'infinity_2b': dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
            'infinity_layer12': dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer16': dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer24': dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer32': dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer40': dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
            'infinity_layer48': dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
        }
        return model_configs[self.args.model_type]
    
    def _get_scale_schedule(self, h_div_w: float) -> List[Tuple[int, int, int]]:
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        
        h_div_w_template = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.args.pn]['scales']
        return [(1, h, w) for (_, h, w) in scale_schedule]
    
    def encode_text(self, prompts: List[str]) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
        """Encode text prompts for conditioning.
        
        Args:
            prompts: List of text prompts (one per batch item)
            
        Returns:
            Tuple of (kv_compact, lens, cu_seqlens_k, max_len)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        tokens = self.text_tokenizer(
            text=prompts, 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)
        
        with torch.no_grad():
            encoder_output = self.text_encoder(
                input_ids=input_ids, 
                attention_mask=mask
            )
            # Handle both dict-like and object access patterns
            if isinstance(encoder_output, dict):
                text_features = encoder_output['last_hidden_state'].float()
            else:
                text_features = encoder_output.last_hidden_state.float()
        
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
        """Encode images to VAE latents and prepare for transformer input.
        
        Args:
            images: Input images tensor (B, C, H, W) in range [-1, 1]
            scale_schedule: List of (pt, ph, pw) tuples for each scale
            
        Returns:
            Tuple of (h, all_bit_indices, x_BLC_wo_prefix)
            - h: Raw VAE features
            - all_bit_indices: Target bit indices for each scale
            - x_BLC_wo_prefix: Transformer input tensor (B, L, C)
        """
        with torch.no_grad():
            h, z, all_indices, all_bit_indices, _, var_inputs = self.vae.encode(
                images, 
                scale_schedule=scale_schedule
            )
            
            # Convert var_inputs (list of (B,d,1,H,W) tensors) to x_BLC_wo_prefix (B,L,C)
            # var_inputs has one tensor per scale except the last
            x_BLC_wo_prefix_list = []
            for var_input in var_inputs:
                # var_input shape: (B, d, 1, H, W) or (B, d, H, W)
                if var_input.dim() == 5:
                    var_input = var_input.squeeze(2)  # (B, d, H, W)
                # Reshape to (B, H*W, d)
                B, d, H, W = var_input.shape
                x_scale = var_input.permute(0, 2, 3, 1).reshape(B, H * W, d)
                x_BLC_wo_prefix_list.append(x_scale)
            
            # Concatenate all scales
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix_list, dim=1)
            
        return h, all_bit_indices, x_BLC_wo_prefix
    
    def compute_loss(
        self,
        logits_BLV: torch.Tensor,
        target_indices: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-entropy loss for VAR training.
        
        In VAR training:
        - Position 0 (SOS input) predicts scale 0 tokens (1 token)
        - Position 1..scale_1_end (var_inputs[0]) predicts scale 1 tokens
        - etc.
        
        So logits and targets have the same length (no shift needed).
        
        Args:
            logits_BLV: Model output logits with shape (B, L, V) where:
                - B = batch size
                - L = total token positions = sum of all scales
                - V = 64 = codebook_dim * 2 = 32 bits * 2 classes
            target_indices: List of bit_indices per scale, each with shape (B, 1, h, w, 32)
                where values are 0 or 1 (binary classification targets)
        
        Returns:
            Dict with 'ce' and 'total' loss values
        """
        losses = {}
        B = logits_BLV.shape[0]
        
        # Flatten targets: each bit_indices has shape (B, 1, h, w, 32) -> (B, h*w, 32)
        # Then concat across scales -> (B, total_tokens, 32)
        target_list = []
        for bit_idx in target_indices:
            # bit_idx shape: (B, 1, h, w, 32) or (B, h, w, 32)
            if bit_idx.dim() == 5:
                bit_idx = bit_idx.squeeze(1)  # (B, h, w, 32)
            # Reshape to (B, h*w, 32)
            B_t, h, w, d = bit_idx.shape
            target_list.append(bit_idx.reshape(B_t, h * w, d))
        
        # Concat across scales: (B, total_tokens, 32)
        target_Bld = torch.cat(target_list, dim=1)
        
        # Get dimensions
        B, L_target, d = target_Bld.shape  # d = 32
        L_logits = logits_BLV.shape[1]
        
        # Verify dimensions match
        if L_logits != L_target:
            raise ValueError(
                f"Logits length ({L_logits}) != target length ({L_target}). "
                f"logits shape: {logits_BLV.shape}, "
                f"target scales: {[idx.shape for idx in target_indices[:3]]}"
            )
        
        # Reshape logits from (B, L, 64) to (B, L, 32, 2) then to (B*L*32, 2)
        # 64 = 32 bits * 2 classes per bit
        logits_reshaped = logits_BLV.reshape(B, L_target, d, 2)  # (B, L, 32, 2)
        logits_flat = logits_reshaped.reshape(-1, 2)  # (B*L*32, 2)
        
        # Flatten targets to (B*L*32,) with values 0 or 1
        target_flat = target_Bld.reshape(-1).long()  # (B*L*32,)
        
        ce_loss = F.cross_entropy(
            logits_flat,
            target_flat,
            reduction='mean'
        )
        losses["ce"] = ce_loss
        losses["total"] = ce_loss
        
        return losses
    
    def train_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, float], torch.Tensor]:
        model = self.infinity.module if hasattr(self.infinity, 'module') else self.infinity
        model.train()
        
        images = batch["image"].to(self.device)
        prompts = batch.get("prompt", ["A 360 degree indoor panorama"] * images.shape[0])
        
        scale_schedule = self._get_scale_schedule(H_DIV_W_ERP)
        
        if images.max() > 1.0:
            images = images / 255.0
        images = images * 2.0 - 1.0
        
        try:
            _, all_bit_indices, var_input = self.encode_image(images, scale_schedule)
        except Exception as e:
            raise RuntimeError(f"encode_image failed: {e}, images shape: {images.shape}")
        
        try:
            text_cond = self.encode_text(prompts)
        except Exception as e:
            raise RuntimeError(f"encode_text failed: {e}, prompts: {prompts[:2] if len(prompts) > 2 else prompts}")
        
        try:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                logits_BLV = self.infinity.forward(
                    label_B_or_BLT=text_cond,
                    x_BLC_wo_prefix=var_input,
                    scale_schedule=scale_schedule,
                    cfg_infer=False,
                )
        except Exception as e:
            raise RuntimeError(f"infinity.forward failed: {e}, text_cond types: {[type(t) for t in text_cond]}, var_input type: {type(var_input)}")
        
        try:
            losses = self.compute_loss(logits_BLV, all_bit_indices)
        except Exception as e:
            raise RuntimeError(f"compute_loss failed: {e}, logits type: {type(logits_BLV)}, indices types: {[type(i) for i in all_bit_indices[:3]]}")
        
        return {k: v.item() for k, v in losses.items()}, losses["total"]
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        model = self.infinity.module if hasattr(self.infinity, 'module') else self.infinity
        return [p for p in model.parameters() if p.requires_grad]


def train_epoch(
    trainer: InfinityDDPTrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    sampler: Optional[DistributedSampler],
) -> Dict[str, float]:
    if sampler is not None:
        sampler.set_epoch(epoch)
    
    epoch_losses: Dict[str, float] = {}
    num_batches = 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        loss_dict = None
        loss_tensor = None
        batch_failed = False
        
        try:
            loss_dict, loss_tensor = trainer.train_step(batch)
        except Exception as e:
            logger.warning(f"Batch {batch_idx} failed: {e}")
            batch_failed = True
        
        if world_size > 1:
            failed_tensor = torch.tensor([1 if batch_failed else 0], device=trainer.device)
            dist.all_reduce(failed_tensor, op=dist.ReduceOp.MAX)
            if failed_tensor.item() > 0:
                torch.cuda.empty_cache()
                continue
        elif batch_failed:
            torch.cuda.empty_cache()
            continue
        
        assert loss_dict is not None and loss_tensor is not None
        scaled_loss = loss_tensor / args.gradient_accumulation_steps
        scaled_loss.backward()
        
        del loss_tensor, scaled_loss
        
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
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        
        if (batch_idx + 1) % args.log_interval == 0 and is_main_process():
            avg_loss = epoch_losses["total"] / num_batches
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f}"
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
    trainer: InfinityDDPTrainer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: PanoConfig,
    output_dir: Path,
    losses: Dict[str, float],
) -> None:
    if not is_main_process():
        return
    
    model = trainer.infinity.module if hasattr(trainer.infinity, 'module') else trainer.infinity
    
    checkpoint = {
        "epoch": epoch,
        "infinity_state_dict": model.state_dict(),
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
    
    rank, world_size, local_rank = setup_distributed()
    
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    config = load_config(args.config, args)
    if is_main_process():
        logger.info(f"Loaded config:\n{config}")
        config.save_yaml(str(output_dir / "config.yaml"))
    
    if is_main_process():
        logger.info(f"Loading dataset from {args.dataset_path}...")
    
    dataloader, sampler = create_hf_pano_dataloader(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        target_height=args.target_height,
        num_workers=args.num_workers,
        shuffle=True,
        distributed=world_size > 1,
        world_size=world_size,
        rank=rank,
    )
    
    if is_main_process():
        logger.info(f"Dataset loaded: {len(dataloader.dataset)} samples, {len(dataloader)} batches per GPU")
    
    trainer = InfinityDDPTrainer(args, config, rank, world_size, local_rank)
    
    trainable_params = trainer.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        if is_main_process():
            logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        model = trainer.infinity.module if hasattr(trainer.infinity, 'module') else trainer.infinity
        model.load_state_dict(checkpoint["infinity_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    
    if is_main_process():
        logger.info(f"Starting training from epoch {start_epoch} with {world_size} GPUs")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        losses = train_epoch(trainer, dataloader, optimizer, epoch, args, sampler)
        
        scheduler.step()
        
        if world_size > 1:
            dist.barrier()
        
        epoch_time = time.time() - epoch_start
        if is_main_process():
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Loss: {losses['total']:.4f}"
            )
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(trainer, optimizer, scheduler, epoch, config, output_dir, losses)
            if world_size > 1:
                dist.barrier()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    cleanup_distributed()
    
    if is_main_process():
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
