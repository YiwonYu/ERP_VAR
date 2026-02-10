#!/usr/bin/env python3
"""
Panorama inference entrypoint for FastVAR.

Generates seamless 360° ERP panoramas or cubemap 6-face outputs.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from pano.config.pano_config import PanoConfig
from pano.models.dir3d_embed import Direction3DEmbedding, compute_erp_directions, compute_cubemap_face_directions
from pano.models.spherical_attention import SphericalAttentionBias
from pano.fastvar.border_keep import compute_merge_with_border_keep
from pano.fastvar.shared_border import synchronize_cubemap_borders
from pano.geometry.cubemap import cubemap_to_erp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate panorama images with FastVAR")
    
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="erp", choices=["erp", "cubemap"])
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./outputs/pano_infer", help="Output directory")
    parser.add_argument("--height", type=int, default=512, help="Output height (ERP: width=2*height)")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--use_fastvar", action="store_true", default=True)
    parser.add_argument("--no_fastvar", action="store_true", help="Disable FastVAR acceleration")
    
    parser.add_argument("--spherical_bias_lambda", type=float, default=1.0)
    parser.add_argument("--border_keep_w", type=int, default=2)
    parser.add_argument("--seam_boost", type=float, default=1.5)
    parser.add_argument("--shared_border_mode", type=str, default="avg", choices=["avg", "copy_owner"])
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_cubemap_combined", action="store_true", help="Save cubemap as single cross-layout image")
    
    return parser.parse_args()


def load_config(config_path: Optional[str], args: argparse.Namespace) -> PanoConfig:
    """Load config from file or create default, then merge CLI args."""
    if config_path:
        path = Path(config_path)
        if path.suffix in [".yaml", ".yml"]:
            config = PanoConfig.from_yaml(config_path)
        elif path.suffix == ".json":
            config = PanoConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    else:
        config = PanoConfig(mode=args.mode)
    
    config.mode = args.mode
    config.spherical_bias_lambda = args.spherical_bias_lambda
    config.use_fastvar = not args.no_fastvar
    config.border_keep_w = args.border_keep_w
    config.seam_boost = args.seam_boost
    config.shared_border_mode = args.shared_border_mode
    
    if args.mode == "erp":
        config.erp_height = args.height
        config.erp_width = args.height * 2
    else:
        config.cubemap_face_size = args.height
    
    config.cfg = args.cfg_scale
    if args.seed is not None:
        config.seed = args.seed
    
    config.validate()
    return config


class DummyGenerator(nn.Module):
    """
    Placeholder generator for demonstration.
    Replace with actual Infinity/HART model loading.
    """
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 3, 3, 1, 1),
            nn.Tanh(),
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


def load_pano_model(
    checkpoint_path: str,
    config: PanoConfig,
    device: str,
) -> nn.Module:
    """Load model with panorama-specific components."""
    model = DummyGenerator()
    
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Using random weights.")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}. Using random weights.")
    
    return model.to(device).eval()


def generate_erp(
    model: nn.Module,
    prompt: str,
    height: int,
    cfg_scale: float,
    seed: Optional[int],
    config: PanoConfig,
    device: str,
) -> torch.Tensor:
    """Generate a single ERP panorama."""
    width = height * 2
    
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn(1, 3, height, width, device=device)
    
    with torch.inference_mode():
        output = model(noise)
    
    output = (output + 1) / 2
    output = output.clamp(0, 1)
    
    return output[0]


def generate_cubemap(
    model: nn.Module,
    prompt: str,
    face_size: int,
    cfg_scale: float,
    seed: Optional[int],
    config: PanoConfig,
    device: str,
) -> torch.Tensor:
    """Generate 6-face cubemap."""
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn(6, 3, face_size, face_size, device=device)
    
    with torch.inference_mode():
        faces = []
        for i in range(6):
            face_output = model(noise[i:i+1])
            faces.append(face_output)
        output = torch.cat(faces, dim=0)
    
    if config.use_shared_border_latent:
        output = output.unsqueeze(0)
        output = synchronize_cubemap_borders(
            output,
            mode=config.shared_border_mode,
            border_width=config.shared_border_width_tokens,
        )
        output = output.squeeze(0)
    
    output = (output + 1) / 2
    output = output.clamp(0, 1)
    
    return output


def save_erp_image(image: torch.Tensor, output_path: Path) -> None:
    """Save ERP image to file."""
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(img_np).save(output_path)


def save_cubemap_images(
    faces: torch.Tensor,
    output_dir: Path,
    prefix: str,
    save_combined: bool = False,
) -> None:
    """Save cubemap faces as separate images or combined."""
    face_names = ["front", "right", "back", "left", "top", "bottom"]
    
    for i, name in enumerate(face_names):
        img_np = (faces[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(output_dir / f"{prefix}_{name}.png")
    
    if save_combined:
        _, H, W = faces.shape[1:]
        cross = torch.zeros(3, H * 3, W * 4, device=faces.device)
        
        cross[:, 0:H, W:2*W] = faces[4]
        cross[:, H:2*H, 0:W] = faces[3]
        cross[:, H:2*H, W:2*W] = faces[0]
        cross[:, H:2*H, 2*W:3*W] = faces[1]
        cross[:, H:2*H, 3*W:4*W] = faces[2]
        cross[:, 2*H:3*H, W:2*W] = faces[5]
        
        img_np = (cross.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(output_dir / f"{prefix}_cross.png")


def main() -> None:
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(args.config, args)
    logger.info(f"Config:\n{config}")
    
    model = load_pano_model(args.checkpoint, config, args.device)
    
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    if not prompts:
        prompts = ["A beautiful 360 panorama landscape"]
        logger.info("No prompt provided, using default")
    
    logger.info(f"Generating {len(prompts)} prompt(s) x {args.num_samples} sample(s)")
    
    total_time = 0.0
    total_images = 0
    
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"[{prompt_idx + 1}/{len(prompts)}] Prompt: {prompt[:50]}...")
        
        for sample_idx in range(args.num_samples):
            seed = args.seed + sample_idx if args.seed is not None else None
            
            start_time = time.time()
            
            if config.mode == "erp":
                image = generate_erp(
                    model, prompt, config.erp_height,
                    config.cfg, seed, config, args.device
                )
                output_path = output_dir / f"erp_{prompt_idx:04d}_{sample_idx:04d}.png"
                save_erp_image(image, output_path)
            else:
                faces = generate_cubemap(
                    model, prompt, config.cubemap_face_size,
                    config.cfg, seed, config, args.device
                )
                prefix = f"cubemap_{prompt_idx:04d}_{sample_idx:04d}"
                save_cubemap_images(faces, output_dir, prefix, args.save_cubemap_combined)
                output_path = output_dir / f"{prefix}_front.png"
            
            gen_time = time.time() - start_time
            total_time += gen_time
            total_images += 1
            
            logger.info(f"  Sample {sample_idx + 1}: {gen_time:.2f}s -> {output_path}")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    logger.info(f"Generated {total_images} images in {total_time:.1f}s ({total_time/max(total_images,1):.2f}s/image)")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
