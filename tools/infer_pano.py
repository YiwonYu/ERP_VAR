#!/usr/bin/env python3
"""
Panorama inference entrypoint for FastVAR with Infinity backend.

Generates seamless 360° ERP panoramas or cubemap 6-face outputs.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Infinity"))

from pano.config.pano_config import PanoConfig
from pano.models.spherical_attention import SphericalAttentionBias

# Note: Cubemap mode is experimental and not the primary use case.
# ERP (equirectangular) mode is the default and fully supported.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

H_DIV_W_ERP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate panorama images with FastVAR + Infinity")
    
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file")
    parser.add_argument("--mode", type=str, default="erp", choices=["erp", "cubemap"])
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--prompts_file", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./outputs/pano_infer", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    parser.add_argument("--pn", type=str, default="1M", choices=["0.06M", "0.25M", "0.60M", "1M"])
    parser.add_argument("--model_path", type=str, required=True, help="Path to Infinity model weights")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to trained checkpoint (e.g., outputs/pano_matterport3d/checkpoint_latest.pt)")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE weights")
    parser.add_argument("--text_encoder_ckpt", type=str, required=True, help="Path to T5 text encoder")
    parser.add_argument("--model_type", type=str, default="infinity_2b")
    parser.add_argument("--vae_type", type=int, default=32)
    
    parser.add_argument("--cfg", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--tau", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--cfg_insertion_layer", type=int, default=0)
    parser.add_argument("--sampling_per_bits", type=int, default=1)
    parser.add_argument("--enable_positive_prompt", type=int, default=0)
    
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
    
    parser.add_argument("--use_fastvar", action="store_true", default=True)
    parser.add_argument("--no_fastvar", action="store_true", help="Disable FastVAR acceleration")
    parser.add_argument("--spherical_bias_lambda", type=float, default=1.0)
    parser.add_argument("--border_keep_w", type=int, default=2)
    parser.add_argument("--seam_boost", type=float, default=1.5)
    parser.add_argument("--shared_border_mode", type=str, default="avg", choices=["avg", "copy_owner"])
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_cubemap_combined", action="store_true")
    
    return parser.parse_args()


def load_config(config_path: Optional[str], args: argparse.Namespace) -> PanoConfig:
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
    config.pn = args.pn
    config.cfg = args.cfg
    config.tau = args.tau
    config.spherical_bias_lambda = args.spherical_bias_lambda
    config.use_fastvar = not args.no_fastvar
    config.border_keep_w = args.border_keep_w
    config.seam_boost = args.seam_boost
    config.shared_border_mode = args.shared_border_mode
    
    if args.seed is not None:
        config.seed = args.seed
    
    config.validate()
    return config


class InfinityPanoGenerator:
    
    def __init__(self, args: argparse.Namespace, config: PanoConfig, device: str = "cuda"):
        self.args = args
        self.config = config
        self.device = device
        
        self._load_models()
        
        if config.use_spherical_attn_bias:
            self.spherical_bias = SphericalAttentionBias(
                lambda_=config.spherical_bias_lambda,
                tau_deg=config.spherical_bias_tau_deg,
            )
        else:
            self.spherical_bias = None
    
    def _load_models(self) -> None:
        from tools.run_infinity import (
            load_tokenizer, load_visual_tokenizer, load_transformer,
            dynamic_resolution_h_w, h_div_w_templates
        )
        
        logger.info("Loading text encoder...")
        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=self.args.text_encoder_ckpt
        )
        
        logger.info("Loading VAE...")
        self.vae = load_visual_tokenizer(self.args)
        
        logger.info("Loading Infinity transformer...")
        self.infinity = load_transformer(self.vae, self.args)
        
        # Load trained checkpoint if provided
        if self.args.checkpoint:
            logger.info(f"Loading trained checkpoint from {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
            
            # Extract model state dict
            if "infinity_state_dict" in checkpoint:
                state_dict = checkpoint["infinity_state_dict"]
                epoch = checkpoint.get("epoch", "unknown")
                logger.info(f"Checkpoint from epoch {epoch}")
            else:
                state_dict = checkpoint
            
            # Load weights
            missing, unexpected = self.infinity.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info(f"Missing keys: {len(missing)} (expected for frozen backbone)")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")
            
            logger.info("Trained checkpoint loaded successfully")
        
        self.dynamic_resolution_h_w = dynamic_resolution_h_w
        self.h_div_w_templates = h_div_w_templates
        
        logger.info("All models loaded successfully")
    
    def _get_scale_schedule(self, h_div_w: float) -> list:
        h_div_w_template = self.h_div_w_templates[
            np.argmin(np.abs(self.h_div_w_templates - h_div_w))
        ]
        scale_schedule = self.dynamic_resolution_h_w[h_div_w_template][self.args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        return scale_schedule
    
    def _encode_prompt(self, prompt: str) -> tuple:
        from tools.run_infinity import encode_prompt
        return encode_prompt(
            self.text_tokenizer, 
            self.text_encoder, 
            prompt,
            enable_positive_prompt=self.args.enable_positive_prompt
        )
    
    def generate_erp(self, prompt: str, seed: Optional[int] = None) -> torch.Tensor:
        from tools.run_infinity import gen_one_img
        
        scale_schedule = self._get_scale_schedule(H_DIV_W_ERP)
        
        g_seed = seed if seed is not None else (
            self.config.seed if self.config.seed >= 0 else int(time.time()) % 10000
        )
        
        with torch.inference_mode():
            with autocast(dtype=torch.bfloat16):
                generated_image = gen_one_img(
                    self.infinity,
                    self.vae,
                    self.text_tokenizer,
                    self.text_encoder,
                    prompt,
                    g_seed=g_seed,
                    gt_leak=0,
                    gt_ls_Bl=None,
                    cfg_list=self.config.cfg,
                    tau_list=self.config.tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[self.args.cfg_insertion_layer],
                    vae_type=self.args.vae_type,
                    sampling_per_bits=self.args.sampling_per_bits,
                    enable_positive_prompt=self.args.enable_positive_prompt,
                )
        
        return generated_image
    
    def generate_cubemap(self, prompt: str, seed: Optional[int] = None) -> torch.Tensor:
        from tools.run_infinity import gen_one_img
        
        scale_schedule = self._get_scale_schedule(1.0)
        
        g_seed = seed if seed is not None else (
            self.config.seed if self.config.seed >= 0 else int(time.time()) % 10000
        )
        
        faces = []
        face_prompts = [
            f"{prompt}, front view",
            f"{prompt}, right side view",
            f"{prompt}, back view",
            f"{prompt}, left side view",
            f"{prompt}, looking up at the sky",
            f"{prompt}, looking down at the ground",
        ]
        
        with torch.inference_mode():
            with autocast(dtype=torch.bfloat16):
                for i, face_prompt in enumerate(face_prompts):
                    face_seed = g_seed + i * 1000
                    face_image = gen_one_img(
                        self.infinity,
                        self.vae,
                        self.text_tokenizer,
                        self.text_encoder,
                        face_prompt,
                        g_seed=face_seed,
                        gt_leak=0,
                        gt_ls_Bl=None,
                        cfg_list=self.config.cfg,
                        tau_list=self.config.tau,
                        scale_schedule=scale_schedule,
                        cfg_insertion_layer=[self.args.cfg_insertion_layer],
                        vae_type=self.args.vae_type,
                        sampling_per_bits=self.args.sampling_per_bits,
                        enable_positive_prompt=self.args.enable_positive_prompt,
                    )
                    faces.append(face_image)
        
        faces_tensor = torch.stack([
            torch.from_numpy(f.cpu().numpy() if isinstance(f, torch.Tensor) else f).float() 
            for f in faces
        ])
        
        if faces_tensor.max() > 1.0:
            faces_tensor = faces_tensor / 255.0
        
        if self.config.use_shared_border_latent:
            # Note: Cubemap border synchronization is not implemented in this version.
            # To enable, uncomment the import at the top of this file and implement
            # the synchronize_cubemap_borders function in pano/fastvar/shared_border.py
            logger.warning(
                "use_shared_border_latent is enabled but cubemap border synchronization "
                "is not available. Skipping border sync. Set use_shared_border_latent=False "
                "in config to suppress this warning."
            )
        
        return faces_tensor


def save_erp_image(image: Union[torch.Tensor, np.ndarray], output_path: Path) -> None:
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = image.cpu().numpy()  # type: np.ndarray
    
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    if img_np.shape[-1] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path), img_np)


def save_cubemap_images(
    faces: torch.Tensor,
    output_dir: Path,
    prefix: str,
    save_combined: bool = False,
) -> None:
    face_names = ["front", "right", "back", "left", "top", "bottom"]
    
    for i, name in enumerate(face_names):
        face = faces[i]
        if isinstance(face, torch.Tensor):
            img_np = face.cpu().numpy()
        else:
            img_np = face
        
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        Image.fromarray(img_np).save(output_dir / f"{prefix}_{name}.png")
    
    if save_combined:
        face_0 = faces[0]
        if isinstance(face_0, torch.Tensor):
            face_h, face_w = int(face_0.shape[-2]), int(face_0.shape[-1])
        else:
            face_h, face_w = int(face_0.shape[0]), int(face_0.shape[1])
        
        cross = np.zeros((face_h * 3, face_w * 4, 3), dtype=np.uint8)
        
        for i, (row, col) in enumerate([(0, 1), (1, 2), (2, 1), (1, 0), (1, 1), (1, 3)]):
            face = faces[i]
            if isinstance(face, torch.Tensor):
                img_np = face.cpu().numpy()
            else:
                img_np = face
            
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            y_start, y_end = row * face_h, (row + 1) * face_h
            x_start, x_end = col * face_w, (col + 1) * face_w
            cross[y_start:y_end, x_start:x_end] = img_np
        
        Image.fromarray(cross).save(output_dir / f"{prefix}_cross.png")


def main() -> None:
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(args.config, args)
    logger.info(f"Config:\n{config}")
    
    generator = InfinityPanoGenerator(args, config, args.device)
    
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    if not prompts:
        prompts = [
            "A 360° equirectangular panorama (ERP), seamless left-right wrap, stable poles. "
            "A high-altitude mountain valley at sunset, glowing clouds, distant snow peaks "
            "forming a continuous horizon, river winding through pine forest, warm rim light, "
            "natural colors, ultra detailed, photorealistic."
        ]
        logger.info("No prompt provided, using default panorama prompt")
    
    logger.info(f"Generating {len(prompts)} prompt(s) x {args.num_samples} sample(s)")
    
    total_time = 0.0
    total_images = 0
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"[{prompt_idx + 1}/{len(prompts)}] Prompt: {prompt[:80]}...")
        
        for sample_idx in range(args.num_samples):
            seed = args.seed + sample_idx if args.seed is not None else None
            
            start_time = time.time()
            
            if config.mode == "erp":
                image = generator.generate_erp(prompt, seed)
                output_path = output_dir / f"erp_{prompt_idx:04d}_{sample_idx:04d}.png"
                save_erp_image(image, output_path)
            else:
                faces = generator.generate_cubemap(prompt, seed)
                prefix = f"cubemap_{prompt_idx:04d}_{sample_idx:04d}"
                save_cubemap_images(faces, output_dir, prefix, args.save_cubemap_combined)
                output_path = output_dir / f"{prefix}_front.png"
            
            gen_time = time.time() - start_time
            total_time += gen_time
            total_images += 1
            
            logger.info(f"  Sample {sample_idx + 1}: {gen_time:.2f}s -> {output_path}")
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    logger.info(f"Generated {total_images} images in {total_time:.1f}s ({total_time/max(total_images,1):.2f}s/image)")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
