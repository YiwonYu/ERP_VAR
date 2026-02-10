"""
Panorama configuration dataclass for FastVAR panorama training and inference.

This module provides a unified configuration interface for all panorama-related
settings, supporting both ERP (Equirectangular Panorama) and cubemap modes.
"""

import argparse
import copy
import json
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

# Handle optional yaml import
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False


@dataclass
class PanoConfig:
    """
    Configuration for panorama training and inference with FastVAR.
    
    Supports two modes:
    - ERP: Equirectangular Panorama (360° seamless, 2:1 aspect ratio)
    - Cubemap: 6-face cubemap with shared border conditioning
    """
    
    # Mode selection
    mode: str = "erp"  # "erp" or "cubemap"
    
    # FastVAR extensions
    use_fastvar: bool = True
    use_border_keep: bool = True
    border_keep_w: int = 2  # border width in tokens to always keep
    seam_boost: float = 1.5  # boost factor for border token importance
    use_shared_border_latent: bool = False  # cubemap only
    shared_border_mode: str = "avg"  # "avg" or "copy_owner"
    shared_border_width_tokens: int = 2
    
    # 3D direction embedding
    use_dir3d_embed: bool = True
    dir3d_embed_dim: int = 64  # intermediate MLP dimension
    
    # Spherical attention bias
    use_spherical_attn_bias: bool = True
    spherical_bias_lambda: float = 1.0  # bias strength
    spherical_bias_tau_deg: Optional[float] = None  # optional temperature in degrees
    spherical_attn_fallback: str = "standard"  # "standard" or "sparse_neighbor"
    
    # Seam losses
    use_seam_losses: bool = True
    wrap_seam_weight: float = 1.0
    pole_consistency_weight: float = 1.0
    pole_band_deg: float = 20.0  # degrees from pole to apply consistency
    pole_tau_deg: float = 10.0  # temperature for pole loss weighting
    pole_sigma_deg: float = 10.0  # sigma for neighbor weighting
    texture_step_boost: float = 1.0  # multiplier for late (texture) scales
    structure_step_weight: float = 0.1  # multiplier for early (structure) scales
    
    # ERP-specific
    erp_height: int = 512
    erp_width: int = 1024  # typically 2x height
    
    # Cubemap-specific
    cubemap_face_size: int = 512
    cubemap_generation_order: List[str] = field(
        default_factory=lambda: ["front", "right", "back", "left", "up", "down"]
    )
    
    # Resolution preset
    pn: str = "1M"  # "0.06M", "0.25M", "0.60M", "1M"
    
    # Generation settings
    cfg: float = 4.0
    tau: float = 0.5
    seed: int = -1  # -1 for random
    
    @classmethod
    def from_yaml(cls, path: str) -> "PanoConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            PanoConfig instance
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If the file doesn't exist
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required to load YAML configs. "
                "Install with: pip install pyyaml"
            )
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)  # type: ignore[union-attr]
        
        return cls.from_dict(data or {})
    
    @classmethod
    def from_json(cls, path: str) -> "PanoConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            PanoConfig instance
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PanoConfig":
        """
        Create configuration from a dictionary.
        
        Unknown keys are ignored with a warning.
        
        Args:
            d: Dictionary with configuration values
            
        Returns:
            PanoConfig instance
        """
        # Get valid field names
        valid_fields = {f.name for f in fields(cls)}
        
        # Filter to valid fields and warn about unknown ones
        filtered = {}
        for key, value in d.items():
            if key in valid_fields:
                filtered[key] = value
            else:
                warnings.warn(f"Unknown config key '{key}' will be ignored")
        
        return cls(**filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary with all configuration values
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Handle non-JSON-serializable types
            if isinstance(value, list):
                value = list(value)
            result[f.name] = value
        return result
    
    def merge_cli_args(self, args: argparse.Namespace) -> "PanoConfig":
        """
        Create a new config with CLI arguments overriding current values.
        
        Only non-None values from args are used to override.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            New PanoConfig instance with merged values
        """
        # Create a copy of current config
        new_config = copy.deepcopy(self)
        
        # Get valid field names
        valid_fields = {f.name for f in fields(self)}
        
        # Override with CLI args
        for key, value in vars(args).items():
            if key in valid_fields and value is not None:
                setattr(new_config, key, value)
        
        return new_config
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Mode validation
        if self.mode not in ("erp", "cubemap"):
            raise ValueError(
                f"mode must be 'erp' or 'cubemap', got '{self.mode}'"
            )
        
        # shared_border_mode validation
        if self.shared_border_mode not in ("avg", "copy_owner"):
            raise ValueError(
                f"shared_border_mode must be 'avg' or 'copy_owner', "
                f"got '{self.shared_border_mode}'"
            )
        
        # spherical_attn_fallback validation
        if self.spherical_attn_fallback not in ("standard", "sparse_neighbor"):
            raise ValueError(
                f"spherical_attn_fallback must be 'standard' or 'sparse_neighbor', "
                f"got '{self.spherical_attn_fallback}'"
            )
        
        # ERP aspect ratio check
        if self.mode == "erp" and self.erp_width != 2 * self.erp_height:
            warnings.warn(
                f"ERP images typically have 2:1 aspect ratio. "
                f"Current: {self.erp_width}x{self.erp_height} "
                f"(expected width={2*self.erp_height})"
            )
        
        # Positive value checks
        if self.border_keep_w < 0:
            raise ValueError(f"border_keep_w must be non-negative, got {self.border_keep_w}")
        
        if self.seam_boost < 0:
            raise ValueError(f"seam_boost must be non-negative, got {self.seam_boost}")
        
        if self.shared_border_width_tokens < 0:
            raise ValueError(
                f"shared_border_width_tokens must be non-negative, "
                f"got {self.shared_border_width_tokens}"
            )
        
        if self.dir3d_embed_dim <= 0:
            raise ValueError(f"dir3d_embed_dim must be positive, got {self.dir3d_embed_dim}")
        
        if self.spherical_bias_lambda < 0:
            raise ValueError(
                f"spherical_bias_lambda must be non-negative, "
                f"got {self.spherical_bias_lambda}"
            )
        
        if self.pole_band_deg < 0 or self.pole_band_deg > 90:
            raise ValueError(
                f"pole_band_deg must be in [0, 90], got {self.pole_band_deg}"
            )
        
        if self.erp_height <= 0:
            raise ValueError(f"erp_height must be positive, got {self.erp_height}")
        
        if self.erp_width <= 0:
            raise ValueError(f"erp_width must be positive, got {self.erp_width}")
        
        if self.cubemap_face_size <= 0:
            raise ValueError(
                f"cubemap_face_size must be positive, got {self.cubemap_face_size}"
            )
        
        # Validate cubemap generation order
        valid_faces = {"front", "right", "back", "left", "up", "down"}
        for face in self.cubemap_generation_order:
            if face not in valid_faces:
                raise ValueError(
                    f"Invalid face name '{face}' in cubemap_generation_order. "
                    f"Valid names: {valid_faces}"
                )
        
        # Check for duplicate faces
        if len(self.cubemap_generation_order) != len(set(self.cubemap_generation_order)):
            raise ValueError(
                "cubemap_generation_order contains duplicate face names"
            )
    
    def save_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save YAML file
            
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required to save YAML configs. "
                "Install with: pip install pyyaml"
            )
        
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)  # type: ignore[union-attr]
    
    def save_json(self, path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path: Path to save JSON file
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        """Pretty representation of config."""
        lines = ["PanoConfig("]
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"    {f.name}={value!r},")
        lines.append(")")
        return "\n".join(lines)
