"""
PyTorch Dataset implementations for Equirectangular Projection (ERP) panorama images.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


class ERPDataset(Dataset[Dict[str, Any]]):
    """
    PyTorch Dataset for loading ERP (Equirectangular Projection) panorama images.
    
    ERP images have a 2:1 aspect ratio (width = 2 * height).
    Images are loaded from a root directory recursively.
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable[..., Any]] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        min_size: int = 256,
        aspect_ratio_tolerance: float = 0.1,
    ):
        """
        Initialize ERPDataset.
        
        Args:
            root: Root directory containing ERP images
            transform: Optional torchvision transform to apply to images
            extensions: Tuple of valid image file extensions (case-insensitive)
            min_size: Minimum image dimension (height) required
            aspect_ratio_tolerance: Tolerance for 2:1 aspect ratio validation (0.0-1.0)
        """
        self.root = Path(root)
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.min_size = min_size
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        
        # Scan directory for valid images
        self.image_paths = self._scan_directory()
        
    def _scan_directory(self) -> List[Path]:
        """
        Recursively scan root directory for valid ERP images.
        
        Returns:
            List of Path objects for valid images, sorted for consistency
        """
        valid_paths = []
        
        for ext in self.extensions:
            # Use glob with recursive search
            for path in self.root.rglob(f"*{ext}"):
                if path.is_file():
                    try:
                        img = Image.open(path)
                        width, height = img.size
                        
                        # Check minimum size
                        if height < self.min_size:
                            continue
                        
                        # Check aspect ratio
                        if self.validate_aspect_ratio(width, height):
                            valid_paths.append(path)
                    except (IOError, OSError):
                        # Skip corrupted images
                        continue
        
        return sorted(valid_paths)
    
    def validate_aspect_ratio(self, width: int, height: int) -> bool:
        """
        Validate if image has 2:1 aspect ratio (ERP requirement).
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            True if aspect ratio is within tolerance of 2:1, False otherwise
        """
        if height == 0:
            return False
        
        aspect_ratio = width / height
        target_ratio = 2.0
        
        # Check if aspect ratio is within tolerance
        tolerance_range = target_ratio * self.aspect_ratio_tolerance
        return abs(aspect_ratio - target_ratio) <= tolerance_range
    
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return an image and metadata.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Dictionary containing:
                - "image": Tensor of shape (C, H, W) with values in [0, 1]
                - "path": String path to the image file
                - "original_size": Tuple of (width, height) before any transforms
        """
        image_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_size = img.size  # (width, height)
        
        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert to tensor without normalization
            img = transforms.ToTensor()(img)
        
        return {
            "image": img,
            "path": str(image_path),
            "original_size": original_size,
        }


class ERPDatasetWithCaption(ERPDataset):
    """
    Extended ERPDataset that loads text captions from sidecar .txt files.
    
    For each image file, a corresponding .txt file with the same basename
    is expected to contain the caption text.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return an image with caption and metadata.
        
        Args:
            idx: Index of the image to load
            
        Returns:
            Dictionary containing:
                - "image": Tensor of shape (C, H, W) with values in [0, 1]
                - "caption": String caption text (empty string if no caption file)
                - "path": String path to the image file
                - "original_size": Tuple of (width, height) before any transforms
        """
        image_path = self.image_paths[idx]
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_size = img.size  # (width, height)
        
        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert to tensor without normalization
            img = transforms.ToTensor()(img)
        
        # Load caption from sidecar .txt file
        caption_path = image_path.with_suffix(".txt")
        caption = ""
        
        if caption_path.exists():
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            except (IOError, OSError):
                # If caption cannot be read, use empty string
                caption = ""
        
        return {
            "image": img,
            "caption": caption,
            "path": str(image_path),
            "original_size": original_size,
        }


def get_erp_transform(target_height: int) -> Callable[..., Any]:
    """
    Create a transform pipeline for ERP panorama images.
    
    Resizes the image to (target_height, target_height * 2) to maintain
    2:1 aspect ratio and applies ImageNet normalization.
    
    Args:
        target_height: Target height for resized images
        
    Returns:
        Callable transform that can be applied to PIL Images
    """
    target_width = target_height * 2
    
    return transforms.Compose([
        transforms.Resize((target_height, target_width), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_erp_dataloader(
    dataset: ERPDataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader[Dict[str, Any]]:
    """
    Create a DataLoader for ERP dataset.
    
    Args:
        dataset: ERPDataset instance
        batch_size: Number of images per batch
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
