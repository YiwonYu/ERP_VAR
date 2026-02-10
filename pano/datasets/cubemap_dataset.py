"""
Cubemap Dataset for loading panorama images as 6-face cubemaps.

Supports three layout formats:
- "folder": 6 separate image files per sample directory
- "cross": Single cross-layout image (4:3 aspect ratio)
- "strip": Single horizontal strip image (6:1 aspect ratio)
"""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# Cubemap face order: front, right, back, left, top, bottom
FACE_ORDER: list[str] = ["front", "right", "back", "left", "top", "bottom"]

# Common image extensions
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class CubemapDataset(Dataset[dict[str, Any]]):
    """
    PyTorch Dataset for cubemap panorama images with multiple layout support.

    Args:
        root: Root directory containing sample folders or images
        transform: Optional torchvision transform to apply to each face
        layout: One of "folder", "cross", or "strip"
            - "folder": {root}/{sample_id}/{face}.{ext}
            - "cross": {root}/{sample_id}/image.{ext} in cross layout (4:3 ratio)
            - "strip": {root}/{sample_id}/image.{ext} in horizontal strip (6:1 ratio)
        face_size: Target size for each cubic face (default 256)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        layout: str = "folder",
        face_size: int = 256,
    ):
        self.root = Path(root)
        self.transform = transform
        self.layout = layout
        self.face_size = face_size

        if layout not in ["folder", "cross", "strip"]:
            raise ValueError(f"layout must be one of 'folder', 'cross', 'strip', got {layout}")

        self.samples: list[dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Discover and index all samples in the root directory."""
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        if self.layout == "folder":
            self._load_folder_layout()
        elif self.layout == "cross":
            self._load_cross_layout()
        elif self.layout == "strip":
            self._load_strip_layout()

    def _load_folder_layout(self) -> None:
        """Load samples where each sample is a folder with 6 face images."""
        for sample_dir in sorted(self.root.iterdir()):
            if not sample_dir.is_dir():
                continue

            # Check if all required faces exist
            faces_found: dict[str, Path] = {}
            for face in FACE_ORDER:
                face_path = self._find_image_file(sample_dir, face)
                if face_path:
                    faces_found[face] = face_path

            if len(faces_found) == 6:
                self.samples.append({"type": "folder", "sample_id": sample_dir.name, "faces": faces_found})

    def _load_cross_layout(self) -> None:
        """Load samples where each sample is an image in cross layout."""
        for sample_dir in sorted(self.root.iterdir()):
            if not sample_dir.is_dir():
                continue

            image_path = self._find_first_image(sample_dir)
            if image_path:
                self.samples.append({"type": "cross", "sample_id": sample_dir.name, "image_path": image_path})

    def _load_strip_layout(self) -> None:
        """Load samples where each sample is an image in horizontal strip layout."""
        for sample_dir in sorted(self.root.iterdir()):
            if not sample_dir.is_dir():
                continue

            image_path = self._find_first_image(sample_dir)
            if image_path:
                self.samples.append({"type": "strip", "sample_id": sample_dir.name, "image_path": image_path})

    def _find_image_file(self, directory: Path, base_name: str) -> Optional[Path]:
        """Find an image file with the given base name in the directory."""
        for ext in IMAGE_EXTENSIONS:
            path = directory / f"{base_name}{ext}"
            if path.exists():
                return path
        return None

    def _find_first_image(self, directory: Path) -> Optional[Path]:
        """Find the first image file in the directory."""
        for file in sorted(directory.iterdir()):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                return file
        return None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load a cubemap sample.

        Returns:
            dict with keys:
            - "faces": Tensor of shape (6, C, H, W) with values in [0, 1]
            - "path": str, path to the sample directory
        """
        sample = self.samples[idx]
        sample_dir = self.root / sample["sample_id"]

        faces_tensor: torch.Tensor
        if sample["type"] == "folder":
            faces_tensor = self._load_folder_faces(sample["faces"])
        elif sample["type"] == "cross":
            faces_tensor = self._load_cross_image(sample["image_path"])
        elif sample["type"] == "strip":
            faces_tensor = self._load_strip_image(sample["image_path"])
        else:
            raise ValueError(f"Unknown sample type: {sample['type']}")

        # Resize each face to target size
        faces_tensor = torch.stack(
            [TF.resize(face, [self.face_size, self.face_size], antialias=True) for face in faces_tensor]
        )

        # Apply transform to all faces if provided
        if self.transform:
            faces_tensor = torch.stack([self.transform(face) for face in faces_tensor])

        return {"faces": faces_tensor, "path": str(sample_dir)}

    def _load_folder_faces(self, faces: dict[str, Path]) -> torch.Tensor:
        """Load 6 face images from individual files in order of FACE_ORDER."""
        face_tensors = []
        for face_name in FACE_ORDER:
            image = Image.open(faces[face_name]).convert("RGB")
            tensor = TF.to_tensor(image)  # Shape: (3, H, W), values in [0, 1]
            face_tensors.append(tensor)
        return torch.stack(face_tensors)  # Shape: (6, 3, H, W)

    def _load_cross_image(self, image_path: Path) -> torch.Tensor:
        """Load cross-layout image and split into 6 faces."""
        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image)  # Shape: (3, H, W)
        faces = split_cross_layout(tensor)  # Shape: (6, 3, H', W')
        return faces

    def _load_strip_image(self, image_path: Path) -> torch.Tensor:
        """Load horizontal strip image and split into 6 faces."""
        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image)  # Shape: (3, H, W)
        faces = split_strip_layout(tensor)  # Shape: (6, 3, H', W')
        return faces


class CubemapDatasetWithCaption(CubemapDataset):
    """
    Cubemap dataset with caption support.

    Expects captions to be stored in {sample_dir}/caption.txt.

    Returns:
        Dict with keys:
        - "faces": Tensor of shape (6, C, H, W)
        - "caption": str, the caption text
        - "path": str, path to the sample directory
    """

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load a cubemap sample with caption."""
        sample = self.samples[idx]
        sample_dir = self.root / sample["sample_id"]

        # Get faces from parent class
        result = super().__getitem__(idx)

        # Load caption if it exists
        caption_path = sample_dir / "caption.txt"
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = ""

        result["caption"] = caption
        return result


def split_cross_layout(image: torch.Tensor) -> torch.Tensor:
    """
    Split a cross-layout image into 6 cubic faces.

    Cross layout (4:3 aspect ratio, W:H = 4:3):
        +-------+
        |  top  |
        +---+---+---+---+
        | L | F | R | B |
        +---+---+---+---+
        | bottom |
        +-------+

    Args:
        image: Tensor of shape (C, H, W) where W:H = 4:3 (e.g., H=300, W=400)

    Returns:
        Tensor of shape (6, C, h, w) with order [front, right, back, left, top, bottom]
    """
    C, H, W = image.shape

    # Calculate face size from width (4 faces across)
    face_w = W // 4
    face_h = H // 3

    if W != face_w * 4 or H != face_h * 3:
        raise ValueError(f"Image dimensions {H}x{W} don't match 4:3 cross layout (got {H}x{W}, expected {face_h*3}x{face_w*4})")

    # Extract faces based on cross layout
    top = image[:, 0:face_h, face_w:2*face_w]  # Row 0, Col 1
    left = image[:, face_h:2*face_h, 0:face_w]  # Row 1, Col 0
    front = image[:, face_h:2*face_h, face_w:2*face_w]  # Row 1, Col 1
    right = image[:, face_h:2*face_h, 2*face_w:3*face_w]  # Row 1, Col 2
    back = image[:, face_h:2*face_h, 3*face_w:4*face_w]  # Row 1, Col 3
    bottom = image[:, 2*face_h:3*face_h, face_w:2*face_w]  # Row 2, Col 1

    # Return in order: front, right, back, left, top, bottom
    return torch.stack([front, right, back, left, top, bottom])


def split_strip_layout(image: torch.Tensor) -> torch.Tensor:
    """
    Split a horizontal strip layout image into 6 cubic faces.

    Strip layout (6:1 aspect ratio):
    +---+---+---+---+---+---+
    | F | R | B | L | T | Bo|
    +---+---+---+---+---+---+

    Args:
        image: Tensor of shape (C, H, W) where W:H = 6:1

    Returns:
        Tensor of shape (6, C, h, w) with order [front, right, back, left, top, bottom]
    """
    C, H, W = image.shape

    # Calculate face size
    face_h = H
    face_w = W // 6

    if W != face_w * 6:
        raise ValueError(f"Image width {W} is not divisible by 6 for strip layout")

    # Extract faces: front, right, back, left, top, bottom
    front = image[:, :, 0*face_w:1*face_w]
    right = image[:, :, 1*face_w:2*face_w]
    back = image[:, :, 2*face_w:3*face_w]
    left = image[:, :, 3*face_w:4*face_w]
    top = image[:, :, 4*face_w:5*face_w]
    bottom = image[:, :, 5*face_w:6*face_w]

    return torch.stack([front, right, back, left, top, bottom])


def create_cubemap_dataloader(
    dataset: CubemapDataset,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader[dict[str, Any]]:
    """
    Create a DataLoader for cubemap dataset.

    Args:
        dataset: CubemapDataset or CubemapDatasetWithCaption instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
