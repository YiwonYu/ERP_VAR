"""
Dataset implementations for panorama training.
"""

from pano.datasets.erp_dataset import (
    ERPDataset,
    ERPDatasetWithCaption,
    get_erp_transform,
    create_erp_dataloader,
)

__all__ = [
    "ERPDataset",
    "ERPDatasetWithCaption",
    "get_erp_transform",
    "create_erp_dataloader",
]

try:
    from pano.datasets.cubemap_dataset import (
        CubemapDataset,
        CubemapDatasetWithCaption,
        create_cubemap_dataloader,
        FACE_ORDER,
    )
    __all__.extend([
        "CubemapDataset",
        "CubemapDatasetWithCaption",
        "create_cubemap_dataloader",
        "FACE_ORDER",
    ])
except ImportError:
    pass
