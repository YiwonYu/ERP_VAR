"""
Geometry utilities for panorama projections.
"""

from pano.geometry.spherical import (
    pixel_to_lonlat,
    lonlat_to_direction,
    pixel_to_direction,
    erp_direction_grid,
    spherical_distance_matrix,
    spherical_distance_approx,
    get_pole_mask,
    get_wrap_seam_indices,
    get_border_mask,
    lonlat_to_pixel,
    direction_to_lonlat,
    direction_to_pixel,
    spherical_neighbor_weights,
)

__all__ = [
    "pixel_to_lonlat",
    "lonlat_to_direction",
    "pixel_to_direction",
    "erp_direction_grid",
    "spherical_distance_matrix",
    "spherical_distance_approx",
    "get_pole_mask",
    "get_wrap_seam_indices",
    "get_border_mask",
    "lonlat_to_pixel",
    "direction_to_lonlat",
    "direction_to_pixel",
    "spherical_neighbor_weights",
]

try:
    from pano.geometry.cubemap import (
        FACE_FRONT,
        FACE_RIGHT,
        FACE_BACK,
        FACE_LEFT,
        FACE_TOP,
        FACE_BOTTOM,
        CUBEMAP_ADJACENCY,
        get_face_direction_vectors,
        get_cubemap_all_directions,
        cubemap_to_erp,
        erp_to_cubemap,
    )
    __all__.extend([
        "FACE_FRONT",
        "FACE_RIGHT",
        "FACE_BACK",
        "FACE_LEFT",
        "FACE_TOP",
        "FACE_BOTTOM",
        "CUBEMAP_ADJACENCY",
        "get_face_direction_vectors",
        "get_cubemap_all_directions",
        "cubemap_to_erp",
        "erp_to_cubemap",
    ])
except ImportError:
    pass
