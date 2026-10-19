"""Loading images and turning the annotated contour masks into point labels."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Gaps between anti-aliased contour strokes create specks of "interior" a few
# pixels big; real operculum interiors are 10x larger even in the small-barnacle image.
MIN_INTERIOR_AREA = 10
# Some small barnacles were drawn with a stroke so thick it closed the interior.
MIN_SOLID_CONTOUR_AREA = 50


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def points_from_mask(mask_path):
    """One (x, y, radius) per annotated barnacle, in full-image pixels.

    Labels the holes inside the strokes rather than the strokes themselves, which
    keeps touching outlines apart. Tiny outlines that the stroke filled in are
    taken as solid blobs. Radius is from the outlined area."""
    stroke = np.array(Image.open(mask_path))[..., 3] > 0
    filled = ndi.binary_fill_holes(stroke)
    interiors, n_interiors = ndi.label(filled & ~stroke)
    blobs, n_blobs = ndi.label(filled, structure=np.ones((3, 3)))

    interior_area = np.bincount(interiors.ravel(), minlength=n_interiors + 1)
    interior_area[0] = 0
    interior_ids = np.flatnonzero(interior_area >= MIN_INTERIOR_AREA)
    interior_centers = np.array(ndi.center_of_mass(filled, interiors, interior_ids))

    # Each interior's blob is one outline, or several touching ones; share its area.
    blob_area = np.bincount(blobs.ravel(), minlength=n_blobs + 1)
    blob_of_interior = ndi.maximum(blobs, interiors, interior_ids).astype(int)  # constant per interior
    outlines_per_blob = np.bincount(blob_of_interior, minlength=n_blobs + 1)
    interior_radius = np.sqrt(
        blob_area[blob_of_interior] / outlines_per_blob[blob_of_interior] / np.pi
    )

    solid_ids = np.flatnonzero((outlines_per_blob == 0) & (blob_area >= MIN_SOLID_CONTOUR_AREA))
    solid_ids = solid_ids[solid_ids > 0]
    solid_centers = np.array(ndi.center_of_mass(filled, blobs, solid_ids)).reshape(-1, 2)
    solid_radius = np.sqrt(blob_area[solid_ids] / np.pi)

    centers = np.vstack([interior_centers, solid_centers])[:, ::-1]  # (row, col) -> (x, y)
    return np.c_[centers, np.concatenate([interior_radius, solid_radius])]


@dataclass
class Sample:
    """An image rectified to the canonical square, with labels if we have them.

    `points` are (x, y, radius) in canonical pixels.
    """

    name: str
    image: np.ndarray  # (CANONICAL_SIZE, CANONICAL_SIZE, 3) float in [0, 1]
    corners: np.ndarray  # frame inner-square corners in the original photo
    points: np.ndarray | None


def load_sample(name, mask_name=None):
    rgb = load_rgb(DATA_DIR / f"{name}.png")
    corners = frame.find_inner_square(rgb)
    transform = frame.homography(corners)
    image = frame.rectify(rgb, transform)

    points = None
    if mask_name is not None:
        points = to_canonical(points_from_mask(DATA_DIR / f"{mask_name}.png"), transform)
    return Sample(name, image, corners, points)


def crop(image, points, box):
    """Cut `box` = (x0, y0, x1, y1) out of a canonical image, keeping the points
    whose centers fall inside it, shifted into the crop's coordinates."""
    x0, y0, x1, y1 = box
    inside = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
    return image[y0:y1, x0:x1], points[inside] - [x0, y0, 0]


def to_canonical(points, transform, size=frame.CANONICAL_SIZE):
    """Map (x, y, radius) points into the square, dropping any outside it. The
    radius is rescaled by how much the warp stretches that spot."""
    xy = transform(points[:, :2])
    shifted = transform(points[:, :2] + np.c_[points[:, 2], np.zeros(len(points))])
    radius = np.linalg.norm(shifted - xy, axis=1)
    inside = np.all((xy >= 0) & (xy < size), axis=1)
    return np.c_[xy, radius][inside]
