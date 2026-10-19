"""Find the frame's inner square and warp it to a fixed-size square.

The frame is the only thing of known size in the photo, so this also puts every
photo at the same scale. Everything downstream works in that square.
"""

from itertools import combinations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.color import rgb2hsv
from skimage.measure import approximate_polygon, find_contours
from skimage.transform import ProjectiveTransform, warp

CANONICAL_SIZE = 1024  # side length, in pixels, of the rectified inner square

# Teal of the wire, in degrees. Rock, algae and the yellow ruler all sit below 90.
FRAME_HUE_RANGE = (150, 220)
WORK_SCALE = 4  # detect the frame on a 4x-downsampled image; it is a coarse object


def frame_mask(rgb):
    hsv = rgb2hsv(rgb)
    hue = hsv[..., 0] * 360
    return (
        (hue >= FRAME_HUE_RANGE[0])
        & (hue <= FRAME_HUE_RANGE[1])
        & (hsv[..., 1] > 0.3)
        & (hsv[..., 2] > 0.1)
    )


def find_inner_square(rgb):
    """Return the 4 corners (x, y) of the frame's inner square in full-resolution
    pixel coordinates, ordered top-left, top-right, bottom-right, bottom-left.

    The inner square is the largest region fully enclosed by wire. Regions that
    touch the image border are open to the outside and can't be it.
    """
    small = rgb[::WORK_SCALE, ::WORK_SCALE]
    wire = ndi.binary_closing(frame_mask(small), iterations=3)
    # Barnacles growing over the wire leave small holes in it; closing bridges them.
    regions, _ = ndi.label(~wire)

    border_labels = np.unique(
        np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]])
    )
    areas = np.bincount(regions.ravel())
    areas[0] = 0
    areas[border_labels] = 0
    if areas.max() == 0:
        raise ValueError("no enclosed region found; is the frame in the picture?")
    cell = ndi.binary_fill_holes(regions == areas.argmax())

    corners = _quadrilateral(cell)
    return _order_corners(corners) * WORK_SCALE


def _quadrilateral(region):
    """Corners of the region, found by intersecting its four side lines.

    The largest quad on the convex hull gives rough corners (without assuming the
    frame is straight in the photo). Those can't be used directly: the cell's
    corners are rounded, so the quad would cut them. Instead each side is the
    longest hull edge between two rough corners."""
    outline = max(find_contours(np.pad(region, 1).astype(float), 0.5), key=len) - 1
    hull = outline[ConvexHull(outline).vertices][:, ::-1]  # (row, col) -> (x, y)

    # Search the max-area quad over a simplified hull to keep it cheap. The
    # simplified vertices are a subset of the hull's, so we can map back.
    simple = approximate_polygon(np.vstack([hull, hull[:1]]), tolerance=1.5)[:-1]
    if len(simple) < 4:
        raise ValueError("inner square outline is degenerate")
    quad = max(combinations(range(len(simple)), 4), key=lambda idx: _area(simple[list(idx)]))
    rough = sorted(int(np.flatnonzero((hull == simple[i]).all(axis=1))[0]) for i in quad)

    sides = []
    for start, end in zip(rough, rough[1:] + [rough[0] + len(hull)]):
        edges = [(hull[k % len(hull)], hull[(k + 1) % len(hull)]) for k in range(start, end)]
        sides.append(max(edges, key=lambda e: np.linalg.norm(e[1] - e[0])))
    return np.array([_intersect(sides[i - 1], sides[i]) for i in range(4)])


def _intersect(line_a, line_b):
    (p, p2), (q, q2) = line_a, line_b
    r, s = p2 - p, q2 - q
    t = np.cross(q - p, s) / np.cross(r, s)
    return p + t * r


def _area(polygon):
    """Shoelace formula; vertices must be in boundary order (hull order guarantees it)."""
    y, x = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _order_corners(points):
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]  # clockwise in image coords, from ~-135 deg
    start = np.argmin(ordered.sum(axis=1))  # top-left has the smallest x + y
    return np.roll(ordered, -start, axis=0)


def homography(corners, size=CANONICAL_SIZE):
    """Transform mapping full-image (x, y) to canonical-square (x, y)."""
    target = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=float)
    transform = ProjectiveTransform()
    if not transform.estimate(corners, target):
        raise ValueError("degenerate frame corners")
    return transform


def rectify(rgb, transform, size=CANONICAL_SIZE):
    """Warp the inner square of `rgb` to a size x size float image in [0, 1]."""
    return warp(rgb, transform.inverse, output_shape=(size, size), order=1)
