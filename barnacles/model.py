"""Barnacle detector: a small U-Net that predicts a heatmap peaking at each
barnacle center. Peaks are the detections, peak height is the confidence.

The Laplacian-of-Gaussian baseline is at the bottom.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from torch import nn

CROP = 256
HEATMAP_SIGMA = 2.0  # pixels; small enough that neighboring barnacles stay separate peaks
PEAK_RADIUS = 3  # suppress non-maxima within this many pixels


def default_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Receptive field is ~100 px, about two shell widths, so the net sees the
    shell around each opening. That's what tells a live barnacle from an empty shell."""

    def __init__(self, width=16):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.down = nn.ModuleList(
            [_conv_block(3, w[0]), _conv_block(w[0], w[1]), _conv_block(w[1], w[2])]
        )
        self.bottom = _conv_block(w[2], w[3])
        self.up = nn.ModuleList(
            [_conv_block(w[3] + w[2], w[2]), _conv_block(w[2] + w[1], w[1]), _conv_block(w[1] + w[0], w[0])]
        )
        self.head = nn.Conv2d(w[0], 1, 1)
        # Start predicting "probably background" everywhere, as CenterNet does;
        # otherwise the first steps are dominated by the huge number of negatives.
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)  # logits


def standardize(image):
    """Per-channel standardization, to take out overall exposure differences
    between photos (img1 is bright and dry, img2 dark and wet)."""
    image = np.asarray(image, dtype=np.float32)
    return (image - image.mean(axis=(0, 1))) / (image.std(axis=(0, 1)) + 1e-6)


def render_heatmap(points_xy, size):
    """Gaussian bump at each point. Bumps are merged with max so touching barnacles
    keep separate peaks, and each center pixel is exactly 1 (the focal loss's positive)."""
    heatmap = np.zeros((size, size), dtype=np.float32)
    r = int(3 * HEATMAP_SIGMA)
    offsets = np.arange(-r, r + 1)
    for x, y in points_xy:
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < size and 0 <= cy < size):
            continue
        gx = np.exp(-((cx + offsets - x) ** 2) / (2 * HEATMAP_SIGMA**2))
        gy = np.exp(-((cy + offsets - y) ** 2) / (2 * HEATMAP_SIGMA**2))
        bump = np.outer(gy, gx)
        y0, x0 = cy - r, cx - r
        ys = slice(max(y0, 0), min(y0 + 2 * r + 1, size))
        xs = slice(max(x0, 0), min(x0 + 2 * r + 1, size))
        patch = bump[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        heatmap[ys, xs] = np.maximum(heatmap[ys, xs], patch)
        heatmap[cy, cx] = 1.0
    return heatmap


def focal_loss(logits, target):
    """CenterNet's focal loss. Pixels near a center are barely penalized, so a peak
    one pixel off isn't treated as a mistake."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * positive
    neg_loss = -torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - positive)
    return (pos_loss.sum() + neg_loss.sum()) / positive.sum().clamp(min=1)


class CropSampler:
    """Random training crops with augmentation: rotation/flips, color and
    brightness, and scale (barnacle size varies by site and age)."""

    def __init__(self, regions, rng, scale_range=(0.7, 1.4)):
        self.regions = [(standardize(img), pts[:, :2]) for img, pts in regions]
        self.rng = rng
        self.scale_range = scale_range
        weights = np.array([img.shape[0] * img.shape[1] for img, _ in self.regions], float)
        self.weights = weights / weights.sum()

    def sample(self):
        image, points = self.regions[self.rng.choice(len(self.regions), p=self.weights)]
        scale = np.exp(self.rng.uniform(*np.log(self.scale_range)))
        src = min(int(round(CROP / scale)), *image.shape[:2])
        y0 = self.rng.integers(0, image.shape[0] - src + 1)
        x0 = self.rng.integers(0, image.shape[1] - src + 1)

        crop = torch.from_numpy(image[y0 : y0 + src, x0 : x0 + src]).permute(2, 0, 1)[None]
        crop = F.interpolate(crop, size=(CROP, CROP), mode="bilinear", align_corners=False)[0]
        pts = (points - [x0, y0]) * (CROP / src)
        heat = torch.from_numpy(render_heatmap(pts, CROP))[None]

        k = int(self.rng.integers(4))
        crop, heat = torch.rot90(crop, k, (1, 2)), torch.rot90(heat, k, (1, 2))
        if self.rng.random() < 0.5:
            crop, heat = crop.flip(2), heat.flip(2)

        gain = torch.from_numpy(self.rng.uniform(0.7, 1.3, (3, 1, 1)).astype(np.float32))
        bias = torch.from_numpy(self.rng.uniform(-0.3, 0.3, (3, 1, 1)).astype(np.float32))
        crop = crop * gain + bias
        if self.rng.random() < 0.2:
            crop = crop.mean(dim=0, keepdim=True).expand(3, -1, -1)
        return crop, heat

    def batch(self, size):
        crops, heats = zip(*(self.sample() for _ in range(size)))
        return torch.stack(crops), torch.stack(heats)


def train(regions, steps=3000, batch_size=12, lr=1e-3, seed=0, device=None, log_every=500):
    """Train a fresh model on `regions`, a list of (canonical image, (x, y, r) points)."""
    device = device or default_device()
    torch.manual_seed(seed)
    sampler = CropSampler(regions, np.random.default_rng(seed))
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)

    model.train()
    running = 0.0
    for step in range(1, steps + 1):
        x, y = sampler.batch(batch_size)
        loss = focal_loss(model(x.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        running += loss.item()
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {running / log_every:.3f}")
            running = 0.0
    return model.eval()


@torch.no_grad()
def heatmap(model, image):
    device = next(model.parameters()).device
    x = torch.from_numpy(standardize(image)).permute(2, 0, 1)[None].to(device)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def peaks(heat, min_confidence=0.05):
    """Local maxima of the heatmap as (x, y, confidence) rows."""
    t = torch.from_numpy(heat)[None, None]
    window = 2 * PEAK_RADIUS + 1
    is_peak = (t == F.max_pool2d(t, window, stride=1, padding=PEAK_RADIUS))[0, 0].numpy()
    ys, xs = np.nonzero(is_peak & (heat >= min_confidence))
    return np.c_[xs, ys, heat[ys, xs]].astype(float)


def detect(model, image, min_confidence=0.05):
    return peaks(heatmap(model, image), min_confidence)


# Baseline: plain blob detection, no learning beyond picking polarity and threshold.

# Barnacle radii are roughly 4-20 px here, and a LoG blob of scale sigma has radius sigma * sqrt(2).
LOG_SIGMAS = np.geomspace(3, 14, 8)


def detect_blobs(image, polarity, min_distance=4):
    """Return (x, y, response) rows for blobs darker (polarity=-1) or brighter (+1)
    than their surroundings."""
    gray = rgb2gray(image)
    gray = (gray - gray.mean()) / gray.std()
    # -sigma^2 * LoG is positive at the center of a bright blob of matching size.
    responses = np.stack([-polarity * s**2 * ndi.gaussian_laplace(gray, s) for s in LOG_SIGMAS])
    best = responses.max(axis=0)
    centers = peak_local_max(best, min_distance=min_distance, threshold_abs=0.0, exclude_border=False)
    return np.c_[centers[:, ::-1], best[centers[:, 0], centers[:, 1]]]
