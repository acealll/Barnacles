"""All the training runs behind the notebook. Saves raw detections to results/
so the notebook doesn't have to retrain.

    python -m barnacles.experiments    # about an hour on an M-series Mac

Models only ever train on the top half of a photo; the bottom half is held out.
"""

import json
from pathlib import Path

import numpy as np

from . import data, evaluate, model

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N = 1024
TOP = (0, 0, N, N // 2)
BOTTOM = (0, N // 2, N, N)
# Labeled patches of increasing size, all inside the top half, as fractions of
# the whole square. This is the "annotate a patch of a new site" scenario.
BUDGETS = {
    "1/16": (0, 0, N // 4, N // 4),
    "1/8": (0, 0, N // 2, N // 4),
    "1/4": (0, 0, N // 2, N // 2),
    "1/2": TOP,
}
SEEDS = (0, 1)
STEPS = 1200


def load_labeled():
    return {
        "img1": data.load_sample("img1", "mask1"),
        "img2": data.load_sample("img2", "mask2"),
    }


def run_label_budget(samples, out):
    """One model per image, budget and seed, trained on a patch of that image only
    and tested on its held-out bottom half."""
    for name, sample in samples.items():
        test_image, _ = data.crop(sample.image, sample.points, BOTTOM)
        for budget, box in BUDGETS.items():
            region = data.crop(sample.image, sample.points, box)
            for seed in SEEDS:
                key = f"budget/{name}/{budget}/seed{seed}"
                print(f"{key}: training on {len(region[1])} labeled barnacles")
                net = model.train([region], steps=STEPS, seed=seed, log_every=0)
                out[key] = model.detect(net, test_image)


def run_cross_image(samples, out):
    """Train on one whole image, test on the whole other image: a new site with no labels."""
    for train_name, test_name in (("img1", "img2"), ("img2", "img1")):
        key = f"cross/{train_name}->{test_name}"
        print(key)
        net = model.train([(samples[train_name].image, samples[train_name].points)], steps=STEPS, log_every=0)
        out[key] = model.detect(net, samples[test_name].image)


def run_pooled(samples, out):
    """One model trained on the top halves of both images. Compare with the
    per-image "1/2" budget to see whether mixing sites helps or hurts."""
    regions = [data.crop(s.image, s.points, TOP) for s in samples.values()]
    for seed in SEEDS:
        net = model.train(regions, steps=STEPS, seed=seed, log_every=0)
        for name, sample in samples.items():
            test_image, _ = data.crop(sample.image, sample.points, BOTTOM)
            out[f"pooled/{name}/seed{seed}"] = model.detect(net, test_image)


def run_baseline(samples, out):
    """LoG blobs (the baseline), with polarity and threshold picked for F1 on the training half.
    The chosen threshold is stored as an extra row so the notebook can apply it."""
    for name, sample in samples.items():
        train_image, train_points = data.crop(sample.image, sample.points, TOP)
        test_image, _ = data.crop(sample.image, sample.points, BOTTOM)
        best = None
        for polarity in (-1, 1):
            blobs = model.detect_blobs(train_image, polarity)
            thresholds = np.quantile(blobs[:, 2], np.linspace(0, 0.99, 100))
            t = evaluate.best_threshold(blobs, train_points, thresholds)
            f1 = evaluate.score(blobs[blobs[:, 2] >= t, :2], train_points).f1
            if best is None or f1 > best[0]:
                best = (f1, polarity, t)
        _, polarity, threshold = best
        out[f"baseline/{name}"] = model.detect_blobs(test_image, polarity)
        out[f"baseline/{name}/threshold"] = np.array([threshold, polarity])


def main():
    RESULTS.mkdir(exist_ok=True)
    samples = load_labeled()
    labeled_counts = {
        name: {b: len(data.crop(s.image, s.points, box)[1]) for b, box in BUDGETS.items()}
        for name, s in samples.items()
    }
    (RESULTS / "labeled_counts.json").write_text(json.dumps(labeled_counts, indent=2))

    out = {}
    for run in (run_baseline, run_cross_image, run_pooled, run_label_budget):
        run(samples, out)
        np.savez_compressed(RESULTS / "detections.npz", **out)  # save as we go


if __name__ == "__main__":
    main()
