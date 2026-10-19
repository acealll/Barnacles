"""Scoring, the simulated review, and the notebook's tables and figure.

A detection is correct if it lands within an annotated barnacle's radius. Each
barnacle can only be matched once (optimal one-to-one assignment).
"""

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

THRESHOLD = 0.5


@dataclass
class Score:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self):
        return self.true_positives / max(self.true_positives + self.false_positives, 1)

    @property
    def recall(self):
        return self.true_positives / max(self.true_positives + self.false_negatives, 1)

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def predicted_count(self):
        return self.true_positives + self.false_positives

    @property
    def true_count(self):
        return self.true_positives + self.false_negatives

    @property
    def count_error(self):
        """Signed relative error of the count; FPs and FNs can cancel here, F1 can't hide them."""
        return (self.predicted_count - self.true_count) / max(self.true_count, 1)

    def as_row(self):
        return {
            "true count": self.true_count,
            "predicted count": self.predicted_count,
            "count error": f"{self.count_error:+.1%}",
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "F1": round(self.f1, 3),
        }


def match(predicted_xy, truth_xyr):
    """Return a boolean per prediction (is it a hit?) and per annotation (was it found?)."""
    predicted_xy = np.asarray(predicted_xy, dtype=float).reshape(-1, 2)
    hit = np.zeros(len(predicted_xy), dtype=bool)
    found = np.zeros(len(truth_xyr), dtype=bool)
    if len(predicted_xy) == 0 or len(truth_xyr) == 0:
        return hit, found

    distance = cdist(predicted_xy, truth_xyr[:, :2])
    allowed = distance <= truth_xyr[None, :, 2]
    # Disallowed pairs get a cost no allowed matching could reach, so the solver
    # only uses them when it has to and we drop them afterwards.
    cost = np.where(allowed, distance, 1e6)
    rows, cols = linear_sum_assignment(cost)
    valid = allowed[rows, cols]
    hit[rows[valid]] = True
    found[cols[valid]] = True
    return hit, found


def score(predicted_xy, truth_xyr):
    hit, found = match(predicted_xy, truth_xyr)
    return Score(int(hit.sum()), int((~hit).sum()), int((~found).sum()))


def kept(detections, threshold=THRESHOLD):
    """(x, y) of the detections at or above the confidence threshold."""
    return detections[detections[:, 2] >= threshold, :2]


def best_threshold(detections, truth_xyr, thresholds):
    f1 = [score(kept(detections, t), truth_xyr).f1 for t in thresholds]
    return thresholds[int(np.argmax(f1))]


def review_curve(detections, truth_xyr, checkpoints, threshold=THRESHOLD, order="uncertainty", seed=0):
    """Simulate someone checking dots one at a time (answering correctly from the
    labels) and return F1 after each number of checks in `checkpoints`.

    "uncertainty" order checks the dots closest to the threshold first; "random"
    is the comparison. Dots below the threshold can be checked too, so a missed
    barnacle can be recovered if the model put any peak on it."""
    is_barnacle, _ = match(detections[:, :2], truth_xyr)
    accepted = detections[:, 2] >= threshold

    if order == "uncertainty":
        queue = np.argsort(np.abs(detections[:, 2] - threshold))
    else:
        queue = np.random.default_rng(seed).permutation(len(detections))

    f1 = []
    reviewed = 0
    for n in checkpoints:
        for i in queue[reviewed:n]:
            accepted[i] = is_barnacle[i]
        reviewed = max(reviewed, n)
        f1.append(score(detections[accepted, :2], truth_xyr).f1)
    return np.array(f1)


# Tables over the detections saved by experiments.py


def load_results(results_dir):
    detections = dict(np.load(results_dir / "detections.npz"))
    labeled_counts = json.loads((results_dir / "labeled_counts.json").read_text())
    return detections, labeled_counts


def baseline_table(detections, held_out):
    rows = []
    for name, truth in held_out.items():
        threshold, polarity = detections[f"baseline/{name}/threshold"]
        s = score(kept(detections[f"baseline/{name}"], threshold), truth)
        rows.append({"image": name, "blob polarity": "dark" if polarity < 0 else "bright", **s.as_row()})
    return pd.DataFrame(rows).set_index("image")


def cross_image_table(detections, samples):
    rows = []
    for key in ("cross/img1->img2", "cross/img2->img1"):
        train, test = key.split("/")[1].split("->")
        s = score(kept(detections[key]), samples[test].points)
        rows.append({"trained on": train, "tested on": test, **s.as_row()})
    return pd.DataFrame(rows).set_index(["trained on", "tested on"])


def budget_table(detections, held_out, labeled_counts, budgets, seeds):
    """Mean and spread over seeds of held-out F1 and count error, per image and budget."""
    rows = []
    for name, truth in held_out.items():
        for budget in budgets:
            scores = [score(kept(detections[f"budget/{name}/{budget}/seed{s}"]), truth) for s in seeds]
            f1 = [s.f1 for s in scores]
            count_error = [s.count_error for s in scores]
            rows.append(
                {
                    "image": name,
                    "labeled area": budget,
                    "labeled barnacles": labeled_counts[name][budget],
                    "F1 mean": np.mean(f1),
                    "F1 min": np.min(f1),
                    "F1 max": np.max(f1),
                    "count error mean": np.mean(count_error),
                    "|count error| max": np.max(np.abs(count_error)),
                }
            )
    return pd.DataFrame(rows)


def pooled_table(detections, held_out, seeds):
    rows = []
    for name, truth in held_out.items():
        for seed in seeds:
            pooled = score(kept(detections[f"pooled/{name}/seed{seed}"]), truth)
            alone = score(kept(detections[f"budget/{name}/1/2/seed{seed}"]), truth)
            rows.append(
                {
                    "image": name,
                    "seed": seed,
                    "F1 trained on this image only": round(alone.f1, 3),
                    "F1 trained on both images": round(pooled.f1, 3),
                }
            )
    return pd.DataFrame(rows).set_index(["image", "seed"])


# Figures

BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8985"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "axes.edgecolor": GRAY,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def show_errors(image, detections, truth_xyr, ax, threshold=THRESHOLD, title=None):
    """Hits as filled dots, false alarms as crosses, missed barnacles as open circles.
    Shapes, not only colors, carry the meaning so it survives colorblindness and print."""
    shown = detections[detections[:, 2] >= threshold]
    hit, found = match(shown[:, :2], truth_xyr)
    ax.imshow(image)
    ax.scatter(*shown[hit, :2].T, s=10, color=AQUA, label=f"correct ({hit.sum()})")
    ax.scatter(*shown[~hit, :2].T, s=28, marker="x", color=RED, linewidths=1.5, label=f"false alarm ({(~hit).sum()})")
    ax.scatter(*truth_xyr[~found, :2].T, s=60, facecolors="none", edgecolors=BLUE, linewidths=1.5,
               label=f"missed ({(~found).sum()})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9)
    ax.set_title(title)
    ax.axis("off")
