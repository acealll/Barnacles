# barnacles

Counting barnacles by hand inside the frame takes forever, often 1,000+ per photo. My idea: have a model put a dot on every live barnacle, and let a person just check the dots it isn't sure about. The details and figures are in [`barnacles.ipynb`](barnacles.ipynb).

## How it works

1. **Find the frame.** The wire is a distinct teal color, so a color threshold finds it. The inner square is the biggest region the wire fully encloses. It gets warped to a 1024×1024 square, which also puts every photo at the same scale.
2. **Labels to points.** The masks outline the opening of *live* barnacles only (empty shells aren't labeled), so each outline becomes one point.
3. **Detect barnacles.** A small U-Net predicts a heatmap that peaks at each barnacle's center. The peaks become dots, and the peak height is the confidence.
4. **Score it.** Train on the top half of a photo, test on the bottom half. A dot counts if it lands inside a labeled barnacle.

## Results

| Setup | img1 F1 | img2 F1 |
|---|---|---|
| Blob detector (baseline) | 0.36 | 0.31 |
| CNN trained on the other photo only | 0.83 | 0.42 |
| CNN trained on the top halves of both photos | 0.94 | 0.91 |

- The CNN does well on photos like the ones it trained on, but doesn't reliably carry over to a new site. Labeling a few hundred barnacles from that site fixes most of it.
- Count error alone is misleading. The img1 → img2 model was only 7% off on the count, even though half its dots were wrong.
- Checking the least confident dots first improves results faster than checking at random, but the gain is modest.
- I haven't timed it, so the actual speed-up is still unknown. More labeled photos from different sites would be the next step.

## Files

- `barnacles.ipynb`: the walkthrough
- `barnacles/frame.py`: finding and straightening the frame
- `barnacles/data.py`: loading images and turning masks into points
- `barnacles/model.py`: the U-Net and the blob-detector baseline
- `barnacles/evaluate.py`: scoring, the review simulation, tables and plots
- `barnacles/experiments.py`: all the training runs; saves to `results/`
- `data/`: the provided images and masks

## Running it

```bash
pip install -r requirements.txt
jupyter notebook barnacles.ipynb
```

The notebook loads saved results, so it runs in under a minute. To retrain everything (about an hour on a Mac with Apple silicon):

```bash
python -m barnacles.experiments
```

I used an AI coding assistant (Claude) heavily on this project, for most of the implementation and writing. I set the goals and made the calls on scope and presentation.
