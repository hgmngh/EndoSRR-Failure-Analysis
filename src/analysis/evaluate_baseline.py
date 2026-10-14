import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ENDOSRR_DIR, RESULTS_DIR, find_pngs, read_binary_mask


def segmentation_metrics(gt, pred):
    if gt.shape != pred.shape:
        raise ValueError(f"Mask sizes differ: {gt.shape} != {pred.shape}")

    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))

    return {
        "IoU": tp / (tp + fp + fn) if (tp + fp + fn) else 1.0,
        "Dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "Precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def evaluate(gt_dir, pred_dir):
    rows = []
    for gt_path in find_pngs(gt_dir):
        pred_path = pred_dir / gt_path.name
        gt = read_binary_mask(gt_path)
        pred = read_binary_mask(pred_path)
        rows.append({"image": gt_path.name, **segmentation_metrics(gt, pred)})

    return pd.DataFrame(rows, columns=["image", "IoU", "Dice", "Precision", "Recall"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against reference masks.")
    parser.add_argument("--gt-dir", type=Path, default=ENDOSRR_DIR / "EndoRR_Dataset/test/mask")
    parser.add_argument("--pred-dir", type=Path, default=ENDOSRR_DIR / "output/final_mask")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "baseline_metrics.csv")
    args = parser.parse_args()

    result = evaluate(args.gt_dir, args.pred_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
