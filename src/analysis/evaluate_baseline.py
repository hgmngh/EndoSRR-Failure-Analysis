
from pathlib import Path

import cv2
import pandas as pd

root = Path(__file__).resolve().parents[3]
gt_dir = root / "EndoSRR/EndoRR_Dataset/test/mask"
pred_dir = root / "EndoSRR/output/final_mask"
save_path = root / "EndoSRR-Research/results/baseline_metrics.csv"

records = []

for gt_path in sorted(gt_dir.glob("*.png")):
    pred_path = pred_dir / gt_path.name

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"파일 누락: {gt_path.name}")
        continue

    if gt.shape != pred.shape:
        print(f"크기 불일치: {gt_path.name}")
        continue

    gt = gt > 0
    pred = pred > 0

    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    records.append({
        "image": gt_path.name,
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
    })

df = pd.DataFrame(records)
save_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(save_path, index=False)

print(df.to_string(index=False))
print(f"\nCSV 저장 완료: {save_path}")