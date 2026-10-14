
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 프로젝트 경로
root = Path(__file__).resolve().parents[3]
gt_dir = root / "EndoSRR/EndoRR_Dataset/test/mask"
pred_dir = root / "EndoSRR/output/final_mask"
save_dir = root / "EndoSRR-Research/results"
save_dir.mkdir(parents=True, exist_ok=True)

records = []

# GT 마스크별 분석
for gt_path in sorted(gt_dir.glob("*.png")):
    pred_path = pred_dir / gt_path.name

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print(f"파일 읽기 실패: {gt_path.name}")
        continue

    if gt.shape != pred.shape:
        print(f"이미지 크기 불일치: {gt_path.name}")
        continue

    gt_binary = (gt > 0).astype(np.uint8)
    pred_binary = pred > 0

    # GT에서 연결된 영역 분리
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        gt_binary, connectivity=8
    )

    for region_id in range(1, count):
        region = labels == region_id
        area = int(stats[region_id, cv2.CC_STAT_AREA])

        # GT 영역 중 예측이 일치한 픽셀 비율
        detected = int(np.count_nonzero(region & pred_binary))
        recall = detected / area

        records.append({
            "image": gt_path.name,
            "region_id": region_id,
            "area_px": area,
            "detected_px": detected,
            "recall": recall
        })

if not records:
    raise RuntimeError("분석된 영역이 없습니다. 입력 경로를 확인하세요.")

df = pd.DataFrame(records)

save_path = save_dir / "gt_component_recall.csv"
df.to_csv(save_path, index=False)

print("GT 연결 요소 총수:", len(df))
print("\n영역별 Recall 통계")
print(df["recall"].describe().round(4))
print("\n완전히 놓친 영역:", (df["recall"] == 0).sum())
print("완전히 검출한 영역:", (df["recall"] == 1).sum())
print("\nCSV 저장 완료:", save_path)

