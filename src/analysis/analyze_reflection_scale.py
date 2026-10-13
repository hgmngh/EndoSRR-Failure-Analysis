
from pathlib import Path

import cv2
import pandas as pd

# 입력 및 결과 저장 경로
project_dir = Path(__file__).resolve().parents[3]
mask_dir = project_dir / "EndoSRR/output/final_mask"
save_dir = project_dir / "EndoSRR-Research/results"
save_dir.mkdir(exist_ok=True)

records = []

for mask_path in sorted(mask_dir.glob("*.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"이미지 읽기 실패: {mask_path.name}")
        continue

    binary = (mask > 0).astype("uint8")

    # 연결된 흰색 영역 구분
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(binary, connectivity=8)
    )

    height, width = binary.shape
    image_area = height * width

    # 0번은 배경이므로 제외
    for region_id in range(1, count):
        area = int(stats[region_id, cv2.CC_STAT_AREA])

        records.append({
            "image": mask_path.name,
            "region_id": region_id,
            "area_px": area,
            "area_ratio_pct": area / image_area * 100
        })

df = pd.DataFrame(
    records,
    columns=["image", "region_id", "area_px", "area_ratio_pct"]
)

save_path = save_dir / "reflection_scale.csv"
df.to_csv(save_path, index=False)

print(f"분석된 영역 수: {len(df)}")
print(f"CSV 저장 완료: {save_path}")
