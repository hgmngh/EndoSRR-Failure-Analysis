
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[3]
image_dir = root / "EndoSRR/image"
mask_dir = root / "EndoSRR/output/final_mask"
save_dir = root / "EndoSRR-Research/results/component_visualizations"
save_dir.mkdir(parents=True, exist_ok=True)

for mask_path in sorted(mask_dir.glob("*.png")):
    image_path = image_dir / mask_path.name

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"파일 읽기 실패: {mask_path.name}")
        continue

    if image.shape[:2] != mask.shape:
        print(f"이미지 크기 불일치: {mask_path.name}")
        continue

    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    visualization = image.copy()

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area <= 10:
            color = (0, 0, 255)       # 빨강: 1~10 px
        elif area <= 100:
            color = (0, 255, 255)     # 노랑: 11~100 px
        else:
            color = (0, 255, 0)       # 초록: 101 px 이상

        cv2.rectangle(
            visualization,
            (x, y),
            (x + w - 1, y + h - 1),
            color,
            1
        )

    save_path = save_dir / mask_path.name
    cv2.imwrite(str(save_path), visualization)

    print(f"{mask_path.name}: {count - 1}개 영역 시각화 완료")

print