import cv2
import numpy as np
import os

# 경로 설정
original_dir = "image"
result_dir = "output/final_inpaint"
save_dir = "output/difference"

os.makedirs(save_dir, exist_ok=True)

# 차이로 판단할 기준값
threshold = 30

for filename in os.listdir(original_dir):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    original_path = os.path.join(original_dir, filename)
    result_path = os.path.join(result_dir, filename)

    if not os.path.exists(result_path):
        continue

    # 원본 / EndoSRR 결과 불러오기
    original = cv2.imread(original_path)
    result = cv2.imread(result_path)

    # 혹시 이미지 크기가 다르면 맞춰주기
    if original.shape != result.shape:
        result = cv2.resize(
            result,
            (original.shape[1], original.shape[0])
        )

    # 픽셀별 RGB 차이 계산
    diff = cv2.absdiff(original, result)

    # RGB 중 하나라도 30 이상 달라졌으면 변경 픽셀로 판단
    changed = np.max(diff, axis=2) > threshold

    # 원본을 복사해서 시각화 이미지 생성
    visualization = original.copy()

    # 변경된 픽셀을 빨간색으로 표시
    visualization[changed] = [0, 0, 255]

    # 저장
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, visualization)

    # 변경된 픽셀 비율도 출력
    changed_ratio = np.mean(changed) * 100

    print(filename)
    print(f"Changed pixels: {changed_ratio:.2f}%")
    print("--------------------")

print("완료!")