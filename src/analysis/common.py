from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
ENDOSRR_DIR = REPO_ROOT.parent / "EndoSRR"
RESULTS_DIR = REPO_ROOT / "results"


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_binary_mask(path):
    return read_image(path, cv2.IMREAD_GRAYSCALE) > 0


def find_pngs(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in: {directory}")
    return paths
