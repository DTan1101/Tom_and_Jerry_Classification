from typing import Tuple

import numpy as np
from PIL import Image
from skimage.feature import hog


def _load_image(path: str, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size))


def extract_raw(path: str, image_size: int) -> np.ndarray:
    image = np.asarray(_load_image(path, image_size), dtype=np.float32) / 255.0
    return image.reshape(-1)


def extract_hog(path: str, image_size: int) -> np.ndarray:
    image = np.asarray(_load_image(path, image_size).convert("L"), dtype=np.float32) / 255.0
    feat = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_sift(path: str, image_size: int, n_keypoints: int = 64) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("SIFT requires opencv-python to be installed.") from exc

    image = np.asarray(_load_image(path, image_size).convert("L"), dtype=np.uint8)
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    _kps, desc = sift.detectAndCompute(image, None)

    if desc is None or len(desc) == 0:
        return np.zeros((n_keypoints * 128,), dtype=np.float32)

    if desc.shape[0] < n_keypoints:
        pad = np.zeros((n_keypoints - desc.shape[0], 128), dtype=np.float32)
        desc = np.vstack([desc.astype(np.float32), pad])
    else:
        desc = desc[:n_keypoints].astype(np.float32)

    return desc.reshape(-1)


def extract_feature(path: str, method: str, image_size: int) -> np.ndarray:
    method = method.lower()
    if method == "raw":
        return extract_raw(path, image_size)
    if method == "hog":
        return extract_hog(path, image_size)
    if method == "sift":
        return extract_sift(path, image_size)
    raise ValueError(f"Unsupported feature method: {method}")


def batch_extract(paths, method: str, image_size: int) -> np.ndarray:
    feats = [extract_feature(path, method=method, image_size=image_size) for path in paths]
    return np.vstack(feats)
