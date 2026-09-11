import os
import cv2
import numpy as np
import pydicom
from PIL import Image
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

# CLAHE 클래스
class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)
        if img.ndim == 3:
            img = img[..., 0]
        return self.clahe.apply(img)

# Histogram Matching 클래스
class HistogramMatching:
    def __init__(self, ref_img: np.ndarray):
        # 참조 이미지는 uint8로 미리 변환
        if ref_img.dtype != np.uint8:
            ref_img = ((ref_img - ref_img.min()) / (ref_img.max() - ref_img.min() + 1e-8) * 255).astype(np.uint8)
        self.ref = ref_img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # 입력 이미지도 명시적으로 uint8로 변환
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        matched = match_histograms(img, self.ref, channel_axis=None)
        return matched.astype(np.uint8)


# DICOM 윈도우 조정
def apply_windowing(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    img = ds.pixel_array.astype(np.float32)
    slope = getattr(ds, 'RescaleSlope', 1)
    intercept = getattr(ds, 'RescaleIntercept', 0)
    img = img * slope + intercept
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if isinstance(wc, pydicom.multival.MultiValue): wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue): ww = float(ww[0])
    if wc is not None and ww is not None:
        img = np.clip(img, wc - ww/2, wc + ww/2)
    return img

# 저장 함수
def save_image(img: np.ndarray, path: str):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(path, img, cmap='gray')

# 메인 실행 함수
def process_and_save(dicom_path, ref_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    ds = pydicom.dcmread(dicom_path)
    img = apply_windowing(ds)

    # 참조 이미지 로드
    ref_ds = pydicom.dcmread(ref_path)
    ref_img = apply_windowing(ref_ds)
    ref_img_resized = np.array(Image.fromarray(ref_img).resize((512, 512), resample=Image.BILINEAR))

    clahe = CLAHE()
    hist_match = HistogramMatching(ref_img_resized)

    # CLAHE 적용
    img_clahe = clahe(img)
    save_image(img_clahe, os.path.join(save_dir, 'clahe_only.png'))

    # Histogram Matching 적용
    img_hist = hist_match(img)
    save_image(img_hist, os.path.join(save_dir, 'histogram_matching_only.png'))

    # CLAHE + Histogram Matching
    img_combined = clahe(hist_match(img))
    save_image(img_combined, os.path.join(save_dir, 'clahe_plus_histogram_matching.png'))

    # 원본 저장
    save_image(img, os.path.join(save_dir, 'original.png'))

# 예시 사용
dicom_path = "data/example/example_stir.dcm"  # 실제 DICOM 경로
ref_path = "data/reference/reference_stir.dcm"  # 참조용 DICOM
# ref_path = "data/reference/reference_fst1.dcm"  # 참조용 DICOM
        # data/reference/reference_fst1.dcm
        # data/reference/reference_stir.dcm
save_dir = "axSpA_classification/result/preprocessing_result/preprocessing_compare"

process_and_save(dicom_path, ref_path, save_dir)
