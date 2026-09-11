import os
import torch
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.exposure import match_histograms, equalize_adapthist
from scipy.ndimage import gaussian_filter
import pandas as pd 

# === 설정 ===
model_path = "runs/detect/yolo12x-singleclass-no-gaussian2/weights/best.pt"
inference_root = Path("data/axSpA_classification_full_slice")
save_dir = Path("bbox_detection/result/20250523")
ref_img_path = "data/reference/reference_stir.dcm"
img_size = 512
save_dir.mkdir(parents=True, exist_ok=True)

# === 기준 이미지 불러오기 ===
ref_ds = pydicom.dcmread(ref_img_path)
ref_arr = ref_ds.pixel_array
ref_arr = ((ref_arr - np.min(ref_arr)) / (np.max(ref_arr) - np.min(ref_arr)) * 255).astype(np.uint8)
ref_img = np.array(Image.fromarray(ref_arr).resize((img_size, img_size)))

# === 기준 이미지 불러오기 ===
ref_ds = pydicom.dcmread(ref_img_path)
ref_arr = ref_ds.pixel_array
ref_arr = ((ref_arr - np.min(ref_arr)) / (np.max(ref_arr) - np.min(ref_arr)) * 255).astype(np.uint8)
ref_img = np.array(Image.fromarray(ref_arr).resize((img_size, img_size)))

# === 전처리 함수 ===
def preprocess_dicom(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    img_resized = np.array(Image.fromarray(img).resize((img_size, img_size)))
    matched = match_histograms(img_resized, ref_img)
    normalized = (matched - np.min(matched)) / (np.max(matched) - np.min(matched))
    clahe = equalize_adapthist(normalized.astype(np.float32), clip_limit=0.01, kernel_size=64)
    # blurred = gaussian_filter(clahe, sigma=1)
    return (clahe * 255).astype(np.uint8), ds

# === 시각화 함수 ===
def draw_prediction(img, left_box, right_box, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray')

    for label, box in [("left", left_box), ("right", right_box)]:
        if box is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        tag = f"{label} {conf:.2f}"
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, tag, color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# === 모델 로드 ===
model = YOLO(model_path)
results_list = []

# === 추론 ===
print(inference_root)
for patient_folder in sorted(inference_root.iterdir()):
    t2_folder = patient_folder / "T1"
    if not t2_folder.exists():
        continue
    print(t2_folder)
    for dcm_file in sorted(t2_folder.glob("*.dcm")):
        try:
            processed, ds = preprocess_dicom(dcm_file)
            img = Image.fromarray(processed).convert("RGB")
            results = model(img)[0]

            left_box, right_box = None, None
            for box in results.boxes:
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                if x_center < img_size / 2:
                    if left_box is None or box.conf[0] > left_box.conf[0]:
                        left_box = box
                else:
                    if right_box is None or box.conf[0] > right_box.conf[0]:
                        right_box = box

            out_name = f"{patient_folder.name}_{str(ds.InstanceNumber)}_pred.png"
            out_path = save_dir / out_name
            draw_prediction(processed, left_box, right_box, out_path)
            
            # === CSV용 데이터 정리 ===
            patient_id = patient_folder.name
            instance_number = str(ds.InstanceNumber)  # DICOM 파일 이름 (보통 Instance Number 역할)

            left_box_coords = ""
            if left_box is not None:
                x1, y1, x2, y2 = left_box.xyxy[0].tolist()
                left_box_coords = f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}"

            right_box_coords = ""
            if right_box is not None:
                x1, y1, x2, y2 = right_box.xyxy[0].tolist()
                right_box_coords = f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}"

            results_list.append({
                "patient_id": patient_id,
                "instance_number": instance_number,
                "left_box": left_box_coords,
                "right_box": right_box_coords
            })

        except Exception as e:
            print(f"❌ 오류 - {dcm_file}: {e}")

print("✅ 외부 데이터 추론 및 결과 저장 완료!")
# === CSV로 저장 ===
csv_save_path = "20250512_bbox_coordinates.csv"
df = pd.DataFrame(results_list)
df.to_csv(csv_save_path, index=False)

print("✅ 외부 데이터 추론 및 결과 저장 완료!")
print(f"✅ 바운딩 박스 CSV 저장 완료: {csv_save_path}")