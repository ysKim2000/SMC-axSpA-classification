import os
import ast
import random
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.exposure import match_histograms, equalize_adapthist
from scipy.ndimage import gaussian_filter

# === 기본 설정 ===
csv_path = "data/bbox_label.csv"
base_dir = "data/axSpA_bbox_dataset"
source_dcm_path = "data/reference/reference_stir.dcm"
output_dir = "bbox_detection/dataset_oneclass_no_gaussian_20250701"
img_size = 512
val_ratio = 0.2
random.seed(42)

# === 기준 이미지 준비 ===
ref_ds = pydicom.dcmread(source_dcm_path)
ref_arr = ref_ds.pixel_array
ref_arr = ((ref_arr - np.min(ref_arr)) / (np.max(ref_arr) - np.min(ref_arr)) * 255).astype(np.uint8)
ref_img = np.array(Image.fromarray(ref_arr, mode='L').resize((img_size, img_size)))

# === 박스 파싱 및 YOLO 변환 ===
def parse_box(box_str):
    if pd.isna(box_str) or box_str in ["[]", ""]:
        return None
    return ast.literal_eval(box_str)

def convert_to_yolo(box, scale_x, scale_y):
    x1, y1, x2, y2 = [b * s for b, s in zip(box, [scale_x, scale_y, scale_x, scale_y])]
    x_center = (x1 + x2) / 2 / img_size
    y_center = (y1 + y2) / 2 / img_size
    w = abs(x2 - x1) / img_size
    h = abs(y2 - y1) / img_size
    return x_center, y_center, w, h

# === CSV 로드 및 환자 ID 정규화 ===
df = pd.read_csv(csv_path)
df['patient_id'] = df['patient_id'].astype(str).str.zfill(7)

# === DICOM 경로 + 라벨 매칭 (slice 단위) ===
image_label_list = []
for _, row in df.iterrows():
    patient_id = row['patient_id']
    slice_num = row['slice_number']
    dcm_dir = os.path.join(base_dir, patient_id, "T2")
    if not os.path.exists(dcm_dir): continue
    for fname in os.listdir(dcm_dir):
        if fname.endswith(".dcm"):
            path = os.path.join(dcm_dir, fname)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                if getattr(ds, 'InstanceNumber', -1) == slice_num:
                    image_label_list.append((patient_id, path, row))
                    break
            except:
                continue

# === 환자 기준으로 분할 ===
from collections import defaultdict

patient_to_items = defaultdict(list)
for pid, path, row in image_label_list:
    patient_to_items[pid].append((path, row))

unique_pids = list(patient_to_items.keys())
random.shuffle(unique_pids)
split_idx = int(len(unique_pids) * (1 - val_ratio))
train_pids = set(unique_pids[:split_idx])
val_pids = set(unique_pids[split_idx:])

train_list = []
val_list = []

for pid in unique_pids:
    if pid in train_pids:
        train_list.extend(patient_to_items[pid])
    else:
        val_list.extend(patient_to_items[pid])

# === 저장 함수 ===
def save_yolo_data(split_list, split_name):
    img_dir = os.path.join(output_dir, "images", split_name)
    label_dir = os.path.join(output_dir, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    count = 0
    for path, row in split_list:
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
            h, w = img.shape
            scale_x, scale_y = img_size / w, img_size / h

            # 전처리
            img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
            img_resized = np.array(Image.fromarray(img, mode='L').resize((img_size, img_size)))
            img_matched = match_histograms(img_resized, ref_img)
            img_norm = (img_matched - np.min(img_matched)) / (np.max(img_matched) - np.min(img_matched))
            img_clahe = equalize_adapthist(img_norm.astype(np.float32), clip_limit=0.01, kernel_size=64)
            img_clahe = (img_clahe * 255).astype(np.uint8)

            # 이미지 저장
            image_id = f"{row['patient_id']}_{row['slice_number']:02d}"
            img_path = os.path.join(img_dir, f"{image_id}.jpg")
            Image.fromarray(img_clahe).convert("L").save(img_path)

            # 단일 클래스 라벨 저장
            label_path = os.path.join(label_dir, f"{image_id}.txt")
            with open(label_path, 'w') as f:
                for box_col in ['left_box', 'right_box']:
                    box = parse_box(row[box_col])
                    if box and len(box) == 4:
                        yolo_box = convert_to_yolo(box, scale_x, scale_y)
                        f.write(f"0 {' '.join(f'{v:.6f}' for v in yolo_box)}\n")

            count += 1
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")
    print(f"✅ {split_name} set: {count} slices saved.")

# === 저장 실행 ===
save_yolo_data(train_list, "train")
save_yolo_data(val_list, "val")

# === data.yaml 생성 ===
with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    f.write(f"""\

path: {output_dir}
train: images/train
val: images/val
nc: 1
names: ['SIJ']
""")

# === 최종 통계 출력 ===
print("📊 Summary")
print(f"Train patients: {len(train_pids)} | slices: {len(train_list)}")
print(f"Val patients  : {len(val_pids)} | slices: {len(val_list)}")