import os
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.ops import box_iou
import numpy as np
from tqdm import tqdm # 진행률 표시를 위해 추가

# === 설정 ===
model_path = "runs/detect/yolo12x-singleclass-no-gaussian2/weights/best.pt"
data_yaml = "bbox_detection/dataset_oneclass_no_gaussian/data.yaml"
val_img_dir = Path("bbox_detection/dataset_oneclass_no_gaussian/images/val")
val_lbl_dir = Path("bbox_detection/dataset_oneclass_no_gaussian/labels/val")
save_dir = Path("yolo12x_singleclass_val_results")
save_dir.mkdir(parents=True, exist_ok=True)

# === 모델 로드 및 검증 ===
model = YOLO(model_path)
metrics = model.val(data=data_yaml, save=True, save_json=True, plots=True, project="yolo_validation_results", name="my_test_run")

# === 평가 함수 (IoU & AP) ===
def calculate_iou(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    iou_matrix = box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    return iou_matrix.max(dim=1)[0].mean().item()

def compute_ap(preds, gts, iou_thresh=0.5):
    """
    단일 클래스에 대한 Average Precision (AP) 계산 - COCO 101-point interpolation 적용
    """
    if not preds:
        return 0.0

    preds = sorted(preds, key=lambda x: x['conf'], reverse=True)
    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))

    gt_matched = {img_idx: np.zeros(len(boxes), dtype=bool) for img_idx, boxes in gts.items()}
    num_total_gts = sum(len(boxes) for boxes in gts.values())

    if num_total_gts == 0:
        return 0.0

    for i, pred in enumerate(preds):
        img_idx = pred['img_idx']
        pred_box = torch.tensor([pred['box']])
        img_gts = gts.get(img_idx, [])

        if len(img_gts) == 0:
            FP[i] = 1
            continue

        gt_boxes = torch.tensor(img_gts)
        ious = box_iou(pred_box, gt_boxes)[0]

        # === 🌟 핵심 수정 포인트: 이미 매칭된 GT는 후보에서 제외 ===
        # 이미 짝이 지어진 GT의 IoU를 0으로 만들어, 다음 순위의 GT와 매칭될 수 있도록 함
        for gt_idx, is_matched in enumerate(gt_matched[img_idx]):
            if is_matched:
                ious[gt_idx] = 0.0
        # ==========================================================

        max_iou, max_idx = torch.max(ious, dim=0)

        # 이제 남아있는 GT들 중 가장 IoU가 높은 것과 비교
        if max_iou >= iou_thresh:
            TP[i] = 1
            gt_matched[img_idx][max_idx.item()] = True
        else:
            FP[i] = 1

    cum_tp = np.cumsum(TP)
    cum_fp = np.cumsum(FP)
    recalls = cum_tp / num_total_gts
    precisions = cum_tp / (cum_tp + cum_fp)

    # === COCO 스타일 101-point interpolation으로 변경 ===
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        # 현재 Recall(t) 이상인 구간에서 가장 높은 Precision 탐색
        valid_precisions = precisions[recalls >= t]
        if len(valid_precisions) == 0:
            p = 0.0
        else:
            p = np.max(valid_precisions)
        ap += p / 101.0
        
    return ap

# === 시각화 함수 ===
# (기존 작성하신 draw_boxes_with_gt 함수 동일하므로 생략 없이 사용하시면 됩니다)
def draw_boxes_with_gt(img_path, selected_boxes, gt_boxes, save_path):
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)

    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, "GT", color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    if selected_boxes:
        for label, box in selected_boxes.items():
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            label_str = f"Pred {conf:.2f}"
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label_str, color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')

# === 검증 데이터 수집 ===
ious = []
preds_by_img = {}  # mAP 계산용 (전체 모델 예측)
gts_by_img = {}    # mAP 계산용 (전체 GT)

image_list = sorted(val_img_dir.glob("*.jpg"))

print("\n이미지별 예측 결과 수집 중...")
for img_idx, img_path in enumerate(tqdm(image_list)):
    results = model(str(img_path), conf=0.001, iou=0.6, imgsz=640, rect=True, verbose=False)[0]
    img = Image.open(img_path)
    iw, ih = img.size

    # 1. mAP 계산을 위한 모델 전체 예측 결과 수집
    preds_by_img[img_idx] = []
    for box in results.boxes:
        preds_by_img[img_idx].append({
            'box': box.xyxy[0].tolist(),
            'conf': float(box.conf[0])
        })

    # 2. GT 박스 로드
    label_path = val_lbl_dir / f"{img_path.stem}.txt"
    gt_boxes = []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())
                x1 = (cx - w / 2) * iw
                y1 = (cy - h / 2) * ih
                x2 = (cx + w / 2) * iw
                y2 = (cy + h / 2) * ih
                gt_boxes.append([x1, y1, x2, y2])
    
    gts_by_img[img_idx] = gt_boxes

    # 3. 작성하신 사용자 정의(좌/우 1개씩) 박스 선택 및 IoU 계산
    center_x = iw / 2
    left_max_conf = right_max_conf = -1
    left_best_box = right_best_box = None
    selected_boxes = {}

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        conf = float(box.conf[0])

        if x_center < center_x and conf > left_max_conf:
            left_max_conf = conf; left_best_box = box
        elif x_center >= center_x and conf > right_max_conf:
            right_max_conf = conf; right_best_box = box

    if left_best_box: selected_boxes['left'] = left_best_box
    if right_best_box: selected_boxes['right'] = right_best_box

    if gt_boxes and selected_boxes:
        gt_tensor = torch.tensor(gt_boxes)
        pred_tensor = torch.stack([box.xyxy[0] for box in selected_boxes.values()])
        ious.append(calculate_iou(pred_tensor, gt_tensor))

    # 시각화 저장
    out_path = save_dir / f"{img_path.stem}_gt_pred.png"
    draw_boxes_with_gt(img_path, selected_boxes, gt_boxes, out_path)

# === 부트스트래핑 (IoU 및 mAP) ===
np.random.seed(42)
n_bootstraps = 1000

# 1. 평균 IoU 부트스트래핑
iou_array = np.array(ious)
if len(iou_array) > 0:
    bootstrapped_ious = [np.mean(np.random.choice(iou_array, len(iou_array), replace=True)) for _ in range(n_bootstraps)]
    iou_ci_lower, iou_ci_upper = np.percentile(bootstrapped_ious, [2.5, 97.5])
else:
    iou_ci_lower = iou_ci_upper = 0.0

# 2. mAP@0.5 및 mAP@0.75 부트스트래핑
print("\nmAP 95% CI 부트스트래핑 계산 중 (1000회)...")
bootstrapped_map50 = []
bootstrapped_map75 = []
img_indices = list(range(len(image_list)))

for _ in tqdm(range(n_bootstraps)):
    # 이미지 인덱스를 복원 추출
    sampled_indices = np.random.choice(img_indices, size=len(img_indices), replace=True)
    
    sampled_preds = []
    sampled_gts = {}
    
    # 추출된 이미지들을 바탕으로 가상의 데이터셋(preds, gts) 구성
    for new_idx, old_idx in enumerate(sampled_indices):
        sampled_gts[new_idx] = gts_by_img[old_idx]
        for p in preds_by_img[old_idx]:
            sampled_preds.append({'img_idx': new_idx, 'box': p['box'], 'conf': p['conf']})
            
    # 해당 샘플 집단에서 mAP@0.5, mAP@0.75 계산
    bootstrapped_map50.append(compute_ap(sampled_preds, sampled_gts, iou_thresh=0.5))
    bootstrapped_map75.append(compute_ap(sampled_preds, sampled_gts, iou_thresh=0.75))

map50_ci_lower, map50_ci_upper = np.percentile(bootstrapped_map50, [2.5, 97.5])
map75_ci_lower, map75_ci_upper = np.percentile(bootstrapped_map75, [2.5, 97.5])

# === 결과 출력 ===
mean_iou = sum(ious) / len(ious) if ious else 0.0

print("\n📊 검증 성능 요약:")
print(f"  Ultralytics mAP@[0.5:0.95]: {metrics.box.map:.4f}")
print(f"  Ultralytics mAP@0.5       : {metrics.box.map50:.4f}")
print(f"  Ultralytics mAP@0.75      : {metrics.box.map75:.4f}")
print("-" * 40)
print(f"  부트스트랩 mAP@0.5 95% CI : [{map50_ci_lower:.4f}, {map50_ci_upper:.4f}]")
print(f"  부트스트랩 mAP@0.75 95% CI: [{map75_ci_lower:.4f}, {map75_ci_upper:.4f}]")
print(f"  평균 IoU                  : {mean_iou:.4f}")
print(f"  평균 IoU 95% CI           : [{iou_ci_lower:.4f}, {iou_ci_upper:.4f}]")