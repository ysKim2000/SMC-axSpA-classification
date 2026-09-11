import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import ast
from tqdm import tqdm
import pydicom
import cv2
from PIL import Image, ImageOps
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns

from original_train import BinaryClassificationModel, MedicalImageDataset


class CLAHE(object):
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')
        return ImageOps.equalize(image)


class HistogramMatching(object):
    def __init__(self, target_image):
        if len(target_image.shape) == 3 and target_image.shape[-1] == 3:
            target_image = target_image[..., 0]
        self.target_image = target_image

    def __call__(self, image):
        image = np.array(image)  # 강제 변환
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image[..., 0]
        return exposure.match_histograms(image, self.target_image)



def compute_metrics(preds, labels):
    from sklearn.metrics import roc_auc_score, confusion_matrix

    tn, fp, fn, tp = confusion_matrix(labels, preds > 0.5).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sens = tp / (tp + fn) if tp + fn > 0 else 0
    spec = tn / (tn + fp) if tn + fp > 0 else 0
    auroc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0
    return acc, sens, spec, auroc


def compute_confusion_matrix(preds, labels, threshold=0.5, save_path='bme_classification/result/cm/confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix

    preds_bin = (preds > threshold).astype(int)
    cm = confusion_matrix(labels, preds_bin)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return cm

def generate_campp_overlay(model, image_tensor, target_layer, device, bbox, full_image_color):
    image_tensor = image_tensor.to(device).requires_grad_()

    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fw = target_layer.register_forward_hook(forward_hook)
    bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output[0]
    model.zero_grad()
    score.backward(retain_graph=True)

    fw.remove()
    bw.remove()

    # --- Grad-CAM++ weights ---
    grad = gradients[0].detach()          # (1, C, H, W)
    act  = activations[0].detach()        # (1, C, H, W)
    grad2, grad3 = grad**2, grad**3
    eps = 1e-8
    weights = grad2 / (2 * grad2 + act * grad3.sum(dim=(2, 3), keepdim=True) + eps)
    weights = weights * torch.relu(grad)

    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / (cam.max() + eps)

    # ----- ROI / 크기·채널·dtype 정렬 -----
    x1, y1, x2, y2 = bbox
    roi = full_image_color[y1:y2, x1:x2]                 # (h, w, 3), uint8
    h, w = roi.shape[:2]

    cam_np = np.power(cam.detach().cpu().numpy(), 0.3)
    cam_resized = cv2.resize(cam_np, (w, h))             # (w, h) 순서

    threshold = 0.3
    cam_mask = (cam_resized >= threshold).astype(np.uint8) * 255   # (h, w) 0/255

    cam_uint8 = np.uint8(255 * cam_resized)              # (h, w) uint8
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)       # (h, w, 3) uint8
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=cam_mask)  # (h, w, 3)

    # 혹시라도 모양/채널이 어긋나면 강제 보정
    if heatmap_masked.ndim == 2:
        heatmap_masked = cv2.cvtColor(heatmap_masked, cv2.COLOR_GRAY2BGR)
    if heatmap_masked.shape[:2] != (h, w):
        heatmap_masked = cv2.resize(heatmap_masked, (w, h))
    if heatmap_masked.dtype != roi.dtype:
        heatmap_masked = heatmap_masked.astype(roi.dtype)

    # ===== 버전 A: 비마스크는 원본 그대로 (마스크 내부만 히트맵 블렌딩) =====
    blended = cv2.addWeighted(roi, 0.6, heatmap_masked, 0.4, 0)    # 동일 크기/채널
    mask2 = (cam_mask > 0)                                         # (h, w) bool
    overlay_masked = roi.copy()
    overlay_masked[mask2] = blended[mask2]                         # 마스크 내부만 교체

    img_masked = full_image_color.copy()
    img_masked[y1:y2, x1:x2] = overlay_masked

    # ===== 버전 B: ROI 파란 패널(비마스크에만) + 히트맵 =====
    # 블루 틴트는 비활성 영역에만 적용해서 weight를 절대 덮지 않음
    blue_roi = np.zeros_like(roi); blue_roi[:] = (255, 0, 0)
    blue_all = cv2.addWeighted(roi, 0.7, blue_roi, 0.3, 0)       # 전체 틴트 결과
    mask3c = np.dstack([mask2]*3)                                  # (h, w, 3) bool
    roi_blue_selective = np.where(mask3c, roi, blue_all)           # 활성부위=원본, 비활성=블루

    # 그 위에 히트맵을 얹음 → weight가 항상 최상단
    overlay_blue = cv2.addWeighted(roi_blue_selective, 0.6, heatmap_masked, 0.4, 0)

    img_blue = full_image_color.copy()
    img_blue[y1:y2, x1:x2] = overlay_blue

    return img_masked, img_blue




def generate_gradcam_overlay(model, image_tensor, target_layer, device, bbox, full_image_color):
    image_tensor = image_tensor.to(device).requires_grad_()
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fw = target_layer.register_forward_hook(forward_hook)
    bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output[0]
    model.zero_grad()
    score.backward(retain_graph=True)

    fw.remove(); bw.remove()

    # ---------- Grad-CAM weights ----------
    grad = gradients[0].detach()      # (1, C, H, W)
    act  = activations[0].detach()    # (1, C, H, W)
    # spatial mean
    if grad.ndim == 4:
        weights = grad.mean(dim=(2, 3), keepdim=True)     # (1,C,1,1)
    elif grad.ndim == 3:
        weights = grad.mean(dim=2, keepdim=True).unsqueeze(-1)
    elif grad.ndim == 2:
        weights = grad.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported grad shape: {grad.shape}")

    cam = (weights * act).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    eps = 1e-8
    cam = cam / (cam.max() + eps)

    # ----- ROI / 크기·채널·dtype 정렬 (Grad-CAM++와 동일) -----
    x1, y1, x2, y2 = bbox
    roi = full_image_color[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    # 동일한 감마/threshold 사용
    cam_np = np.power(cam.detach().cpu().numpy(), 0.3)
    cam_resized = cv2.resize(cam_np, (w, h))

    threshold = 0.3
    cam_mask = (cam_resized >= threshold).astype(np.uint8) * 255

    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=cam_mask)

    if heatmap_masked.ndim == 2:
        heatmap_masked = cv2.cvtColor(heatmap_masked, cv2.COLOR_GRAY2BGR)
    if heatmap_masked.shape[:2] != (h, w):
        heatmap_masked = cv2.resize(heatmap_masked, (w, h))
    if heatmap_masked.dtype != roi.dtype:
        heatmap_masked = heatmap_masked.astype(roi.dtype)

    # ===== 버전 A: 비마스크는 원본 그대로 (마스크 내부만 블렌딩) =====
    blended = cv2.addWeighted(roi, 0.6, heatmap_masked, 0.4, 0)
    mask2 = (cam_mask > 0)
    overlay_masked = roi.copy()
    overlay_masked[mask2] = blended[mask2]
    img_masked = full_image_color.copy()
    img_masked[y1:y2, x1:x2] = overlay_masked

    # ===== 버전 B: ROI 파란 패널(비마스크만) + 히트맵 =====
    blue_roi = np.zeros_like(roi); blue_roi[:] = (255, 0, 0)  # BGR Blue
    # 왼쪽처럼 선명도 높게: 배경 약간 어둡게 + 블루 진하게
    roi_dim  = (roi.astype(np.float32) * 0.85).astype(np.uint8)
    blue_all = cv2.addWeighted(roi_dim, 0.7, blue_roi, 0.3, 0)

    mask3c = np.dstack([mask2]*3)
    roi_blue_selective = np.where(mask3c, roi, blue_all)  # 활성=원본 유지, 비활성=블루
    overlay_blue = cv2.addWeighted(roi_blue_selective, 0.6, heatmap_masked, 0.4, 0)

    img_blue = full_image_color.copy()
    img_blue[y1:y2, x1:x2] = overlay_blue

    return img_masked, img_blue




if __name__ == '__main__':
    # Paths
    bbox_csv = 'data/new_csv/bbox_label.csv'
    bme_csv = 'data/bme_label.csv'
    test_dir = 'bme_classification/dataset/test'
    ckpt = 'bme_classification/checkpoints/20250509_convnext_large_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt'
    save_root = 'bme_classification/result/gradcam/'

    # Preprocess
    clahe = CLAHE()
    target_dcm = pydicom.dcmread('data/reference/reference_stir.dcm')
    hist_matcher = HistogramMatching(((target_dcm.pixel_array - target_dcm.pixel_array.min()) /
                                     (target_dcm.pixel_array.max() - target_dcm.pixel_array.min()) * 255).astype(np.uint8))

    bbox_df = pd.read_csv(bbox_csv)
    bme_df = pd.read_csv(bme_csv)

    model = BinaryClassificationModel('convnext_large').get_model()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    gradcam_model = model.module
    # target_layer = list(model.module.children())[-1]
    target_layer = model.module.features[-1]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    for pid in os.listdir(test_dir):
        dicom_dir = os.path.join(test_dir, pid, 'T2')
        for fname in sorted(os.listdir(dicom_dir)):
            if not fname.endswith('.dcm'):
                continue

            dcm_path = os.path.join(dicom_dir, fname)
            dcm = pydicom.dcmread(dcm_path)
            slice_number = int(dcm.InstanceNumber)
            full_img = dcm.pixel_array
            full_img = ((full_img - full_img.min()) / (full_img.max() - full_img.min()) * 255).astype(np.uint8)
            full_img = clahe(full_img)
            full_img = hist_matcher(full_img)
            full_img = full_img.astype(np.uint8)
            full_img_color = cv2.cvtColor(full_img, cv2.COLOR_GRAY2BGR)

            row = bbox_df[(bbox_df.patient_id == int(pid)) & (bbox_df.slice_number == slice_number)]
            if row.empty:
                continue

            matched_rows = bme_df[bme_df.patient_id == int(pid)]
            if matched_rows.empty:
                continue  # 🔒 라벨 없는 경우 건너뜀

            bboxes, labels, preds, patch_tensors = {}, {}, {}, {}

            for side in ['left', 'right']:
                try:
                    box_str = row.iloc[0][f'{side}_box']
                    bbox = list(map(int, ast.literal_eval(box_str)))
                    x1, y1, x2, y2 = bbox
                    patch = full_img[y1:y2, x1:x2].astype(np.uint8)
                    if patch.shape[0] == 0 or patch.shape[1] == 0:
                        continue
                    patch_tensor = transform(patch).unsqueeze(0)
                    label_dict = ast.literal_eval(matched_rows.iloc[0][f'{side}_bme'])
                    label = int(label_dict.get(str(slice_number).zfill(4), 0))
                    with torch.no_grad():
                        out = model(patch_tensor.to(device))
                        prob = torch.sigmoid(out)[0].item()
                        pred = int(prob > 0.5)
                    bboxes[side] = bbox
                    labels[side] = label
                    preds[side] = pred
                    patch_tensors[side] = patch_tensor
                except Exception as e:
                    print(f"{pid}-{slice_number}-{side} Error: {e}")

            # 저장 조건: 예측 == 라벨인 경우만
            # True Positive 조건만 시각화 + 저장
            # True Positive 조건만 시각화 + 저장
            # True Positive 조건만 시각화 + 저장
            if ((preds.get('left') == 1 and labels.get('left') == 1) or
                (preds.get('right') == 1 and labels.get('right') == 1)):

                save_dir = os.path.join(save_root, pid)
                os.makedirs(save_dir, exist_ok=True)

                preprocessed_img = full_img_color.copy()  # 💡 CAM 적용 전 저장용 이미지

                # --- Grad-CAM / Grad-CAM++ 각 2종( masked / roiBlue ) 누적 캔버스 준비 ---
                gc_masked   = preprocessed_img.copy()
                gc_blue     = preprocessed_img.copy()
                gcpp_masked = preprocessed_img.copy()
                gcpp_blue   = preprocessed_img.copy()

                for side in ['left', 'right']:
                    if side not in bboxes:
                        continue
                    if preds.get(side) == 1 and labels.get(side) == 1:
                        # Grad-CAM
                        gc_masked, _ = generate_gradcam_overlay(
                            gradcam_model, patch_tensors[side], target_layer, device,
                            bboxes[side], gc_masked
                        )
                        _, gc_blue   = generate_gradcam_overlay(
                            gradcam_model, patch_tensors[side], target_layer, device,
                            bboxes[side], gc_blue
                        )
                        # Grad-CAM++
                        gcpp_masked, _ = generate_campp_overlay(
                            gradcam_model, patch_tensors[side], target_layer, device,
                            bboxes[side], gcpp_masked
                        )
                        _, gcpp_blue   = generate_campp_overlay(
                            gradcam_model, patch_tensors[side], target_layer, device,
                            bboxes[side], gcpp_blue
                        )

                # original 저장
                cv2.imwrite(os.path.join(save_dir, f'{pid}_{slice_number}_original.png'), preprocessed_img)

                # 결과 저장 (파일명 구분: GC / GCPP)
                to_save = [
                    (gc_masked,   "GC_overlay_masked"),
                    (gc_blue,     "GC_overlay_roiBlue"),
                    (gcpp_masked, "GCPP_overlay_masked"),
                    (gcpp_blue,   "GCPP_overlay_roiBlue"),
                ]
                for img, tag in to_save:
                    cv2.imwrite(os.path.join(save_dir, f'{pid}_{slice_number}_{tag}.png'), img)

                # ROI crop (TP만)
                for side in ['left', 'right']:
                    if side in bboxes and preds.get(side) == 1 and labels.get(side) == 1:
                        patch_img = patch_tensors[side].squeeze().cpu().numpy() * 255
                        patch_img = patch_img.astype(np.uint8).transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(save_dir, f'{pid}_{slice_number}_roi_{side}.png'), patch_img)