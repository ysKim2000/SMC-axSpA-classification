import os
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import pydicom
import matplotlib.pyplot as plt
import cv2

from train_fullslice_v2 import (
    PatientSliceDataset as BaseDataset,
    MultiModalModel,
    create_transforms,
)


def save_overlay(cam, raw, path, pred_label, true_label, alpha=0.4, thr=0):
    """
    cam          : torch.Tensor  [1,H,W] or [H,W]      (0~1, NaN 가능)
    raw          : np.ndarray    [H,W]   uint8 or float
    pred_label   : str ('normal' or 'axSpA')
    true_label   : str ('normal' or 'axSpA')
    """
    # ── CAM 준비 ─────────────────────────────────────────
    cam = cam.squeeze().detach().cpu()
    if torch.isnan(cam).any() or cam.max() < thr:
        cam_arr = None
    else:
        cam_arr = cam.numpy()  # (H,W)

    # ── RAW 준비 : dtype/range 정규화 → 0-1
    if raw.dtype == np.uint8:
        raw_disp = raw.astype(np.float32) / 255.0
    else:
        rmin, rmax = raw.min(), raw.max()
        raw_disp = (raw - rmin) / (rmax - rmin + 1e-8)

    # ── 512×512 로 리사이즈 ───────────────────────────────
    raw_disp = cv2.resize(raw_disp, (256, 256), interpolation=cv2.INTER_LINEAR)

    # ── 그림 그리기 ─────────────────────────────────────
    plt.figure(figsize=(5, 5))
    plt.imshow(raw_disp, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    if cam_arr is not None:
        plt.imshow(cam_arr, cmap='jet', alpha=alpha, vmin=0, vmax=1, interpolation='bilinear')

    # ── 오른쪽 하단에 텍스트 추가 ────────────────────────
    ax = plt.gca()
    text = f"Pred: {pred_label}\nLabel: {true_label}"
    ax.text(
        0.95, 0.05, text,
        transform=ax.transAxes,
        fontsize=10,
        color='white',
        ha='right', va='bottom',
        bbox=dict(facecolor='black', alpha=0.6, pad=2)
    )

    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300)
    plt.close()


def make_cam(act: torch.Tensor, grad: torch.Tensor, out_size=(256, 256)):
    # grad를 act와 같은 디바이스로 옮깁니다
    grad = grad.to(act.device, non_blocking=True)

    # 이제 act.device == grad.device
    w   = grad.mean((2, 3), keepdim=True)
    cam = (w * act).sum(1, keepdim=True).relu_()
    cam = F.interpolate(cam, out_size, mode='bilinear', align_corners=False)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam.detach()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations, self.gradients = None, None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, _, __, output):
        self.activations = output

    def save_gradient(self, _, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x1, x2, bme, slice_h, slice_w):
        self.model.zero_grad()
        out = self.model(x1, x2, bme)
        pred = out[0] if isinstance(out, tuple) else out  # 유연하게 처리

        pred.sum().backward()

        w = self.gradients.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, (slice_h, slice_w), mode='bilinear', align_corners=False)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max + 1e-8)
        return cam



class GradCamDataset(BaseDataset):
    def __getitem__(self, idx):
        t1, t2, label_fda, bme, pid = super().__getitem__(idx)
        t1s_list, t2s_list, _, _ = self.data[idx]

        raw_t1 = np.stack([arr for arr, _ in t1s_list], axis=0)
        raw_t2 = np.stack([arr for arr, _ in t2s_list], axis=0)

        return t1, t2, label_fda, bme, pid, raw_t1, raw_t2


def collate_gradcam(batch):
    t1s, t2s, labs, bmes, pids, raws1, raws2 = zip(*batch)
    return (
      default_collate(t1s),
      default_collate(t2s),
      default_collate(labs),
      default_collate(bmes),
      list(pids),
      list(raws1),
      list(raws2),
    )


def get_last_conv(backbone):
    for m in reversed(list(backbone.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("Conv2d layer not found")


def evaluate(model, loader, device, out_dir, alpha=0.4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # 마지막 Conv 레이어 추출
    backbone1 = model.module.t1_feature_extractor[0] if isinstance(model, torch.nn.DataParallel) else model.t1_feature_extractor[0]
    backbone2 = model.module.t2_feature_extractor[0] if isinstance(model, torch.nn.DataParallel) else model.t2_feature_extractor[0]

    last_t1 = get_last_conv(backbone1)
    last_t2 = get_last_conv(backbone2)
    cam_t1 = GradCAM(model, last_t1)
    cam_t2 = GradCAM(model, last_t2)

    for p in model.parameters():
        p.requires_grad_(False)
    last_t1.requires_grad_(True)
    last_t2.requires_grad_(True)

    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
    y_true, y_pred, y_score = [], [], []

    for x1, x2, lab, bme, pids, raws1, raws2 in tqdm(loader):
        B, S = x1.size(0), x1.size(1)  # B = batch_size(=1), S = num_slices
        x1, x2, bme = x1.to(device), x2.to(device), bme.to(device)

        # 환자-level 예측용 전체 slice forward pass만 수행 (no_grad로 메모리 절약)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                out_all, _ = model(x1, x2, bme)
                prob_all = torch.sigmoid(out_all)

        for bi in range(B):
            pid = pids[bi]
            batch_lab = lab[bi].item()
            prob_slices = prob_all[bi]
            patient_prob = prob_slices.mean().item()
            pred_str = "axSpA" if patient_prob > 0.5 else "normal"
            label_str = "axSpA" if batch_lab == 1 else "normal"

            # metric 저장용
            y_true.append(batch_lab)
            y_pred.append(1 if patient_prob > 0.5 else 0)
            y_score.append(patient_prob)

            # slice-level로 CAM 생성
            for si in range(S):
                x1_slice = x1[bi, si].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                x2_slice = x2[bi, si].unsqueeze(0).unsqueeze(0)
                bme_slice = bme[bi, si].view(1, 1)

                # Grad-CAM 계산
                cam_t1_map = cam_t1(x1_slice, x2_slice, bme_slice, 256, 256).squeeze(0)
                cam_t2_map = cam_t2(x1_slice, x2_slice, bme_slice, 256, 256).squeeze(0)

                # CAM 오버레이 저장
                save_overlay(
                    cam_t1_map,
                    raws1[bi][si],
                    os.path.join(out_dir, f'{pid}_slice{si:02d}_T1.png'),
                    pred_str,
                    label_str,
                    alpha
                )
                save_overlay(
                    cam_t2_map,
                    raws2[bi][si],
                    os.path.join(out_dir, f'{pid}_slice{si:02d}_T2.png'),
                    pred_str,
                    label_str,
                    alpha
                )

                # 메모리 정리
                del x1_slice, x2_slice, bme_slice, cam_t1_map, cam_t2_map
                torch.cuda.empty_cache()

        # 배치 후처리
        del x1, x2, bme, prob_all, out_all
        torch.cuda.empty_cache()

    # metrics 출력
    print(f"► ACC:{accuracy_score(y_true, y_pred):.4f}  Sens:{recall_score(y_true, y_pred):.4f}  Spec:{recall_score(y_true, y_pred, pos_label=0):.4f}  AUC:{roc_auc_score(y_true, y_score):.4f}")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dicom_dir = 'axSpA_classification/dataset/test_fullslice'
    label_csv = 'data/AxSpA_label.csv'
    bme_csv = 'data/Combined_BME_Labels.csv'
    ckpt_path = 'axSpA_classification/check_points/20250523/convnext_large_v2_new_version_BCE_fullslices_new_classifier_v4.pt'
    out_dir = 'axSpA_classification/result/cams_overlay_5'

    tf_t1, tf_t2 = create_transforms(apply_augmentations=False)

    ds = GradCamDataset(dicom_dir, label_csv, bme_csv, tf_t1, tf_t2)
    print(f"Test patients: {len(ds)} | class dist: {Counter([int(l) for _,_,l,_,_,_,_ in ds])}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=8,
        collate_fn=collate_gradcam, pin_memory=True
    )

    model = MultiModalModel('convnext_large', hidden_dim=256).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    evaluate(model, loader, device, out_dir, alpha=0.4)