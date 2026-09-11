import os
import random
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
# import pytorch_model_summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from PIL import ImageOps, Image
from skimage.exposure import match_histograms
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts, CyclicLR,
                                      ReduceLROnPlateau)
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from torchvision.transforms import ToPILImage
from sklearn.metrics import roc_auc_score, confusion_matrix


# --- Preprocessing utilities ---
class CLAHE:
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)
        if img.ndim == 3:
            img = img[..., 0]
        return self.clahe.apply(img)


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

# --- Dataset ---
class PatientSliceDataset(Dataset):
    def __init__(self, dicom_dir, label_csv, bme_csv, transform_t1=None, transform_t2=None):
        self.dicom_dir = dicom_dir
        self.labels = pd.read_csv(label_csv)
        self.bme = pd.read_csv(bme_csv)

        reference_image_t1 = apply_windowing(pydicom.dcmread("data/reference/reference_fst1.dcm"))
        reference_image_t2 = apply_windowing(pydicom.dcmread("data/reference/reference_stir.dcm"))

        self.ref_hist_t1 = HistogramMatching(np.array(Image.fromarray(reference_image_t1).resize((256, 256), resample=Image.BILINEAR)))
        self.ref_hist_t2 = HistogramMatching(np.array(Image.fromarray(reference_image_t2).resize((256, 256), resample=Image.BILINEAR)))
        
        self.labels['patient_id'] = self.labels['patient_id'].astype(str).str.zfill(7)
        self.bme['patient_id'] = self.bme['patient_id'].astype(str).str.zfill(7)
        self.folders = sorted([os.path.join(dicom_dir,f) for f in os.listdir(dicom_dir)
                               if os.path.isdir(os.path.join(dicom_dir,f))])

        self.clahe = CLAHE()
        self.transform_t1 = transform_t1
        self.transform_t2 = transform_t2
        self.data = []
        self._prepare()

    def _prepare(self):
        for folder in self.folders:
            pid = os.path.basename(folder)
            t1f, t2f = os.path.join(folder,'T1'), os.path.join(folder,'T2')
            if not os.path.exists(t1f) or not os.path.exists(t2f): continue
            t1s = self._load_series(t1f, pid, 't1')
            t2s = self._load_series(t2f, pid, 't2')
            if not t1s or not t2s: continue
            row = self.labels[self.labels['patient_id']==pid]
            if row.empty: continue
            label = float(row['FDA'].values[0])
            self.data.append((t1s, t2s, label, pid))

    def _load_series(self, path, pid, modality):
        files = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dcm')])
        out = []
        for f in files:
            ds = pydicom.dcmread(f)
            if ds.PatientID.zfill(7)!=pid: continue
            img = apply_windowing(ds)
            hist = self.ref_hist_t1 if modality=='t1' else self.ref_hist_t2
            if hist: img = hist(img)
            img = self.clahe(img)
            slice_no = int(ds.InstanceNumber)
            bme_row = self.bme[(self.bme['patient_id']==pid)&(self.bme['slice_number']==slice_no)]
            bme_val = float(bme_row['bme'].values[0]) if not bme_row.empty else 0.0
            out.append((img.astype(np.float32), bme_val))
        return out

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        t1s, t2s, label, pid = self.data[idx]

        imgs1,bme1 = zip(*t1s)
        imgs2,_ = zip(*t2s)
        if self.transform_t1:
            imgs1=[self.transform_t1(i) for i in imgs1]
        if self.transform_t2:
            imgs2=[self.transform_t2(i) for i in imgs2]
            
        t1 = torch.stack(imgs1)
        t2 = torch.stack(imgs2)
        
        bme = torch.tensor(bme1,dtype=torch.float32)
        
        return t1, t2, torch.tensor(label), bme, pid
    
    # def save_preprocessed(self, save_root):
    #     """
    #     Raw 전처리 단계(CLAHE + 히스토그램 매칭)까지 완료된
    #     T1/T2 슬라이스를 8-bit 그레이스케일 PNG로 저장합니다.

    #     결과 구조:
    #     save_root/
    #       └── {patient_id}/
    #           ├── T1/
    #           └── T2/
    #     """
    #     os.makedirs(save_root, exist_ok=True)

    #     for t1s, t2s, _, pid in self.data:
    #         # 1) pad / trim to self.num_slices
    #         if len(t1s) < self.num_slices:
    #             t1s_pad = t1s + [(np.zeros_like(t1s[0][0]), 0)] * (self.num_slices - len(t1s))
    #         else:
    #             t1s_pad = t1s[: self.num_slices]

    #         if len(t2s) < self.num_slices:
    #             t2s_pad = t2s + [(np.zeros_like(t2s[0][0]), 0)] * (self.num_slices - len(t2s))
    #         else:
    #             t2s_pad = t2s[: self.num_slices]

    #         # 2) extract just the numpy images
    #         imgs1 = [img for img, _ in t1s_pad]
    #         imgs2 = [img for img, _ in t2s_pad]

    #         # 3) 폴더 생성
    #         patient_dir = os.path.join(save_root, pid)
    #         t1_dir = os.path.join(patient_dir, "T1")
    #         t2_dir = os.path.join(patient_dir, "T2")
    #         os.makedirs(t1_dir, exist_ok=True)
    #         os.makedirs(t2_dir, exist_ok=True)

    #         # 4) helper: np.ndarray → 8bit 그레이스케일 PIL
    #         def to_grayscale_pil(arr: np.ndarray) -> Image.Image:
    #             # normalize to [0,255]
    #             a = arr.astype(np.float32)
    #             a = a - a.min()
    #             if a.max() != 0:
    #                 a = a / a.max() * 255
    #             a = a.astype(np.uint8)
    #             return Image.fromarray(a, mode="L")

    #         # 5) T1 저장
    #         for idx, arr in enumerate(imgs1, start=1):
    #             pil = to_grayscale_pil(arr)
    #             fname = f"{pid}_T1_slice_{idx:02d}.png"
    #             pil.save(os.path.join(t1_dir, fname))

    #         # 6) T2 저장
    #         for idx, arr in enumerate(imgs2, start=1):
    #             pil = to_grayscale_pil(arr)
    #             fname = f"{pid}_T2_slice_{idx:02d}.png"
    #             pil.save(os.path.join(t2_dir, fname))

    #         print(f"[Saved] Patient {pid}: {len(imgs1)} T1, {len(imgs2)} T2 slices")


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        attn_scores = self.attention(features)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_features = features * attn_weights
        combined_features = torch.sum(weighted_features, dim=1)
        return combined_features, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / (embed_dim ** 0.5)
    
    def forward(self, x, mask=None):
        # x: [B, S, D], mask: [B, S]
        B, S, D = x.size()
        Q = self.query(x)  # [B,S,D]
        K = self.key(x)    # [B,S,D]
        V = self.value(x)  # [B,S,D]

        scores = torch.bmm(Q, K.transpose(1,2)) * self.scale  # [B,S,S]

        if mask is not None:
            # 마스크를 (B,S,S)로 확장
            mask2d = mask.unsqueeze(1).expand(-1, S, -1)  # [B,S,S]
            scores = scores.masked_fill(mask2d == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)          # [B,S,S]
        attn_output  = torch.bmm(attn_weights, V)         # [B,S,D]
        return attn_output, attn_weights



class MultiModalModel(nn.Module):
    def __init__(self, model_type='resnet50', hidden_dim=256, dropout_rate=0.7):
        super(MultiModalModel, self).__init__()

        if 'wide_resnet' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_wide_resnet_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_wide_resnet_extractor(model_type)
        elif 'resnext' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_resnext_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_resnext_extractor(model_type)
        elif 'resnet' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_resnet_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_resnet_extractor(model_type)
        elif 'efficientnet' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_efficientnet_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_efficientnet_extractor(model_type)
        elif 'densenet' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_densenet_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_densenet_extractor(model_type)
        elif 'swin' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_swin_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_swin_extractor(model_type)
        elif 'vit' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_vit_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_vit_extractor(model_type)
        elif 'convnext' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_convnext_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_convnext_extractor(model_type)
        elif 'vgg' in model_type:
            self.t1_feature_extractor, self.feature_dim = self.create_vgg_extractor(model_type)
            self.t2_feature_extractor, _ = self.create_vgg_extractor(model_type)
        else:
            raise ValueError("Unsupported model type. Choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'resnext101_32x8d', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'densenet121', 'densenet169', 'densenet201', 'swin_t', 'swin_s', 'swin_b', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'vgg11', 'vgg13', 'vgg16', 'vgg19']")

        # self.bme_fc = nn.Linear(1, hidden_dim)
        
        # self.bme_fc_output_dim = hidden_dim // 2
        # self.bme_fc = nn.Sequential(
        #     nn.Linear(1, hidden_dim // 4),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(hidden_dim // 4, hidden_dim // 2),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(hidden_dim // 2, hidden_dim // 2)  # 최종 출력 크기 확대
        # )
        
        # BME embedding
        self.bme_fc_output_dim = 128
        self.bme_fc = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(True), nn.Dropout(dropout_rate),
            nn.Linear(64, self.bme_fc_output_dim), nn.ReLU(True), nn.Dropout(dropout_rate)
        )
        # project t1/t2 features to BME dim
        self.t1_proj = nn.Linear(self.feature_dim, self.bme_fc_output_dim)
        self.t2_proj = nn.Linear(self.feature_dim, self.bme_fc_output_dim)
        # modality attention scorer on unified dim
        self.modality_attention = nn.Sequential(
            nn.Linear(self.bme_fc_output_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        # temporal self-attention on fused features of size bme_fc_output_dim
        self.attention = SelfAttention(self.bme_fc_output_dim)
        # classifier taking pooled vector of dim bme_fc_output_dim
        self.classifier = self.create_classifier_module(self.bme_fc_output_dim, hidden_dim)
        
        self.modality_weights_list = []
        self.attn_w_list = []

    def create_resnet_extractor(self, resnet_type):
        resnet = getattr(models, resnet_type)(pretrained=True)
        feature_dim = 512 if resnet_type in ['resnet18', 'resnet34'] else 2048
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        return nn.Sequential(feature_extractor, nn.Flatten()), feature_dim

    # def create_efficientnet_extractor(self, efficientnet_type):
    #     efficientnet = getattr(models, efficientnet_type)(pretrained=True)
    #     feature_dim = efficientnet.classifier[1].in_features
    #     efficientnet.classifier = nn.Identity()
    #     return nn.Sequential(efficientnet, nn.Flatten()), feature_dim
    
    def create_efficientnet_extractor(self, efficientnet_type):
        efficientnet = timm.create_model(efficientnet_type, pretrained=True)
        feature_dim = efficientnet.num_features
        efficientnet.reset_classifier(0)  # classifier layer를 nn.Identity()로 바꿈
        # 4. return
        return nn.Sequential(efficientnet, nn.Flatten()), feature_dim

    def create_densenet_extractor(self, densenet_type):
        densenet = getattr(models, densenet_type)(pretrained=True)
        feature_dim = densenet.classifier.in_features
        densenet.classifier = nn.Identity()
        return nn.Sequential(densenet, nn.Flatten()), feature_dim

    def create_wide_resnet_extractor(self, wide_resnet_type):
        wide_resnet = getattr(models, wide_resnet_type)(pretrained=True)
        feature_dim = wide_resnet.fc.in_features
        wide_resnet.fc = nn.Identity()
        return nn.Sequential(wide_resnet, nn.Flatten()), feature_dim

    def create_resnext_extractor(self, resnext_type):
        resnext = getattr(models, resnext_type)(pretrained=True)
        feature_dim = resnext.fc.in_features
        resnext.fc = nn.Identity()
        return nn.Sequential(resnext, nn.Flatten()), feature_dim

    def create_vit_extractor(self, vit_type):
        vit = getattr(models, vit_type)(pretrained=True)
        feature_dim = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        return nn.Sequential(vit, nn.Flatten()), feature_dim

    def create_convnext_extractor(self, convnext_type):
        convnext = getattr(models, convnext_type)(pretrained=True)
        feature_dim = convnext.classifier[2].in_features
        convnext.classifier = nn.Identity()
        return nn.Sequential(convnext, nn.Flatten()), feature_dim

    def create_vgg_extractor(self, vgg_type):
        vgg = getattr(models, vgg_type)(pretrained=True)
        feature_dim = vgg.classifier[0].in_features * 7 * 7  # Assuming input size is 224x224
        vgg.classifier = nn.Identity()
        return nn.Sequential(vgg, nn.Flatten()), feature_dim

    def create_swin_extractor(self, swin_type):
        if swin_type == 'swin_t':
            swin = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        elif swin_type == 'swin_s':
            swin = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        elif swin_type == 'swin_b':
            swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported Swin Transformer type. Choose from ['swin_t', 'swin_s', 'swin_b']")
       
        feature_dim = swin.head.in_features
        swin.head = nn.Identity()
        return nn.Sequential(swin, nn.Flatten()), feature_dim

    def create_attention_module(self, feature_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    # def create_classifier_module(self, feature_dim, hidden_dim):
    #     return nn.Sequential(
    #         nn.Linear(feature_dim, hidden_dim),
    #         nn.ReLU(True),
    #         nn.Linear(hidden_dim, 1)
    #     )
        
    def create_classifier_module(self, feature_dim, hidden_dim):
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
    # def create_classifier_module(self, feature_dim, hidden_dim):
    #     return nn.Sequential(
    #         nn.LayerNorm(feature_dim), 
    #         nn.Dropout(0.5),
    #         nn.Linear(feature_dim, hidden_dim), 
    #         nn.ReLU(), 
    #         nn.Dropout(0.5),
    #         nn.Linear(hidden_dim, 1)
    #     )
            
    def forward(self, t1, t2, bme, mask=None):
        B, S, C, H, W = t1.size()
        t1 = t1.view(B*S, C, H, W)
        t2 = t2.view(B*S, C, H, W)

        # 1) raw features 추출
        t1_raw = self.t1_feature_extractor[0](t1).view(B, S, -1)
        t2_raw = self.t2_feature_extractor[0](t2).view(B, S, -1)

        # 2) BME 임베딩
        bme_emb = self.bme_fc(bme.view(B*S,1)).view(B, S, -1)

        # 3) modality별 피처 투영
        t1_feat = self.t1_proj(t1_raw)
        t2_feat = self.t2_proj(t2_raw)

        # 4) modality weights 계산
        w1 = self.modality_attention(t1_feat)
        w2   = self.modality_attention(t2_feat)
        w3 = self.modality_attention(bme_emb)
        weights = torch.softmax(torch.cat([w1, w2, w3], dim=-1), dim=-1)

        # 5) fuse features
        w1_, w2_, w3_ = weights[...,0:1], weights[...,1:2], weights[...,2:3]
        fused = t1_feat * w1_ + t2_feat * w2_ + bme_emb * w3_

        # 6) 기록: modality weight
        self.modality_weights_list.append(weights.detach().cpu())

        # 7) temporal attention & 기록
        attn_out, attn_w = self.attention(fused, mask)
        self.attn_w_list.append(attn_w.detach().cpu())

        # 8) pooling
        if mask is not None:
            lengths = mask.sum(1, True).clamp(min=1)
            pooled = (attn_out * mask.unsqueeze(-1)).sum(1) / lengths
        else:
            pooled = attn_out.mean(1)

        # 9) classification
        logits = self.classifier(pooled)
        return logits, attn_w

        
    # def forward(self, t1, t2, bme, mask=None):
    #     # t1/t2: [B,S,C,H,W], bme: [B,S], mask: [B,S]
    #     B, S, C, H, W = t1.size()
    #     t1 = t1.view(B*S, C, H, W)
    #     t2 = t2.view(B*S, C, H, W)

    #     t1_feats = self.t1_feature_extractor[0](t1).view(B, S, -1)
    #     t2_feats = self.t2_feature_extractor[0](t2).view(B, S, -1)
    #     bme_feats = self.bme_fc(bme.view(B*S,1)).view(B, S, -1)

    #     # Save for cosine alignment loss
    #     self.t1_attn_input = torch.cat((t1_feats, bme_feats), dim=2)
    #     self.t2_attn_input = torch.cat((t2_feats, bme_feats), dim=2)

    #     combined = torch.cat((t1_feats, t2_feats, bme_feats), dim=2)  # [B,S,D]
    #     attn_out, attn_w = self.attention(combined, mask=mask)

    #     if mask is not None:
    #         lengths = mask.sum(dim=1, keepdim=True)
    #         pooled = (attn_out * mask.unsqueeze(-1)).sum(dim=1) / lengths
    #     else:
    #         pooled = attn_out.mean(dim=1)

    #     logits = self.classifier(pooled)
    #     return logits, attn_w
    
    def plot_modality_weights(modality_weights_list, save_path=None):
        """
        modality_weights_list: list of tensors [epochs][B, S, 3]
        """
        import matplotlib.pyplot as plt
        avg_weights = [w.mean(dim=(0,1)).cpu().numpy() for w in modality_weights_list]
        epochs = range(1, len(avg_weights)+1)
        w1 = [w[0] for w in avg_weights]
        w2 = [w[1] for w in avg_weights]
        w3 = [w[2] for w in avg_weights]
        plt.figure()
        plt.plot(epochs, w1, label='T1')
        plt.plot(epochs, w2, label='T2')
        plt.plot(epochs, w3, label='BME')
        plt.xlabel('Epoch')
        plt.ylabel('Average Modality Weight')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


    def plot_temporal_attention(attn_w_list, slice_idx=0, save_path=None):
        """
        attn_w_list: list of tensors [epochs][B, S, S]
        slice_idx: which sample in batch to visualize
        """
        import matplotlib.pyplot as plt
        num_epochs = len(attn_w_list)
        fig, axes = plt.subplots(1, num_epochs, figsize=(3*num_epochs,3))
        for i, w in enumerate(attn_w_list):
            mat = w[slice_idx].cpu().numpy()
            ax = axes[i] if num_epochs>1 else axes
            im = ax.imshow(mat, aspect='auto')
            ax.set_title(f'Epoch {i+1}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()



def cosine_attention_loss(attn_input1, attn_input2, mask=None):
    # Normalize along feature dim
    attn_input1 = F.normalize(attn_input1, dim=-1)
    attn_input2 = F.normalize(attn_input2, dim=-1)

    # Cosine similarity per slice
    sim = (attn_input1 * attn_input2).sum(dim=-1)  # [B, S]

    # 1 - cosine similarity is the loss
    loss = 1 - sim

    if mask is not None:
        loss = loss * mask  # ignore padded parts
        lengths = mask.sum(dim=1).clamp(min=1)  # prevent division by zero
        loss = loss.sum(dim=1) / lengths  # average per sample
    else:
        loss = loss.mean(dim=1)

    return loss.mean()




# Loss: Hybrid Focal-Tversky + BCE
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3, eps=1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        targets = targets.view_as(probs).float()

        # Compute TP, FP, FN
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        # Tversky index with stability eps
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        # Clamp to avoid exactly 0 or 1
        tversky = tversky.clamp(min=self.eps, max=1.0 - self.eps)

        # Focal Tversky loss
        loss = (1.0 - tversky) ** self.gamma
        return loss


class HybridFocalTverskyBCE(nn.Module):
    def __init__(self, lam=0.7, ft_alpha=0.7, ft_beta=0.3, ft_gamma=1.3, pos_weight=None, device=None):
        super().__init__()
        self.lam   = lam
        self.ft    = FocalTverskyLoss(ft_alpha, ft_beta, ft_gamma)
        
        # pos_weight가 float이면 Tensor로 바꿔줍니다.
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        # device 정보가 있으면 옮겨줍니다.
        if pos_weight is not None and device is not None:
            pos_weight = pos_weight.to(device)
        
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.lam * self.ft(logits, targets) + (1 - self.lam) * self.bce(logits, targets)


# Balanced Focal Loss
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha    = alpha
        self.gamma    = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t   = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        mod_factor = (1 - p_t) ** self.gamma
        loss = alpha_t * mod_factor * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()


# Generalized Cross Entropy Loss (Zhang & Sabuncu, 2018)
class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q=0.7, reduction='mean'):
        super().__init__()
        assert q > 0 and q <= 1, "q should be in (0,1]"
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets: {0,1}, logits → prob for class=1
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        # L_q = (1 - p_t^q) / q
        loss = (1 - p_t.pow(self.q)) / self.q
        return loss.mean() if self.reduction=='mean' else loss.sum()


# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, lambda_focal=0.5, alpha=0.25, gamma=2.0, q=0.7, pos_weight=None, device=None):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.focal = BalancedFocalLoss(alpha=alpha, gamma=gamma)
        self.gce   = GeneralizedCrossEntropy(q=q)

        if pos_weight is not None:
            # wrap float -> tensor, send to device
            if not isinstance(pos_weight, torch.Tensor):
                pw = torch.tensor(pos_weight, dtype=torch.float32)
            else:
                pw = pos_weight
            if device is not None:
                pw = pw.to(device)
            self.weighted_bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.weighted_bce = None

    def forward(self, logits, targets):
        loss_f = self.focal(logits, targets)
        if self.weighted_bce is not None:
            loss_b = self.weighted_bce(logits, targets)
            return self.lambda_focal * loss_f + (1 - self.lambda_focal) * loss_b
        else:
            loss_g = self.gce(logits, targets)
            return self.lambda_focal * loss_f + (1 - self.lambda_focal) * loss_g


def create_dataloaders(dicom_dir, label_csv, bme_csv, transform_t1, transform_t2, batch_size=1, random_state=42):
    print('Data Loading...')
   
    # Correctly pass the separate transformations for T1 and T2
    dataset = PatientSliceDataset(dicom_dir, label_csv, bme_csv, transform_t1=transform_t1, transform_t2=transform_t2)
    # dataset.save_preprocessed("axSpA_classification/result/preprocessing_result")
    
    print(f'Dataset size: {len(dataset)}')
   
    torch.manual_seed(random_state)
    random.seed(random_state)
   
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
    # 클래스 가중치 계산
    class_counts = [0, 0]
    for _, _, label_fda, _, _ in train_dataset:
        class_counts[int(label_fda)] += 1
    class_weights = [1.0 / count for count in class_counts]
   
    # 샘플 가중치 설정
    train_weights = [class_weights[int(label_fda)] for _, _, label_fda, _,_ in train_dataset]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=variable_length_collate, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=variable_length_collate, num_workers=16, pin_memory=True)
   
    save_preprocessed_images_for_debug(train_loader.dataset.dataset, save_root='axSpA_classification/result/check_image')
   
    train_labels = [label.item() for _, _, label,_,_ in train_loader.dataset]
    val_labels = [label.item() for _, _, label,_,_ in val_loader.dataset]

    print("Train set class distribution: ", Counter(train_labels))
    print("Validation set class distribution: ", Counter(val_labels))
   
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
   
    return dataloaders, class_weights


def save_preprocessed_images_for_debug(dataset, save_root='axSpA_classification/result/check_image'):
    """
    학습에 사용되는 T1, T2 전처리 이미지를 확인용으로 저장합니다.
    
    Args:
        dataset: PatientSliceDataset (train_loader.dataset.dataset)
        save_root: 저장할 최상위 폴더 경로
    """
    os.makedirs(save_root, exist_ok=True)

    def to_uint8(arr):
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr *= 255
        return arr.astype(np.uint8)

    for i in range(len(dataset)):
        t1s, t2s, label, bme, pid = dataset[i]
        pid = str(pid)
        t1_dir = os.path.join(save_root, pid, 'T1')
        t2_dir = os.path.join(save_root, pid, 'T2')
        os.makedirs(t1_dir, exist_ok=True)
        os.makedirs(t2_dir, exist_ok=True)

        for j in range(t1s.size(0)):
            img = to_uint8(t1s[j][0].cpu().numpy())  # [1, H, W] → [H, W]
            Image.fromarray(img).save(os.path.join(t1_dir, f"{pid}_T1_slice_{j:02d}.png"))

        for j in range(t2s.size(0)):
            img = to_uint8(t2s[j][0].cpu().numpy())
            Image.fromarray(img).save(os.path.join(t2_dir, f"{pid}_T2_slice_{j:02d}.png"))

    print(f"[✓] Saved debug images to: {save_root}")
    

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          num_epochs=10, patience=3, checkpoint_path='best_model.pth',
          model_type=None, scheduler_name='', device='cuda'):
    
    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train, y_score_train = [], [], []
        correct_train, total_train = 0, 0

        for t1, t2, labels_fda, labels_bme, mask, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            labels_fda = labels_fda.float().to(device)
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits, attn1 = model(t1, t2, labels_bme.to(device), mask)
                _, attn2 = model(t2, t1, labels_bme.to(device), mask)  # swapped
                task_loss = criterion(logits, labels_fda)
                align_loss = cosine_attention_loss(attn1, attn2)
                loss = task_loss + 0.1 * align_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            truths = labels_fda.cpu().numpy().astype(int).flatten()
            
            # after forward pass
            if torch.isnan(logits).any():
                print(f"[Epoch {epoch+1}] NaN in logits")

            # after loss
            if torch.isnan(loss).any():
                print(f"[Epoch {epoch+1}] NaN in loss ")

            # after sigmoid
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            if np.isnan(probs).any():
                print(f"[Epoch {epoch+1}] NaN in probs: {probs}")
  

            y_true_train.extend(truths.tolist())
            y_pred_train.extend(preds.tolist())
            y_score_train.extend(probs.tolist())
            correct_train += (preds == truths).sum()
            total_train += truths.shape[0]

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_train).ravel()
        sens_train = tp / (tp + fn) if (tp + fn) else 0
        spec_train = tn / (tn + fp) if (tn + fp) else 0
        auc_train = roc_auc_score(y_true_train, y_score_train)

        # === Validation ===
        model.eval()
        running_loss = 0.0
        y_true_val, y_pred_val, y_score_val = [], [], []
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for t1, t2, labels_fda, labels_bme, mask, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]'):
                labels_fda = labels_fda.float().to(device)
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

                with torch.cuda.amp.autocast():
                    outputs, _ = model(t1, t2, labels_bme.to(device), mask)
                    loss = criterion(outputs, labels_fda)

                running_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                truths = labels_fda.cpu().numpy().astype(int).flatten()

                y_true_val.extend(truths.tolist())
                y_pred_val.extend(preds.tolist())
                y_score_val.extend(probs.tolist())
                correct_val += (preds == truths).sum()
                total_val += truths.shape[0]

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        tn, fp, fn, tp = confusion_matrix(y_true_val, y_pred_val).ravel()
        sens_val = tp / (tp + fn) if (tp + fn) else 0
        spec_val = tn / (tn + fp) if (tn + fp) else 0
        auc_val = roc_auc_score(y_true_val, y_score_val)

        print(f"[{model_type} - {scheduler_name}] Epoch {epoch+1}/{num_epochs}")
        print(f" Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Sens: {sens_train:.3f} | Spec: {spec_train:.3f} | AUROC: {auc_train:.3f}")
        print(f" Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Sens: {sens_val:.3f} | Spec: {spec_val:.3f} | AUROC: {auc_val:.3f}")

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f" Saving checkpoint at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f" Early stop count: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(" Early stopping triggered.")
            break

    return best_val_loss, best_val_acc




# Albumentations 파이프라인 
# alb_pipeline = A.Compose([
#     A.HorizontalFlip(p=0.1),                              # 줄임
#     A.Rotate(limit=3, p=0.1),                             # limit 축소 + 확률 낮춤
#     A.ColorJitter(brightness=0.05, contrast=0.05, 
#                   saturation=0.05, hue=0.01, p=0.1),      # 색 변화 축소
#     A.GaussianBlur(blur_limit=(3,3), p=0.1),              # blur 약화
#     # A.ElasticTransform(alpha=1, sigma=50, p=0.2),       # 임시 제거
#     A.RandomGamma(p=0.1),
#     # A.RandomResizedCrop(height=512, width=512, scale=(0.8,1.0), p=0.2),
#     # A.CoarseDropout(max_holes=5, max_height=50, max_width=50, p=0.2),
# ])


class VerticalStripMask(A.ImageOnlyTransform):
    def __init__(self, width_ratio=0.2, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)  # 👈 named arguments로 수정
        self.width_ratio = width_ratio

    def apply(self, img, **params):
        h, w = img.shape[:2]
        center = w // 2
        half_width = int((w * self.width_ratio) / 2)
        img[:, center - half_width:center + half_width] = 0
        return img


alb_pipeline = A.Compose([
    VerticalStripMask(width_ratio=0.2, p=0.8),  # ✅ 가장 먼저
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=3, p=0.3),
    A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=5, p=0.1),
    A.RandomGamma(gamma_limit=(90, 110), p=0.1),
])



def create_transforms(apply_augmentations=True):
    pil_pre = [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  
    ]
    if apply_augmentations:
        pil_pre += [
            # transforms.RandomHorizontalFlip(p=0.1),
            # transforms.RandomApply([transforms.RandomRotation(3)], p=0.1),
            # transforms.RandomApply([transforms.ColorJitter(0.05, 0.05, 0.05, 0.01)], p=0.1),
            # transforms.RandomAdjustSharpness(1.5, p=0.1),
            transforms.Lambda(lambda img: Image.fromarray(
                alb_pipeline(image=np.array(img))['image']
            )),
        ]

    tensor_post = [
        transforms.ToTensor(),  # 결과는 [1, H, W]
        transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min() + 1e-6) if t.max() > t.min() else t),
        # single→3채널 복사
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ]

    transform_t1 = transforms.Compose(pil_pre + tensor_post)
    transform_t2 = transforms.Compose(pil_pre + tensor_post)
    return transform_t1, transform_t2



# def create_transforms(apply_augmentations=True):
#     # 2) PIL augmentations ───────────────────────────────
#     pil_pre = [transforms.ToPILImage()]
#     if apply_augmentations:
#         pil_pre += [
#             transforms.RandomHorizontalFlip(p=0.2),
#             transforms.RandomApply([transforms.RandomRotation(5)], p=0.2),
#             transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.2),
#             transforms.RandomApply(
#                 [transforms.RandomAffine(degrees=0,
#                                          translate=(0.1, 0.1),
#                                          shear=(10, 10),
#                                          scale=(0.8, 1.2))], p=0.2),
#             transforms.RandomPerspective(distortion_scale=0.4, p=0.2),
#             transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
#             transforms.RandomGrayscale(p=0.1),
#             # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
#             # Albumentations
#             transforms.Lambda(lambda img: alb_pipeline(image=np.array(img))['image']),
#         ]

#     # 3) resize → tensor → normalize → channel-repeat
#     tensor_post = [
#         transforms.Resize((512, 512)),
#         transforms.Lambda(lambda img: img if isinstance(img, torch.Tensor)
#                            else transforms.ToTensor()(img)),
#         transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min())
#                            if t.max() > t.min() else t),
        
        
#         transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
#         # ImageNet pre-trained 모델이 기대하는 입력 분포
#         transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                              std =(0.229, 0.224, 0.225)),
#     ]

#     # ─── RandomErasing은 augmentations 켜졌을 때만 추가 ───
#     # if apply_augmentations:
#     #     tensor_post.append(
#     #         transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
#     #     )

#     transform_t1 = transforms.Compose(pil_pre + tensor_post)
#     transform_t2 = transforms.Compose(pil_pre + tensor_post)
#     return transform_t1, transform_t2


def train_with_augmentations(model_type, scheduler_name, dicom_dir, label_csv, bme_csv, batch_size, random_state, device):
    # Set up transformations for training with augmentations
    transform_train_t1, transform_train_t2 = create_transforms(apply_augmentations=True)
   
    # Set up transformations for validation without augmentations
    transform_val_t1, transform_val_t2 = create_transforms(apply_augmentations=False)

    # Create dataloaders
    dataloaders, class_weights = create_dataloaders(dicom_dir, label_csv, bme_csv, transform_train_t1, transform_train_t2, batch_size, random_state)

    # Update validation dataset with no augmentations, only preprocessing
    dataloaders['val'].dataset.dataset.transform_t1 = transform_val_t1
    dataloaders['val'].dataset.dataset.transform_t2 = transform_val_t2

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # Set up model
    model = MultiModalModel(model_type=model_type, hidden_dim=1024).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Set up loss function and optimizer
    pos_weight = class_weights[1] / class_weights[0]
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    # criterion = nn.BCEWithLogitsLoss()
    criterion = HybridFocalTverskyBCE(
        lam       = 0.5,   # Focal-Tversky 50 % + BCE 50 %
        ft_alpha  = 0.3,   # FP 패널티 (낮춰서 specificity ↓)
        ft_beta   = 0.7,   # FN 패널티 (높여서 sensitivity ↑)
        ft_gamma  = 1.3,   # focal 세기
        pos_weight= pos_weight, 
        device=device
    )
    # criterion = CombinedLoss(
    #     lambda_focal=0.7,
    #     alpha=0.75,
    #     gamma=2.0,
    #     q=0.6,
    #     pos_weight=pos_weight,   # this can now be a float
    #     device=device            # pass through your training device
    # )

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)

    # Set up scheduler 
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif scheduler_name == 'CyclicLR': 
        scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0001, step_size_up=5, mode='triangular2', cycle_momentum=False)
    elif scheduler_name == 'ReduceLROnPlateau': 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Define checkpoint filename with augmentation label 
    save_path = f'axSpA_classification/check_points/20250525' 
    if not os.path.exists(save_path): 
        os.makedirs(save_path) 

    checkpoint_filename = f'{save_path}/{model_type}_modality_attention_classifier_v7.pt' 

    # print(pytorch_model_summary.summary(model.module, torch.zeros(1, 12, 3, 256, 256).to(device), torch.zeros(1, 12, 3, 256, 256).to(device), torch.zeros(1,12,1).to(device), show_input=True))
    # Train the model
    best_val_loss, best_val_acc = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=200, patience=20, checkpoint_path=checkpoint_filename,
        model_type=model_type, scheduler_name=scheduler_name
    )

    return best_val_loss, best_val_acc, model


def save_debug_image(image, save_dir, filename, cmap='gray'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    
    if isinstance(image, np.ndarray):
        img = image
    elif isinstance(image, torch.Tensor):
        img = image.cpu().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # [C,H,W]
            img = np.transpose(img, (1, 2, 0))  # [H,W,C]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
    else:
        raise ValueError("Unsupported image type")

    # Normalize for visualization
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    plt.imsave(path, img, cmap=cmap)


def variable_length_collate(batch):
    bs = len(batch)
    # 시퀀스별 길이
    seq_lens = [item[0].size(0) for item in batch]
    S_max = max(seq_lens)
    C, H, W = batch[0][0].size()[1:]  # assuming all channels/heights/widths same

    # 패딩 텐서, 마스크, 레이블, pid 준비
    t1_pad = torch.zeros(bs, S_max, C, H, W)
    t2_pad = torch.zeros_like(t1_pad)
    bme_pad = torch.zeros(bs, S_max)
    mask   = torch.zeros(bs, S_max, dtype=torch.bool)
    labels = torch.zeros(bs, 1)
    pids   = []

    for i, (t1, t2, lbl, bme, pid) in enumerate(batch):
        L = t1.size(0)
        t1_pad[i, :L] = t1
        t2_pad[i, :L] = t2
        bme_pad[i, :L] = bme
        mask[i, :L]   = 1
        labels[i,0]   = lbl
        pids.append(pid)

    return t1_pad, t2_pad, labels, bme_pad, mask, pids


if __name__ == "__main__":
    dicom_dir  = "axSpA_classification/dataset/train_fullslice"
    label_csv  = "data/AxSpA_label.csv"
    bme_csv    = "data/Combined_BME_Labels.csv"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_types = ['convnext_large', 'convnext_base', 'convnext_small', 'convnext_tiny', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']
    schedulers  = ['CosineAnnealingWarmRestarts']
    batch_size  = 4
    random_state= 17

    results = []

    for model_type in model_types:
        for scheduler_name in schedulers:
            # ─── 훈련 및 모델 반환 ──────────────────────────────────────
            best_val_loss, best_val_acc = train_with_augmentations(
                model_type=model_type,
                scheduler_name=scheduler_name,
                dicom_dir=dicom_dir,
                label_csv=label_csv,
                bme_csv=bme_csv,
                batch_size=batch_size,
                random_state=random_state,
                device=device
            )

            # base_model = model.module if isinstance(model, nn.DataParallel) else model

            # save_dir = f"axSpA_classification/result/visuals/{model_type}_{scheduler_name}"
            # os.makedirs(save_dir, exist_ok=True)
            
            # base_model.plot_modality_weights(
            #     base_model.modality_weights_list,
            #     save_path=os.path.join(save_dir, "modality_weights.png")
            # )
            
            # base_model.plot_temporal_attention(
            #     base_model.attn_w_list,
            #     slice_idx=0,
            #     save_path=os.path.join(save_dir, "temporal_attention.png")
            # )

            # ─── 결과 기록 및 출력 ─────────────────────────────────────
            results.append([model_type, scheduler_name, best_val_loss, best_val_acc])
            print(f"[{model_type} | {scheduler_name}] Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%\n")

    # ─── 전체 요약 ─────────────────────────────────────────────────
    print("Model Performance Summary:")
    print(f"{'Model Type':<20}{'Scheduler':<30}{'Best Val Loss':<20}{'Best Val Acc':<20}")
    for mt, sc, vl, va in results:
        print(f"{mt:<20}{sc:<30}{vl:<20.4f}{va:<20.2f}%")