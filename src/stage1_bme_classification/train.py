import os
import numpy as np
import torch
import pydicom
from torchvision import transforms, models
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import random
from collections import Counter
import torchio as tio 
from skimage import exposure  
from sklearn.metrics import roc_auc_score, confusion_matrix

import numpy as np, torch
from torchvision import transforms as tv
import monai.transforms as mt
import torchio as tio
import cv2

# class CLAHE(object):
#     def __call__(self, image):
#         if isinstance(image, torch.Tensor):
#             image = transforms.ToPILImage()(image)
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image)
#         if image.mode != 'L':
#             image = image.convert('L')
#         return ImageOps.equalize(image)


class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = image.numpy().squeeze()
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.clahe.apply(image.astype(np.uint8))



class HistogramMatching(object):
    def __init__(self, target_image):
        # Ensure the target image is single-channel (grayscale)
        if len(target_image.shape) == 3 and target_image.shape[-1] == 3:
            target_image = target_image[..., 0]  # Use only one channel if it's RGB
        self.target_image = target_image
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # Ensure the input image is single-channel (grayscale)
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = image[..., 0]  # Use only one channel if it's RGB
            matched = exposure.match_histograms(image, self.target_image, channel_axis=None)
            return matched
        else:
            raise ValueError("Image should be a numpy array")


class MedicalImageDataset(Dataset):
    def __init__(self, bbox_csv, bme_csv, images_dir, image_type='T2', transform=None, target_image=None):
        self.bbox_labels = pd.read_csv(bbox_csv)
        self.bme_labels = pd.read_csv(bme_csv)
        self.images_dir = images_dir
        self.image_type = image_type
        self.transform = transform
        self.histogram_matching = HistogramMatching(target_image) if target_image is not None else None
        self.data = self._load_data()

    def _load_data(self):
        data = []
        self.patient_ids = set()
        self.slice_count_per_patient = {}

        for patient_dir in os.listdir(self.images_dir):
            patient_path = os.path.join(self.images_dir, patient_dir, self.image_type)
            if os.path.isdir(patient_path):
                for image_file in os.listdir(patient_path):
                    if image_file.endswith('.dcm'):
                        image_path = os.path.join(patient_path, image_file)
                        dicom_image = pydicom.dcmread(image_path)
                        image = dicom_image.pixel_array
                        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

                        if self.histogram_matching:
                            image = self.histogram_matching(image)

                        patient_id = int(dicom_image.PatientID)
                        slice_number = int(dicom_image.InstanceNumber)

                        # 환자 수, 슬라이스 수 기록
                        self.patient_ids.add(patient_id)
                        if patient_id not in self.slice_count_per_patient:
                            self.slice_count_per_patient[patient_id] = set()
                        self.slice_count_per_patient[patient_id].add(slice_number)

                        bbox_row = self.bbox_labels[(self.bbox_labels['patient_id'] == patient_id) &
                                                    (self.bbox_labels['slice_number'] == slice_number)]

                        if patient_id in self.bme_labels['patient_id'].values:
                            left_bme_str = self.bme_labels.loc[self.bme_labels['patient_id'] == patient_id, 'left_bme'].values[0]
                            right_bme_str = self.bme_labels.loc[self.bme_labels['patient_id'] == patient_id, 'right_bme'].values[0]
                            left_bme_dict = ast.literal_eval(left_bme_str)
                            right_bme_dict = ast.literal_eval(right_bme_str)
                            left_value = left_bme_dict.get(str(slice_number).zfill(4), 0)
                            right_value = right_bme_dict.get(str(slice_number).zfill(4), 0)
                        else:
                            left_value = 0
                            right_value = 0

                        if not bbox_row.empty:
                            left_box = list(map(int, eval(bbox_row.iloc[0]['left_box'])))
                            right_box = list(map(int, eval(bbox_row.iloc[0]['right_box'])))

                            if self.is_valid_bbox(left_box) and self.is_valid_bbox(right_box):
                                clahe = CLAHE()
                                image_clahe = image.astype(np.uint8)

                                left_img = image_clahe[left_box[1]:left_box[3], left_box[0]:left_box[2]]
                                right_img = image_clahe[right_box[1]:right_box[3], right_box[0]:right_box[2]]
                                right_img = np.fliplr(right_img).copy()
                                data.append((left_img, left_value))
                                data.append((right_img, right_value))
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
        return image, label

    def is_valid_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 > x1) and (y2 > y1) and (x2 - x1) > 1 and (y2 - y1) > 1


class BinaryClassificationModel:
    def __init__(self, model_type):
        self.model_type = model_type.lower()
        self.model = self._initialize_model()

    def _initialize_model(self):
        if 'vgg' in self.model_type:
            model = self._get_vgg()
        elif 'wide_resnet' in self.model_type:
            model = self._get_wide_resnet()
        elif 'resnext' in self.model_type:
            model = self._get_resnext()
        elif 'resnet' in self.model_type:
            model = self._get_resnet()
        elif 'densenet' in self.model_type:
            model = self._get_densenet()
        elif 'efficientnet' in self.model_type:
            model = self._get_efficientnet()
        elif 'vit' in self.model_type:
            model = self._get_vit()
        elif 'swin' in self.model_type:
            model = self._get_swin_transformer()
        elif 'convnext' in self.model_type:
            model = self._get_convnext()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model


    def _get_vgg(self):
        if self.model_type == 'vgg11':
            model = models.vgg11(pretrained=True)
        elif self.model_type == 'vgg13':
            model = models.vgg13(pretrained=True)
        elif self.model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif self.model_type == 'vgg19':
            model = models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG model type: {self.model_type}")

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 1)
        return model

    def _get_resnet(self):
        if self.model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif self.model_type == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif self.model_type == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.model_type == 'resnet101':
            model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet model type: {self.model_type}")

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        return model

    def _get_densenet(self):
        if self.model_type == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif self.model_type == 'densenet161':
            model = models.densenet161(pretrained=True)
        elif self.model_type == 'densenet169':
            model = models.densenet169(pretrained=True)
        elif self.model_type == 'densenet201':
            model = models.densenet201(pretrained=True)
        else:
            raise ValueError(f"Unsupported DenseNet model type: {self.model_type}")

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
        return model

    def _get_wide_resnet(self):
        if self.model_type == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=True)
        elif self.model_type == 'wide_resnet101_2':
            model = models.wide_resnet101_2(pretrained=True)
        else:
            raise ValueError(f"Unsupported Wide ResNet model type: {self.model_type}")

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        return model

    def _get_resnext(self):
        if self.model_type == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=True)
        elif self.model_type == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNeXt model type: {self.model_type}")

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        return model

    def _get_efficientnet(self):
        if self.model_type == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
        elif self.model_type == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=True)
        elif self.model_type == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=True)
        elif self.model_type == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=True)
        elif self.model_type == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=True)
        elif self.model_type == 'efficientnet_b5':
            model = models.efficientnet_b5(pretrained=True)
        elif self.model_type == 'efficientnet_b6':
            model = models.efficientnet_b6(pretrained=True)
        elif self.model_type == 'efficientnet_b7':
            model = models.efficientnet_b7(pretrained=True)
        else:
            raise ValueError(f"Unsupported EfficientNet model type: {self.model_type}")
        
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        return model

    def _get_vit(self):
        if self.model_type == 'vit_b_16':
            model = models.vit_b_16(pretrained=True)
        elif self.model_type == 'vit_b_32':
            model = models.vit_b_32(pretrained=True)
        elif self.model_type == 'vit_l_16':
            model = models.vit_l_16(pretrained=True)
        elif self.model_type == 'vit_l_32':
            model = models.vit_l_32(pretrained=True)
        else:
            raise ValueError(f"Unsupported ViT model type: {self.model_type}")

        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 1)
        return model

    def _get_swin_transformer(self):
        if self.model_type == 'swin_t':
            model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        elif self.model_type == 'swin_s':
            model = models.swin_s(weights=models.Swin_T_Weights.DEFAULT)
        elif self.model_type == 'swin_b':
            model = models.swin_b(weights=models.Swin_T_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported Swin Transformer model type: {self.model_type}")

        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 1)
        return model

    def _get_convnext(self):
        if self.model_type == 'convnext_tiny':
            model = models.convnext_tiny(pretrained=True)
        elif self.model_type == 'convnext_small':
            model = models.convnext_small(pretrained=True)
        elif self.model_type == 'convnext_base':
            model = models.convnext_base(pretrained=True)
        elif self.model_type == 'convnext_large':
            model = models.convnext_large(pretrained=True)
        else:
            raise ValueError(f"Unsupported ConvNeXt model type: {self.model_type}")

        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 1)

        return model


    def get_model(self):
        return self.model


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def compute_metrics(preds, labels):
    preds_bin = (preds > 0.5).astype(int)
    labels_bin = labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_bin, preds_bin, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(labels_bin, preds) if len(set(labels_bin)) > 1 else 0.0
    return sensitivity, specificity, auroc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, patience=10, device='cuda', check_point_path='bme_classification/checkpoints/bme_train_v1_checkpoint.pt', model_type=None):
    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_sensitivities, val_specificities, val_aurocs = [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        train_loss, train_corrects, total_train = 0.0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                preds = torch.sigmoid(outputs) > 0.5
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(preds == labels.unsqueeze(1)).item()
                total_train += labels.size(0)
                tepoch.set_postfix(loss=train_loss / total_train, accuracy=100. * train_corrects / total_train, lr=optimizer.param_groups[0]['lr'])

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_corrects / total_train
        model.eval()
        val_loss, val_corrects, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
                val_loss += loss.item() * inputs.size(0)
                val_corrects += ((preds > 0.5) == labels.cpu().numpy()).sum()
                total_val += labels.size(0)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_corrects / total_val
        sensitivity, specificity, auroc = compute_metrics(np.array(all_preds), np.array(all_labels))
        scheduler.step(epoch_val_loss)
        print(f'[{model_type}] Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(f'Val Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, AUROC: {auroc:.4f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        val_sensitivities.append(sensitivity)
        val_specificities.append(specificity)
        val_aurocs.append(auroc)

        if epoch_val_loss < best_loss:
            print(f'Validation loss decreased ({best_loss:.6f} --> {epoch_val_loss:.6f}). Saving model...')
            print(f"Saving checkpoint path: {check_point_path}...")
            best_loss, best_acc = epoch_val_loss, epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), check_point_path)
        else:
            patience_counter += 1
            print(f'Early stop counts: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

    print(f'Best Validation Loss: {best_loss:.4f}, Best Validation Acc: {best_acc:.4f}')
    figure_path = f'bme_classification/result/figures/{model_type}'
    if not os.path.exists(figure_path): os.makedirs(figure_path)
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{figure_path}/{model_type}_v1_loss_curve.png')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracies,label="train")
    plt.plot(val_accuracies,label="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'{figure_path}/{model_type}_v1_accuracy_curve.png')
    plt.show()

    return best_loss, best_acc


def create_dataloaders(bbox_csv, bme_csv, images_dir, image_type='T2', target_image=None, train_transforms=None, validation_transforms=None, batch_size=1, random_state=42):
    dataset = MedicalImageDataset(bbox_csv, bme_csv, images_dir, image_type, target_image=target_image)
    print("Complete data load")
    print(f'Dataset size before filtering: {len(dataset)}')

    valid_data = [item for item in dataset if item is not None]
    dataset.data = valid_data

    print(f'Dataset size after filtering: {len(dataset)}')
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=random_state)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = validation_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=16)

    train_labels = [label.item() for _, label in train_loader.dataset]
    val_labels = [label.item() for _, label in val_loader.dataset]

    print("Train set class distribution: ", Counter(train_labels))
    print("Validation set class distribution: ", Counter(val_labels))
    return train_loader, val_loader



class HybridFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(HybridFocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # 기본 BCE 로스 (로짓 입력)

    def forward(self, inputs, targets):
        # BCE loss 계산
        bce_loss = self.bce(inputs, targets)

        # 시그모이드 후 확률로 변환
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)  # pt: 예측 확률 (정답 클래스 쪽으로)

        # Focal weight 적용
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HybridFocalWeightedBCELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # BCE Loss
        bce = self.bce_loss(logits, targets)

        # Focal Loss
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p_t
        focal_term = (1 - pt) ** self.gamma
        focal = (-(targets * torch.log(probs + 1e-8) + (1 - targets) * torch.log(1 - probs + 1e-8)))
        focal = (focal_term * focal).mean()

        return self.alpha * focal + (1 - self.alpha) * bce


class FocalTverskyLoss(nn.Module):
    """
    α : FP(특이도) 패널티,  β : FN(민감도) 패널티
    γ : focal-power (1 이상이면 hard sample 집중)
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, eps=1e-7):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, logits, targets):
        probs  = torch.sigmoid(logits)
        targets = targets.view_as(probs)          # (N,1) → (N,1)

        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()

        tversky = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps)
        loss = (1 - tversky) ** self.gamma
        return loss


class HybridFocalTverskyBCE(nn.Module):
    """
    lam ∈ [0,1]  → 0이면 Pure BCE, 1이면 Pure Focal-Tversky
    pos_weight : BCE 불균형 가중치 (torch.tensor, device 포함)
    """
    def __init__(self,
                 lam=0.6,
                 ft_alpha=0.3, ft_beta=0.7, ft_gamma=1.3,
                 pos_weight=None):
        super().__init__()
        self.lam = lam
        self.ft  = FocalTverskyLoss(ft_alpha, ft_beta, ft_gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.lam * self.ft(logits, targets) + (1 - self.lam) * self.bce(logits, targets)


if __name__ == "__main__":
    # ─── A. torchvision 초기 전처리 ───
    tv_base = tv.Compose([
        tv.ToPILImage(),                  # NumPy → PIL
        tv.Resize((512, 512)),
        tv.RandomHorizontalFlip(),
        tv.RandomRotation(10),
        tv.Lambda(lambda x: np.array(x)), # PIL → NumPy (H,W,C)
    ])

    # ─── B. MONAI 변형 (확률 0.3~0.5 권장) ───
    monai_aug = mt.Compose([
        mt.RandFlip(prob=0.5, spatial_axis=1),                 # 좌우
        mt.RandRotate(range_z=np.pi/18, prob=0.4),             # ±10°
        mt.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        mt.RandGridDistortion(prob=0.2, distort_limit=0.03),   # 미세 왜곡
        mt.RandGaussianNoise(prob=0.3, mean=0., std=0.015),
        mt.RandBiasField(prob=0.25),                           # RF inhomogeneity
        mt.RandHistogramShift(prob=0.3, num_control_points=3),
        mt.EnsureType(),                                       # NumPy → torch.Tensor
    ])

    # ─── C. torchio artifact ───
    tio_aug = tv.Lambda(lambda x: tio.Compose([
            tio.RandomMotion(),
            tio.RandomAnisotropy(),
        ])(x)
    )

    # ─── D. 마지막 Tensor / 채널 맞추기 ───
    to_tensor = tv.Compose([
        tv.Lambda(lambda x: x if torch.is_tensor(x) else torch.from_numpy(x)),
        tv.Lambda(lambda x: x.unsqueeze(0) if x.ndim==2 else x.permute(2,0,1)), # C,H,W
        tv.ConvertImageDtype(torch.float32),
        tv.Lambda(lambda x: x.repeat(3,1,1)),   # (1,H,W) → (3,H,W)  (모델은 RGB 기대)
    ])

    # ─── 최종 Compose ───
    train_transforms = tv.Compose([
        tv_base,
        tv.Lambda(lambda x: monai_aug(x)),  # MONAI 변형
        tio_aug,
        to_tensor,
    ])

    validation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    target_dicom_file = 'data/reference/reference_stir.dcm'
    target_dicom = pydicom.dcmread(target_dicom_file)
    target_image = target_dicom.pixel_array
    target_image = ((target_image - np.min(target_image)) / (np.max(target_image) - np.min(target_image)) * 255).astype(np.uint8)

    images_dir = 'bme_classification/dataset/train'
    bbox_csv = 'data/new_csv/bbox_label.csv'
    bme_csv = 'data/bme_label.csv'

    # model_type = 'resnet101' # 18, 34, 50, 101
    # model_type = 'efficientnet_b7' # b0 - b7
    # model_type = 'densenet201' # 121, 161, 169, 201
    # model_type = 'wide_resnet50_2' # 50_2, 101_2
    # model_type = 'resnext101_32x8d' # 50_32x4d, 101_32x8d
    # model_type = 'vit_l_32' # b_16, b_32, l_16, l_32
    # model_type = 'swin_t' # t, s, b
    # model_type = 'convnext_tiny' # convnext_tiny, convnext_small, convnext_base, convnext_large
    # model_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'densenet121', 
    #                'densenet161', 'densenet169', 'densenet201', 'wide_resnet50_2', 'wide_resnet101_2']

    model_types = ['convnext_large']

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = []

    for model_type in model_types:
        # save_path = f'02_BME_Classification/check_points_slice_new/{model_type}'
        save_path = f'bme_classification/checkpoints/'

        if not os.path.exists(save_path): os.makedirs(save_path)
        
        check_point_path = f'{save_path}/__20250509_{model_type}_monaiAugmentation_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt'
        train_loader, val_loader = create_dataloaders(bbox_csv, bme_csv, images_dir, image_type='T2',
                                                     target_image=target_image, train_transforms=train_transforms, validation_transforms=validation_transforms, batch_size=14)

        model_instance = BinaryClassificationModel(model_type)
        model = model_instance.get_model().to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for BME Classification model!")
            model = nn.DataParallel(model)

        pos_weight_value = 2471 / 487  # 약 5.07
        pos_weight = torch.tensor([5.07]).to(device)
        
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = HybridFocalBCELoss(alpha=0.25, gamma=2.0)
        # criterion = HybridFocalWeightedBCELoss(
        #     alpha=0.7,      # focal 쪽 비중을 더 줌
        #     gamma=2.0,      # 더 강하게 hard sample 집중
        #     pos_weight=pos_weight
        # )
        criterion = HybridFocalTverskyBCE(
            lam       = 0.7,   # Focal-Tversky 70 % + BCE 30 %
            ft_alpha  = 0.3,   # FP 패널티 (낮춰서 specificity ↓)
            ft_beta   = 0.7,   # FN 패널티 (높여서 sensitivity ↑)
            ft_gamma  = 1.3,   # focal 세기
            pos_weight= pos_weight
        )



        optimizer = optim.AdamW(model.parameters(), lr=0.000005, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


        print(f"Model: {model_type}")

        best_val_loss, best_val_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=300, patience=20, device=device, check_point_path=check_point_path, model_type=model_type)
        result.append([model_type, best_val_loss, best_val_acc])

    print("모델별 성능 비교:")
    print(f"{'Model Type':<20}{'Best Val Loss':<20}{'Best Val Acc':<20}")

    for result_item in result:
        model_type, best_val_loss, best_val_acc = result_item
        print(f"{model_type:<20}{best_val_loss:<20.4f}{best_val_acc:<20.2f}%")

