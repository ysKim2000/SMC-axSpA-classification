import os
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import ast
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import pydicom
from skimage import exposure  
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
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = image[..., 0]
            return exposure.match_histograms(image, self.target_image)
        else:
            raise ValueError("Image should be numpy array")


# class MedicalImageDataset(Dataset):
#     def __init__(self, bbox_csv, bme_csv, images_dir, image_type='T2', transform=None, target_image=None):
#         self.bbox_labels = pd.read_csv(bbox_csv)
#         self.bme_labels = pd.read_csv(bme_csv)
#         self.images_dir = images_dir
#         self.image_type = image_type
#         self.transform = transform
#         self.histogram_matching = HistogramMatching(target_image) if target_image is not None else None
#         self.data = self._load_data()

#     def _load_data(self):
#         data = []
#         for patient in os.listdir(self.images_dir):
#             path = os.path.join(self.images_dir, patient, self.image_type)
#             if not os.path.isdir(path):
#                 continue
#             for f in os.listdir(path):
#                 if not f.endswith('.dcm'): continue
#                 dicom = pydicom.dcmread(os.path.join(path, f))
#                 img = dicom.pixel_array
#                 img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
#                 if self.histogram_matching:
#                     img = self.histogram_matching(img)
#                 pid = int(dicom.PatientID)
#                 sn = int(dicom.InstanceNumber)
#                 row = self.bbox_labels[(self.bbox_labels.patient_id==pid)&(self.bbox_labels.slice_number==sn)]
#                 if row.empty: continue
#                 left_box = eval(row.iloc[0].left_box)
#                 right_box = eval(row.iloc[0].right_box)
#                 clahe = CLAHE()
#                 img_eq = np.array(clahe(img))
#                 for box_key, bme_side in [(left_box, 'left_bme'), (right_box, 'right_bme')]:
#                     x1,y1,x2,y2 = box_key
#                     patch = img_eq[y1:y2, x1:x2]
#                     label_str = self.bme_labels.loc[self.bme_labels.patient_id==pid, bme_side].values
#                     if len(label_str)==0:
#                         lbl=0
#                     else:
#                         d = ast.literal_eval(label_str[0])
#                         lbl = d.get(str(sn).zfill(4), 0)
#                     data.append((patch, lbl))
#         return data

#     def __len__(self): return len(self.data)
#     def __getitem__(self, idx):
#         img, lbl = self.data[idx]
#         if self.transform:
#             img = self.transform(img)
#         return img, torch.tensor(lbl, dtype=torch.float32)


def compute_metrics(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds>0.5).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0.0
    sens = tp/(tp+fn) if tp+fn>0 else 0
    spec = tn/(tn+fp) if tn+fp>0 else 0
    auroc = roc_auc_score(labels, preds) if len(np.unique(labels))>1 else 0
    return acc, sens, spec, auroc

import matplotlib.pyplot as plt
import seaborn as sns

def compute_confusion_matrix(preds, labels, threshold=0.5, save_path='bme_classification/result/cm/confusion_matrix.png'):
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return cm



if __name__ == '__main__':
    # Paths
    bbox_csv = 'data/new_csv/bbox_label.csv'
    bme_csv = 'data/bme_label.csv'
    test_dir = 'bme_classification/dataset/test'
    ckpt = 'bme_classification/checkpoints/20250509_convnext_large_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt' # Accuracy: 0.8244, Sensitivity: 0.8938, Specificity: 0.8020, AUROC: 0.9186
    # ckpt = 'bme_classification/checkpoints/20250509_convnext_large_monaiAugmentation_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt' # Accuracy: 0.8405, Sensitivity: 0.8540, Specificity: 0.8362, AUROC: 0.9110

    # Transforms
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1))
    ])

    # Load target image for histogram matching
    targ = pydicom.dcmread('data/reference/reference_stir.dcm').pixel_array
    targ = ((targ - targ.min())/(targ.max()-targ.min())*255).astype(np.uint8)

    # Dataset and loader
    dataset = MedicalImageDataset(bbox_csv, bme_csv, test_dir, transform=transf, target_image=targ)

    # 총 환자 수
    print("총 환자 수:", len(dataset.patient_ids))

    # 총 슬라이스 수
    total_slices = sum(len(slices) for slices in dataset.slice_count_per_patient.values())
    print("총 슬라이스 수:", total_slices)

    # 환자별 슬라이스 수 (선택적)
    for pid, slices in dataset.slice_count_per_patient.items():
        print(f"환자 {pid} 슬라이스 수: {len(slices)}")

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)

    # Model
    model = BinaryClassificationModel('convnext_large').get_model()
    model = torch.nn.DataParallel(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(ckpt))
    model.to(device).eval()

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            prob = torch.sigmoid(out).view(-1).cpu().numpy()
            all_preds.extend(prob.tolist())
            all_labels.extend(lbls.numpy().flatten().tolist())
            
    save_csv_path = 'bme_classification/result/bme_predictions_for_ci.csv'
    
    # 저장 경로 디렉토리 생성
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)

    # DataFrame 생성 및 저장
    df_results = pd.DataFrame({
        'true_label': all_labels,
        'predicted_prob': all_preds
    })
    
    # 1차원 확인 (혹시 모를 에러 방지)
    df_results['true_label'] = df_results['true_label'].values.ravel()
    df_results['predicted_prob'] = df_results['predicted_prob'].values.ravel()

    # Threshold 0.5 기준 예측 라벨 추가 (선택)
    df_results['predicted_label'] = (df_results['predicted_prob'] > 0.5).astype(int)

    df_results.to_csv(save_csv_path, index=False)
    print(f"\n[Info] Predictions saved to: {save_csv_path}")
    
    # Metrics
    acc, sens, spec, auroc = compute_metrics(np.array(all_preds), np.array(all_labels))
    print(f"Accuracy: {acc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, AUROC: {auroc:.4f}")

    # Confusion Matrix
    cm = compute_confusion_matrix(np.array(all_preds), np.array(all_labels), threshold=0.5)
    print("Confusion Matrix:\n", cm)


