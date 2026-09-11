import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
import pydicom
from skimage import exposure
from tqdm import tqdm
from original_train import BinaryClassificationModel
from torchvision.transforms.functional import to_pil_image



# === 전처리 유틸 ===
class CLAHE:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')
        return ImageOps.equalize(image)


class HistogramMatching:
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
            raise ValueError("Image must be a NumPy array")


# === DICOM 기반 Patch 추출 Dataset ===
class InferenceDataset(Dataset):
    def __init__(self, bbox_csv, image_root, image_type='T2', transform=None, target_image=None, save_dir=None):
        self.bbox_df = pd.read_csv(bbox_csv)
        self.bbox_df['patient_id'] = self.bbox_df['patient_id'].astype(str).str.zfill(7)
        self.image_root = image_root
        self.image_type = image_type
        self.transform = transform
        self.save_dir = save_dir
        self.histogram_matching = HistogramMatching(target_image) if target_image is not None else None
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for pid in os.listdir(self.image_root):
            t2_path = os.path.join(self.image_root, pid, self.image_type)
            if not os.path.isdir(t2_path):
                continue
            for fname in os.listdir(t2_path):
                if not fname.endswith('.dcm'):
                    continue
                fpath = os.path.join(t2_path, fname)
                dcm = pydicom.dcmread(fpath)
                img = dcm.pixel_array

                # 1. 전체 이미지 512x512로 리사이즈
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                img = Image.fromarray(img).resize((512, 512), resample=Image.BILINEAR)
                img = np.array(img)

                # 2. 히스토그램 매칭
                if self.histogram_matching:
                    img = self.histogram_matching(img)

                # 3. CLAHE 적용 안함! 
                clahe = CLAHE()
                # img_eq = np.array(clahe(img))
                img_eq = img.astype(np.uint8)

                # 4. BBox 자르기
                patient_id = str(dcm.PatientID).zfill(7)
                slice_number = int(dcm.InstanceNumber)
                row = self.bbox_df[
                    (self.bbox_df['patient_id'] == patient_id) &
                    (self.bbox_df['instance_number'] == slice_number)
                ]
                if row.empty:
                    continue

                left_box = eval(row.iloc[0]['left_box'])
                right_box = eval(row.iloc[0]['right_box'])

                for side, box in [('left', left_box), ('right', right_box)]:
                    x1, y1, x2, y2 = map(int, box)
                    patch = img_eq[y1:y2, x1:x2]

                    # Transform 적용 (512x512 등)
                    if self.transform:
                        patch_tensor = self.transform(patch)
                    else:
                        patch_tensor = transforms.ToTensor()(patch)

                    # # === 저장 ===
                    # if self.save_dir:
                    #     save_path = Path(self.save_dir) / patient_id
                    #     save_path.mkdir(parents=True, exist_ok=True)
                    #     out_img = to_pil_image(patch_tensor)
                    #     out_img.save(save_path / f"{side}_{slice_number}.png")

                    # 모델 입력용
                    data.append((patch_tensor, patient_id, slice_number, side))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



# === 후처리 + 저장 함수 ===
def save_processed_predictions(preds, infos, save_path):
    df = pd.DataFrame(infos, columns=["patient_id", "instance_number", "side"])
    df["probability"] = preds
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(7)
    df["instance_number"] = df["instance_number"].astype(int)

    # pivot
    pivot_df = df.pivot_table(index=['patient_id', 'instance_number'],
                              columns='side',
                              values='probability').reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={'left': 'left_prob', 'right': 'right_prob'})

    pivot_df['left_bme'] = (pivot_df['left_prob'] >= 0.5).astype(int)
    pivot_df['right_bme'] = (pivot_df['right_prob'] >= 0.5).astype(int)
    pivot_df['bme'] = ((pivot_df['left_bme'] == 1) | (pivot_df['right_bme'] == 1)).astype(int)

    pivot_df.to_csv(save_path, index=False)
    print(f"[✓] Saved: {save_path}")


# === 메인 실행 ===
if __name__ == '__main__':
    # Path 설정
    bbox_csv = '20250515_bbox_coordinates_filled.csv'
    image_root = 'data/axspa_classification_external_sliced'
    # ckpt_path = 'bme_classification/checkpoints/20250509_convnext_large_monaiAugmentation_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt'
    ckpt_path = 'bme_classification/checkpoints/20250509_convnext_large_HybridFocalTverskyBCE_no_clahe_minus_augmentation_5e-6_cosine_512_512_AdamW_1e-5_checkpoint.pt' # Accuracy: 0.8244, Sensitivity: 0.8938, Specificity: 0.8020, AUROC: 0.9186
    save_path = 'bme_classification/inference_result/bme_inference_v2_50per.csv'

    # Transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    # Target image for histogram matching
    target_dcm = pydicom.dcmread('data/reference/reference_stir.dcm').pixel_array
    target_img = ((target_dcm - target_dcm.min()) / (target_dcm.max() - target_dcm.min()) * 255).astype(np.uint8)

    # Dataset
    # Dataset 생성 시 save_dir 지정
    dataset = InferenceDataset(
        bbox_csv=bbox_csv,
        image_root=image_root,
        transform=transform,
        target_image=target_img,
        save_dir='bme_classification/inference_image_result'
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

    # Load model
    model = BinaryClassificationModel('convnext_large').get_model()
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device).eval()

    # Inference
    all_probs = []
    all_infos = []
    with torch.no_grad():
        for imgs, pids, sns, sides in tqdm(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()
            if probs.ndim == 0: probs = [probs.item()]
            all_probs.extend(probs)
            all_infos.extend(zip(pids, sns, sides))

    # Save processed result
    save_processed_predictions(all_probs, all_infos, save_path)
