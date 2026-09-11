
import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageOps
import pydicom
from skimage import exposure
from collections import Counter
import cv2

class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.ndim == 3:
            image = image[..., 0]  # take grayscale if accidentally RGB

        if image.dtype != np.uint8:
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            image = image.astype(np.uint8)

        clahe_applied = self.clahe.apply(image)
        return clahe_applied

class HistogramMatching(object):
    def __init__(self, target_image):
        if len(target_image.shape) == 3 and target_image.shape[-1] == 3:
            target_image = target_image[..., 0]
        self.target_image = target_image

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = image[..., 0]
            matched = exposure.match_histograms(image, self.target_image, channel_axis=False)
            return matched
        else:
            raise ValueError("Image should be a numpy array")

class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, target_dicom_file=None):
        self.labels = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.mode = 'train'

        if target_dicom_file:
            target_dicom = pydicom.dcmread(target_dicom_file)
            target_image = target_dicom.pixel_array
            target_image = ((target_image - np.min(target_image)) / (np.max(target_image) - np.min(target_image)) * 255).astype(np.uint8)
            target_image = Image.fromarray(target_image).resize((512, 512), Image.BILINEAR)
            target_image = np.array(target_image)
            self.histogram_matching = HistogramMatching(target_image)
        else:
            self.histogram_matching = None

        self.clahe = CLAHE()
        self.samples = []  # 💡 left/right를 각각 독립 sample로 저장
        self.skipped = 0
        self.skipped_info = []

        dicom_files = sorted(list(self.images_dir.glob("**/T2/*.dcm")))
        for dicom_path in dicom_files:
            try:
                dicom_image = pydicom.dcmread(str(dicom_path))
                patient_id = int(dicom_image.PatientID)
                slice_number = int(dicom_image.InstanceNumber)

                row = self.labels[
                    (self.labels['patient_id'] == patient_id) &
                    (self.labels['slice_number'] == slice_number)
                ]

                if row.empty:
                    self.skipped += 1
                    self.skipped_info.append((patient_id, slice_number))
                    continue

                row = row.iloc[0]
                left_box = list(map(int, ast.literal_eval(row['left_box'])))
                right_box = list(map(int, ast.literal_eval(row['right_box'])))
                left_bme = int(row['left_bme'])
                right_bme = int(row['right_bme'])

                image = dicom_image.pixel_array
                image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
                image = Image.fromarray(image).resize((512, 512), Image.BILINEAR)
                image = np.array(image)

                if self.histogram_matching:
                    image = self.histogram_matching(image)
                image = self.clahe(image)

                # 좌우 crop
                left_crop = image[left_box[1]:left_box[3], left_box[0]:left_box[2]]
                right_crop = np.fliplr(image[right_box[1]:right_box[3], right_box[0]:right_box[2]]).copy()

                left_crop_resized = Image.fromarray(left_crop).resize((512, 512), Image.BILINEAR)
                right_crop_resized = Image.fromarray(right_crop).resize((512, 512), Image.BILINEAR)

                self.samples.append((left_crop_resized, left_bme))
                self.samples.append((right_crop_resized, right_bme))
            except Exception as e:
                self.skipped += 1
                continue

    def set_mode(self, mode):
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        self.mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

        image = transform(image)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


if __name__ == "__main__":
    bbox_csv = 'data/Updated_BBox_Data.csv'
    images_dir = 'bme_classification/dataset'
    target_dicom_file = 'data/reference/reference_stir.dcm'

    dataset = MedicalImageDataset(
        csv_path=bbox_csv,
        images_dir=images_dir,
        mode='train',
        target_dicom_file=target_dicom_file
    )

    print(f"Total valid samples: {len(dataset)}")
    print(f"Total skipped (no label): {dataset.skipped}")
    print("Skipped samples (patient_id, slice_number):")
    for info in dataset.skipped_info:
        print(info)

    all_labels = []
    for i in range(len(dataset)):
        _, lbl = dataset[i]
        all_labels.append(int(lbl.item()))

    label_counts = Counter(all_labels)
    print(f"Label distribution: {label_counts}")
