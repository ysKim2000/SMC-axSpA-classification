import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from dataset import MedicalImageDataset  # 사용자 정의 dataset.py에서 import


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
        model = getattr(models, self.model_type)(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
        return model

    def _get_resnet(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    def _get_densenet(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    def _get_wide_resnet(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    def _get_resnext(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    def _get_efficientnet(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model

    def _get_vit(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
        return model

    def _get_swin_transformer(self):
        name_parts = self.model_type.split('_')
        weight_enum = getattr(models, f"{name_parts[0].capitalize()}_{name_parts[1].capitalize()}_Weights").DEFAULT
        model = getattr(models, self.model_type)(weights=weight_enum)
        model.head = nn.Linear(model.head.in_features, 1)
        return model

    def _get_convnext(self):
        model = getattr(models, self.model_type)(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
        return model

    def get_model(self):
        return self.model


def collate_fn(batch):
    images, labels = zip(*batch)  # batch: list of (image, label)
    return torch.stack(images), torch.stack(labels)



if __name__ == "__main__":
    # === 설정 ===
    csv_path = 'data/Updated_BBox_Data.csv'
    images_dir = 'bme_classification/dataset'
    target_dicom = 'data/reference/reference_stir.dcm'
    model_type = 'convnext_base'
    batch_size = 4

    # === 데이터셋 로딩 (소량만 사용) ===
    full_dataset = MedicalImageDataset(csv_path, images_dir, mode='train', target_dicom_file=target_dicom)
    small_dataset = Subset(full_dataset, list(range(4)))  # 디버깅용으로 4개만 추출
    loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # === 모델 로딩 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassificationModel(model_type, input_channels=6).get_model().to(device)
    model.eval()

    # === 테스트 추론 ===
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

        print("Logits:", outputs.cpu().numpy())
        print("Predictions:", preds.cpu().numpy())
        print("Ground Truth:", labels.cpu().numpy())
        break
