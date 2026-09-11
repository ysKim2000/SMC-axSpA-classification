import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import torch
from train import (CLAHE, HistogramMatching, MinMaxNormalize,
                                     MultiModalModel, PatientSliceDataset,
                                     create_transforms)
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


def evaluate_model(model, test_loader, device, label_csv, save_confusion_matrix_path=None, save_auc_roc_path=None, save_filtered_csv_path=None, save_attention_weights_csv=None):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    patient_ids = []
    attention_data = []  # CSV 파일로 저장할 데이터를 담을 리스트

    with torch.no_grad():
        for t1, t2, labels_fda, labels_bme, patient_ids_batch in tqdm(test_loader, desc='Testing'):
            labels_fda = labels_fda.unsqueeze(1).float().to(device)
            labels_bme = labels_bme.float().to(device)
            t1, t2 = t1.to(device), t2.to(device)
            outputs, attn_weights = model(t1, t2, labels_bme)  # Attention 가중치 가져오기
            outputs = torch.sigmoid(outputs)
            y_true.extend(labels_fda.cpu().numpy())
            y_pred.extend((outputs > 0.8).cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
           
            # Attention 가중치와 patient_id, label_fda, label_bme를 함께 저장
            attn_weights = attn_weights.cpu().numpy()  # Tensor를 numpy 배열로 변환
            labels_bme = labels_bme.cpu().numpy()  # label_bme를 numpy로 변환
            
            for idx, patient_id in enumerate(patient_ids_batch):
                label_fda_value = labels_fda[idx].item()  # 양성(1) 또는 음성(0) 라벨
                for slice_idx, weight in enumerate(attn_weights[idx]):
                    attention_data.append({
                        "patient_id": str(patient_id).zfill(7),
                        "slice_index": slice_idx,
                        "attention_weight": weight,
                        "label_fda": label_fda_value,
                        "label_bme": labels_bme[idx, slice_idx]  # 각 슬라이스별 label_bme 값 추가
                    })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Attention 가중치를 CSV로 저장
    if save_attention_weights_csv:
        df_attention = pd.DataFrame(attention_data)
        df_attention.to_csv(save_attention_weights_csv, index=False)
        print(f"Attention weights per slice saved to {save_attention_weights_csv}")

    # Attention 가중치를 양성/음성별로 시각화
    if attention_data:
        plot_attention_weights(attention_data, save_path="03_AxSpA_Classification/result/attention_weights_per_slice.png")
    else:
        print("No attention weights were collected. Skipping visualization.")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC Score: {auc:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if save_confusion_matrix_path:
        plt.savefig(save_confusion_matrix_path)
    plt.show()

    # AUROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if save_auc_roc_path:
        plt.savefig(save_auc_roc_path)

    # Load label_csv to add predictions
    df = pd.read_csv(label_csv)
    df['patient_id'] = df['patient_id'].astype(str).str.zfill(7)

    # Add predictions to the original dataframe
    df['prediction'] = df['patient_id'].map(dict(zip(patient_ids, y_pred.flatten())))

    # Filter the dataframe to only include test patients
    if save_filtered_csv_path:
        filtered_df = df[df['patient_id'].isin(patient_ids)]
        filtered_df.to_csv(save_filtered_csv_path, index=False)
        print(f"Filtered test data saved to {save_filtered_csv_path}")

    return accuracy, sensitivity, specificity, auc


def plot_attention_weights(attention_data, save_path="attention_weights_per_slice.png"):
    # DataFrame으로 변환
    df_attention = pd.DataFrame(attention_data)

    # 양성(1)과 음성(0) 데이터로 분리한 후, 각 slice_index별로 평균 attention weight 계산
    # df_attention['attention_weight'] = pd.to_numeric(df_attention['attention_weight'], errors='coerce')
    
    print("df_attention: ", df_attention)
    
    positive_weights = df_attention[df_attention['label_fda'] == 1.0].groupby('slice_index')['attention_weight'].mean()
    negative_weights = df_attention[df_attention['label_fda'] == 0.0].groupby('slice_index')['attention_weight'].mean()

    print("positive_weights: ", positive_weights)
    print("negative_weights: ", negative_weights)
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(positive_weights.index, positive_weights.values.mean(), label='Positive (1)', color='blue')
    plt.plot(negative_weights.index, negative_weights.values.mean(), label='Negative (0)', color='red')
    plt.xlabel("Slice Index")
    plt.ylabel("Average Attention Weight")
    plt.title("Average Attention Weights per Slice (Positive vs. Negative)")
    plt.legend()

    # 이미지 파일로 저장
    plt.savefig(save_path)
    plt.show()
    print(f"Attention weights visualization saved to {save_path}")


def create_dataloaders(dicom_dir, label_csv, bme_csv, transform_t1, transform_t2, batch_size=1):
    test_dataset = PatientSliceDataset(dicom_dir, label_csv, bme_csv, transform_t1=transform_t1, transform_t2=transform_t2)
    print(f"Dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_labels = [label.item() for _, _, label, _, _ in test_loader.dataset]
    print("Test set class distribution: ", Counter(test_labels))

    return test_loader


if __name__ == "__main__":
    # Test data paths and parameters
    dicom_dir = "data/axspa_classification_external_sliced"
    label_csv = "data/axSpA_external_label.csv"

    # BME CSV paths
    bme_csv_paths = [
        "bme_classification/inference_results/external_full_slice_predictions.csv" 
    ]

    batch_size = 4

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1,0"
    model_types = ['convnext_base']

    # Load reference images for T1 and T2
    reference_image_t1_path = "data/reference/reference_fst1.dcm"
    reference_image_t2_path = "data/reference/reference_stir.dcm"

    reference_image_t1 = pydicom.dcmread(reference_image_t1_path).pixel_array.astype(np.uint8)
    reference_image_t2 = pydicom.dcmread(reference_image_t2_path).pixel_array.astype(np.uint8)

    reference_image_t1 = np.array(Image.fromarray(reference_image_t1).resize((512, 512), resample=Image.BILINEAR))
    reference_image_t2 = np.array(Image.fromarray(reference_image_t2).resize((512, 512), resample=Image.BILINEAR))

    transform_test_t1, transform_test_t2 = create_transforms(reference_image_t1, reference_image_t2, apply_augmentations=False)

    for model_type in model_types:
        print("Model: ", model_type)

        checkpoint_filename = f'axSpA_classification/check_points/20250519/{model_type}_HybridFocalTverskyBCE_v10.pt'
        # checkpoint_filename = f'03_AxSpA_Classification/ckpt_mlp_pca_20241107/resnet34/resnet34_pca_multimodal_T1T2_checkpoint.pt'
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiModalModel(model_type=model_type, hidden_dim=512).to(device)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # Load the best checkpoint
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate the model with each BME CSV path
        for bme_csv in bme_csv_paths:
            print(f"Evaluating with BME CSV: {bme_csv}")
            test_loader = create_dataloaders(dicom_dir, label_csv, bme_csv, transform_test_t1, transform_test_t2, batch_size=batch_size)

            # Save confusion matrix and filtered CSV with patient IDs from the test set
            confusion_matrix_save_path = f'03_AxSpA_Classification/result/confusion_matrix/_confusion_matrix_external_test_{model_type}_{os.path.basename(bme_csv).split(".")[0]}.png'
            save_auc_roc_save_path = f'03_AxSpA_Classification/result/auroc/_auc_roc_external_test_{model_type}_{os.path.basename(bme_csv).split(".")[0]}.png'
            filtered_csv_save_path = f'03_AxSpA_Classification/result/_filtered_external_test_data_{model_type}_{os.path.basename(bme_csv).split(".")[0]}.csv'
            save_attention_weights_path = f'03_AxSpA_Classification/result/attention_weight_external_test_{model_type}_{os.path.basename(bme_csv).split(".")[0]}.csv'
           
            evaluate_model(model, test_loader, device, label_csv, save_confusion_matrix_path=confusion_matrix_save_path, save_auc_roc_path=save_auc_roc_save_path, save_filtered_csv_path=filtered_csv_save_path, save_attention_weights_csv=save_attention_weights_path)