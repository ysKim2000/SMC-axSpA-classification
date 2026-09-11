import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, confusion_matrix, recall_score,
    roc_auc_score, roc_curve
)
from torch.utils.data import DataLoader

from train_fullslice_v2 import (
    MultiModalModel,
    PatientSliceDataset,
    create_transforms,
    variable_length_collate
)


def evaluate_model(
    model,
    test_loader,
    device,
    label_csv,
    save_confusion_matrix_path=None,
    save_auc_roc_path=None,
    save_filtered_csv_path=None,
    save_attention_weights_csv=None
):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    patient_ids = []
    attention_data = []

    with torch.no_grad():
        for t1, t2, labels_fda, labels_bme, mask, pids in tqdm(test_loader, desc='Testing'):
            t1, t2, labels_fda, labels_bme, mask = t1.to(device), t2.to(device), labels_fda.to(device), labels_bme.to(device), mask.to(device)

            outputs, attn_weights = model(t1, t2, labels_bme, mask=mask)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            truths = labels_fda.cpu().numpy().astype(int).flatten()

            y_true.extend(truths.tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs.tolist())
            patient_ids.extend([str(pid).zfill(7) for pid in pids])

            attn_np = attn_weights.cpu().numpy()  # [B, S, S]
            bme_np = labels_bme.cpu().numpy()
            for i, pid in enumerate(pids):
                for slice_idx, weight in enumerate(attn_np[i].mean(axis=0)):
                    attention_data.append({
                        'patient_id': str(pid).zfill(7),
                        'slice_index': slice_idx,
                        'attention_weight': float(weight),
                        'label_fda': int(truths[i]),
                        'label_bme': float(bme_np[i, slice_idx])
                    })
    # save attention weights
    if save_attention_weights_csv:
        df_attn = pd.DataFrame(attention_data)
        df_attn.to_csv(save_attention_weights_csv, index=False)
        print(f"Saved attention weights to {save_attention_weights_csv}")

    # compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_scores)

    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"AUC: {auc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_confusion_matrix_path:
        plt.savefig(save_confusion_matrix_path)
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    if save_auc_roc_path:
        plt.savefig(save_auc_roc_path)
    plt.show()

    # add predictions to label csv
    df = pd.read_csv(label_csv)
    df['patient_id'] = df['patient_id'].astype(str).str.zfill(7)
    df['prediction'] = df['patient_id'].map(dict(zip(patient_ids, y_pred)))
    if save_filtered_csv_path:
        df_test = df[df['patient_id'].isin(patient_ids)]
        df_test.to_csv(save_filtered_csv_path, index=False)
        print(f"Saved filtered predictions to {save_filtered_csv_path}")


if __name__ == '__main__':
    dicom_dir = 'axSpA_classification/dataset/test_fullslice'
    label_csv = 'data/AxSpA_label.csv'
    bme_csv   = 'data/Combined_BME_Labels.csv'
    checkpoint = 'axSpA_classification/check_points/20250525/convnext_large_modality_attention_classifier_v3.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_t1, transform_t2 = create_transforms(apply_augmentations=False)

    test_dataset = PatientSliceDataset(dicom_dir, label_csv, bme_csv, transform_t1=transform_t1, transform_t2=transform_t2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=variable_length_collate)

    model = MultiModalModel(model_type='convnext_large', hidden_dim=1024).to(device)
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])

    evaluate_model(
        model, test_loader, device, label_csv,  
        save_confusion_matrix_path='axSpA_classification/result/confusion_matrix/confusion_matrix_best.png',
        save_auc_roc_path='axSpA_classification/result/auroc/auc_roc_best.png',
        save_filtered_csv_path='axSpA_classification/result/filtered_test_data.csv',
        save_attention_weights_csv='axSpA_classification/result/attention_weight.csv'
    )