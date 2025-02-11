import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix

def read_csv_files(snr, pattern='anomaly_score_*.csv', test=False):
    if test:
        return glob.glob(os.path.join(snr, 'test', pattern))
    else:
        return glob.glob(os.path.join(snr, pattern))

# 파일 이름에서 레이블을 결정하는 함수
def get_label(filename):
    if 'abnormal' in filename:
        return 1
    else:
        return 0

def get_machine_id(filename):
    parts = filename.split('_')
    return f"{parts[-3]}_{parts[-1].split('.')[0]}"  # e.g., "slider_id_04"

def get_labels_and_scores(csv_files):
    actual_labels = []
    anomaly_scores = []
    machine_ids = []
    
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        df.columns = ['filename', 'anomaly_score']
        df['label'] = df['filename'].apply(get_label)
        df['machine_id'] = file.split('/')[-1]  # 파일 이름에서 machine_id 추출
        actual_labels.extend(df['label'].tolist())
        anomaly_scores.extend(df['anomaly_score'].tolist())
        machine_ids.extend([get_machine_id(file)] * len(df))
    
    return pd.Series(actual_labels), pd.Series(anomaly_scores), pd.Series(machine_ids)

def calculate_metrics(actual_labels, anomaly_scores):
    fpr, tpr, thresholds = roc_curve(actual_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    predicted_labels = (anomaly_scores > optimal_threshold).astype(int)
    f1 = f1_score(actual_labels, predicted_labels)
    return predicted_labels, fpr, tpr, optimal_idx, roc_auc, optimal_threshold, f1

def draw_confusion_matrix(actual_labels, predicted_labels, snr, machine_id, test=False):
    cm = confusion_matrix(actual_labels, predicted_labels, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=['normal', 'abnormal'],
                yticklabels=['normal', 'abnormal'], annot_kws={"size":20})
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)

    if test:
        plt.title(f'{snr}/{machine_id}/test', pad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./per_id/{snr}/testfig/confusion_matrix_{snr}_{machine_id}.png')
    else:
        plt.title(f'{snr}/{machine_id}', pad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./per_id/{snr}/confusion_matrix_{snr}_{machine_id}.png')
    plt.close()

def draw_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, snr, machine_id, test=False):
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right")

    if test:
        plt.title(f'{snr}/{machine_id}/test', pad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./per_id/{snr}/testfig/ROC_curve_{snr}_{machine_id}.png')
    else:
        plt.title(f'{snr}/{machine_id}', pad=20, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./per_id/{snr}/ROC_curve_{snr}_{machine_id}.png')
    
    plt.close()

def process_machine_id(snr, machine_id, actual_labels, anomaly_scores, test=False):
    print(f'Processing {machine_id} in {snr}...')
    results = calculate_metrics(actual_labels, anomaly_scores)
    predicted_labels, fpr, tpr, optimal_idx, roc_auc, optimal_threshold, f1 = results
    
    print(f'Optimal threshold: {optimal_threshold}')
    print(f'F1 Score: {f1}')
    
    draw_confusion_matrix(actual_labels, predicted_labels, snr, machine_id, test)
    draw_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, snr, machine_id, test)

def process_snr(snr, test=False):
    if not os.path.isdir('./per_id'):
        os.mkdir('./per_id')
    if not os.path.isdir(f'./per_id/{snr}'):
        os.mkdir(f'./per_id/{snr}')
    if not os.path.isdir(f'./per_id/{snr}/testfig'):
        os.mkdir(f'./per_id/{snr}/testfig')
    
    csv_files = read_csv_files(snr, test=test)
    actual_labels, anomaly_scores, machine_ids = get_labels_and_scores(csv_files)
    # print(machine_ids)
    
    unique_machine_ids = machine_ids.unique()
    
    for machine_id in unique_machine_ids:
        mask = machine_ids == machine_id
        process_machine_id(snr, machine_id, actual_labels[mask], anomaly_scores[mask], test)

def main():
    snr_list = ['6_dB', '-6_dB', '0_dB']
    for snr in snr_list:
        process_snr(snr, test=False)

if __name__ == "__main__":
    main()
