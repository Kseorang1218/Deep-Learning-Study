import os
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 이름에서 레이블을 결정하는 함수
def get_label(filename):
    if 'abnormal' in filename:
        return 1
    else:
        return 0

# CSV 파일이 있는 디렉토리
csv_directory = '.'

# result.csv를 제외한 모든 CSV 파일 목록 생성
csv_files = glob.glob(os.path.join(csv_directory, 'anomaly_score_*.csv'))

# 실제 레이블과 anomaly score를 저장할 리스트 초기화
actual_labels = []
anomaly_scores = []

# 각 CSV 파일 읽기
for file in csv_files:
    df = pd.read_csv(file, header=None)
    df.columns = ['filename', 'anomaly_score']
    df['label'] = df['filename'].apply(get_label)
    actual_labels.extend(df['label'].tolist())
    anomaly_scores.extend(df['anomaly_score'].tolist())

# print(df['label'].value_counts())
# 리스트를 sklearn과 호환되는 pandas Series로 변환
actual_labels = pd.Series(actual_labels)
anomaly_scores = pd.Series(anomaly_scores)

plt.hist(anomaly_scores, bins=50)
plt.xlabel('Anomaly Score')
plt.xlim([0,2])
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.savefig('./anomaly_score_distribution.png')  # 플롯을 파일로 저장

# 여러 임계값을 테스트하여 최적의 임계값을 찾는 예제
thresholds = np.arange(0.0, 5.0, 0.01)
f1_scores = []

for threshold in thresholds:
    predicted_labels = (anomaly_scores > threshold).astype(int)
    f1 = f1_score(actual_labels, predicted_labels)
    f1_scores.append(f1)
# print(f1_scores)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f'Optimal threshold: {optimal_threshold}, F1 score: {max(f1_scores)}')

# 최적의 임계값을 사용하여 점수를 이진화
predicted_labels = (anomaly_scores > optimal_threshold).astype(int)

# 최적의 임계값으로 F1 점수 계산
f1 = f1_score(actual_labels, predicted_labels)
print(f'최적 임계값 사용 시 F1 점수: {f1}')

# 혼동 행렬 생성
cm = confusion_matrix(actual_labels, predicted_labels, normalize='true')
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', xticklabels=['normal', 'abnormal'], 
            yticklabels=['normal', 'abnormal'], annot_kws={"size":20})
plt.xlabel('pred', fontsize=20)
plt.ylabel('actual',fontsize=20)

# 플롯을 파일로 저장
plt.savefig('./confusion_matrix.png')

# plt.show()는 주석 처리 또는 삭제
# plt.show()
