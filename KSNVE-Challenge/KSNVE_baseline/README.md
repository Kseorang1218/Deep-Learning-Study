
# KSNVE AI Challenge

이 문서는 한국소음진동공학회 제1회 소음진동 AI 챌린지의 baseline system에 대한 문서이다.



## Description

본 소음진동 AI 챌린지의 목적은 가변 속도 조건 하에서 진동 데이터를 활용한 이상 신호를 감지하는 것이다.



Baseline system은 총 4개의 python code로 이루어져 있다.

- ```preprocess.py```: 데이터 전처리 코드

- ```train.py```: 모델 학습 코드

- ```eval.py```: 학습된 모델의 결함 별 성능을 추출하는 코드

- ```test.py```: 평가 데이터에 대한 anomaly score를 추출하는 코드


## Baseline system

Baseline system은 spectrogram 기반의 AutoEncoder 모델을 활용한다.
AutoEncoder는 데이터의 압축 및 복원 과정을 통해 특징을 학습하며, 비정상 데이터는 데이터를 복원하지 못하는 것을 기대해 이상 진단을 수행한다.
입력 spectrogram $X = \{ X_t \}_{t=1}^T$ 이고, $X_t \in \mathbb{R}^{F}$ 이며, 이 때 $T$와 $F$는 각각 spectrogram의 time frame과 frequency bin 개수이다. AutoEncoder 모델을 $\psi_{\phi}$라 할 때, 각 데이터에 대한 anomaly score는 아래와 같다.

$\text{Score}(X) = \frac{1}{TF} \sum_{t=1}^T \|X_t - \psi_{\phi}(X_t) \|_2^2$




![alt text](model.png)



## Dataset

데이터셋은 크게 train, eval, test로 나누어져 있다.

- train: 모델의 학습 시에 사용하는 데이터셋으로, 정상 데이터 1470개로 구성되어 있다.
- eval: 모델의 평가 시에 사용하는 데이터셋으로, 정상 데이터 630개, 결함 3종(inner, outer, ball)별 630개씩 총 2520개로 구성되어 있다.
- test: 모델의 평가 시에 사용하는 데이터셋으로, 정상과 결함 데이터 총 6331개로 구성되어 있다.





## Usage


**1. 데이터셋 다운로드**

학습에 사용할 데이터셋을 아래 링크에서 다운받을 수 있다.

https://drive.google.com/drive/folders/1NEb2vchotohz1CUvVOS2hvo_V7nG3EoQ?usp=sharing




**2. 데이터셋 구성**

데이터셋은 아래 폴더들로 구성되어 있다.

    - ./data
      - /train
        - /vibration_normal_0_0.csv
        - /vibration_normal_0_1.csv
        - ...
        - /vibration_normal_4_269.csv
  
      -  /eval
         - /vibration_ball_0_0.csv
         - /vibration_ball_0_1.csv
         - ...
         - /vibration_ball_2_29.csv
         - /vibration_inner_0_0.csv
         - ...
         - /vibration_inner_2_29.csv
         - /vibration_normal_4_270.csv
         - ...
         - /vibration_normal_6_299.csv
         - /vibration_outer_0_0.csv
         - ...
         - /vibration_outer_2_29.csv

      -  /test
         - /test_0000.csv
         - /test_0001.csv
         - ...
         - /test_6330.csv



다운받은 train, eval, test 데이터셋을 각각 './data/train', './data/eval', './data/test' 에 저장한다.



**3. Parameter 설정**

학습과 관련된 파라미터는 `param.yaml` 에 저장되어 있다.







**4. 코드 실행**

4.1. Preprocessing

각 데이터셋을 전처리하는 코드이다.
실행 명령은 아래와 같다.

```
$ python preprocess.py
```

전처리된 데이터는 './dataset' 폴더에 저장된다.


4.2. Model training

Baseline 모델을 학습하는 코드이다.
코드 실행 명령은 아래와 같다.

```
$ python train.py
```

학습이 완료된 모델은 './model' 폴더에 저장된다.


4.3. Model evaluation


eval 데이터셋으로 모델의 성능 및 anomaly score를 추출하는 코드이다.
코드 실행 명령은 아래와 같다.

```
$ python eval.py
```

eval 데이터에 대한 anomaly score 결과는 './results/eval_score.csv' 로 저장된다.


4.4. Model test

test 데이터셋으로 anomaly score를 추출하는 코드이다.
코드 실행 명령은 아래와 같다.


```
$ python test.py
```

test 데이터에 대한 anomaly score 결과는 './results/test_score.csv' 로 저장된다.





## Submission

![alt text](overview.png)




Challenge 참여를 위해 참가자들은 총 3종의 파일을 제출해야 한다.

- eval, test 데이터에 대한 anomaly score 파일 (eval_score.csv, test_score.csv)
- 학습 및 평가가 재현 가능한 코드
- Technical report

학습이 완료된 모델을 eval.py와 test.py로 추출한 anomaly score 파일을 제출해야 한다.
이번 challenge는 eval 데이터셋으로 추출한 결과 (eval_score.csv), test 데이터셋으로 추출한 결과 (test_score.csv)를 기준으로 참가자들의 모델 성능을 평가한다.


학습 및 평가 과정이 재현이 가능한 코드를 제출해야 한다. 제출한 코드로 모델의 학습과 함께 test data에 대한 score 결과를 추출할 수 있어야 한다.


Technical report는 제안하는 이상 진단 모델의 구조, 모델의 학습, 이상 진단 방법에 대한 설명이 포함되어야 한다. 1~2페이지 내외의 자유 형식의 문서를 제출해야 한다.






## Requirements

~~~
pip install torch
pip install librosa
pip install numpy
pip install tqdm
~~~
