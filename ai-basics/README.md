# 인공지능 기초 실습 과정 — 픽셀에서 CNN까지

이미지가 숫자 배열이라는 것에서 출발해, 직접 만든 3×3 필터 → 퍼셉트론 → 다층 신경망 → 역전파 → TensorFlow → CNN까지 **원리를 직접 구현하며** 따라가는 15단원 실습 과정입니다.
각 단원은 독립된 Jupyter 노트북이라, 진도에 맞춰 하루 또는 이틀 이상으로 유연하게 운영합니다.

> 바탕: 2017–2018 SK하이닉스 인공지능 실무 강의 자료(이재희)를 TensorFlow 2 / Keras 기준으로 새로 구성.

## 운영 원칙

- **시간보다 이해가 우선.** 하루에 끝나지 않으면 다음 날 이어서 진행합니다. 단원을 건너뛰지 않습니다.
- **개념 → 따라하기 → 관찰 질문 → 연습문제** 의 순서를 모든 단원에서 반복합니다 (원래 강의의 "개념 · 따라하기 · 연습문제 · 정답" 흐름).
- 같은 문제를 **두 번** 풉니다: NumPy로 직접(5·9단원) → TensorFlow로(10단원). 차이가 곧 프레임워크가 해 주는 일입니다.
- 3단원에서 **직접 만든 3×3 필터** 를 13단원에서 **CNN이 스스로 배운 커널** 과 나란히 비교합니다. 과정의 처음과 끝이 이어집니다.
- **TensorBoard** 는 첫 TensorFlow 모델(10단원)부터 끝까지 씁니다. **Seaborn** 은 데이터와 평가 결과를 그릴 때 씁니다.

## 15단원

| # | 주제 | 주요 실습 · 도구 | 노트북 |
|:---:|---|---|---|
| 1 | 디지털 이미지 기초 | 픽셀, 해상도, RGB, Grayscale, HSV, 히스토그램 | [01](notebooks/01_digital_image.ipynb) |
| 2 | 이미지 프로세싱 | 평활화, 블러, 샤프닝, Canny, 잡음 제거, 회전 | [02](notebooks/02_image_processing.ipynb) |
| 3 | 필터와 합성곱 | NumPy로 합성곱 구현, Sobel, Gaussian, 패딩·스트라이드 | [03](notebooks/03_filters_convolution.ipynb) |
| 4 | 신경망의 역사 | 퍼셉트론 → XOR → 홉필드 → 역전파 → 딥러닝, 홉필드 기억 복원 실습 | [04](notebooks/04_neural_network_history.ipynb) |
| 5 | 퍼셉트론 | NumPy로 AND · OR · NAND, 학습 규칙, 결정 경계, XOR의 벽과 2층 해결 | [05](notebooks/05_perceptron.ipynb) |
| 6 | Python 데이터 처리 | NumPy 배열·행렬 곱·브로드캐스팅, Pandas DataFrame, Seaborn | [06](notebooks/06_numpy_pandas.ipynb) |
| 7 | 머신러닝 기초 | Scikit-learn, 학습/테스트 분리, 선형 분류, 결정 경계, 표준화 | [07](notebooks/07_machine_learning_basics.ipynb) |
| 8 | 다층 신경망 | MLP 순전파, 가중치·편향·파라미터 수, 활성화 함수 | [08](notebooks/08_multilayer_perceptron.ipynb) |
| 9 | 신경망 학습 원리 | 손실 함수, 경사하강법, 학습률, **NumPy로 역전파 구현** | [09](notebooks/09_learning_backprop.ipynb) |
| 10 | TensorFlow 실습 | 텐서, 자동 미분, Keras로 XOR, TensorBoard 시작 | [10](notebooks/10_tensorflow_xor.ipynb) |
| 11 | TensorBoard | 학습률 · 옵티마이저 비교, 검증 손실, 과대적합 | [11](notebooks/11_tensorboard_optimizers.ipynb) |
| 12 | MNIST 이미지 분류 | 전처리(정규화·펼치기·원-핫), 소프트맥스, MLP 학습 | [12](notebooks/12_mnist_mlp.ipynb) |
| 13 | CNN | 합성곱·풀링·특징 맵, **배운 커널 vs 3단원 커널** | [13](notebooks/13_cnn.ipynb) |
| 14 | CNN 발전 과정 | LeNet-5 학습, AlexNet·VGG·ResNet 파라미터, 기울기 소실과 숏컷 | [14](notebooks/14_cnn_history.ipynb) |
| 15 | 모델 평가와 분석 | 혼동 행렬(Seaborn), 정밀도·재현율·F1, 확신도, MLP vs CNN | [15](notebooks/15_model_evaluation.ipynb) |

### 단원 사이의 연결

```text
1 이미지=숫자 ─▶ 2 영상 처리 ─▶ 3 필터·합성곱 ───────────────────────────┐
                                                                          ▼
4 역사 ─▶ 5 퍼셉트론(XOR의 벽) ─▶ 8 다층 신경망 ─▶ 9 역전파 ─▶ 10 TensorFlow ─▶ 12 MNIST ─▶ 13 CNN ─▶ 14 유명 모델 ─▶ 15 평가
              │                                                   │
              └──── 6 NumPy·Pandas, 7 Scikit-learn ────────────── 11 TensorBoard (10단원부터 계속)
```

## 실습 환경 — Docker Compose

교육생마다 같은 환경을 쓰도록 JupyterLab과 TensorBoard를 컨테이너로 띄웁니다.

```bash
cd ai-basics/docker
docker compose up -d --build
```

| 서비스 | 주소 | 하는 일 |
|---|---|---|
| JupyterLab | <http://localhost:8888> | 모든 실습 (`notebooks/` 폴더가 그대로 연결됨) |
| TensorBoard | <http://localhost:6006> | `logs/` 폴더의 학습 기록을 자동으로 보여 줌 |

- 이미지 안에 TensorFlow(CPU), Scikit-learn, OpenCV, Seaborn, 한글 글꼴(나눔고딕)이 들어 있고, **MNIST도 미리 받아 둡니다**(교실 네트워크가 느려도 됨).
- 노트북에서 만든 로그(`logs/`), 모델(`models/`), 커널(`data/`)은 호스트 폴더에 남습니다.
- 끝낼 때: `docker compose down`
- ⚠️ 토큰 없이 열리도록 설정했습니다. **자기 노트북(localhost)에서만** 쓰고, 공유 서버에 띄울 때는 토큰을 설정하세요.

Docker 없이 할 때:

```bash
python3 -m venv ~/ai-basics && source ~/ai-basics/bin/activate
pip install -r ai-basics/docker/requirements.txt
cd ai-basics/notebooks && jupyter lab
```

## 폴더 구성

```text
ai-basics/
├── notebooks/   # 교육생이 여는 노트북 15개 (출력 없이 배포)
├── src/         # 노트북 원본 (.py, percent 형식) — 수정은 여기서
├── tools/       # py2nb.py: src → notebooks 변환
├── docker/      # Dockerfile, docker-compose.yml, requirements.txt
├── data/ logs/ models/   # 실습 중 생기는 파일 (git에는 올리지 않음)
├── INSTRUCTOR.md         # 강사 가이드: 단원별 시간, 핵심 메시지, 막히는 지점
└── README.md
```

노트북을 고칠 때는 `src/`의 `.py`를 수정한 뒤 다시 만듭니다.

```bash
python3 ai-basics/tools/py2nb.py
```

## 검증

15개 노트북 모두 TensorFlow 2.21 · Scikit-learn 1.9 · OpenCV 5.0 환경에서 처음부터 끝까지 실행해 확인했습니다 (CPU, MNIST 학습은 에포크를 줄여 단원당 수십 초).
대표 결과: XOR 역전파 손실 0.343 → 0.0003, MNIST MLP 약 97.4% · CNN 약 98.2% (3 에포크), CNN 첫 층 커널과 Sobel X/Y의 상관계수 약 0.85 / 0.92.
