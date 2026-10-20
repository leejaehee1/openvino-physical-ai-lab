# %% [markdown]
# # 12. MNIST 이미지 분류 — 데이터 전처리와 MLP 학습
#
# > 인공지능 기초 실습 과정 · 단원 12/15
#
# **학습 목표**
# - MNIST 데이터의 구조(28×28 흑백, 라벨 0~9)를 이해한다.
# - 정규화, 펼치기(flatten), 원-핫 인코딩 같은 **전처리** 를 직접 한다.
# - 소프트맥스와 교차 엔트로피로 10개 클래스를 분류하는 MLP를 학습시키고 TensorBoard로 기록한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | MNIST, 전처리, 소프트맥스, 교차 엔트로피 |
# | 따라하기 | 데이터 살펴보기 → 전처리 → MLP 학습 → 평가 → 틀린 예 보기 → 저장 |
# | 연습문제 | 은닉층 크기와 드롭아웃 |

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from tensorflow import keras

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
EPOCHS = 5            # 수업 시간이 넉넉하면 10~15로 늘려 보세요

# %% [markdown]
# ## 1. 개념
#
# ### MNIST
# | 항목 | 내용 |
# |---|---|
# | 내용 | 손으로 쓴 숫자 0~9 (1998, 르쿤 등) |
# | 이미지 | 28 × 28 흑백, 값 0~255 |
# | 규모 | 학습 60,000장 · 테스트 10,000장 |
# | 의미 | 딥러닝의 "Hello, World" (원래 강의의 TF 1.x 예제는 학습 55,000 + 검증 5,000으로 나눠 썼음) |
#
# ### 소프트맥스 — 점수를 확률로
# 출력 뉴런 10개의 점수(로짓)를 **합이 1인 확률** 로 바꿉니다.
#
# $$ \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_j e^{z_j}} $$
#
# 예: 점수 [2.0, 1.0, 0.1] → 확률 약 [0.66, 0.24, 0.10]

# %%
z = np.array([2.0, 1.0, 0.1])
print("소프트맥스:", (np.exp(z) / np.exp(z).sum()).round(2))

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 데이터 불러와 살펴보기

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("학습:", x_train.shape, y_train.shape, " 테스트:", x_test.shape, y_test.shape)
print("자료형:", x_train.dtype, " 값 범위:", x_train.min(), "~", x_train.max())

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, lab in zip(axes.ravel(), x_train[:16], y_train[:16]):
    ax.imshow(img, cmap="gray"); ax.set_title(f"정답 {lab}"); ax.axis("off")
plt.show()

# %%
print("숫자 하나를 숫자 행렬로 (1단원의 '이미지 = 숫자 배열'):")
print(np.array2string(x_train[0][4:24, 4:24] // 26, max_line_width=200))   # 0~9로 줄여서 출력

pd.Series(y_train).value_counts().sort_index().plot.bar(figsize=(8, 2.5), title="학습 데이터의 숫자별 개수")
plt.show()

# %% [markdown]
# ### 2-2. 전처리
#
# | 단계 | 이유 | 코드 |
# |---|---|---|
# | 정규화 | 0~255 → 0~1. 값이 크면 학습이 불안정 (6단원 브로드캐스팅) | `x / 255.0` |
# | 펼치기 | MLP는 1차원 입력을 받음: 28×28 → 784 | `reshape(-1, 784)` |
# | 원-핫 인코딩 | 라벨 3 → `[0,0,0,1,0,0,0,0,0,0]` | `to_categorical` |

# %%
x_train_f = (x_train / 255.0).reshape(-1, 784).astype("float32")
x_test_f = (x_test / 255.0).reshape(-1, 784).astype("float32")
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh = keras.utils.to_categorical(y_test, 10)
print("입력:", x_train_f.shape, " 라벨:", y_train_oh.shape)
print("라벨", y_train[0], "→", y_train_oh[0])

# %% [markdown]
# > 펼치는 순간 "어떤 픽셀이 어떤 픽셀 옆에 있었는지"라는 **공간 정보가 사라집니다**. 이것이 MLP의 한계이고, 13단원 CNN이 푸는 문제입니다.
#
# ### 2-3. 모델 만들기 — 784 → 128 → 10 (8단원에서 파라미터를 세 본 그 구조)

# %%
keras.utils.set_random_seed(0)
mlp = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),                     # 학습 때 20%를 무작위로 끔 (과대적합 방지)
    keras.layers.Dense(10, activation="softmax"),
], name="mnist_mlp")
mlp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
mlp.summary()

# %% [markdown]
# ### 2-4. 학습 — TensorBoard 기록, 학습 데이터의 10%는 검증용

# %%
tb = keras.callbacks.TensorBoard(log_dir="../logs/12_mnist/mlp_128")
history = mlp.fit(x_train_f, y_train_oh, epochs=EPOCHS, batch_size=128, validation_split=0.1, callbacks=[tb], verbose=2)

# %%
h = pd.DataFrame(history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
h[["loss", "val_loss"]].plot(ax=axes[0], title="손실"); h[["accuracy", "val_accuracy"]].plot(ax=axes[1], title="정확도")
plt.show()

# %% [markdown]
# ### 2-5. 테스트 데이터로 평가

# %%
test_loss, test_acc = mlp.evaluate(x_test_f, y_test_oh, verbose=0)
print(f"테스트 정확도: {test_acc:.4f}  (10,000장 중 약 {round((1 - test_acc) * 10000)}장 틀림)")

probs = mlp.predict(x_test_f[:1], verbose=0)[0]
pd.DataFrame({"숫자": range(10), "확률": probs.round(4)}).set_index("숫자").T

# %% [markdown]
# ### 2-6. 무엇을 틀렸나

# %%
pred = mlp.predict(x_test_f, verbose=0).argmax(axis=1)
wrong = np.where(pred != y_test)[0]
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, i in zip(axes.ravel(), wrong[:16]):
    ax.imshow(x_test[i], cmap="gray"); ax.set_title(f"정답 {y_test[i]} / 예측 {pred[i]}", fontsize=9); ax.axis("off")
plt.show()

# %% [markdown]
# 사람이 봐도 헷갈리는 글씨가 많습니다. 어떤 숫자끼리 많이 헷갈리는지는 15단원에서 **혼동 행렬** 로 정리합니다.
#
# ### 2-7. 저장 (15단원에서 CNN과 비교)

# %%
os.makedirs("../models", exist_ok=True)
mlp.save("../models/mnist_mlp.keras")
print("저장: ../models/mnist_mlp.keras")

# %% [markdown]
# **관찰 질문**
# 1. 학습 정확도와 검증 정확도의 차이는 몇 %p인가요? 과대적합의 신호가 있나요?
# 2. 정규화(`/ 255.0`)를 빼고 학습시키면 어떻게 될까요?
# 3. 숫자 이미지를 오른쪽으로 3픽셀 옮기면 MLP 입장에서 784개 입력이 얼마나 바뀔까요? (→ 13단원 CNN의 장점)

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 은닉층을 32, 128, 512개로 바꿔 테스트 정확도와 파라미터 수를 표로 비교하세요. 로그는 `../logs/12_mnist/mlp_32` 처럼 나눠 TensorBoard에서 겹쳐 보세요.
#
# **문제 2.** 드롭아웃을 빼고(또는 0.5로 키우고) 학습시켜, 학습/검증 곡선의 간격이 어떻게 변하는지 보세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# rows = []
# for units in [32, 128, 512]:
#     keras.utils.set_random_seed(0)
#     m = keras.Sequential([keras.Input(shape=(784,)), keras.layers.Dense(units, activation="relu"),
#                           keras.layers.Dropout(0.2), keras.layers.Dense(10, activation="softmax")])
#     m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
#     m.fit(x_train_f, y_train_oh, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=0,
#           callbacks=[keras.callbacks.TensorBoard(f"../logs/12_mnist/mlp_{units}")])
#     rows.append({"은닉": units, "파라미터": m.count_params(), "테스트 정확도": m.evaluate(x_test_f, y_test_oh, verbose=0)[1]})
# print(pd.DataFrame(rows))
# # 파라미터는 크게 늘지만 정확도 향상은 점점 작아진다
# ```
# </details>
