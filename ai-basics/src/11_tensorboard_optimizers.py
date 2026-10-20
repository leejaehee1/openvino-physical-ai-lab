# %% [markdown]
# # 11. TensorBoard — 손실·정확도, 학습률과 옵티마이저 비교
#
# > 인공지능 기초 실습 과정 · 단원 11/15
#
# **학습 목표**
# - TensorBoard로 여러 실험을 **겹쳐서 비교** 하는 법을 익힌다.
# - 학습률이 너무 작거나 클 때 손실 곡선이 어떻게 달라지는지 확인한다 (9단원 1차원 실험의 신경망 버전).
# - SGD, 모멘텀, RMSprop, Adam 옵티마이저의 차이를 결과로 비교한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 실험 관리, 학습률, 옵티마이저, 검증 데이터 |
# | 따라하기 | 같은 모델 · 같은 데이터로 학습률 4가지 → 옵티마이저 4가지 → TensorBoard에서 겹쳐 보기 |
# | 연습문제 | 배치 크기 비교 실험 설계 |

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow import keras

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
LOG_ROOT = "../logs/11_compare"

# %% [markdown]
# ## 1. 개념
#
# ### 공정한 비교의 규칙
# **한 번에 하나만 바꾼다.** 데이터, 모델 구조, 초기값(시드), 에포크 수는 고정하고 비교할 것 하나만 바꿉니다. 실험마다 로그 폴더 이름을 다르게 주면 TensorBoard가 자동으로 겹쳐 그려 줍니다.
#
# ### 옵티마이저 — 경사하강법의 변형들
#
# | 옵티마이저 | 아이디어 | 비유 |
# |---|---|---|
# | **SGD** | 기울기 반대 방향으로 학습률만큼 (9단원) | 매 걸음 발밑 경사만 보고 걷기 |
# | **모멘텀** | 이전에 움직이던 방향을 이어 감 | 굴러가는 공 — 작은 턱은 넘어감 |
# | **RMSprop** | 많이 흔들리는 방향은 보폭을 줄임 | 길 상태에 맞춰 보폭 조절 |
# | **Adam** | 모멘텀 + RMSprop | 지금 가장 많이 쓰는 기본값 |
#
# ### 검증 데이터
# 학습 데이터의 손실만 보면 과대적합을 놓칩니다. 학습에 쓰지 않는 **검증 데이터** 의 손실(`val_loss`)을 함께 봅니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 데이터와 모델 — 7·8단원의 초승달

# %%
X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)


def make_model():
    keras.utils.set_random_seed(42)               # 모든 실험에서 같은 초기 가중치
    return keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


def run(name, optimizer, epochs=60, batch_size=32):
    model = make_model()
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    h = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0,
                  callbacks=[keras.callbacks.TensorBoard(os.path.join(LOG_ROOT, name))])
    return pd.DataFrame(h.history)

# %% [markdown]
# ### 2-2. 실험 1 — 학습률 (Adam 고정)

# %%
lr_results = {f"lr={lr}": run(f"lr/{lr}", keras.optimizers.Adam(learning_rate=lr))
              for lr in [0.0001, 0.001, 0.01, 0.5]}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for name, h in lr_results.items():
    axes[0].plot(h["loss"], label=name); axes[1].plot(h["val_accuracy"], label=name)
axes[0].set_title("학습 손실"); axes[1].set_title("검증 정확도")
for a in axes: a.legend(); a.set_xlabel("epoch")
plt.show()
pd.DataFrame({k: {"마지막 학습 손실": v["loss"].iloc[-1], "마지막 검증 정확도": v["val_accuracy"].iloc[-1]}
              for k, v in lr_results.items()}).T.round(3)

# %% [markdown]
# - 0.0001: 거의 안 움직임 (9단원 "너무 작은 학습률")
# - 0.001 ~ 0.01: 안정적으로 내려감
# - 0.5: 출렁이거나 나쁜 지점에 머묾 (9단원 "너무 큰 학습률")
#
# ### 2-3. 실험 2 — 옵티마이저 (학습률은 각각의 흔한 기본값)

# %%
opt_results = {
    "SGD":     run("opt/SGD", keras.optimizers.SGD(learning_rate=0.05)),
    "모멘텀":   run("opt/Momentum", keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)),
    "RMSprop": run("opt/RMSprop", keras.optimizers.RMSprop(learning_rate=0.005)),
    "Adam":    run("opt/Adam", keras.optimizers.Adam(learning_rate=0.005)),
}
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for name, h in opt_results.items():
    axes[0].plot(h["loss"], label=name); axes[1].plot(h["val_loss"], label=name)
axes[0].set_title("학습 손실"); axes[1].set_title("검증 손실")
for a in axes: a.legend(); a.set_xlabel("epoch")
plt.show()

# %% [markdown]
# ### 2-4. TensorBoard에서 겹쳐 보기
#
# 1. **http://localhost:6006** 접속 (Docker Compose의 TensorBoard)
# 2. 왼쪽 실행(run) 목록에서 `11_compare/lr/...`만 골라 **학습률 비교**, `11_compare/opt/...`만 골라 **옵티마이저 비교**
# 3. **Smoothing** 막대로 곡선을 부드럽게 해서 추세를 봅니다
# 4. `train`(학습)과 `validation`(검증) 곡선이 벌어지기 시작하는 지점 = 과대적합의 시작
#
# 노트북 안에서 보려면 아래 주석을 풀고 실행합니다.

# %%
# %load_ext tensorboard
# %tensorboard --logdir ../logs/11_compare

# %% [markdown]
# **관찰 질문**
# 1. 모멘텀을 더한 SGD는 순수 SGD보다 얼마나 빨리 내려가나요?
# 2. 학습 손실은 계속 내려가는데 검증 손실이 올라가기 시작하는 실험이 있나요? 그 지점에서 무엇을 해야 할까요?
# 3. "Adam이 항상 최고"라고 말할 수 있을까요? 이 실험만으로 결론 내리기에 부족한 점은?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** Adam(학습률 0.005)을 고정하고 **배치 크기** 를 8, 32, 128, 700으로 바꾸는 실험을 설계·실행하세요. 로그 폴더는 `batch/8` 처럼 나눕니다.
# - 손실 곡선의 출렁임과 1 에포크에 걸리는 시간은 어떻게 다른가요?
#
# **문제 2.** 과대적합을 일부러 만들어 보세요. 학습 데이터를 100개로 줄이고 은닉층을 128-128로 키운 뒤 300 에포크 학습시켜, TensorBoard에서 학습/검증 곡선이 벌어지는 것을 확인하세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# import time
# for bs in [8, 32, 128, 700]:
#     t0 = time.time()
#     h = run(f"batch/{bs}", keras.optimizers.Adam(0.005), batch_size=bs)
#     print(bs, "배치:", round((time.time() - t0) / 60 * 1000), "ms/epoch,", "검증 정확도", round(h["val_accuracy"].iloc[-1], 3))
# # 배치가 작으면 1 에포크에 갱신을 많이 해서 빨리 내려가지만 출렁이고, 크면 매끈하지만 느리게 내려간다
#
# keras.utils.set_random_seed(0)
# big = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(128, activation="relu"),
#                         keras.layers.Dense(128, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
# big.compile(keras.optimizers.Adam(0.005), "binary_crossentropy", metrics=["accuracy"])
# big.fit(X_tr[:100], y_tr[:100], validation_data=(X_val, y_val), epochs=300, verbose=0,
#         callbacks=[keras.callbacks.TensorBoard(os.path.join(LOG_ROOT, "overfit"))])
# ```
# </details>
