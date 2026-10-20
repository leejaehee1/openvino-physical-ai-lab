# %% [markdown]
# # 10. TensorFlow 실습 — Keras로 XOR 해결
#
# > 인공지능 기초 실습 과정 · 단원 10/15
#
# **학습 목표**
# - 텐서의 랭크(rank) · 모양(shape) · 자료형(dtype)을 다룬다.
# - TensorFlow가 기울기를 **자동으로** 계산하는 것(자동 미분)을 9단원의 손 계산과 비교한다.
# - 5·9단원에서 NumPy로 푼 XOR을 **Keras 몇 줄** 로 다시 풀고, 처음부터 **TensorBoard** 로 학습을 기록한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 텐서 · 자동 미분 · 모델/층/옵티마이저/손실 |
# | 따라하기 | 텐서 기초 → GradientTape → Keras XOR → TensorBoard 기록 → NumPy 버전과 비교 |
# | 연습문제 | OR · NAND도 같은 코드로 / 은닉층 바꾸기 |
#
# > 원래 강의(TensorFlow 1.x)의 `Session`, `placeholder`, `Saver`는 TensorFlow 2에서 사라졌습니다. 지금은 코드를 쓰는 즉시 실행되고(즉시 실행 모드), 모델 저장은 `model.save()` 한 줄입니다.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import font_manager
from tensorflow import keras

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
print("TensorFlow", tf.__version__)
LOG_ROOT = "../logs"

# %% [markdown]
# ## 1. 개념 — TensorFlow = Tensor + Flow
#
# | 용어 | 뜻 |
# |---|---|
# | **텐서** | n차원 배열 (6단원의 스칼라·벡터·행렬·텐서). NumPy 배열과 거의 같고 GPU에서 계산 가능 |
# | **자동 미분** | 계산 과정을 기록해 두었다가 기울기를 자동으로 구함 (9단원 역전파를 대신해 줌) |
# | **층 (Layer)** | `Dense` = 8단원의 $f(XW + b)$ 한 층 |
# | **모델 (Model)** | 층을 쌓은 것 |
# | **손실 · 옵티마이저** | 9단원의 손실 함수와 경사하강법 |
# | **TensorBoard** | 학습 과정을 웹 화면으로 보여 주는 도구 (기본 주소 `localhost:6006`) |

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 텐서 기초 — 랭크, 모양, 자료형

# %%
tensors = {
    "스칼라": tf.constant(3.0),
    "벡터": tf.constant([1.0, 2.0, 3.0]),
    "행렬": tf.constant([[1, 2], [3, 4]]),
    "3차원 텐서": tf.zeros((2, 28, 28)),
}
pd.DataFrame([{"이름": k, "랭크": len(t.shape), "모양": tuple(t.shape), "자료형": t.dtype.name} for k, t in tensors.items()])

# %%
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0], [1.0]])
print("행렬 곱 a @ b:\n", (a @ b).numpy())
print("NumPy로 바꾸기:", type(a.numpy()))
w = tf.Variable([[0.5], [-0.5]])          # 학습으로 바뀌는 값은 Variable
print("Variable:", w.numpy().ravel())

# %% [markdown]
# ### 2-2. 자동 미분 — 9단원 손 계산과 비교
# 9단원에서 $L(w) = (w-3)^2$ 의 기울기를 $2(w-3)$ 으로 손으로 구했습니다. TensorFlow는 계산만 적어 주면 기울기를 알아서 구합니다.

# %%
w = tf.Variable(-4.0)
with tf.GradientTape() as tape:
    loss = (w - 3) ** 2
print("자동 미분:", tape.gradient(loss, w).numpy(), "  손 계산 2(w-3):", 2 * (-4.0 - 3))

# 9단원의 경사하강법을 TensorFlow로
opt = keras.optimizers.SGD(learning_rate=0.3)
for _ in range(20):
    with tf.GradientTape() as tape:
        loss = (w - 3) ** 2
    opt.apply_gradients([(tape.gradient(loss, w), w)])
print("20번 후 w =", round(float(w.numpy()), 4))

# %% [markdown]
# ### 2-3. Keras로 XOR 모델 만들기
# 9단원과 같은 구조: 입력 2 → 은닉 4 → 출력 1

# %%
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

keras.utils.set_random_seed(0)
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(4, activation="tanh", name="hidden"),
    keras.layers.Dense(1, activation="sigmoid", name="output"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# %% [markdown]
# `summary()`의 파라미터 수 17 = (2×4 + 4) + (4×1 + 1). 8단원에서 손으로 센 방법과 같습니다.
#
# ### 2-4. 학습 — 처음부터 TensorBoard에 기록

# %%
log_dir = os.path.join(LOG_ROOT, "10_xor", "adam_tanh")
tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50)
history = model.fit(X, Y, epochs=500, verbose=0, callbacks=[tb])
print("마지막 손실:", round(history.history["loss"][-1], 4), " 정확도:", history.history["accuracy"][-1])
pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "예측 확률": model.predict(X, verbose=0).ravel().round(3), "정답": Y.ravel()})

# %%
pd.DataFrame(history.history).plot(figsize=(8, 3.5), title="학습 기록 (같은 내용을 TensorBoard에서도 볼 수 있음)")
plt.xlabel("epoch")
plt.show()

# %% [markdown]
# ### 2-5. TensorBoard 열기
#
# | 실행 환경 | 방법 |
# |---|---|
# | Docker Compose (이 과정 기본) | 브라우저에서 **http://localhost:6006** — `logs/` 폴더를 자동으로 읽음 |
# | 노트북 안에서 바로 | 아래 셀의 주석을 풀고 실행 |
#
# 볼 곳: **Scalars**(손실·정확도 곡선), **Graphs**(모델 구조), **Histograms**(가중치 분포가 학습하며 바뀌는 모습)

# %%
# %load_ext tensorboard
# %tensorboard --logdir ../logs

# %% [markdown]
# ### 2-6. 학습된 가중치 꺼내 보기

# %%
W1, b1 = model.get_layer("hidden").get_weights()
W2, b2 = model.get_layer("output").get_weights()
print("W1 모양", W1.shape, " b1 모양", b1.shape, " W2 모양", W2.shape, " b2 모양", b2.shape)
H = np.tanh(X @ W1 + b1)
print("직접 계산한 출력:", (1 / (1 + np.exp(-(H @ W2 + b2)))).ravel().round(3))   # model.predict와 같음

# %% [markdown]
# ### 2-7. NumPy 버전(9단원)과 비교
#
# | 하는 일 | NumPy (9단원) | Keras |
# |---|---|---|
# | 가중치 만들기 | `rng.normal(...)` 직접 | `Dense(4)`가 알아서 |
# | 순전파 | `sigmoid(X @ W1 + b1)` 직접 | `model(X)` |
# | 역전파 | `d2`, `d1` 식을 직접 유도 | **자동 미분** |
# | 갱신 | `W -= lr * grad` | 옵티마이저(`Adam`) |
# | 기록·시각화 | 리스트에 모아 matplotlib | **TensorBoard** 콜백 |
#
# **원리는 똑같고, 반복되는 일을 프레임워크가 대신할 뿐입니다.** 그래서 9단원을 먼저 직접 해 본 것입니다.
#
# ### 2-8. 저장과 불러오기 (원래 강의의 `Saver`)

# %%
os.makedirs("../models", exist_ok=True)
model.save("../models/xor.keras")
loaded = keras.models.load_model("../models/xor.keras")
print("불러온 모델 예측:", loaded.predict(X, verbose=0).ravel().round(3))

# %% [markdown]
# **관찰 질문**
# 1. 은닉층 활성화를 `"relu"`, 학습률을 0.005로 바꾸면 500 에포크 안에 풀리나요? TensorBoard에서 두 실행을 겹쳐 보세요.
# 2. Histograms 탭에서 가중치 분포는 학습이 진행되며 어떻게 변하나요?
# 3. 9단원은 MSE, 여기서는 이진 교차 엔트로피를 썼습니다. 0/1 문제에서 교차 엔트로피를 쓰는 이유는?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 같은 코드로 **OR** 와 **NAND** 를 학습시키세요. 은닉층 없이 `Dense(1, activation="sigmoid")` 하나로도 되나요? XOR은요?
#
# **문제 2.** XOR을 은닉 뉴런 2개, 8개로 학습시켜 TensorBoard 로그 폴더를 각각 다르게 저장하고(`../logs/10_xor/h2`, `h8`) 손실 곡선을 비교하세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# def single_layer(y):
#     m = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1, activation="sigmoid")])
#     m.compile(keras.optimizers.Adam(0.1), "binary_crossentropy", metrics=["accuracy"])
#     m.fit(X, y, epochs=300, verbose=0)
#     return m.predict(X, verbose=0).ravel().round(2)
#
# print("OR  :", single_layer(np.array([[0], [1], [1], [1]], np.float32)))   # 풀림 (직선으로 나뉨)
# print("NAND:", single_layer(np.array([[1], [1], [1], [0]], np.float32)))   # 풀림
# print("XOR :", single_layer(Y))                                            # 못 풂 (5단원과 같은 이유)
#
# for h in [2, 8]:
#     keras.utils.set_random_seed(0)
#     m = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(h, activation="tanh"),
#                           keras.layers.Dense(1, activation="sigmoid")])
#     m.compile(keras.optimizers.Adam(0.05), "binary_crossentropy", metrics=["accuracy"])
#     m.fit(X, Y, epochs=500, verbose=0,
#           callbacks=[keras.callbacks.TensorBoard(os.path.join(LOG_ROOT, "10_xor", f"h{h}"))])
# ```
# </details>
