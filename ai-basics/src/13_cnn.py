# %% [markdown]
# # 13. CNN — 합성곱, 풀링, 특징 맵
#
# > 인공지능 기초 실습 과정 · 단원 13/15
#
# **학습 목표**
# - 합성곱층 · 풀링층 · 완전연결층으로 이루어진 CNN을 만들어 MNIST를 분류한다.
# - CNN이 **스스로 배운 3×3 커널** 을 꺼내, 3단원에서 사람이 만든 Sobel·Gaussian·나만의 커널과 나란히 비교한다.
# - 층을 지날 때마다 이미지가 어떻게 바뀌는지(특징 맵) 직접 본다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 합성곱층 = 학습하는 커널, 풀링, 특징 맵, 출력 크기 공식 |
# | 따라하기 | CNN 만들기 → 학습 → MLP와 비교 → **배운 커널 vs 3단원 커널** → 특징 맵 |
# | 연습문제 | 위치 이동에 대한 강건성 비교 |

# %%
import os

import cv2
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
EPOCHS = 3

# %% [markdown]
# ## 1. 개념 — 3단원의 커널을 "학습"시키면
#
# | 3단원 (사람이 설계) | CNN (데이터로 학습) |
# |---|---|
# | Sobel, Gaussian처럼 커널 숫자를 사람이 정함 | 커널 숫자가 **가중치** — 역전파로 학습 (9단원) |
# | 커널 1~2개 | 층마다 커널 수십~수백 개 |
# | 경계·블러 같은 정해진 효과 | 분류에 **도움이 되는 특징** 을 스스로 찾음 |
#
# ```
# 입력 28×28×1
#  → 합성곱 3×3 × 16개 (ReLU)  → 26×26×16   특징 맵 16장
#  → 최대 풀링 2×2              → 13×13×16   크기 절반
#  → 합성곱 3×3 × 32개 (ReLU)  → 11×11×32
#  → 최대 풀링 2×2              →  5×5×32
#  → 펼치기 800 → 완전연결 64 → 10 (소프트맥스)
# ```
#
# - 출력 크기는 3단원의 공식 $(N - F + 2P)/S + 1$: $(28 - 3 + 0)/1 + 1 = 26$
# - **풀링**: 2×2 영역에서 가장 큰 값만 남김 → 크기는 절반, 약간 움직여도 결과가 비슷 (원래 강의의 "Subsampling")
# - **특징 맵**: 커널 하나를 이미지 전체에 적용한 결과 이미지

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 데이터 — 이번엔 펼치지 않는다

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_c = (x_train / 255.0)[..., None].astype("float32")    # (60000, 28, 28, 1) — 채널 차원 추가
x_test_c = (x_test / 255.0)[..., None].astype("float32")
print(x_train_c.shape)

# %% [markdown]
# ### 2-2. CNN 만들기

# %%
keras.utils.set_random_seed(0)
cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv1"),
    keras.layers.MaxPooling2D((2, 2), name="pool1"),
    keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2"),
    keras.layers.MaxPooling2D((2, 2), name="pool2"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
], name="mnist_cnn")
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.summary()

# %% [markdown]
# `conv1`의 파라미터 160 = 커널 16개 × (3×3 + 편향 1). **커널 하나가 3단원의 3×3 필터 하나** 입니다.
# 라벨은 원-핫 대신 정수 그대로 쓰고 손실을 `sparse_categorical_crossentropy`로 바꿨습니다(결과는 같음).
#
# ### 2-3. 학습 (TensorBoard 기록)

# %%
history = cnn.fit(x_train_c, y_train, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=2,
                  callbacks=[keras.callbacks.TensorBoard("../logs/12_mnist/cnn", histogram_freq=1)])
test_acc = cnn.evaluate(x_test_c, y_test, verbose=0)[1]
print(f"CNN 테스트 정확도: {test_acc:.4f}")
os.makedirs("../models", exist_ok=True)
cnn.save("../models/mnist_cnn.keras")

# %% [markdown]
# ### 2-4. MLP(12단원)와 비교

# %%
rows = [{"모델": "CNN (이번 단원)", "파라미터": cnn.count_params(), "테스트 정확도": round(test_acc, 4)}]
if os.path.exists("../models/mnist_mlp.keras"):
    mlp = keras.models.load_model("../models/mnist_mlp.keras")
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    acc = mlp.evaluate((x_test / 255.0).reshape(-1, 784), y_test_oh, verbose=0)[1]
    rows.insert(0, {"모델": "MLP (12단원)", "파라미터": mlp.count_params(), "테스트 정확도": round(acc, 4)})
pd.DataFrame(rows)

# %% [markdown]
# CNN은 MLP보다 **파라미터가 적은데도** 더 정확합니다. 같은 커널을 이미지 전체에 재사용(가중치 공유)하고, 이웃 픽셀의 관계(공간 정보)를 유지하기 때문입니다.
# TensorBoard의 `12_mnist` 아래에서 `mlp_128`과 `cnn`을 겹쳐 보세요.
#
# ### 2-5. 배운 커널 vs 3단원에서 사람이 만든 커널

# %%
learned = cnn.get_layer("conv1").get_weights()[0][:, :, 0, :]     # (3, 3, 16)

human = {
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
    "Gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16,
}
if os.path.exists("../data/my_kernel.npy"):                          # 3단원 연습문제에서 저장한 커널
    human["나만의 커널 (3단원)"] = np.load("../data/my_kernel.npy")

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for i, ax in enumerate(axes.ravel()):
    k = learned[:, :, i]
    ax.imshow(k, cmap="bwr", vmin=-np.abs(learned).max(), vmax=np.abs(learned).max())
    ax.set_title(f"배운 커널 {i}"); ax.axis("off")
plt.suptitle("CNN이 MNIST에서 스스로 배운 3×3 커널 16개 (빨강 +, 파랑 −)")
plt.show()

fig, axes = plt.subplots(1, len(human), figsize=(3.2 * len(human), 3))
for ax, (name, k) in zip(np.atleast_1d(axes), human.items()):
    ax.imshow(k, cmap="bwr", vmin=-np.abs(k).max(), vmax=np.abs(k).max()); ax.set_title(name); ax.axis("off")
plt.suptitle("3단원에서 사람이 만든 커널")
plt.show()

# %% [markdown]
# 배운 커널 중 **한쪽은 빨강, 반대쪽은 파랑** 인 것이 있나요? 그것은 Sobel처럼 특정 방향의 경계에 반응하는 커널입니다. 아무도 Sobel을 가르쳐 주지 않았는데, 숫자를 구분하려다 보니 경계 검출기가 저절로 생긴 것입니다.
#
# 가장 Sobel과 닮은 커널을 숫자로 찾아봅니다(정규화 후 상관계수).

# %%
def similarity(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


rows = []
for name, hk in human.items():
    if np.allclose(hk, hk.mean()):
        continue
    scores = [abs(similarity(learned[:, :, i], hk)) for i in range(16)]
    best = int(np.argmax(scores))
    rows.append({"사람이 만든 커널": name, "가장 닮은 배운 커널": best, "|상관계수|": round(scores[best], 3)})
pd.DataFrame(rows)

# %% [markdown]
# ### 2-6. 같은 숫자에 두 종류의 커널 적용하기

# %%
digit = x_test[0].astype(np.float32)
best_sobel = int(np.argmax([abs(similarity(learned[:, :, i], human["Sobel X"])) for i in range(16)]))
pairs = [("원본", digit), ("Sobel X (사람)", cv2.filter2D(digit, -1, human["Sobel X"])),
         (f"배운 커널 {best_sobel}", cv2.filter2D(digit, -1, learned[:, :, best_sobel]))]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, (t, im) in zip(axes, pairs):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
plt.show()

# %% [markdown]
# ### 2-7. 특징 맵 — 층을 지나며 바뀌는 모습

# %%
feature_model = keras.Model(cnn.inputs, [cnn.get_layer(n).output for n in ["conv1", "pool1", "conv2"]])
maps = feature_model.predict(x_test_c[:1], verbose=0)
for name, fmap in zip(["conv1 (26×26×16)", "pool1 (13×13×16)", "conv2 (11×11×32)"], maps):
    n = 8
    fig, axes = plt.subplots(1, n, figsize=(14, 2))
    for i, ax in enumerate(axes):
        ax.imshow(fmap[0, :, :, i], cmap="viridis"); ax.axis("off")
    fig.suptitle(f"{name} — 앞 {n}장")
    plt.show()

# %% [markdown]
# 첫 층의 특징 맵은 아직 숫자 모양이 보이고(경계, 획), 깊어질수록 작아지고 추상적이 됩니다. "얕은 층은 선, 깊은 층은 부분과 물체"라는 CNN의 특징을 직접 본 것입니다(14단원에서 더 깊은 모델로 이어짐).
#
# **관찰 질문**
# 1. 배운 커널 16개 중 의미가 분명해 보이는 것과 잡음처럼 보이는 것은 몇 개씩인가요?
# 2. `conv2`의 커널은 3×3×**16** 크기입니다. 왜 깊이가 16일까요?
# 3. 시드를 바꿔 다시 학습시키면 같은 커널이 나올까요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 테스트 이미지를 오른쪽으로 3픽셀 옮겼을 때(`np.roll`) MLP와 CNN의 정확도가 각각 얼마나 떨어지는지 비교하세요.
#
# **문제 2.** `conv1`의 커널을 8개, 32개로 바꿔 학습시키고 정확도·파라미터 수·배운 커널 모양을 비교하세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# shifted = np.roll(x_test, 3, axis=2) / 255.0
# print("CNN:", cnn.evaluate(shifted[..., None], y_test, verbose=0)[1])
# if os.path.exists("../models/mnist_mlp.keras"):
#     print("MLP:", mlp.evaluate(shifted.reshape(-1, 784), keras.utils.to_categorical(y_test, 10), verbose=0)[1])
# # 둘 다 떨어지지만 CNN이 덜 떨어진다 (같은 커널을 모든 위치에 쓰고, 풀링이 작은 이동을 흡수)
# ```
# </details>
