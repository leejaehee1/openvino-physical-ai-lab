# %% [markdown]
# # 14. CNN 발전 과정 — LeNet, AlexNet, VGG, ResNet
#
# > 인공지능 기초 실습 과정 · 단원 14/15
#
# **학습 목표**
# - LeNet → AlexNet → VGG → ResNet이 **앞 모델의 어떤 문제를 풀었는지** 설명할 수 있다.
# - LeNet-5를 직접 만들어 MNIST로 학습시킨다.
# - 유명한 모델들의 파라미터 수를 직접 세어 보고, 무게가 어디에 몰려 있는지 확인한다.
# - 깊은 망에서 기울기가 사라지는 문제와 ResNet의 **숏컷 연결** 이 그것을 푸는 모습을 숫자로 본다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | ImageNet 대회와 오류율의 변화, 모델별 핵심 아이디어 |
# | 따라하기 | LeNet-5 학습 → AlexNet·VGG·ResNet 파라미터 세기 → 3×3 쌓기 계산 → 기울기 소실과 숏컷 실험 |
# | 연습문제 | VGG 스타일 블록 만들기 |

# %%
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import font_manager
from tensorflow import keras
from tensorflow.keras import layers

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념 — 문제와 해결의 연쇄
#
# | 연도 | 모델 | 풀어낸 문제 | 핵심 아이디어 |
# |---|---|---|---|
# | 1998 | **LeNet-5** | 특징을 사람이 설계해야 했음 | 합성곱 → 풀링 → 분류의 기본 골격, 필터를 **학습** |
# | 2012 | **AlexNet** | 실제 사진(ImageNet)을 인식하기엔 데이터·계산이 부족 | **GPU** 학습, ReLU, 드롭아웃, 데이터 증강 |
# | 2014 | **VGG** | 층마다 필터 크기가 제각각 | **3×3만** 깊게 쌓기 (16·19층) |
# | 2014 | GoogLeNet | 깊고 넓어질수록 계산량 폭증 | 여러 크기 필터를 나란히, **1×1 합성곱** 으로 압축 |
# | 2015 | **ResNet** | 깊게 쌓으면 오히려 학습이 안 됨 | **숏컷 연결** $F(x) + x$ → 152층 |
#
# ### ImageNet 대회(ILSVRC) Top-5 오류율의 변화 (대략)
#
# | 연도 | 우승 모델 | Top-5 오류율 |
# |---|---|---|
# | 2011 | 사람이 설계한 특징 방식 | 약 26% |
# | 2012 | AlexNet | 약 15% |
# | 2014 | GoogLeNet (VGG 2위 약 7%) | 약 7% |
# | 2015 | ResNet | 약 3.6% |
#
# Top-5 오류율: 상위 5개 예측 안에 정답이 없는 비율. 1,000개 클래스, 학습 이미지 약 120만 장.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. LeNet-5 만들고 학습시키기 (1998)
# 원래 입력은 32×32라서, 28×28 MNIST에는 첫 합성곱에 `padding="same"`을 줍니다. 활성화는 당시처럼 tanh, 풀링은 평균 풀링.

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 255.0)[..., None].astype("float32")
x_test = (x_test / 255.0)[..., None].astype("float32")

keras.utils.set_random_seed(0)
lenet = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(6, 5, padding="same", activation="tanh", name="C1"),
    layers.AveragePooling2D(2, name="S2"),
    layers.Conv2D(16, 5, activation="tanh", name="C3"),
    layers.AveragePooling2D(2, name="S4"),
    layers.Flatten(),
    layers.Dense(120, activation="tanh", name="F5"),
    layers.Dense(84, activation="tanh", name="F6"),
    layers.Dense(10, activation="softmax", name="output"),
], name="LeNet5")
lenet.summary()

# %%
lenet.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
lenet.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2,
          callbacks=[keras.callbacks.TensorBoard("../logs/12_mnist/lenet5")])
print("LeNet-5 테스트 정확도:", round(lenet.evaluate(x_test, y_test, verbose=0)[1], 4))

# %% [markdown]
# 파라미터 약 **6만 개** 로, 2 에포크만 학습해도 MNIST를 96~97% 맞힙니다(더 학습하면 98% 이상). 1998년에 이미 은행 수표의 숫자를 읽는 데 쓰였습니다. 그런데 왜 10년 넘게 조용했을까요? → 실제 사진을 학습할 **데이터와 계산력** 이 없었기 때문입니다.
#
# ### 2-2. AlexNet, VGG16, ResNet50 — 파라미터 세기
# 학습은 하지 않고 구조만 만들어 크기를 봅니다(`weights=None`이라 다운로드 없음). AlexNet은 Keras에 없어서 논문 구조대로 직접 쌓습니다(원래는 GPU 2장에 나눠 계산한 구조를 하나로 합친 형태).

# %%
def alexnet():
    return keras.Sequential([
        keras.Input(shape=(227, 227, 3)),
        layers.Conv2D(96, 11, strides=4, activation="relu"), layers.MaxPooling2D(3, 2),
        layers.Conv2D(256, 5, padding="same", activation="relu"), layers.MaxPooling2D(3, 2),
        layers.Conv2D(384, 3, padding="same", activation="relu"),
        layers.Conv2D(384, 3, padding="same", activation="relu"),
        layers.Conv2D(256, 3, padding="same", activation="relu"), layers.MaxPooling2D(3, 2),
        layers.Flatten(),
        layers.Dense(4096, activation="relu"), layers.Dropout(0.5),
        layers.Dense(4096, activation="relu"), layers.Dropout(0.5),
        layers.Dense(1000, activation="softmax"),
    ], name="AlexNet")


rows = [{"모델": "LeNet-5 (1998)", "층 수(대략)": 7, "파라미터": lenet.count_params()}]
builders = [("AlexNet (2012)", 8, alexnet),
            ("VGG16 (2014)", 16, lambda: keras.applications.VGG16(weights=None)),
            ("ResNet50 (2015)", 50, lambda: keras.applications.ResNet50(weights=None)),
            ("MobileNetV2 (2018, 엣지용)", 53, lambda: keras.applications.MobileNetV2(weights=None))]
vgg_fc1 = None
for name, depth, build in builders:
    m = build()
    rows.append({"모델": name, "층 수(대략)": depth, "파라미터": m.count_params()})
    if name.startswith("VGG16"):
        vgg_fc1 = m.get_layer("fc1").count_params()
    del m; keras.backend.clear_session(); gc.collect()

params = pd.DataFrame(rows)
params["파라미터(백만)"] = (params["파라미터"] / 1e6).round(2)
params

# %%
ax = params.plot.barh(x="모델", y="파라미터", logx=True, legend=False, figsize=(8, 3.5))
ax.set_xlabel("파라미터 수 (로그 눈금)")
plt.show()
print(f"VGG16의 첫 완전연결층(fc1) 하나: {vgg_fc1:,}개 = 전체의 {vgg_fc1 / params.loc[2, '파라미터'] * 100:.0f}%")

# %% [markdown]
# - VGG16 파라미터의 대부분은 **마지막 완전연결층** 에 몰려 있습니다. 합성곱층은 의외로 가볍습니다.
# - ResNet50은 VGG16보다 3배 깊은데 파라미터는 5분의 1 수준입니다(1×1 합성곱, 거대한 완전연결층 제거).
# - MobileNetV2는 휴대폰·엣지 기기를 위해 만든 가벼운 CNN입니다.
#
# ### 2-3. VGG의 아이디어 — 3×3을 쌓으면 왜 좋은가

# %%
C = 64
for n_small, big in [(2, 5), (3, 7)]:
    print(f"3×3 {n_small}번 = {big}×{big} 범위.  파라미터 {n_small * 9 * C * C:,} vs {big * big * C * C:,}"
          f"  (약 {big * big / (n_small * 9):.1f}배 절약), ReLU {n_small}번 vs 1번")

# %% [markdown]
# **같은 범위를 더 적은 파라미터로, 더 많은 비선형으로** 봅니다.
#
# ### 2-4. 깊게 쌓으면 왜 학습이 안 될까 — 기울기 소실 실험
# 시그모이드 층 20개를 쌓은 망에서, 출력의 손실 기울기가 **앞쪽 층까지 얼마나 전달되는지** 잽니다. 숏컷 연결을 넣은 망과 비교합니다(학습은 하지 않고 기울기 크기만 봄).

# %%
def deep_net(n_layers=20, residual=False, units=32):
    inp = keras.Input(shape=(784,))
    x = layers.Dense(units, activation="sigmoid")(inp)
    for _ in range(n_layers):
        h = layers.Dense(units, activation="sigmoid")(x)
        x = layers.Add()([x, h]) if residual else h          # ResNet: 출력 = F(x) + x
    out = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inp, out)


def grad_norms(model, x, y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, model(x)))
    grads = tape.gradient(loss, model.trainable_weights)
    # 층마다 가중치 행렬(kernel)의 기울기 크기만 모은다 (편향 제외)
    return [float(tf.norm(g)) for g, w in zip(grads, model.trainable_weights) if "kernel" in getattr(w, "path", w.name)]


xb, yb = x_train[:256].reshape(-1, 784), y_train[:256]
keras.utils.set_random_seed(0)
plain = grad_norms(deep_net(residual=False), xb, yb)
keras.utils.set_random_seed(0)
resid = grad_norms(deep_net(residual=True), xb, yb)

plt.figure(figsize=(9, 3.5))
plt.semilogy(plain, "o-", label="일반 (plain)"); plt.semilogy(resid, "s-", label="숏컷 연결 (residual)")
plt.xlabel("층 번호 (0 = 입력 쪽)"); plt.ylabel("기울기 크기 (로그)"); plt.legend(); plt.title("층별 기울기 크기")
plt.show()
print(f"입력 쪽 첫 층 기울기 — 일반: {plain[0]:.2e}, 숏컷: {resid[0]:.2e}  (약 10^{np.log10(resid[0] / plain[0]):.0f}배 차이)")

# %% [markdown]
# 일반 망은 입력 쪽으로 갈수록 기울기가 **급격히 작아져** 앞쪽 층이 거의 학습되지 않습니다(기울기 소실). 숏컷 연결은 더하기 경로로 기울기를 그대로 흘려보내 앞쪽 층까지 학습 신호가 닿습니다.
# 이것이 ResNet이 152층까지 쌓을 수 있었던 이유이고, 지금은 트랜스포머(GPT)·로봇 정책(ACT)에도 똑같이 들어 있습니다.
#
# ### 2-5. ResNet의 잔차 블록 직접 만들기

# %%
def residual_block(x, filters):
    shortcut = x
    h = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    h = layers.Conv2D(filters, 3, padding="same")(h)
    if shortcut.shape[-1] != filters:                         # 채널 수가 다르면 1×1 합성곱으로 맞춤
        shortcut = layers.Conv2D(filters, 1)(shortcut)
    return layers.Activation("relu")(layers.Add()([h, shortcut]))


inp = keras.Input(shape=(28, 28, 1))
x = residual_block(inp, 16)
x = layers.MaxPooling2D()(x)
x = residual_block(x, 32)
x = layers.GlobalAveragePooling2D()(x)
mini_resnet = keras.Model(inp, layers.Dense(10, activation="softmax")(x), name="mini_resnet")
print("미니 ResNet 파라미터:", mini_resnet.count_params())

# %% [markdown]
# **관찰 질문**
# 1. AlexNet은 LeNet보다 약 1,000배 큽니다. 이런 모델을 학습시키려면 무엇이 필요했나요?
# 2. VGG16에서 완전연결층을 없애면 파라미터가 얼마나 줄어들까요?
# 3. 기울기 소실 실험에서 시그모이드 대신 ReLU를 쓰면 일반 망의 결과는 어떻게 달라질까요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** LeNet-5의 `C1`, `C3` 파라미터 수를 손으로 계산해 `summary()`와 맞춰 보세요. (힌트: (커널 가로 × 세로 × 입력 채널 + 1) × 커널 수)
#
# **문제 2.** "3×3 합성곱 2번 → 최대 풀링"을 한 블록으로 하는 **미니 VGG**(블록 2개)를 만들어 MNIST로 2 에포크 학습시키고, LeNet-5·13단원 CNN과 정확도와 파라미터 수를 비교하세요.
#
# **문제 3.** 2-5의 `mini_resnet`을 MNIST로 2 에포크 학습시켜 보세요.

# %%
# 문제 1~3 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# print("C1:", (5 * 5 * 1 + 1) * 6, "  C3:", (5 * 5 * 6 + 1) * 16)   # 156, 2416
#
# keras.utils.set_random_seed(0)
# vgg_mini = keras.Sequential([
#     keras.Input(shape=(28, 28, 1)),
#     layers.Conv2D(16, 3, padding="same", activation="relu"), layers.Conv2D(16, 3, padding="same", activation="relu"),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding="same", activation="relu"), layers.Conv2D(32, 3, padding="same", activation="relu"),
#     layers.MaxPooling2D(),
#     layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10, activation="softmax")])
# vgg_mini.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# vgg_mini.fit(x_train, y_train, epochs=2, batch_size=128, verbose=2)
# print(vgg_mini.count_params(), vgg_mini.evaluate(x_test, y_test, verbose=0)[1])
#
# mini_resnet.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# mini_resnet.fit(x_train, y_train, epochs=2, batch_size=128, verbose=2)
# ```
# </details>
