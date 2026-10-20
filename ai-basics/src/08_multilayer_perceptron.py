# %% [markdown]
# # 08. 다층 신경망 (MLP) — 가중치, 편향, 활성화 함수
#
# > 인공지능 기초 실습 과정 · 단원 8/15
#
# **학습 목표**
# - 입력층 · 은닉층 · 출력층으로 이루어진 다층 신경망의 **순전파** 를 NumPy 행렬 곱으로 구현한다.
# - 가중치·편향의 모양과 파라미터 개수를 계산할 수 있다.
# - 활성화 함수(계단, 시그모이드, tanh, ReLU)가 왜 필요한지 설명할 수 있다.
# - Scikit-learn의 MLP로 7단원의 초승달 데이터를 곡선 경계로 나눈다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 층 · 뉴런 · 가중치 행렬 · 활성화 함수 |
# | 따라하기 | 활성화 함수 그리기 → XOR 순전파 → 파라미터 세기 → 활성화 함수가 없으면? → MLP로 초승달 |
# | 연습문제 | 은닉 뉴런 수에 따른 결정 경계 변화 |

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념 — 퍼셉트론을 층으로 쌓으면
#
# ```
#   입력층        은닉층          출력층
#   x1 ─┐      ┌─ h1 ─┐
#       ├─ W1 ─┤      ├─ W2 ─▶ y
#   x2 ─┘      └─ h2 ─┘
# ```
#
# | 계산 | 식 | 모양 (입력 4개 기준) |
# |---|---|---|
# | 은닉층 | $H = f(X W_1 + b_1)$ | (4, 2) · (2, 2) + (2,) → (4, 2) |
# | 출력층 | $Y = f(H W_2 + b_2)$ | (4, 2) · (2, 1) + (1,) → (4, 1) |
#
# - **가중치 행렬** $W$: 열 하나가 뉴런 하나 (6단원 2-4)
# - **편향** $b$: 뉴런마다 하나
# - **활성화 함수** $f$: 층마다 가중합에 적용하는 비선형 함수
#
# 5단원에서는 OR·NAND·AND를 따로 학습해 붙였습니다. 다층 신경망은 이것을 **하나의 모델** 로 본 것입니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 활성화 함수 그리기

# %%
z = np.linspace(-5, 5, 400)
acts = {
    "계단 (step)": (z >= 0).astype(float),
    "시그모이드 (sigmoid)": 1 / (1 + np.exp(-z)),
    "tanh": np.tanh(z),
    "ReLU": np.maximum(0, z),
}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (name, v) in zip(axes, acts.items()):
    ax.plot(z, v, lw=2); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    ax.set_title(name)
plt.show()

# %% [markdown]
# | 함수 | 출력 범위 | 특징 |
# |---|---|---|
# | 계단 | 0 또는 1 | 퍼셉트론(5단원). **기울기가 0** 이라 경사하강법으로 학습 불가 (9단원) |
# | 시그모이드 | 0 ~ 1 | 부드러운 계단. 확률처럼 해석 가능. 양 끝에서 기울기가 거의 0 |
# | tanh | −1 ~ 1 | 0을 중심으로 대칭 |
# | ReLU | 0 ~ ∞ | 계산이 단순하고 깊은 망에서도 학습이 잘 됨 → 지금의 기본 |
#
# ### 2-2. XOR을 푸는 2층 신경망 — 순전파 직접 계산
# 5단원의 OR · NAND · AND 가중치를 그대로 행렬에 넣습니다. 시그모이드에 큰 가중치를 쓰면 계단 함수와 비슷하게 동작합니다.

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
W1 = np.array([[20, -20],          # 열 0: OR,  열 1: NAND
               [20, -20]], dtype=float)
b1 = np.array([-10, 30], dtype=float)
W2 = np.array([[20], [20]], dtype=float)   # AND
b2 = np.array([-30], dtype=float)

H = sigmoid(X @ W1 + b1)          # 은닉층 (4, 2)
Y = sigmoid(H @ W2 + b2)          # 출력층 (4, 1)
pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "h1(OR)": H[:, 0].round(3), "h2(NAND)": H[:, 1].round(3),
              "출력": Y[:, 0].round(3), "XOR 정답": [0, 1, 1, 0]})

# %% [markdown]
# ### 2-3. 파라미터 세기
# 파라미터 = 학습으로 바뀌는 숫자 = **모든 가중치 + 모든 편향**.

# %%
def count_params(layers):
    """layers: [입력 수, 은닉1, 은닉2, ..., 출력 수]"""
    rows = []
    for i in range(len(layers) - 1):
        w, b = layers[i] * layers[i + 1], layers[i + 1]
        rows.append({"층": f"{layers[i]} → {layers[i + 1]}", "가중치": w, "편향": b, "합계": w + b})
    df = pd.DataFrame(rows)
    df.loc["전체"] = ["", df["가중치"].sum(), df["편향"].sum(), df["합계"].sum()]
    return df


print("XOR 신경망 [2, 2, 1]"); print(count_params([2, 2, 1]))
print("\nMNIST용 신경망 [784, 128, 10] (12단원)"); print(count_params([784, 128, 10]))

# %% [markdown]
# ### 2-4. 활성화 함수가 없으면?
# 활성화 함수 없이 층을 쌓으면 $X W_1 W_2 = X (W_1 W_2)$ — 행렬 곱 두 번은 **행렬 곱 한 번과 같습니다**. 아무리 쌓아도 직선(선형) 하나만 표현할 수 있어 XOR을 못 풉니다.

# %%
rng = np.random.default_rng(0)
A, B = rng.normal(size=(2, 3)), rng.normal(size=(3, 1))
print("두 층을 차례로:", (X @ A @ B).ravel().round(3))
print("한 층으로 합침:", (X @ (A @ B)).ravel().round(3))

# %% [markdown]
# **비선형 활성화 함수가 있어야 층을 쌓는 의미가 생깁니다.**
#
# ### 2-5. Scikit-learn MLP로 초승달 나누기
# 7단원에서 선형 모델이 못 나눈 데이터입니다. 이번에는 가중치를 사람이 정하지 않고 **학습** 시킵니다(원리는 9단원).

# %%
Xm, ym = make_moons(n_samples=400, noise=0.2, random_state=0)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)


def plot_decision(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 300),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, hidden in zip(axes, [(1,), (3,), (16, 16)]):
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", max_iter=3000, random_state=0).fit(Xm_tr, ym_tr)
    plot_decision(ax, mlp, Xm, ym, f"은닉층 {hidden}\n테스트 정확도 {accuracy_score(ym_te, mlp.predict(Xm_te)):.2f}")
plt.show()

# %% [markdown]
# 은닉 뉴런이 1개면 직선에 가깝고, 늘릴수록 **곡선 경계** 를 만들 수 있습니다. 은닉층이 입력 공간을 접고 구부려 나눌 수 있게 만드는 것입니다.
#
# **관찰 질문**
# 1. 2-2에서 가중치 20을 2로 줄이면 출력이 어떻게 바뀌나요? 왜 그럴까요?
# 2. [784, 128, 10] 신경망의 파라미터 중 대부분은 어느 층에 있나요?
# 3. 은닉 뉴런이 많을수록 항상 좋을까요? (7단원의 과대적합을 떠올려 보세요)

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 2-2의 XOR 신경망에서 활성화 함수를 ReLU로 바꾸면 그대로 동작할까요? 가중치를 어떻게 바꿔야 할까요? (출력층은 시그모이드 유지)
#
# **문제 2.** 초승달 데이터에서 `activation`을 `"logistic"`(시그모이드), `"tanh"`, `"relu"`로 바꿔 (16, 16) MLP의 테스트 정확도와 결정 경계를 비교하세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# relu = lambda z: np.maximum(0, z)
# W1r = np.array([[1, 1], [1, 1]], float); b1r = np.array([0, -1], float)   # h1 = x1+x2, h2 = max(0, x1+x2-1)
# H = relu(X @ W1r + b1r)                                                    # (0,0)→(0,0) (0,1)→(1,0) (1,1)→(2,1)
# Y = sigmoid((H @ np.array([[1], [-2]], float) - 0.5) * 20)                 # h1 − 2·h2 = XOR
# print(Y.ravel().round(3))
#
# for act in ["logistic", "tanh", "relu"]:
#     m = MLPClassifier((16, 16), activation=act, max_iter=3000, random_state=0).fit(Xm_tr, ym_tr)
#     print(act, round(accuracy_score(ym_te, m.predict(Xm_te)), 3))
# ```
# </details>
