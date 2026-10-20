# %% [markdown]
# # 09. 신경망 학습 원리 — 손실 함수, 경사하강법, 역전파
#
# > 인공지능 기초 실습 과정 · 단원 9/15
#
# **학습 목표**
# - 손실 함수가 "얼마나 틀렸나"를 숫자 하나로 나타낸다는 것을 이해한다.
# - 경사하강법으로 손실이 줄어드는 과정을 그림으로 보고, 학습률의 효과를 확인한다.
# - XOR 신경망의 **역전파를 NumPy로 직접 구현** 해 가중치가 스스로 찾아지는 것을 본다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 손실 함수(MSE, 교차 엔트로피), 기울기, 학습률, 연쇄 법칙 |
# | 따라하기 | 1차원 경사하강법 → 학습률 비교 → XOR 역전파 구현 → 손실 곡선 |
# | 연습문제 | 학습률·은닉 뉴런 수 바꿔 보기 |
#
# 8단원에서는 가중치를 사람이 정했습니다. 이번에는 **무작위 가중치에서 출발해 데이터로 찾게** 합니다.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념
#
# ### 손실 함수 — 얼마나 틀렸나
#
# | 이름 | 식 | 언제 |
# |---|---|---|
# | 평균 제곱 오차 (MSE) | $\frac{1}{n}\sum (y - \hat{y})^2$ | 숫자를 맞힐 때 (회귀) |
# | 이진 교차 엔트로피 | $-\frac{1}{n}\sum [y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | 0/1을 맞힐 때 |
# | 교차 엔트로피 | $-\sum y_k \log \hat{y}_k$ | 여러 클래스 중 하나 (12단원 MNIST) |
#
# ### 경사하강법 — 내리막으로 조금씩
# 손실을 가중치에 대해 미분한 **기울기** 는 "손실이 가장 빨리 커지는 방향"입니다. 그 **반대 방향** 으로 조금씩 움직이면 손실이 줄어듭니다.
#
# $$ w \leftarrow w - \eta \frac{\partial L}{\partial w} \qquad (\eta: \text{학습률}) $$
#
# ### 역전파 — 기울기를 뒤에서부터 계산
# 출력층에서 생긴 오차를 **연쇄 법칙** 으로 앞쪽 층까지 거꾸로 전달하며 각 가중치의 기울기를 구합니다(1986년 루멜하트·힌턴·윌리엄스, 4단원).

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 1차원 경사하강법
# 손실이 $L(w) = (w - 3)^2$ 인 가장 단순한 경우. 최솟값은 $w = 3$, 기울기는 $2(w - 3)$.

# %%
def gd(lr, w0=-4.0, steps=20):
    ws = [w0]
    for _ in range(steps):
        grad = 2 * (ws[-1] - 3)
        ws.append(ws[-1] - lr * grad)
    return np.array(ws)


w_axis = np.linspace(-5, 11, 200)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, lr in zip(axes, [0.05, 0.3, 0.9, 1.05]):
    ws = gd(lr)
    ax.plot(w_axis, (w_axis - 3) ** 2, c="gray")
    ax.plot(ws, (ws - 3) ** 2, "o-", c="tab:red", ms=4)
    ax.set_ylim(-2, 70); ax.set_title(f"학습률 {lr}  → 마지막 w = {ws[-1]:.2f}")
plt.show()

# %% [markdown]
# | 학습률 | 결과 |
# |---|---|
# | 너무 작음 (0.05) | 천천히 가서 20번 안에 도착 못 함 |
# | 적당함 (0.3) | 빠르게 3에 도착 |
# | 큼 (0.9) | 좌우로 크게 튀면서 겨우 도착 |
# | 너무 큼 (1.05) | **발산** — 점점 멀어짐 |
#
# 11단원 TensorBoard에서 학습률과 옵티마이저를 바꿔 가며 같은 현상을 손실 곡선으로 봅니다.
#
# ### 2-2. XOR 신경망 학습 — 역전파 직접 구현
#
# 구조: 입력 2 → 은닉 4 (시그모이드) → 출력 1 (시그모이드), 손실: MSE
#
# | 단계 | 계산 |
# |---|---|
# | 순전파 | $H = \sigma(XW_1 + b_1)$, $\hat{Y} = \sigma(HW_2 + b_2)$ |
# | 손실 | $L = \text{mean}((\hat{Y} - Y)^2)$ |
# | 출력층 오차 | $\delta_2 = (\hat{Y} - Y) \cdot \hat{Y}(1-\hat{Y})$ ← 시그모이드의 미분 $\sigma(1-\sigma)$ |
# | 은닉층 오차 | $\delta_1 = (\delta_2 W_2^T) \cdot H(1-H)$ ← 오차를 **뒤로** 전달 |
# | 기울기 | $\partial L/\partial W_2 = H^T\delta_2$, $\partial L/\partial W_1 = X^T\delta_1$ |
# | 갱신 | $W \leftarrow W - \eta \cdot \text{기울기}$ |

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def train_xor(hidden=4, lr=1.0, epochs=5000, seed=0):
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(0, 1, (2, hidden)), np.zeros(hidden)
    W2, b2 = rng.normal(0, 1, (hidden, 1)), np.zeros(1)
    losses = []
    for _ in range(epochs):
        # 순전파
        H = sigmoid(X @ W1 + b1)
        Y_hat = sigmoid(H @ W2 + b2)
        losses.append(np.mean((Y_hat - Y) ** 2))
        # 역전파 (상수 2/n은 학습률에 흡수)
        d2 = (Y_hat - Y) * Y_hat * (1 - Y_hat)
        d1 = (d2 @ W2.T) * H * (1 - H)
        # 갱신
        W2 -= lr * H.T @ d2;  b2 -= lr * d2.sum(axis=0)
        W1 -= lr * X.T @ d1;  b1 -= lr * d1.sum(axis=0)
    return (W1, b1, W2, b2), np.array(losses)


params, losses = train_xor()
W1, b1, W2, b2 = params
Y_hat = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
print("처음 손실:", losses[0].round(4), " 마지막 손실:", losses[-1].round(5))
pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "예측": Y_hat.ravel().round(3), "정답": Y.ravel()})

# %%
plt.figure(figsize=(7, 3.5))
plt.plot(losses)
plt.yscale("log"); plt.xlabel("반복 (epoch)"); plt.ylabel("손실 (MSE, 로그 눈금)"); plt.title("XOR 학습 손실 곡선")
plt.show()

# %% [markdown]
# **무작위 가중치에서 출발했는데 XOR을 풀었습니다.** 5단원에서 사람이 OR·NAND·AND로 나눠 생각했던 것을, 은닉층이 스스로 찾아낸 것입니다.
#
# ### 2-3. 은닉층은 무엇을 배웠나

# %%
H = sigmoid(X @ W1 + b1)
pd.DataFrame(H.round(2), columns=[f"h{i + 1}" for i in range(H.shape[1])]).assign(XOR=Y.ravel())

# %% [markdown]
# 은닉 뉴런마다 입력을 다르게 받아들입니다. 어떤 뉴런은 OR처럼, 어떤 뉴런은 NAND처럼 반응하는지 찾아보세요. 같은 결과라도 매번 다른 방식으로 풀 수 있습니다.
#
# ### 2-4. 수치 미분으로 역전파 검산하기
# 역전파로 구한 기울기가 맞는지, 가중치를 아주 조금 바꿔 손실 변화를 직접 재서 확인합니다.

# %%
def loss_of(W1, b1, W2, b2):
    return np.mean((sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2) - Y) ** 2)


rng = np.random.default_rng(1)
W1, b1, W2, b2 = rng.normal(size=(2, 4)), np.zeros(4), rng.normal(size=(4, 1)), np.zeros(1)
H = sigmoid(X @ W1 + b1); Y_hat = sigmoid(H @ W2 + b2)
d2 = (Y_hat - Y) * Y_hat * (1 - Y_hat) * 2 / len(X)       # 이번엔 상수 2/n까지 정확히
grad_backprop = (X.T @ ((d2 @ W2.T) * H * (1 - H)))[0, 0]

eps = 1e-6
W1p = W1.copy(); W1p[0, 0] += eps
W1m = W1.copy(); W1m[0, 0] -= eps
grad_numeric = (loss_of(W1p, b1, W2, b2) - loss_of(W1m, b1, W2, b2)) / (2 * eps)
print(f"역전파 기울기 {grad_backprop:.8f}  vs  수치 미분 {grad_numeric:.8f}")

# %% [markdown]
# 두 값이 거의 같다면 역전파 구현이 맞습니다. TensorFlow 같은 프레임워크는 이 계산을 **자동으로** 해 줍니다(자동 미분) → 10단원
#
# **관찰 질문**
# 1. 계단 함수를 쓰면 왜 역전파로 학습할 수 없을까요? (8단원 표의 "기울기가 0")
# 2. 손실 곡선이 처음에 한동안 평평하다가 갑자기 떨어지는 이유는 무엇일까요?
# 3. 시드(seed)를 바꾸면 결과가 항상 같나요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** `train_xor`의 학습률을 0.1, 1.0, 5.0으로 바꿔 손실 곡선을 한 그림에 그리세요.
#
# **문제 2.** 은닉 뉴런을 1, 2, 4, 8개로 바꿔 보세요. 몇 개부터 XOR을 안정적으로 푸나요? 시드 5개로 반복해 성공 횟수를 세 보세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# for lr in [0.1, 1.0, 5.0]:
#     plt.plot(train_xor(lr=lr)[1], label=f"lr={lr}")
# plt.yscale("log"); plt.legend(); plt.show()
#
# for h in [1, 2, 4, 8]:
#     ok = 0
#     for s in range(5):
#         (W1, b1, W2, b2), _ = train_xor(hidden=h, seed=s)
#         pred = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2) > 0.5
#         ok += np.array_equal(pred.astype(int), Y.astype(int))
#     print(f"은닉 {h}개: 5번 중 {ok}번 성공")
# # 1개로는 못 풀고, 2개는 시드에 따라 실패(나쁜 지점에 갇힘), 4개 이상이면 안정적
# ```
# </details>
