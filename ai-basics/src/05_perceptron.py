# %% [markdown]
# # 05. 퍼셉트론 — AND, OR, 그리고 XOR
#
# > 인공지능 기초 실습 과정 · 단원 5/15
#
# **학습 목표**
# - 퍼셉트론의 계산 $y = \text{step}(w_1x_1 + w_2x_2 + b)$를 **NumPy만으로** 구현한다.
# - AND, OR, NAND를 손으로 정한 가중치와 **학습 규칙** 두 가지 방법으로 만든다.
# - 단층 퍼셉트론이 **XOR을 못 푸는 이유** 를 결정 경계 그림으로 확인하고, 층을 쌓아 해결한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 입력 · 가중치 · 가중합 · 계단 함수 |
# | 따라하기 | 손으로 정한 AND → 퍼셉트론 학습 규칙 → 결정 경계 → XOR 실패 → 게이트 조합으로 XOR |
# | 연습문제 | OR · NAND 학습시키기 |
#
# 이 단원의 NumPy 코드는 **10단원에서 TensorFlow로 같은 문제를 풀 때** 다시 비교합니다.

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

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])          # 입력 4가지
targets = {
    "AND":  np.array([0, 0, 0, 1]),
    "OR":   np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR":  np.array([0, 1, 1, 0]),
}
pd.DataFrame(X, columns=["x1", "x2"]).assign(**targets)

# %% [markdown]
# ## 1. 개념 — 퍼셉트론 한 개
#
# ```
#  x1 ──(w1)──┐
#             ├──▶ Σ (가중합 + b) ──▶ step ──▶ y (0 또는 1)
#  x2 ──(w2)──┘
# ```
#
# | 단계 | 계산 |
# |---|---|
# | 입력 (inputs) | $x_1, x_2$ |
# | 가중치 (weights) | $w_1, w_2$ — 입력이 얼마나 중요한가 |
# | 가중합 (weighted sum) | $z = w_1x_1 + w_2x_2 + b$ ($b$: 편향, bias) |
# | 계단 함수 (step function) | $z \ge 0$ 이면 1, 아니면 0 |
#
# 편향 $b$는 "얼마나 쉽게 1을 내보낼지"를 정하는 문턱 값입니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 손으로 가중치를 정해 AND 만들기
# $w_1 = w_2 = 1$, $b = -1.5$ 이면 두 입력이 **모두 1일 때만** $z \ge 0$이 됩니다.

# %%
def step(z):
    return (z >= 0).astype(int)


def perceptron(X, w, b):
    return step(X @ w + b)       # X @ w : 4개 입력을 한 번에 계산 (6단원에서 자세히)


w, b = np.array([1.0, 1.0]), -1.5
print("AND 출력:", perceptron(X, w, b), " 정답:", targets["AND"])

# %% [markdown]
# ### 2-2. 퍼셉트론 학습 규칙 — 가중치를 데이터로 찾기
#
# 틀릴 때마다 가중치를 조금씩 고칩니다.
#
# $$ w \leftarrow w + \eta\,(y_{정답} - y_{예측})\,x, \qquad b \leftarrow b + \eta\,(y_{정답} - y_{예측}) $$
#
# - 맞히면 $(y_{정답} - y_{예측}) = 0$ → 그대로
# - 1이어야 하는데 0을 냈으면 → 가중치를 입력 방향으로 키움
# - $\eta$(에타): 학습률, 한 번에 얼마나 고칠지

# %%
def train_perceptron(X, y, lr=0.1, epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    w, b = rng.normal(0, 0.5, 2), 0.0
    log = []
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = int(xi @ w + b >= 0)
            delta = yi - pred
            w += lr * delta * xi
            b += lr * delta
            errors += int(delta != 0)
        log.append({"epoch": epoch + 1, "틀린 개수": errors, "w1": w[0], "w2": w[1], "b": b})
        if errors == 0:
            break
    return w, b, pd.DataFrame(log)


w_and, b_and, log = train_perceptron(X, targets["AND"])
print("학습된 가중치:", w_and.round(2), " 편향:", round(b_and, 2))
print("출력:", perceptron(X, w_and, b_and))
log

# %% [markdown]
# 틀린 개수가 0이 되면 학습이 끝납니다. 손으로 정한 값과 다른 가중치가 나와도 **같은 문제를 풉니다**. 정답 가중치는 하나가 아닙니다.
#
# ### 2-3. 결정 경계 — 퍼셉트론은 "직선 하나"로 나눈다
# $w_1x_1 + w_2x_2 + b = 0$ 은 평면 위의 **직선** 입니다. 퍼셉트론은 이 직선의 한쪽은 1, 반대쪽은 0이라고 판단합니다.

# %%
def plot_boundary(ax, w, b, y, title):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    zz = step(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    for xi, yi in zip(X, y):
        ax.scatter(*xi, s=200, c="tab:red" if yi else "tab:blue", edgecolors="k")
    ax.set_title(title); ax.set_xlabel("x1"); ax.set_ylabel("x2")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, name in zip(axes, ["AND", "OR", "NAND"]):
    w_, b_, _ = train_perceptron(X, targets[name])
    plot_boundary(ax, w_, b_, targets[name], f"{name}: 출력 {perceptron(X, w_, b_)}")
plt.show()

# %% [markdown]
# ### 2-4. XOR — 직선 하나로는 안 된다

# %%
w_x, b_x, log_x = train_perceptron(X, targets["XOR"], epochs=100)
print("XOR 출력:", perceptron(X, w_x, b_x), " 정답:", targets["XOR"])
print("마지막 에포크 틀린 개수:", log_x["틀린 개수"].iloc[-1], f"(100 에포크 동안 한 번도 0이 안 됨: {(log_x['틀린 개수'] > 0).all()})")

fig, ax = plt.subplots(figsize=(5, 4.5))
plot_boundary(ax, w_x, b_x, targets["XOR"], "XOR: 어떤 직선도 빨강·파랑을 못 나눔")
plt.show()

# %% [markdown]
# (0,1)·(1,0)은 1이고 (0,0)·(1,1)은 0입니다. 대각선으로 엇갈려 있어서 **직선 하나로는 절대 나눌 수 없습니다**(선형 분리 불가능).
# 1969년 민스키·페퍼트가 지적한 바로 그 문제이고, 첫 번째 인공지능 겨울의 원인이 되었습니다(4단원).
#
# ### 2-5. 해결 — 퍼셉트론을 두 층으로 쌓기
# XOR = (OR) AND (NAND). 이미 만들 줄 아는 게이트를 **조합** 하면 됩니다.
#
# ```
#  x1 ─┬─▶ [OR]   ─┐
#      │            ├─▶ [AND] ─▶ XOR
#  x2 ─┴─▶ [NAND] ─┘
#  입력층    은닉층      출력층
# ```

# %%
w_or, b_or, _ = train_perceptron(X, targets["OR"])
w_nand, b_nand, _ = train_perceptron(X, targets["NAND"])

hidden = np.stack([perceptron(X, w_or, b_or), perceptron(X, w_nand, b_nand)], axis=1)   # 은닉층 출력
xor_out = perceptron(hidden, w_and, b_and)

pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "OR": hidden[:, 0], "NAND": hidden[:, 1],
              "AND(OR, NAND)": xor_out, "XOR 정답": targets["XOR"]})

# %% [markdown]
# **층을 하나 더 쌓았더니 XOR이 풀렸습니다.** 은닉층이 입력을 **새로운 좌표(OR, NAND)** 로 바꿔 주어, 그 좌표에서는 직선 하나로 나눌 수 있게 된 것입니다.
#
# 남은 문제: 여기서는 OR·NAND를 **따로따로** 학습시켜 붙였습니다. 은닉층이 무엇을 배워야 할지 모를 때, 두 층을 **한꺼번에** 학습시키려면? → 8단원(다층 신경망), 9단원(역전파)
#
# **관찰 질문**
# 1. 학습률을 1.0이나 0.01로 바꾸면 학습에 걸리는 에포크 수가 어떻게 바뀌나요?
# 2. NAND는 AND와 결정 경계가 어떤 관계인가요?
# 3. 은닉층의 출력 `hidden`을 평면에 찍어 보면, XOR의 네 점이 어떻게 배치되나요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** `train_perceptron`을 쓰지 말고, **직접 손으로** OR과 NAND의 가중치·편향을 정해 출력이 맞는지 확인하세요.
#
# **문제 2.** 은닉층 출력 `hidden`의 네 점을 산점도로 그리고, XOR 정답 색으로 칠해 보세요. 직선 하나로 나눌 수 있게 바뀌었나요?

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# print("OR  :", perceptron(X, np.array([1.0, 1.0]), -0.5))     # 하나라도 1이면 z ≥ 0
# print("NAND:", perceptron(X, np.array([-1.0, -1.0]), 1.5))    # AND의 부호를 모두 뒤집음
#
# plt.scatter(hidden[:, 0], hidden[:, 1], c=targets["XOR"], cmap="coolwarm", s=200, edgecolors="k")
# plt.xlabel("OR 출력"); plt.ylabel("NAND 출력"); plt.title("은닉층이 만든 새 좌표")
# plt.show()
# # 입력 (0,1)과 (1,0)은 둘 다 (OR=1, NAND=1) 한 점으로 모이고, (0,0)→(0,1), (1,1)→(1,0)으로 간다.
# # 빨강(XOR=1)은 (1,1) 한 점뿐이므로 직선 하나(= AND)로 나눌 수 있다.
# ```
# </details>
