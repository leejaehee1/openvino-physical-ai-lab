# %% [markdown]
# # 06. Python 데이터 처리 — NumPy 배열·행렬, Pandas DataFrame
#
# > 인공지능 기초 실습 과정 · 단원 6/15
#
# **학습 목표**
# - 5단원에서 쓴 `X @ w + b`가 **행렬 곱** 이라는 것을 이해하고, 반복문 없이 계산하는 법(벡터화)을 익힌다.
# - 배열의 모양(shape), 형태 바꾸기(reshape), 브로드캐스팅을 다룬다.
# - Pandas DataFrame으로 데이터를 정리·요약하고, Seaborn으로 그린다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 스칼라 · 벡터 · 행렬 · 텐서, 행렬 곱의 규칙 |
# | 따라하기 | 배열 만들기 → 모양 바꾸기 → 브로드캐스팅 → 행렬 곱으로 퍼셉트론 → DataFrame → Seaborn |
# | 연습문제 | 붓꽃 데이터 요약하기 |

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.datasets import load_iris

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념 — 숫자를 담는 그릇들
#
# | 이름 | 차원(rank) | 모양 예 | 예 |
# |---|---|---|---|
# | 스칼라 | 0 | `()` | 학습률 0.1 |
# | 벡터 | 1 | `(3,)` | 입력 하나 `[x1, x2, x3]` |
# | 행렬 | 2 | `(4, 2)` | 입력 4개 × 특징 2개 |
# | 텐서 | 3 이상 | `(512, 512, 3)` | 컬러 이미지 (1단원) |
#
# **행렬 곱 규칙**: $(m \times n) \cdot (n \times p) = (m \times p)$ — 앞 행렬의 **열 수** 와 뒤 행렬의 **행 수** 가 같아야 합니다.
# 결과의 $(i, j)$ 칸 = 앞 행렬 $i$번째 행과 뒤 행렬 $j$번째 열의 **점곱**(원소끼리 곱해 더하기).

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 배열 만들기와 모양

# %%
a = np.array([1, 2, 3])
M = np.array([[1, 2, 3], [4, 5, 6]])
print("a:", a, " shape:", a.shape, " ndim:", a.ndim)
print("M:\n", M, "\n shape:", M.shape, " ndim:", M.ndim, " dtype:", M.dtype)
print("zeros:\n", np.zeros((2, 3)))
print("arange:", np.arange(0, 10, 2), " linspace:", np.linspace(0, 1, 5))

# %% [markdown]
# ### 2-2. 인덱싱과 모양 바꾸기

# %%
img_like = np.arange(24).reshape(2, 3, 4)     # 텐서 (2, 3, 4)
print("원래 모양:", img_like.shape)
print("첫 번째 2차원 조각:\n", img_like[0])
print("reshape(6, 4):", img_like.reshape(6, 4).shape)
print("flatten:", img_like.flatten().shape)        # 12단원 MNIST에서 28×28 → 784로 펼칠 때
print("전치 M.T:\n", M.T, M.T.shape)

# %% [markdown]
# ### 2-3. 브로드캐스팅 — 모양이 달라도 자동으로 맞춰 계산

# %%
print("M + 10:\n", M + 10)                       # 스칼라를 모든 칸에
print("M + [100, 200, 300]:\n", M + np.array([100, 200, 300]))   # 행마다 같은 벡터를
gray = np.random.default_rng(0).integers(0, 256, (4, 4))
print("0~1로 정규화:\n", (gray / 255).round(2))   # 12단원에서 이미지 전처리로 다시 씀

# %% [markdown]
# ### 2-4. 행렬 곱으로 퍼셉트론 계산 — 5단원의 `X @ w + b`
#
# 입력 4개(4×2)와 가중치(2,)를 곱하면 가중합 4개가 **한 번에** 나옵니다. 편향 b는 브로드캐스팅으로 4칸 모두에 더해집니다.

# %%
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
w = np.array([1.0, 1.0])
b = -1.5

z_loop = [X[i, 0] * w[0] + X[i, 1] * w[1] + b for i in range(4)]   # 반복문
z_vec = X @ w + b                                                    # 행렬 곱
print("반복문:", z_loop)
print("행렬 곱:", z_vec)

# %% [markdown]
# 뉴런이 여러 개면 가중치도 행렬이 됩니다. 입력 (4, 2) · 가중치 (2, **3**) = (4, 3): **뉴런 3개의 출력이 한 번에** 나옵니다. 8단원 다층 신경망의 핵심 계산입니다.

# %%
W = np.array([[1.0, -1.0, 0.5],
              [1.0, -1.0, 0.5]])           # (2, 3): 입력 2개 → 뉴런 3개
B = np.array([-1.5, 1.5, -0.2])
print("출력 모양:", (X @ W + B).shape)
print((X @ W + B >= 0).astype(int))         # 열마다 AND, NAND, OR

# %% [markdown]
# ### 2-5. 왜 벡터화인가 — 속도 비교

# %%
big = np.random.default_rng(0).random((200_000, 2))
t0 = time.perf_counter(); _ = [r[0] * w[0] + r[1] * w[1] + b for r in big]; t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); _ = big @ w + b;                                   t_vec = time.perf_counter() - t0
print(f"반복문 {t_loop * 1000:.1f} ms  vs  행렬 곱 {t_vec * 1000:.2f} ms  → 약 {t_loop / t_vec:.0f}배")

# %% [markdown]
# 딥러닝이 GPU를 쓰는 이유도 같습니다. 신경망 계산은 거의 전부 **큰 행렬 곱** 이라, 한꺼번에 계산할수록 빠릅니다.
#
# ### 2-6. Pandas DataFrame — 표로 정리하기
# 5단원 퍼셉트론 결과를 표로 만들고, 실제 데이터셋(붓꽃)을 다뤄 봅니다.

# %%
df = pd.DataFrame(X, columns=["x1", "x2"])
df["z"] = X @ w + b
df["AND"] = (df["z"] >= 0).astype(int)
df

# %%
iris = load_iris(as_frame=True)
flowers = iris.frame.rename(columns={
    "sepal length (cm)": "꽃받침 길이", "sepal width (cm)": "꽃받침 너비",
    "petal length (cm)": "꽃잎 길이", "petal width (cm)": "꽃잎 너비", "target": "품종번호"})
flowers["품종"] = flowers["품종번호"].map(dict(enumerate(iris.target_names)))
print(flowers.shape)
flowers.head()

# %%
flowers.describe().round(2)

# %%
flowers.groupby("품종")[["꽃잎 길이", "꽃잎 너비"]].mean().round(2)

# %% [markdown]
# ### 2-7. Seaborn으로 한눈에 보기

# %%
sns.pairplot(flowers.drop(columns="품종번호"), hue="품종", height=2.2)
plt.show()

# %% [markdown]
# 꽃잎 길이·너비만 봐도 setosa는 **직선 하나로** 나머지와 나뉩니다. 7단원에서 이것을 퍼셉트론과 머신러닝 모델로 직접 나눠 봅니다.
#
# **관찰 질문**
# 1. (4, 2) 행렬과 (3, 2) 행렬은 그대로 곱할 수 있나요? 어떻게 하면 곱할 수 있을까요?
# 2. `X @ W`에서 W의 열 하나가 뉴런 하나에 해당한다는 것을 설명해 보세요.
# 3. pairplot에서 versicolor와 virginica는 어느 특징으로 가장 잘 나뉘어 보이나요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 품종별로 4개 특징의 **평균과 표준편차** 를 한 표로 구하세요.
#
# **문제 2.** 꽃잎 길이가 5cm 이상인 꽃은 몇 송이이고, 품종별로 몇 송이인가요?
#
# **문제 3.** 붓꽃 4개 특징 데이터(150×4)에 (4, 3) 크기의 무작위 가중치 행렬을 곱하면 결과 모양은? 직접 계산해 확인하세요.

# %%
# 문제 1~3 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# feats = ["꽃받침 길이", "꽃받침 너비", "꽃잎 길이", "꽃잎 너비"]
# print(flowers.groupby("품종")[feats].agg(["mean", "std"]).round(2))
#
# big = flowers[flowers["꽃잎 길이"] >= 5]
# print(len(big)); print(big["품종"].value_counts())
#
# Wr = np.random.default_rng(0).normal(size=(4, 3))
# print((flowers[feats].to_numpy() @ Wr).shape)   # (150, 3)
# ```
# </details>
