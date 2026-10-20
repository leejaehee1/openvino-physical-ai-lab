# %% [markdown]
# # 07. 머신러닝 기초 — Scikit-learn, 선형 분류, 결정 경계
#
# > 인공지능 기초 실습 과정 · 단원 7/15
#
# **학습 목표**
# - 인공지능 · 머신러닝 · 딥러닝의 관계를 설명할 수 있다.
# - 학습 데이터와 테스트 데이터를 나누는 이유를 알고, Scikit-learn으로 **학습 → 예측 → 평가** 흐름을 실행한다.
# - 선형 분류기(퍼셉트론, 로지스틱 회귀)의 **결정 경계** 를 그려 보고, 직선으로 못 나누는 데이터에서 한계를 확인한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | AI ⊃ 머신러닝 ⊃ 딥러닝, 지도학습, 학습/테스트 분리 |
# | 따라하기 | 붓꽃 분류 → 결정 경계 → 초승달 데이터에서 선형 모델의 한계 |
# | 연습문제 | 특징 고르기와 모델 비교 |

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.datasets import load_iris, make_moons
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념
#
# ```
#  인공지능 (사람처럼 판단하는 기계 전체)
#   └─ 머신러닝 (규칙을 사람이 쓰지 않고 데이터에서 배움)
#        └─ 딥러닝 (여러 층의 신경망으로 배움)
# ```
#
# | 구분 | 규칙 기반 프로그램 | 머신러닝 |
# |---|---|---|
# | 입력 | 데이터 + **규칙** | 데이터 + **정답** |
# | 출력 | 답 | **규칙(모델)** |
# | 예 | 2단원 Canny의 임계값을 사람이 정함 | 붓꽃 품종을 가르는 경계를 데이터로 찾음 |
#
# **지도학습**: 정답(라벨)이 있는 데이터로 배우는 방식. 이 과정의 거의 모든 실습이 지도학습입니다.
#
# **왜 학습/테스트를 나누나?** 공부한 문제로만 시험을 보면 실력을 알 수 없습니다. 모델이 **처음 보는 데이터** 에서도 맞히는지 보려고 일부를 떼어 둡니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 데이터 준비 — 붓꽃, 특징 2개만

# %%
iris = load_iris()
X = iris.data[:, 2:4]            # 꽃잎 길이, 꽃잎 너비 (그림으로 그리기 쉽게 2개만)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("학습:", X_train.shape, " 테스트:", X_test.shape)

# %% [markdown]
# ### 2-2. Scikit-learn의 공통 사용법 — `fit` → `predict` → 평가
# 어떤 모델이든 사용법이 같습니다. 5단원에서 손으로 만든 퍼셉트론도 `Perceptron` 한 줄입니다.

# %%
models = {
    "퍼셉트론": Perceptron(random_state=0),
    "로지스틱 회귀": LogisticRegression(max_iter=1000),
    "k-최근접 이웃 (k=5)": KNeighborsClassifier(n_neighbors=5),
}
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)                        # 학습
    pred = m.predict(X_test)                       # 예측
    rows.append({"모델": name, "학습 정확도": accuracy_score(y_train, m.predict(X_train)),
                 "테스트 정확도": accuracy_score(y_test, pred)})
pd.DataFrame(rows).round(3)

# %% [markdown]
# ### 2-3. 결정 경계 그리기
# 평면의 모든 점에 대해 모델이 어떤 품종이라고 답하는지 색칠하면, 모델이 배운 **규칙의 모양** 이 보입니다.

# %%
def plot_decision(ax, model, X, y, title):
    pad = 0.5
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=30)
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, m) in zip(axes, models.items()):
    plot_decision(ax, m, X, y, name)
    ax.set_xlabel("꽃잎 길이"); ax.set_ylabel("꽃잎 너비")
plt.show()

# %% [markdown]
# 퍼셉트론과 로지스틱 회귀의 경계는 **직선** 입니다(선형 분류기). k-최근접 이웃은 주변 점을 보고 정하기 때문에 경계가 구불구불합니다.
#
# 퍼셉트론의 정확도가 유난히 낮은 것도 눈여겨보세요. versicolor와 virginica는 직선으로 완전히 나뉘지 않는데, 퍼셉트론은 "틀리면 고친다"를 반복할 뿐 **덜 틀리는 선** 을 찾는 기준이 없어 흔들리다 멈춥니다. 로지스틱 회귀는 손실 함수(9단원)를 줄이는 방식이라 겹치는 데이터에서도 안정적인 선을 찾습니다.
#
# ### 2-4. 로지스틱 회귀 — 확률을 내는 선형 분류기
# 퍼셉트론은 0/1만 내지만, 로지스틱 회귀는 **시그모이드** 로 "1일 확률"을 냅니다. 8단원 활성화 함수, 12단원 소프트맥스로 이어집니다.

# %%
logit = models["로지스틱 회귀"]
sample = X_test[:5]
proba = logit.predict_proba(sample)
pd.DataFrame(proba, columns=iris.target_names).assign(정답=[iris.target_names[t] for t in y_test[:5]]).round(3)

# %% [markdown]
# ### 2-5. 직선으로 못 나누는 데이터 — 초승달

# %%
Xm, ym = make_moons(n_samples=400, noise=0.2, random_state=0)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)

moon_models = {"로지스틱 회귀 (선형)": LogisticRegression(), "k-최근접 이웃 (비선형)": KNeighborsClassifier(15)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, m) in zip(axes, moon_models.items()):
    m.fit(Xm_tr, ym_tr)
    plot_decision(ax, m, Xm, ym, f"{name}\n테스트 정확도 {accuracy_score(ym_te, m.predict(Xm_te)):.2f}")
plt.show()

# %% [markdown]
# 초승달 두 개는 5단원의 XOR처럼 **직선 하나로 나눌 수 없습니다**. 선형 모델은 한계에 부딪히고, 곡선 경계를 만드는 모델이 필요합니다.
# → 8단원에서 **다층 신경망** 으로 같은 데이터를 풀어 봅니다.
#
# ### 2-6. 특징의 크기 맞추기 — 표준화
# 특징마다 단위가 다르면(예: cm와 mm) 큰 값이 결과를 좌우합니다. 평균 0, 표준편차 1로 맞추는 것이 **표준화** 입니다. 학습 데이터로 기준을 정하고, 테스트 데이터에는 **같은 기준** 을 적용합니다.

# %%
scaler = StandardScaler().fit(X_train)
print("표준화 전 평균:", X_train.mean(axis=0).round(2), " 표준편차:", X_train.std(axis=0).round(2))
Xs = scaler.transform(X_train)
print("표준화 후 평균:", Xs.mean(axis=0).round(2), " 표준편차:", Xs.std(axis=0).round(2))

# %% [markdown]
# **관찰 질문**
# 1. 학습 정확도와 테스트 정확도가 크게 다르다면 무슨 뜻일까요?
# 2. 초승달 데이터에서 k를 1로 줄이면 경계가 어떻게 바뀌나요? 테스트 정확도는?
# 3. 테스트 데이터로 표준화 기준을 정하면 왜 안 될까요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** 붓꽃의 **꽃받침** 길이·너비(특징 0, 1)만 써서 같은 세 모델을 학습시키고, 꽃잎을 쓸 때와 정확도를 비교하세요. 어느 특징이 더 유용한가요?
#
# **문제 2.** 초승달 데이터에서 k-최근접 이웃의 k를 1, 5, 15, 50으로 바꿔 학습·테스트 정확도를 표로 만드세요. 과대적합이 보이는 k는?

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(iris.data[:, :2], y, test_size=0.3, random_state=42, stratify=y)
# for name, m in models.items():
#     m.fit(Xs_tr, ys_tr); print(name, round(accuracy_score(ys_te, m.predict(Xs_te)), 3))
# # 꽃받침만 쓰면 versicolor와 virginica가 많이 겹쳐 정확도가 떨어진다
#
# rows = []
# for k in [1, 5, 15, 50]:
#     m = KNeighborsClassifier(k).fit(Xm_tr, ym_tr)
#     rows.append({"k": k, "학습": accuracy_score(ym_tr, m.predict(Xm_tr)), "테스트": accuracy_score(ym_te, m.predict(Xm_te))})
# print(pd.DataFrame(rows).round(3))
# # k=1: 학습 1.0, 테스트는 더 낮음 → 과대적합
# ```
# </details>
