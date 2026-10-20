# %% [markdown]
# # 15. 모델 평가와 분석 — 혼동 행렬, 정밀도·재현율, 오류 분석
#
# > 인공지능 기초 실습 과정 · 단원 15/15
#
# **학습 목표**
# - 정확도 하나로는 알 수 없는 것을 **혼동 행렬** 로 찾아낸다.
# - 정밀도(Precision) · 재현율(Recall) · F1을 클래스별로 계산하고 해석한다.
# - MLP(12단원)와 CNN(13단원)을 같은 기준으로 비교하고, 틀린 예와 확신도를 분석한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 혼동 행렬, 정밀도, 재현율, F1, 확신도 |
# | 따라하기 | 모델 불러오기 → 혼동 행렬(Seaborn) → 분류 리포트 → 많이 헷갈린 쌍 → 확신도 분석 → 두 모델 비교 |
# | 연습문제 | "모르겠다"고 답하는 모델 만들기 |

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념
#
# ### 혼동 행렬 — 무엇을 무엇으로 착각했나
# 행 = 정답, 열 = 예측. 대각선은 맞힌 개수, 나머지는 **착각한 개수** 입니다.
#
# ### 클래스 하나(예: 숫자 9)를 기준으로
#
# | | 예측 = 9 | 예측 ≠ 9 |
# |---|---|---|
# | **정답 = 9** | TP (맞게 찾음) | FN (놓침) |
# | **정답 ≠ 9** | FP (잘못 찾음) | TN |
#
# | 지표 | 식 | 질문 |
# |---|---|---|
# | **정밀도 (Precision)** | TP / (TP + FP) | "9라고 한 것 중 진짜 9는?" |
# | **재현율 (Recall)** | TP / (TP + FN) | "진짜 9 중에 9라고 찾은 것은?" |
# | **F1** | 둘의 조화 평균 | 둘의 균형 |
#
# 불량 검출이라면 놓치면 안 되니 **재현율**, 스팸 차단이라면 정상 메일을 막으면 안 되니 **정밀도** 가 중요합니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 데이터와 모델 준비
# 12·13단원에서 저장한 모델을 불러옵니다. 없으면 여기서 빠르게 학습합니다.

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
xtr, xte = (x_train / 255.0).astype("float32"), (x_test / 255.0).astype("float32")


def get_mlp():
    if os.path.exists("../models/mnist_mlp.keras"):
        return keras.models.load_model("../models/mnist_mlp.keras")
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input(shape=(784,)), keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dropout(0.2), keras.layers.Dense(10, activation="softmax")])
    m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    m.fit(xtr.reshape(-1, 784), keras.utils.to_categorical(y_train, 10), epochs=3, batch_size=128, verbose=0)
    return m


def get_cnn():
    if os.path.exists("../models/mnist_cnn.keras"):
        return keras.models.load_model("../models/mnist_cnn.keras")
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input(shape=(28, 28, 1)),
                          keras.layers.Conv2D(16, 3, activation="relu"), keras.layers.MaxPooling2D(),
                          keras.layers.Conv2D(32, 3, activation="relu"), keras.layers.MaxPooling2D(),
                          keras.layers.Flatten(), keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(xtr[..., None], y_train, epochs=2, batch_size=128, verbose=0)
    return m


mlp, cnn = get_mlp(), get_cnn()
probs = {"MLP": mlp.predict(xte.reshape(-1, 784), verbose=0), "CNN": cnn.predict(xte[..., None], verbose=0)}
preds = {k: p.argmax(axis=1) for k, p in probs.items()}
for k, p in preds.items():
    print(f"{k} 정확도: {(p == y_test).mean():.4f}")

# %% [markdown]
# ### 2-2. 혼동 행렬 — Seaborn 히트맵

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (name, p) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, p)
    off = cm.copy(); np.fill_diagonal(off, 0)                 # 대각선을 빼야 착각이 잘 보임
    sns.heatmap(off, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax)
    ax.set_title(f"{name} — 틀린 것만 (대각선 제외)"); ax.set_xlabel("예측"); ax.set_ylabel("정답")
plt.show()

# %% [markdown]
# ### 2-3. 클래스별 정밀도·재현율·F1

# %%
reports = {k: pd.DataFrame(classification_report(y_test, p, output_dict=True)).T.iloc[:10] for k, p in preds.items()}
compare = pd.concat({k: r[["precision", "recall", "f1-score"]] for k, r in reports.items()}, axis=1).round(3)
compare.index.name = "숫자"
compare

# %%
recall = pd.DataFrame({k: r["recall"] for k, r in reports.items()})
recall.plot.bar(figsize=(10, 3.5), ylim=(recall.min().min() - 0.02, 1.0), title="숫자별 재현율")
plt.xlabel("숫자"); plt.show()

# %% [markdown]
# ### 2-4. 가장 많이 헷갈린 쌍

# %%
def top_confusions(y_true, y_pred, n=5):
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0][:n]
    return pd.DataFrame([{"정답": t, "예측": p, "횟수": cm[t, p]} for t, p in idx])


pd.concat({k: top_confusions(y_test, p) for k, p in preds.items()}, axis=1)

# %%
t, p = top_confusions(y_test, preds["CNN"]).iloc[0][["정답", "예측"]]
idx = np.where((y_test == t) & (preds["CNN"] == p))[0][:8]
fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.4))
for ax, i in zip(np.atleast_1d(axes), idx):
    ax.imshow(x_test[i], cmap="gray"); ax.set_title(f"확신 {probs['CNN'][i].max():.2f}", fontsize=9); ax.axis("off")
plt.suptitle(f"CNN이 {t}를 {p}로 착각한 예")
plt.show()

# %% [markdown]
# ### 2-5. 확신도 — 맞힐 때와 틀릴 때
# 소프트맥스의 가장 큰 확률을 "확신도"로 봅니다. 모델이 틀릴 때 스스로 덜 확신하는지 확인합니다.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
for ax, (name, pr) in zip(axes, probs.items()):
    conf = pr.max(axis=1)
    correct = preds[name] == y_test
    sns.histplot(conf[correct], bins=30, stat="density", color="tab:blue", label="맞힘", ax=ax, alpha=0.5)
    sns.histplot(conf[~correct], bins=30, stat="density", color="tab:red", label="틀림", ax=ax, alpha=0.5)
    ax.set_title(f"{name} 확신도 분포"); ax.set_xlabel("가장 큰 확률"); ax.legend()
plt.show()

# %% [markdown]
# 틀린 예는 확신도가 낮은 쪽에 몰려 있습니다. 그렇다면 **확신이 낮을 때는 답하지 않게** 할 수 있습니다(연습문제).
#
# ### 2-6. 두 모델 한 장 요약

# %%
summary = pd.DataFrame([{
    "모델": k,
    "파라미터": m.count_params(),
    "정확도": (preds[k] == y_test).mean(),
    "가장 낮은 재현율(숫자)": f"{reports[k]['recall'].min():.3f} ({reports[k]['recall'].idxmin()})",
    "틀린 개수": int((preds[k] != y_test).sum()),
} for k, m in [("MLP", mlp), ("CNN", cnn)]])
summary.round(4)

# %% [markdown]
# **관찰 질문**
# 1. 두 모델이 공통으로 많이 헷갈리는 쌍이 있나요? 사람이 봐도 헷갈리는 쌍인가요?
# 2. 정확도가 99%라도 특정 숫자의 재현율이 낮다면, 그 모델을 우편번호 인식기에 써도 될까요?
# 3. 이 과정의 1단원에서 본 "이미지 = 숫자 배열"이, 15단원에서는 어떤 판단까지 이어졌나요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** CNN에서 확신도가 기준값(예: 0.9) 미만이면 "모름"으로 처리하세요. 기준값을 0.5, 0.7, 0.9, 0.99로 바꿔 가며
# - 답한 비율(커버리지)
# - 답한 것 중 정확도
#
# 를 표로 만드세요. 현장(예: 불량 검사)에서는 "모름"을 사람이 다시 보게 할 수 있습니다.
#
# **문제 2.** 재현율이 가장 낮은 숫자 하나를 골라, 그 숫자를 놓친 예(FN)를 16개 그려 보고 공통점을 찾으세요.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# conf = probs["CNN"].max(axis=1)
# rows = []
# for th in [0.5, 0.7, 0.9, 0.99]:
#     ans = conf >= th
#     rows.append({"기준": th, "답한 비율": ans.mean(), "답한 것의 정확도": (preds["CNN"][ans] == y_test[ans]).mean()})
# print(pd.DataFrame(rows).round(4))
# # 기준을 높일수록 답하는 비율은 줄고, 답한 것의 정확도는 올라간다 (트레이드오프)
#
# d = int(reports["CNN"]["recall"].astype(float).idxmin())
# miss = np.where((y_test == d) & (preds["CNN"] != d))[0][:16]
# fig, axes = plt.subplots(2, 8, figsize=(14, 4))
# for ax, i in zip(axes.ravel(), miss):
#     ax.imshow(x_test[i], cmap="gray"); ax.set_title(f"예측 {preds['CNN'][i]}"); ax.axis("off")
# plt.show()
# ```
# </details>
#
# ---
#
# ## 과정을 마치며
#
# | 단원 | 배운 것 | 이어진 곳 |
# |---|---|---|
# | 1–3 | 이미지는 숫자 배열, 필터는 3×3 숫자 | 13단원 CNN이 그 필터를 **학습** |
# | 4–5 | 퍼셉트론과 XOR의 벽 | 8–10단원 다층 신경망과 역전파로 돌파 |
# | 6–7 | NumPy·Pandas·Scikit-learn | 모든 실습의 도구 |
# | 8–11 | 신경망 학습 원리, TensorFlow, TensorBoard | 12–14단원 MNIST와 CNN |
# | 12–15 | MLP → CNN → 유명 모델 → 평가 | 실제 프로젝트(검출, 이상 탐지, 로봇 정책)의 출발점 |
