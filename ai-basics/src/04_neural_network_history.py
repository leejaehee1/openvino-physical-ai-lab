# %% [markdown]
# # 04. 신경망의 역사 — 퍼셉트론에서 노벨상까지
#
# > 인공지능 기초 실습 과정 · 단원 4/15
#
# **학습 목표**
# - 인공 신경망이 **어떤 문제에 막히고, 어떻게 돌파했는지** 를 흐름으로 설명할 수 있다.
# - 퍼셉트론, 홉필드 네트워크, 역전파, 딥러닝 부활에서 힌턴(Hinton)과 홉필드(Hopfield)의 기여를 안다.
# - 홉필드 네트워크로 "기억에서 패턴을 복원"하는 것을 직접 실행해 본다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 생물 뉴런 → 인공 뉴런, 두 번의 겨울과 두 번의 봄 |
# | 따라하기 | 홉필드 네트워크로 깨진 글자 복원 |
# | 연습문제 | 패턴을 몇 개까지 기억할 수 있을까 |

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념 — 생물 뉴런에서 인공 뉴런으로
#
# | 생물 뉴런 | 인공 뉴런 |
# |---|---|
# | 수상돌기: 다른 뉴런의 신호를 받음 | 입력 $x_1, x_2, \dots$ |
# | 시냅스의 세기: 신호를 얼마나 강하게 전달할지 | 가중치 $w_1, w_2, \dots$ |
# | 세포체: 신호를 모음 | 가중합 $\sum w_i x_i + b$ |
# | 일정 이상이면 발화 | 활성화 함수 (계단, 시그모이드 …) |
# | 축삭: 다음 뉴런으로 전달 | 출력 $y$ |
#
# > 시냅스 결합이 강해지는 것이 학습과 기억의 원리로 여겨집니다(헵의 법칙). 인공 신경망의 "학습"은 곧 **가중치를 바꾸는 것** 입니다.

# %% [markdown]
# ## 2. 연표 — 두 번의 겨울과 두 번의 봄
#
# | 연도 | 사건 | 의미 |
# |---|---|---|
# | 1943 | 매컬러·피츠, 최초의 인공 뉴런 수학 모델 | 뉴런을 논리 회로처럼 표현 |
# | 1949 | 헵(Hebb)의 학습 법칙 | "함께 발화하는 뉴런은 연결이 강해진다" |
# | 1957–58 | **로젠블랫의 퍼셉트론** | 데이터로 가중치를 **학습** 하는 첫 기계 (5단원) |
# | 1969 | 민스키·페퍼트 『Perceptrons』 | 단층 퍼셉트론은 **XOR을 못 푼다** → **첫 번째 겨울** |
# | 1982 | **홉필드 네트워크** | 에너지가 낮아지는 쪽으로 움직이며 **기억을 복원** 하는 신경망 (아래 실습) |
# | 1985 | 힌턴 등, 볼츠만 머신 | 확률로 데이터의 분포를 배우는 신경망 |
# | 1986 | **루멜하트·힌턴·윌리엄스, 역전파** 보급 | 여러 층을 학습시키는 방법 → XOR 해결 (9단원) |
# | 1998 | 르쿤의 LeNet-5 | 합성곱 신경망으로 손글씨 인식 (14단원) |
# | 1990–2000년대 | 데이터·계산력 부족, 깊은 망 학습 어려움 | **두 번째 겨울** |
# | 2006 | **힌턴, 심층 신뢰망** — 층별 사전 학습 | "딥러닝"이라는 이름과 함께 부활 |
# | 2012 | **AlexNet** (힌턴 연구실) | GPU + 대규모 데이터로 이미지 인식 대회 압도 (14단원) |
# | 2018 | 튜링상: 힌턴 · 르쿤 · 벤지오 | 딥러닝의 세 개척자 |
# | 2024 | **노벨 물리학상: 홉필드 · 힌턴** | "인공 신경망 기계학습을 가능하게 한 기초 발견과 발명" |
#
# **흐름으로 보면**
# - 퍼셉트론(학습하는 뉴런) → XOR에 막힘 → **층을 쌓고 역전파로 학습** 해 돌파
# - 깊게 쌓으면 학습이 안 됨 → **사전 학습, ReLU, GPU, 큰 데이터** 로 돌파
# - 이 과정의 5 → 8 → 9 → 12 → 13단원이 이 역사를 그대로 따라갑니다.

# %% [markdown]
# ## 3. 따라하기 — 홉필드 네트워크로 깨진 글자 복원하기
#
# 홉필드 네트워크는 **기억 장치** 처럼 동작합니다.
# 1. **저장**: 기억할 패턴들(±1로 된 그림)로 가중치를 만든다 — 헵의 법칙 $W = \sum p\,p^T$ (대각선은 0)
# 2. **복원**: 일부가 깨진 그림을 넣으면, 각 뉴런이 $\text{sign}(W x)$로 바뀌기를 반복하며 가장 가까운 기억으로 수렴한다
#
# 이웃 뉴런들의 "의견"에 맞춰 자기 값을 바꾸다 보면 에너지가 낮아지는 쪽(저장된 기억)으로 굴러가는 원리입니다.

# %%
letters = {
    "H": ["X...X", "X...X", "XXXXX", "X...X", "X...X"],
    "T": ["XXXXX", "..X..", "..X..", "..X..", "..X.."],
    "O": [".XXX.", "X...X", "X...X", "X...X", ".XXX."],
}


def to_vec(rows):
    return np.array([1 if ch == "X" else -1 for r in rows for ch in r], dtype=np.float32)


patterns = np.stack([to_vec(r) for r in letters.values()])      # (3, 25)
N = patterns.shape[1]

# 저장: 헵의 법칙
W = sum(np.outer(p, p) for p in patterns) / N
np.fill_diagonal(W, 0)
print("가중치 행렬 모양:", W.shape)


def recall(x, steps=10):
    x = x.copy()
    history = [x.copy()]
    for _ in range(steps):
        for i in np.random.default_rng(0).permutation(N):      # 한 뉴런씩 차례로 갱신
            x[i] = 1 if W[i] @ x >= 0 else -1
        history.append(x.copy())
        if np.array_equal(history[-1], history[-2]):
            break
    return x, history


def show_row(vecs, titles):
    fig, axes = plt.subplots(1, len(vecs), figsize=(2.2 * len(vecs), 2.4))
    for ax, v, t in zip(axes, vecs, titles):
        ax.imshow(v.reshape(5, 5), cmap="gray_r", vmin=-1, vmax=1)
        ax.set_title(t); ax.axis("off")
    plt.show()


show_row(list(patterns), [f"기억 '{k}'" for k in letters])

# %%
rng = np.random.default_rng(1)
target = patterns[0]                                  # 'H'
noisy = target.copy()
flip = rng.choice(N, size=4, replace=False)           # 25칸 중 4칸 뒤집기
noisy[flip] *= -1

restored, history = recall(noisy)
show_row([noisy, restored, target], ["깨진 입력 (4칸 뒤집힘)", f"복원 ({len(history) - 1}회 반복)", "원래 기억"])
print("원래 기억과 같은가?", np.array_equal(restored, target))

# %% [markdown]
# **관찰 질문**
# 1. 이 네트워크는 "H"라는 정답을 따로 저장하지 않았습니다. 그런데 어떻게 H로 돌아갔을까요?
# 2. 5단원의 퍼셉트론이 "입력 → 정답"을 배우는 것과, 홉필드가 "패턴 자체를 기억"하는 것은 어떻게 다른가요?
# 3. 힌턴의 볼츠만 머신은 홉필드 네트워크에 **확률** 을 더한 것입니다. 왜 확률이 도움이 될까요? (힌트: 잘못된 기억에 갇히지 않기)

# %% [markdown]
# ## 4. 연습문제
#
# **문제 1.** 뒤집는 칸 수를 2, 4, 6, 8, 10으로 늘려 가며 무작위로 20번씩 시험해 복원 성공률을 구하세요. 몇 칸부터 눈에 띄게 실패하나요?
#
# **문제 2.** 글자를 하나 더(예: "L" 모양) 추가해 4개를 기억시키고, 복원이 여전히 잘 되는지 보세요.
# 홉필드 네트워크는 뉴런 수의 약 14%(25개면 3~4개) 정도까지만 안정적으로 기억한다고 알려져 있습니다.

# %%
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 예시</summary>
#
# ```python
# for k in [2, 4, 6, 8, 10]:
#     ok = 0
#     for seed in range(20):
#         x = target.copy()
#         x[np.random.default_rng(seed).choice(N, k, replace=False)] *= -1
#         ok += np.array_equal(recall(x)[0], target)
#     print(f"{k}칸 뒤집기: 20번 중 {ok}번 성공")
#
# letters["L"] = ["X....", "X....", "X....", "X....", "XXXXX"]
# patterns = np.stack([to_vec(r) for r in letters.values()])
# W = sum(np.outer(p, p) for p in patterns) / N
# np.fill_diagonal(W, 0)
# # 같은 실험을 반복 → 기억이 많을수록, 서로 닮을수록 복원이 어려워진다
# ```
# </details>
