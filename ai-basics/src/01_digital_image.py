# %% [markdown]
# # 01. 디지털 이미지 기초
#
# > 인공지능 기초 실습 과정 · 단원 1/15
#
# **학습 목표**
# - 컴퓨터에게 이미지는 **숫자 배열** 이라는 것을 직접 확인한다.
# - 픽셀, 해상도, 채널(RGB), Grayscale, HSV를 설명할 수 있다.
# - 색 공간을 바꾸면 어떤 일이 쉬워지는지(색으로 물체 찾기) 체험한다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 픽셀 · 해상도 · 채널 · 색 공간 |
# | 따라하기 | 이미지 열기 → 픽셀 읽기 → 해상도 바꾸기 → RGB 분리 → Grayscale → HSV |
# | 연습문제 | HSV로 특정 색 영역만 골라내기 |
#
# 이 단원에서 만든 감각(이미지 = 숫자 행렬)은 3단원 합성곱, 13단원 CNN까지 그대로 이어집니다.

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from skimage import data

for f in ["NanumGothic", "AppleGothic", "Malgun Gothic"]:
    if any(f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# %% [markdown]
# ## 1. 개념 — 이미지는 숫자 격자다
#
# | 용어 | 뜻 | 예 |
# |---|---|---|
# | **픽셀 (pixel)** | 이미지를 이루는 가장 작은 점 하나 | 한 칸 |
# | **해상도** | 가로 × 세로 픽셀 수 | 640 × 480 |
# | **채널** | 픽셀 하나가 가진 값의 개수 | 흑백 1개, 컬러 3개(R, G, B) |
# | **비트 깊이** | 값 하나의 범위 | 8비트 → 0(어두움) ~ 255(밝음) |
#
# 컬러 이미지 한 장 = **높이 × 너비 × 3** 크기의 숫자 배열입니다.
#
# > 실습 이미지: scikit-image에 들어 있는 `astronaut`(NASA 우주비행사 사진, 퍼블릭 도메인)와 `camera` 사진을 씁니다.
# > 인터넷 없이도 열리고 저작권 걱정이 없습니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 이미지 열고 모양 보기

# %%
img = data.astronaut()          # RGB 순서
print("모양 (높이, 너비, 채널):", img.shape)
print("자료형:", img.dtype, " 최솟값:", img.min(), " 최댓값:", img.max())
plt.imshow(img)
plt.title("astronaut")
plt.axis("off")
plt.show()

# %% [markdown]
# ### 2-2. 픽셀 하나 읽어 보기
# `img[행, 열]` → 그 위치의 `[R, G, B]` 값입니다. 행이 y(세로), 열이 x(가로)라는 점에 주의하세요.

# %%
y, x = 100, 200
print(f"({y}, {x}) 위치의 [R, G, B] =", img[y, x])

# 일부 영역만 잘라 보기 (관심 영역, ROI)
roi = img[50:150, 150:250]
print("잘라낸 영역 모양:", roi.shape)

# 아주 작은 영역의 숫자를 직접 출력
print(img[100:105, 200:205, 0])   # R 채널 5×5

# %% [markdown]
# ### 2-3. 해상도 바꾸기
# 해상도를 낮추면 숫자 개수가 줄고(계산이 빨라지고), 세부 정보가 사라집니다.
# 딥러닝 모델은 보통 입력 크기를 고정하기 때문에(예: 224 × 224) 이 과정을 항상 거칩니다.

# %%
sizes = [512, 128, 32, 16]
fig, axes = plt.subplots(1, len(sizes), figsize=(12, 3))
for ax, s in zip(axes, sizes):
    small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
    ax.imshow(small)
    ax.set_title(f"{s}×{s}  ({small.size:,}개 숫자)")
    ax.axis("off")
plt.show()

# %% [markdown]
# ### 2-4. RGB 채널 나눠 보기

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, (ax, name) in enumerate(zip(axes, ["R (빨강)", "G (초록)", "B (파랑)"])):
    ax.imshow(img[:, :, i], cmap="gray")
    ax.set_title(name)
    ax.axis("off")
plt.show()

# %% [markdown]
# 빨간 성조기 부분은 R 채널에서 밝고, B 채널에서 어둡습니다. **각 채널은 그 색이 얼마나 강한지를 나타내는 흑백 이미지** 입니다.
#
# ### 2-5. Grayscale — 밝기만 남기기
# 사람 눈은 초록에 가장 민감해서, 단순 평균이 아니라 **가중 평균** 을 씁니다.
#
# $$ \text{Gray} = 0.299R + 0.587G + 0.114B $$

# %%
gray_manual = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
gray_cv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print("모양:", gray_cv.shape, " 직접 계산과 OpenCV의 최대 차이:", np.abs(gray_manual.astype(int) - gray_cv).max())

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(gray_manual, cmap="gray"); axes[0].set_title("직접 계산")
axes[1].imshow(gray_cv, cmap="gray");     axes[1].set_title("OpenCV")
for a in axes: a.axis("off")
plt.show()

# %% [markdown]
# ### 2-6. HSV — 색을 "색상 · 채도 · 명도"로 나누기
#
# | 채널 | 뜻 | OpenCV 범위 |
# |---|---|---|
# | **H (Hue, 색상)** | 무슨 색인가 (빨강 → 노랑 → 초록 → 파랑 → 빨강) | 0 ~ 179 |
# | **S (Saturation, 채도)** | 얼마나 선명한가 | 0 ~ 255 |
# | **V (Value, 명도)** | 얼마나 밝은가 | 0 ~ 255 |
#
# RGB에서는 "빨간색"을 찾으려면 세 값을 동시에 따져야 하지만, HSV에서는 **H 하나로 색을 고를 수 있어** 조명이 조금 바뀌어도 덜 흔들립니다.

# %%
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, (ax, name) in enumerate(zip(axes, ["H 색상", "S 채도", "V 명도"])):
    ax.imshow(hsv[:, :, i], cmap="hsv" if i == 0 else "gray")
    ax.set_title(name)
    ax.axis("off")
plt.show()

# %% [markdown]
# ### 2-7. 밝기 분포 — 히스토그램
# 히스토그램은 "0~255 각 값이 몇 번 나왔나"입니다. 너무 어둡거나 밝은 사진을 숫자로 진단할 수 있습니다.

# %%
plt.figure(figsize=(8, 3))
for i, c in enumerate(["r", "g", "b"]):
    plt.hist(img[:, :, i].ravel(), bins=64, color=c, alpha=0.4, label=c.upper())
plt.legend(); plt.title("RGB 히스토그램"); plt.xlabel("값 (0~255)"); plt.ylabel("픽셀 수")
plt.show()

# %% [markdown]
# **관찰 질문**
# 1. 512×512 컬러 이미지 한 장은 숫자 몇 개인가요?
# 2. 16×16으로 줄였을 때도 "사람"이라는 것을 알아볼 수 있나요? 모델 입력 크기를 정할 때 무엇을 고려해야 할까요?
# 3. Grayscale에서 초록 가중치가 가장 큰 이유는?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** HSV로 이미지에서 **빨간색 영역만** 흰색(255)으로 표시한 마스크를 만들어 보세요.
# - 힌트: 빨강은 H가 0 근처와 179 근처 **양쪽** 에 있습니다. `cv2.inRange`를 두 번 쓰고 합칩니다.
#
# **문제 2.** 마스크를 이용해 원본에서 빨간 부분만 남기고 나머지는 검게 만드세요.

# %%
# 문제 1, 2 — 여기에 작성
lower1, upper1 = np.array([0, 100, 80]), np.array([10, 255, 255])
# mask = ...
# result = ...

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# lower1, upper1 = np.array([0, 100, 80]),   np.array([10, 255, 255])
# lower2, upper2 = np.array([170, 100, 80]), np.array([179, 255, 255])
# mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
# result = cv2.bitwise_and(img, img, mask=mask)
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# axes[0].imshow(mask, cmap="gray"); axes[1].imshow(result)
# plt.show()
# ```
# </details>
#
# **더 생각해 보기**: M5 로봇 수업에서 "빨간 큐브 찾기"를 딥러닝 없이 한다면 이 방법을 쓸 수 있습니다. 어떤 상황에서 실패할까요? (조명, 비슷한 색의 배경 …) → 그래서 학습하는 모델이 필요해집니다.
