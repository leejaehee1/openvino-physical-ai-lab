# %% [markdown]
# # 02. 이미지 프로세싱
#
# > 인공지능 기초 실습 과정 · 단원 2/15
#
# **학습 목표**
# - 블러, 샤프닝, 윤곽선 검출, 평활화, 잡음 제거를 직접 적용해 결과를 비교한다.
# - "이웃 픽셀을 어떻게 섞느냐"에 따라 전혀 다른 효과가 난다는 것을 관찰한다. (→ 3단원 필터와 합성곱)
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 디지털 영상 처리로 할 수 있는 일 |
# | 따라하기 | 평활화 → 블러 → 샤프닝 → 윤곽선 → 잡음 제거 → 회전 |
# | 연습문제 | 잡음 종류에 따라 알맞은 필터 고르기 |
#
# > 원래 강의에서는 고전 실습 이미지인 Lena를 썼지만, 이 과정은 출처와 사용 허락이 분명한 scikit-image 내장 이미지(`camera`)를 씁니다.
# > Lena를 쓰고 싶다면 이미지 파일을 `../data/`에 넣고 `cv2.imread`로 바꿔 열면 됩니다.

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


def show(images, titles, cmap="gray", size=4):
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for ax, im, t in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    plt.show()


gray = data.camera()            # 512×512 흑백
print(gray.shape, gray.dtype)

# %% [markdown]
# ## 1. 개념 — 영상 처리로 할 수 있는 일
#
# | 작업 | 하는 일 | 쓰이는 곳 |
# |---|---|---|
# | 평활화 (Equalization) | 밝기 분포를 고르게 펴서 대비를 높임 | 어두운 사진 보정 |
# | 블러 (Blurring) | 이웃과 평균 → 부드럽게 | 잡음 감소, 전처리 |
# | 샤프닝 (Sharpening) | 이웃과의 차이를 키움 → 또렷하게 | 흐린 사진 보정 |
# | 윤곽선 검출 (Edge) | 밝기가 급하게 바뀌는 곳 찾기 | 물체 경계, 특징 추출 |
# | 잡음 제거 (Denoising) | 튀는 값 없애기 | 센서 잡음 |
# | 기하 변환 | 회전, 이동, 크기 변경 | 데이터 증강 |
#
# 대부분은 **"한 픽셀의 새 값 = 주변 픽셀들의 어떤 조합"** 으로 만들어집니다. 그 "조합 방법"이 3단원의 **필터(커널)** 입니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 히스토그램 평활화

# %%
dark = (gray * 0.4).astype(np.uint8)                 # 일부러 어둡게
eq = cv2.equalizeHist(dark)
show([dark, eq], ["어두운 원본", "평활화"])

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(dark.ravel(), bins=64); axes[0].set_title("평활화 전 분포")
axes[1].hist(eq.ravel(), bins=64);   axes[1].set_title("평활화 후 분포")
plt.show()

# %% [markdown]
# ### 2-2. 블러 — 평균 블러와 가우시안 블러

# %%
mean5 = cv2.blur(gray, (5, 5))                   # 5×5 영역 단순 평균
gauss5 = cv2.GaussianBlur(gray, (5, 5), 1.5)     # 가까울수록 큰 가중치
mean15 = cv2.blur(gray, (15, 15))
show([gray, mean5, gauss5, mean15], ["원본", "평균 5×5", "가우시안 5×5", "평균 15×15"])

# %% [markdown]
# ### 2-3. 샤프닝
# 원본에 "원본 − 블러(= 세부 성분)"를 더하면 경계가 또렷해집니다. (언샤프 마스킹)

# %%
blur = cv2.GaussianBlur(gray, (0, 0), 3)
sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)     # 1.8·원본 − 0.8·블러
show([gray, sharp], ["원본", "샤프닝"])

# %% [markdown]
# ### 2-4. 윤곽선 검출 — Canny

# %%
edges_lo = cv2.Canny(gray, 50, 100)
edges_hi = cv2.Canny(gray, 150, 300)
show([gray, edges_lo, edges_hi], ["원본", "Canny (50, 100)", "Canny (150, 300)"])

# %% [markdown]
# 임계값을 높이면 강한 경계만 남습니다. **어디까지를 "경계"로 볼지 사람이 정하는** 방식이라는 점을 기억하세요. CNN은 이것을 데이터로 배웁니다(13단원).
#
# ### 2-5. 잡음 넣고 없애기

# %%
rng = np.random.default_rng(0)
gauss_noise = np.clip(gray + rng.normal(0, 25, gray.shape), 0, 255).astype(np.uint8)  # 가우시안 잡음
sp = gray.copy()                                                                        # 소금·후추 잡음
mask = rng.random(gray.shape)
sp[mask < 0.03] = 0
sp[mask > 0.97] = 255

show([gauss_noise, cv2.GaussianBlur(gauss_noise, (5, 5), 0), cv2.medianBlur(gauss_noise, 5)],
     ["가우시안 잡음", "→ 가우시안 블러", "→ 중앙값 필터"])
show([sp, cv2.GaussianBlur(sp, (5, 5), 0), cv2.medianBlur(sp, 3)],
     ["소금·후추 잡음", "→ 가우시안 블러", "→ 중앙값 필터"])

# %% [markdown]
# ### 2-6. 회전과 뒤집기 — 데이터 증강의 기초

# %%
h, w = gray.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h))
show([gray, rotated, cv2.flip(gray, 1)], ["원본", "30° 회전", "좌우 반전"])

# %% [markdown]
# 사진을 돌리거나 뒤집어도 "카메라맨"은 카메라맨입니다. 모델 학습 때 이런 변형으로 데이터를 늘리는 것을 **데이터 증강** 이라고 합니다(14단원 AlexNet).
#
# **관찰 질문**
# 1. 평균 블러와 가우시안 블러의 결과는 어떻게 다른가요?
# 2. 소금·후추 잡음에는 왜 중앙값 필터가 훨씬 잘 듣나요?
# 3. 블러 → 샤프닝은 서로 반대 작업처럼 보입니다. 원본으로 완전히 되돌릴 수 있을까요?

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** `sp`(소금·후추 잡음 이미지)를 원본과 가장 비슷하게 복원하는 필터와 크기를 찾아보세요. 원본과의 평균 절대 오차(MAE)로 비교합니다.
#
# **문제 2.** 잡음 제거 후 Canny 윤곽선을 그려, 잡음 제거를 했을 때와 안 했을 때를 비교하세요.

# %%
def mae(a, b):
    return np.abs(a.astype(float) - b.astype(float)).mean()

print("잡음 이미지 MAE:", round(mae(sp, gray), 2))
# 문제 1, 2 — 여기에 작성

# %% [markdown]
# <details>
# <summary>정답 보기</summary>
#
# ```python
# for k in [3, 5, 7]:
#     print("median", k, round(mae(cv2.medianBlur(sp, k), gray), 2),
#           "| gaussian", k, round(mae(cv2.GaussianBlur(sp, (k, k), 0), gray), 2))
# # 보통 중앙값 3×3이 가장 낮다 (튀는 값을 평균에 섞지 않고 버리기 때문)
#
# show([cv2.Canny(sp, 100, 200), cv2.Canny(cv2.medianBlur(sp, 3), 100, 200)],
#      ["잡음 그대로 Canny", "중앙값 후 Canny"])
# ```
# </details>
