# %% [markdown]
# # 03. 필터와 합성곱
#
# > 인공지능 기초 실습 과정 · 단원 3/15
#
# **학습 목표**
# - 커널(필터)과 합성곱(convolution)을 NumPy로 **직접 구현** 한다.
# - Sobel, Gaussian 커널이 무엇을 계산하는지 숫자로 이해한다.
# - 출력 크기 공식(패딩, 스트라이드)을 계산할 수 있다.
#
# | 진행 | 내용 |
# |---|---|
# | 개념 | 커널 · 합성곱 · 패딩 · 스트라이드 |
# | 따라하기 | 손 계산 → NumPy 구현 → OpenCV와 비교 → Sobel · Gaussian → 크기 공식 |
# | 연습문제 | 나만의 3×3 커널 만들기 (13단원에서 CNN이 배운 커널과 비교) |

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
gray = data.camera().astype(np.float32)

# %% [markdown]
# ## 1. 개념 — 작은 창을 미끄러뜨리며 곱하고 더하기
#
# 1. 작은 **커널**(예: 3×3 숫자 표)을 이미지의 왼쪽 위에 올린다.
# 2. 겹친 칸끼리 **곱해서 모두 더한다**.
# 3. 그 값을 출력의 한 칸에 적는다.
# 4. 한 칸 옮겨서 반복한다.
#
# 2단원의 블러, 샤프닝, 윤곽선은 모두 **커널의 숫자만 다른 같은 연산** 입니다.
#
# > 엄밀히 말하면 수학의 합성곱은 커널을 뒤집어서 곱하고, 딥러닝·OpenCV는 뒤집지 않고 곱합니다(상호상관). 딥러닝에서는 커널을 학습하기 때문에 둘을 구분하지 않고 "합성곱"이라고 부릅니다.

# %% [markdown]
# ## 2. 따라하기
# ### 2-1. 손으로 한 칸 계산하기

# %%
image = np.array([[3, 1, 1, 2, 8, 4],
                  [1, 0, 7, 3, 2, 6],
                  [2, 3, 5, 1, 1, 3],
                  [1, 4, 1, 2, 6, 5],
                  [3, 2, 1, 3, 7, 2],
                  [9, 2, 6, 2, 5, 1]], dtype=np.float32)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)      # 왼쪽 열 − 오른쪽 열 = 세로 경계

patch = image[0:3, 0:3]
print("왼쪽 위 3×3:\n", patch)
print("곱해서 더한 값:", (patch * kernel).sum())     # (3+1+2) − (1+7+5) = −7

# %% [markdown]
# ### 2-2. NumPy로 합성곱 함수 만들기

# %%
def conv2d(img, k, stride=1, padding=0):
    """2차원 합성곱 (딥러닝 방식: 커널을 뒤집지 않음)."""
    if padding:
        img = np.pad(img, padding)
    kh, kw = k.shape
    oh = (img.shape[0] - kh) // stride + 1
    ow = (img.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            r, c = i * stride, j * stride
            out[i, j] = (img[r:r + kh, c:c + kw] * k).sum()
    return out


print(conv2d(image, kernel))

# %% [markdown]
# 손 계산한 왼쪽 위 값 −7이 첫 칸에 나왔나요? 출력이 6×6이 아니라 **4×4** 인 이유는 아래 2-5에서 봅니다.
#
# ### 2-3. OpenCV와 결과 비교
# `cv2.filter2D`도 같은 계산(뒤집지 않는 방식)을 합니다. 가장자리 처리를 빼고 비교합니다.

# %%
ours = conv2d(gray, kernel)
cv_out = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_CONSTANT)[1:-1, 1:-1]
print("모양:", ours.shape, cv_out.shape, " 최대 차이:", np.abs(ours - cv_out).max())

# %% [markdown]
# 직접 만든 함수와 라이브러리가 **같은 결과** 를 냅니다. 다만 파이썬 반복문은 느리기 때문에 실제로는 라이브러리(그리고 GPU)를 씁니다.
#
# ### 2-4. 유명한 커널들

# %%
kernels = {
    "평균 블러": np.ones((3, 3), np.float32) / 9,
    "샤프닝": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "Sobel X (세로 경계)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
    "Sobel Y (가로 경계)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32),
}
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, (name, k) in enumerate(kernels.items()):
    axes[0, col].imshow(k, cmap="bwr", vmin=-2, vmax=2)
    for (r, c), v in np.ndenumerate(k):
        axes[0, col].text(c, r, f"{v:.2g}", ha="center", va="center")
    axes[0, col].set_title(name); axes[0, col].axis("off")
    out = cv2.filter2D(gray, -1, k)
    axes[1, col].imshow(np.abs(out) if "Sobel" in name else out, cmap="gray")
    axes[1, col].axis("off")
plt.show()

# %% [markdown]
# ### 2-5. Sobel로 경계의 세기와 방향 구하기
# 가로 방향 변화(Gx)와 세로 방향 변화(Gy)를 합치면 경계의 **세기** 가 됩니다: $\sqrt{G_x^2 + G_y^2}$

# %%
gx = cv2.filter2D(gray, -1, kernels["Sobel X (세로 경계)"])
gy = cv2.filter2D(gray, -1, kernels["Sobel Y (가로 경계)"])
mag = np.sqrt(gx ** 2 + gy ** 2)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, im, t in zip(axes, [np.abs(gx), np.abs(gy), mag], ["|Gx|", "|Gy|", "세기 √(Gx²+Gy²)"]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
plt.show()

# %% [markdown]
# ### 2-6. 가우시안 커널 직접 만들기
# 가운데일수록 큰 값, 합이 1. σ(시그마)가 클수록 넓게 퍼져 더 흐려집니다.

# %%
def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


g = gaussian_kernel(5, 1.0)
print(np.round(g, 3), " 합:", g.sum().round(3))
print("OpenCV 가우시안과 최대 차이:",
      np.abs(cv2.filter2D(gray, -1, g) - cv2.GaussianBlur(gray, (5, 5), 1.0)).max().round(3))

# %% [markdown]
# ### 2-7. 출력 크기 공식 — 패딩과 스트라이드
#
# $$ \text{출력 크기} = \frac{N - F + 2P}{S} + 1 $$
#
# | 기호 | 뜻 |
# |---|---|
# | N | 입력 크기 |
# | F | 커널 크기 |
# | P | 패딩 (가장자리에 0을 덧대는 칸 수) |
# | S | 스트라이드 (한 번에 움직이는 칸 수) |

# %%
for P, S in [(0, 1), (1, 1), (0, 2), (1, 2)]:
    out = conv2d(image, kernel, stride=S, padding=P)
    print(f"패딩 {P}, 스트라이드 {S} → 출력 {out.shape}   (공식: {(6 - 3 + 2 * P) // S + 1})")

# %% [markdown]
# - 패딩 1을 주면 3×3 커널에서도 **크기가 유지** 됩니다. CNN에서 `padding="same"`이 이것입니다.
# - 스트라이드 2는 크기를 약 **절반** 으로 줄입니다. 13단원의 풀링도 비슷한 역할을 합니다.
#
# **관찰 질문**
# 1. Sobel X와 Sobel Y는 숫자 배치가 어떻게 다르고, 결과는 어떻게 다른가요?
# 2. 샤프닝 커널의 숫자를 모두 더하면 얼마인가요? 평균 블러는? 합이 1이라는 것은 무슨 뜻일까요?
# 3. 지금까지 커널의 숫자는 **사람이 정했습니다**. 사진에서 "고양이 귀"를 찾는 커널은 어떻게 정할 수 있을까요? → 13단원

# %% [markdown]
# ## 3. 연습문제
#
# **문제 1.** **대각선(↘) 방향 경계** 에 강하게 반응하는 3×3 커널을 직접 설계하고 `camera` 이미지에 적용해 보세요.
#
# **문제 2.** 직접 만든 커널을 아래처럼 저장해 두세요. **13단원에서 CNN이 스스로 배운 커널과 나란히 비교** 합니다.

# %%
my_kernel = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]], dtype=np.float32)   # ← 바꿔 보세요

out = cv2.filter2D(gray, -1, my_kernel)
plt.imshow(np.abs(out), cmap="gray"); plt.axis("off"); plt.show()

import os
os.makedirs("../data", exist_ok=True)
np.save("../data/my_kernel.npy", my_kernel)
print("저장: ../data/my_kernel.npy")

# %% [markdown]
# <details>
# <summary>정답 예시</summary>
#
# ```python
# my_kernel = np.array([[ 2,  1,  0],
#                       [ 1,  0, -1],
#                       [ 0, -1, -2]], dtype=np.float32)
# # 왼쪽 위가 밝고 오른쪽 아래가 어두운 곳(↘ 방향으로 밝기가 떨어지는 경계)에서 큰 값
# ```
# </details>
