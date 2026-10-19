#!/usr/bin/env python3
"""LLM의 앞부분(토큰화 → 입력·타깃 → 임베딩 → 셀프 어텐션 → 코잘 마스크)을 numpy만으로 따라가 보는 실습.

사용법:
    python3 mini_llm_demo.py

필요 패키지: numpy (M1에서 OpenCV와 함께 설치되어 있음)
숫자는 무작위 초기값이라 의미가 없고, '모양(shape)'과 '흐름'을 보는 것이 목적이다.
"""
import numpy as np

np.random.seed(123)
np.set_printoptions(precision=2, suppress=True)


def title(text):
    print(f"\n{'=' * 8} {text} {'=' * 8}")


# 1. 토큰화 — 텍스트를 토큰으로 자르고, 토큰마다 번호(ID)를 붙인다
text = "로봇이 큐브를 집는다 로봇이 큐브를 상자에 넣는다 로봇이 상자를 닫는다"
tokens = text.split()                                  # 실제 LLM은 BPE로 더 잘게 자른다
vocab = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
ids = [vocab[t] for t in tokens]

title("1. 토큰화")
print("어휘사전:", vocab)
print("토큰 ID :", ids)

# 2. 입력·타깃 쌍 — 한 칸씩 밀어서 '다음 토큰'을 정답으로 삼는다 (슬라이딩 윈도)
context = 3
title(f"2. 입력과 타깃 (문맥 길이 {context})")
id2tok = {i: t for t, i in vocab.items()}
for i in range(len(ids) - context):
    x, y = ids[i:i + context], ids[i + 1:i + context + 1]
    print(f"입력 {[id2tok[t] for t in x]}  →  타깃 {[id2tok[t] for t in y]}")

# 3. 임베딩 — 토큰 ID를 벡터로 바꾸고(룩업), 위치 정보를 더한다
emb_dim = 4
tok_emb = 0.5 * np.random.randn(len(vocab), emb_dim)    # 학습으로 바뀌는 표
pos_emb = 0.5 * np.random.randn(context, emb_dim)       # 위치마다 하나씩
x_ids = ids[:context]
X = tok_emb[x_ids] + pos_emb                            # (토큰 수, 임베딩 차원)

title("3. 임베딩")
print("입력 토큰:", [id2tok[t] for t in x_ids])
print("토큰 임베딩 + 위치 임베딩 = 입력 임베딩, 모양", X.shape)
print(X)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# 4. 가장 단순한 셀프 어텐션 — 점곱으로 닮은 정도를 재고, 소프트맥스로 비율을 만들고, 가중합
scores = X @ X.T                                        # 모든 토큰 쌍의 점곱 (유사도)
weights = softmax(scores)                               # 행마다 합이 1
context_vecs = weights @ X

title("4. 단순 셀프 어텐션")
print("어텐션 가중치 (행: 지금 토큰, 열: 참고하는 토큰, 행 합 = 1)")
print(weights)
print("행 합:", weights.sum(axis=1))
print("문맥 벡터 모양:", context_vecs.shape)

# 5. 쿼리·키·값 + 스케일링 + 코잘 마스크 — GPT가 실제로 쓰는 방식
d_k = 2
W_q, W_k, W_v = (np.random.randn(emb_dim, d_k) for _ in range(3))
Q, K, V = X @ W_q, X @ W_k, X @ W_v
scores = Q @ K.T / np.sqrt(d_k)                         # √d_k로 나눠 값이 너무 커지지 않게
mask = np.triu(np.ones((context, context)), k=1).astype(bool)
scores[mask] = -np.inf                                  # 미래 토큰은 볼 수 없게
weights = softmax(scores)
out = weights @ V

title("5. 쿼리·키·값 + 코잘 마스크")
print("코잘 어텐션 가중치 (대각선 위 = 미래 = 0)")
print(weights)
print("출력 모양:", out.shape, "← 다음 층으로 전달")

# 6. 규모 감각 — GPT-2 small과 엣지 배포
title("6. 규모 감각")
params = 124_000_000
for name, nbytes in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
    print(f"GPT-2 124M 가중치 {name}: 약 {params * nbytes / 1e6:,.0f} MB")
