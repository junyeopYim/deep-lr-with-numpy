"""손으로 만든 작은 데이터셋.

다운로드가 필요 없고 즉시 만들어지므로, 새 레이어가 "학습이 되긴 하는지"
확인하는 데 쓴다. gradcheck가 미분의 정확성을 본다면, 이쪽은 최적화가
실제로 손실을 내리는지를 본다.
"""

from __future__ import annotations

import numpy as np


def spiral(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """나선 분류 문제.

    클래스들이 원점 주변으로 얽혀 있어 선형 분리가 불가능하다. 은닉층 없는
    모델은 정확도 50% 근처에서 멈추고, 은닉층을 하나 넣으면 95%를 넘긴다.
    비선형성이 정말로 필요한지 확인하는 가장 싼 실험이다.

    Returns:
        X: (n_per_class * n_classes, 2), y: (...,) 정수 라벨.
    """
    rng = np.random.default_rng(seed)
    N, K = n_per_class, n_classes
    X = np.zeros((N * K, 2))
    y = np.zeros(N * K, dtype=np.int64)
    for k in range(K):
        ix = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1.0, N)                      # 반지름
        t = np.linspace(k * 4, (k + 1) * 4, N) + rng.standard_normal(N) * noise  # 각도
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def moons(n_samples: int = 200, noise: float = 0.15, seed: int = 0):
    """맞물린 반달 두 개. 이진 분류용."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    X = np.vstack([
        np.c_[np.cos(t_out), np.sin(t_out)],
        np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5],
    ])
    X += rng.standard_normal(X.shape) * noise
    y = np.hstack([np.zeros(n_out, dtype=np.int64), np.ones(n_in, dtype=np.int64)])
    return X, y


def sine(n_samples: int = 200, noise: float = 0.1, seed: int = 0):
    """1차원 회귀용. y = sin(2πx) + 잡음."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n_samples))[:, None]
    y = np.sin(2 * np.pi * x) + rng.standard_normal((n_samples, 1)) * noise
    return x, y


def shapes(n_samples: int = 600, size: int = 16, noise: float = 0.6,
           jitter: float = 2.0, seed: int = 0):
    """원 vs 십자 이진 분류용 합성 이미지.

    코세라 C1W2의 고양이/비고양이 과제를 대신한다. 다운로드가 필요 없고
    난이도를 조절할 수 있으며, 이미지라서 **학습된 가중치를 그림으로 볼 수 있다**.
    로지스틱 회귀가 무엇을 보고 판단하는지 확인하는 것이 요점이다.

    중심 위치와 크기가 무작위로 흔들리고 가우시안 잡음이 얹혀서,
    선형 분류기가 완벽하게는 풀지 못한다 (검증 정확도 90%대).

    Args:
        n_samples: 샘플 수.
        size: 이미지 한 변의 픽셀 수.
        noise: 가우시안 잡음의 표준편차. 크면 어려워진다.
        jitter: 도형 중심이 흔들리는 최대 픽셀 수.

    Returns:
        X: (n_samples, size, size) 실수 이미지.
        y: (n_samples,) 0 = 원, 1 = 십자.
    """
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n_samples, size, size))
    y = rng.integers(0, 2, n_samples)
    yy, xx = np.mgrid[0:size, 0:size]

    for i, label in enumerate(y):
        cx = (size - 1) / 2 + rng.uniform(-jitter, jitter)
        cy = (size - 1) / 2 + rng.uniform(-jitter, jitter)
        if label == 0:  # 원: 중심에서의 거리가 반지름 근처인 픽셀
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            radius, thick = rng.uniform(3.5, 5.5), rng.uniform(0.8, 1.4)
            imgs[i] = (np.abs(r - radius) < thick).astype(float)
        else:           # 십자: 가로 막대와 세로 막대의 합집합
            thick, arm = rng.uniform(0.8, 1.6), rng.uniform(4.0, 6.5)
            horizontal = (np.abs(yy - cy) < thick) & (np.abs(xx - cx) < arm)
            vertical = (np.abs(xx - cx) < thick) & (np.abs(yy - cy) < arm)
            imgs[i] = (horizontal | vertical).astype(float)

    imgs += rng.standard_normal(imgs.shape) * noise
    return imgs, y.astype(np.int64)


def blobs(n_samples: int = 200, noise: float = 0.7, seed: int = 0):
    """2차원 가우시안 두 덩어리. 선형분리 가능해서 결정경계를 그려 보기 좋다."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    n0 = n_samples - n1
    X = np.vstack([rng.standard_normal((n0, 2)) * noise + [-1.5, -1.0],
                   rng.standard_normal((n1, 2)) * noise + [1.5, 1.0]])
    y = np.hstack([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
