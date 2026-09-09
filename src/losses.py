"""손실 함수.

노트북에서 유도하고 검증한 것만 옮겨 온다.
"""

from __future__ import annotations

import numpy as np

# a 가 정확히 0 또는 1 이 되면 log 가 발산한다. |z| > 37 이면 float64 에서
# 실제로 그렇게 되므로(시그모이드 구현과 무관한 한계), 손실에서 잘라 준다.
EPS = 1e-12


def binary_cross_entropy(a: np.ndarray, y: np.ndarray) -> float:
    """L = -mean[ y log a + (1-y) log(1-a) ].

    최대우도에서 나온다. P(y|x) = a^y (1-a)^{1-y} 에 로그를 씌우고 부호를 뒤집은 것.
    """
    a_safe = np.clip(a, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(a_safe) + (1 - y) * np.log(1 - a_safe)))


def binary_cross_entropy_grad(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂L/∂z = a - y.  (시그모이드까지 묶은 기울기)

    σ' 의 a(1-a) 와 log 미분의 1/a 가 약분되어 이렇게 정리된다.
    da 는 a→0 에서 발산하지만 이 식은 항상 [-1, 1] 안에 있다.
    그래서 두 단계를 절대 따로 구현하지 않는다.
    """
    return a - y
