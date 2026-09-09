"""여러 곳에서 공유하는 수치 안정 함수.

노트북에서 유도하고 검증한 뒤 옮겨 온 것들만 있다.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^{-z}).

    부호에 따라 나누어 계산해 exp 넘침 경고를 피한다. 두 식은 분자·분모에
    e^z 를 곱한 것뿐이라 수학적으로 같다.

        z >= 0 : 1 / (1 + e^{-z})
        z <  0 : e^{z} / (1 + e^{z})

    유도: σ'(z) = σ(z)(1 - σ(z)).  notebooks/01_logistic_regression.ipynb 1절 참고.
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_grad(a: np.ndarray) -> np.ndarray:
    """σ'(z) 를 **출력** a = σ(z) 로부터 계산한다. a(1-a)."""
    return a * (1.0 - a)
