"""numpy만으로 밑바닥부터 — 퍼셉트론에서 Transformer까지.

이 패키지는 **도구만** 담는다. 신경망 구성요소(레이어, 손실, 옵티마이저)는
`notebooks/`에서 유도하고 검증한 뒤 여기로 옮겨 온다. 비어 있는 항목이 많은 것이 정상이다.

담긴 것:
    utils.gradcheck  수치미분 검증기 — 손유도한 backward가 맞는지 확인한다
    utils.plotting   학습 곡선, 결정경계, gradcheck 막대그래프
    data             DataLoader, 장난감 데이터셋, MNIST 로더
    functional       sigmoid                        (노트북 01에서 정착)
    losses           binary_cross_entropy           (노트북 01에서 정착)
"""

from . import data, functional, losses, utils
from .functional import sigmoid, sigmoid_grad
from .losses import binary_cross_entropy, binary_cross_entropy_grad
from .utils import (
    GradCheckError,
    check_function,
    check_layer,
    check_loss,
    numerical_gradient,
    rel_error,
)

__version__ = "0.1.0"

__all__ = [
    "check_function", "check_layer", "check_loss",
    "numerical_gradient", "rel_error", "GradCheckError",
    "sigmoid", "sigmoid_grad",
    "binary_cross_entropy", "binary_cross_entropy_grad",
    "utils", "data", "functional", "losses",
]
