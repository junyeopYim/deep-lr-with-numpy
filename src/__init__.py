"""개념·수식·NumPy 코드를 연결하는 학습 프로젝트의 공통 도구.

모델, 손실, 역전파, 학습 루프는 각 노트북에서 직접 구현합니다.
이 패키지는 수치미분 검산, 데이터, 시각화 보조 기능을 제공합니다.
"""

from . import data, utils
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
    "utils", "data",
]
