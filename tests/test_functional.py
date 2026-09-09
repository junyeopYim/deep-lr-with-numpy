"""노트북 01에서 정착시킨 것들의 회귀 테스트.

유도는 notebooks/01_logistic_regression.ipynb 에 있다.
여기서는 그 유도가 이후 리팩터링에서 깨지지 않는지만 지킨다.
"""

import warnings

import numpy as np
import pytest

from src import (
    binary_cross_entropy,
    binary_cross_entropy_grad,
    check_function,
    sigmoid,
    sigmoid_grad,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- sigmoid ------------------------------------------------------------------

def test_sigmoid_known_values():
    assert np.allclose(sigmoid(np.array([0.0])), 0.5)
    assert np.allclose(sigmoid(np.array([-np.inf, np.inf])), [0.0, 1.0])


def test_sigmoid_is_in_open_unit_interval(rng):
    a = sigmoid(rng.standard_normal(200) * 5)
    assert np.all(a > 0) and np.all(a < 1)


def test_sigmoid_does_not_overflow_on_extreme_inputs():
    """순진한 1/(1+exp(-z)) 는 여기서 넘침 경고를 낸다. 분기 버전은 조용해야 한다."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 어떤 경고든 실패로 만든다
        out = sigmoid(np.array([-800.0, -50.0, 0.0, 50.0, 800.0]))
    assert np.all(np.isfinite(out))


def test_sigmoid_matches_naive_where_naive_is_safe(rng):
    z = rng.standard_normal(100) * 3
    assert np.allclose(sigmoid(z), 1.0 / (1.0 + np.exp(-z)))


def test_sigmoid_symmetry(rng):
    """σ(-z) = 1 - σ(z)."""
    z = rng.standard_normal(50) * 4
    assert np.allclose(sigmoid(-z), 1 - sigmoid(z))


def test_sigmoid_derivative(rng):
    """σ'(z) = σ(z)(1-σ(z)) 를 수치미분으로 확인."""
    check_function(sigmoid,
                   lambda dout, z: dout * sigmoid_grad(sigmoid(z)),
                   rng.standard_normal(8) * 2)


def test_sigmoid_grad_max_is_quarter():
    """도함수의 최댓값은 z=0 에서 0.25. 기울기 소실의 원인이 되는 상한이다."""
    z = np.linspace(-10, 10, 2001)
    assert np.isclose(sigmoid_grad(sigmoid(z)).max(), 0.25, atol=1e-6)


# --- binary cross entropy -----------------------------------------------------

def test_bce_at_chance_is_log_two():
    """모든 예측이 0.5면 손실은 정확히 log 2. 학습 시작점의 기준값이다."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert np.isclose(binary_cross_entropy(np.full(4, 0.5), y), np.log(2))


def test_bce_is_near_zero_for_confident_correct():
    y = np.array([1.0, 0.0])
    assert binary_cross_entropy(np.array([1 - 1e-9, 1e-9]), y) < 1e-8


def test_bce_stays_finite_at_saturation():
    """a 가 정확히 0/1 이어도 nan 이 되면 안 된다 (클리핑)."""
    loss = binary_cross_entropy(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert np.isfinite(loss) and loss > 0


def test_bce_gradient_is_prediction_minus_target(rng):
    """dL/dz = a - y 를, z 를 흔들어 수치미분으로 확인한다."""
    y = rng.integers(0, 2, 6).astype(float)
    check_function(lambda z: np.array(binary_cross_entropy(sigmoid(z), y)),
                   lambda dout, z: dout * binary_cross_entropy_grad(sigmoid(z), y) / len(y),
                   rng.standard_normal(6) * 2)


def test_bce_gradient_is_bounded(rng):
    """da 는 a→0 에서 발산하지만 dz = a - y 는 항상 [-1, 1] 안에 있다."""
    a = sigmoid(rng.standard_normal(100) * 20)
    y = rng.integers(0, 2, 100).astype(float)
    g = binary_cross_entropy_grad(a, y)
    assert np.all(np.abs(g) <= 1.0)
