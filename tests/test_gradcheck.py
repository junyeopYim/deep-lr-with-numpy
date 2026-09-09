"""gradient checker 자신을 검증한다.

검증 도구가 조용히 통과만 시켜 준다면 이 프로젝트에는 안전망이 없는 셈이다.
그래서 "맞는 걸 통과시키는지"만이 아니라 **"틀린 걸 잡아내는지"**를 같이 본다.

여기 쓰이는 레이어들은 이 파일 안에서 정의한 최소 구현이다. `src`에는 아직
Layer 추상화가 없고(노트북에서 만들어 옮길 예정), `check_layer`는 덕 타이핑으로
동작하므로 그럴 필요도 없다. 아래 `_MiniLayer`가 곧 `check_layer`가 요구하는
인터페이스의 전부다.
"""

import numpy as np
import pytest

from src import check_function, check_layer, numerical_gradient, rel_error
from src.utils import GradCheckError


class _Param:
    """check_layer가 파라미터에 요구하는 것: .data 와 .grad 뿐이다."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)


class _MiniLayer:
    """check_layer가 레이어에 요구하는 인터페이스의 전부.

    forward / backward / zero_grad / named_parameters.
    """

    def __init__(self):
        self._params: dict[str, _Param] = {}

    def named_parameters(self):
        return self._params.items()

    def zero_grad(self):
        for p in self._params.values():
            p.grad.fill(0.0)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


# --- 수치미분 자체 ------------------------------------------------------------

def test_numerical_gradient_matches_known_derivative():
    # f(x) = Σ x³ 의 도함수는 3x²
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    grad = numerical_gradient(lambda: float(np.sum(x ** 3)), x)
    assert rel_error(grad, 3 * x ** 2) < 1e-8


def test_numerical_gradient_restores_input():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    numerical_gradient(lambda: float(np.sum(x ** 2)), x)
    assert np.array_equal(x, before), "수치미분이 입력을 원상복구하지 않았습니다"


def test_numerical_gradient_rejects_integer_array():
    with pytest.raises(TypeError):
        numerical_gradient(lambda: 0.0, np.array([1, 2, 3]))


def test_rel_error_zero_for_identical():
    a = np.array([1e-8, 5.0, -3.0])
    assert rel_error(a, a.copy()) == 0.0


def test_rel_error_handles_both_zero():
    z = np.zeros(4)
    assert rel_error(z, z) == 0.0  # 0/0 이 나오면 안 된다


def test_rel_error_is_scale_normalized():
    """작은 원소 하나가 자기 자신으로 나뉘어 튀지 않아야 한다.

    텐서 스케일이 1인데 어떤 원소의 참값이 1e-9라면 중심차분에는 그 원소를
    해상할 유효숫자가 없다. 전체 스케일로 정규화하는 이유다.
    """
    a = np.array([1.0, 1e-9])
    b = np.array([1.0, 2e-9])          # 절대오차 1e-9, 텐서 스케일 대비 1e-9
    assert rel_error(a, b) < 1e-8
    assert rel_error(a, b, elementwise=True) > 0.4   # 원소별로는 튄다


# --- 일부러 틀린 레이어들 ------------------------------------------------------

class _Square(_MiniLayer):
    """y = x². dx = 2x·dout."""

    def forward(self, x):
        self._x = x
        return x ** 2

    def backward(self, dout):
        return 2 * self._x * dout


class _WrongSquare(_Square):
    """상수를 틀린 버전. 검사기가 반드시 잡아야 한다."""

    def backward(self, dout):
        return 3 * self._x * dout  # 2가 맞다


class _WrongOnlyForOnes(_Square):
    """dout이 전부 1일 때만 우연히 맞는 버전.

    상류 기울기를 무작위로 뽑아야 하는 이유를 보여준다.
    """

    def backward(self, dout):
        return 2 * self._x * np.mean(dout)


class _WrongParamGrad(_MiniLayer):
    """dx는 맞지만 파라미터 기울기가 틀린 버전. 흔한 실수다."""

    def __init__(self):
        super().__init__()
        self._params["w"] = _Param([2.0, -1.0, 0.5])

    @property
    def w(self):
        return self._params["w"]

    def forward(self, x):
        self._x = x
        return x * self.w.data

    def backward(self, dout):
        self.w.grad += np.sum(dout, axis=0)  # x를 곱하는 걸 빠뜨렸다
        return dout * self.w.data


# --- check_layer --------------------------------------------------------------

def test_check_layer_passes_correct_backward():
    x = np.random.default_rng(0).standard_normal((4, 3))
    assert check_layer(_Square(), x)["dx"] < 1e-8


def test_check_layer_catches_wrong_constant():
    x = np.random.default_rng(0).standard_normal((4, 3))
    with pytest.raises(GradCheckError):
        check_layer(_WrongSquare(), x)


def test_check_layer_catches_backward_that_only_works_for_ones():
    x = np.random.default_rng(0).standard_normal((4, 3))
    layer = _WrongOnlyForOnes()
    # 상류 기울기를 전부 1로 주면 이 틀린 구현도 통과해 버린다.
    assert check_layer(layer, x, dout=np.ones((4, 3)))["dx"] < 1e-8
    # 무작위 상류 기울기(기본값)에서는 잡힌다.
    with pytest.raises(GradCheckError):
        check_layer(layer, x)


def test_check_layer_catches_wrong_parameter_gradient():
    x = np.random.default_rng(0).standard_normal((5, 3))
    with pytest.raises(GradCheckError) as exc:
        check_layer(_WrongParamGrad(), x)
    assert "dw" in str(exc.value)


def test_check_layer_does_not_mutate_caller_arrays():
    x = np.random.default_rng(0).standard_normal((4, 3))
    before = x.copy()
    check_layer(_Square(), x)
    assert np.array_equal(x, before)


def test_check_layer_zeroes_grads_first():
    """이미 쓰레기 값이 들어 있는 .grad 때문에 검사가 틀리면 안 된다."""
    dirty = _WrongParamGrad()
    dirty.w.grad += 999.0
    clean = _WrongParamGrad()
    x = np.random.default_rng(1).standard_normal((5, 3))
    assert (check_layer(dirty, x, raise_on_fail=False)
            == check_layer(clean, x, raise_on_fail=False))


def test_correct_and_wrong_are_separated_by_orders_of_magnitude():
    """허용치 1e-7이 애매한 구간에 놓이지 않는지 확인한다."""
    x = np.random.default_rng(0).standard_normal((4, 3))
    good = check_layer(_Square(), x, raise_on_fail=False)["dx"]
    bad = check_layer(_WrongSquare(), x, raise_on_fail=False)["dx"]
    assert good < 1e-8 < 1e-3 < bad


# --- check_function: 클래스로 만들기 전 단계의 검증 ----------------------------

def test_check_function_passes_correct_pair():
    err = check_function(lambda x: np.maximum(x, 0.0),
                         lambda dout, x: dout * (x > 0),
                         np.array([1.0, -2.0, 3.0, -0.5]))
    assert err < 1e-8


def test_check_function_catches_wrong_pair():
    with pytest.raises(GradCheckError):
        check_function(np.tanh,
                       lambda dout, x: dout * (1 - np.tanh(x)),  # 제곱을 빠뜨림
                       np.array([0.3, -1.2, 0.8]))


def test_check_function_rejects_wrong_output_shape():
    with pytest.raises(ValueError, match="모양"):
        check_function(lambda x: x, lambda dout, x: dout.sum(), np.ones((3, 2)))


def test_check_function_does_not_mutate_input():
    x = np.array([0.5, -1.5, 2.0])
    before = x.copy()
    check_function(np.tanh, lambda dout, x: dout * (1 - np.tanh(x) ** 2), x)
    assert np.array_equal(x, before)


# --- 그림 설정 ----------------------------------------------------------------

def test_setup_plots_picks_a_usable_font():
    """setup_plots 가 rcParams 를 실제로 바꾸는지, 폰트를 고르는지."""
    import matplotlib
    from src.utils import setup_plots

    before = dict(matplotlib.rcParams)
    try:
        font = setup_plots()
        assert matplotlib.rcParams["axes.grid"] is True
        assert matplotlib.rcParams["figure.autolayout"] is True
        assert matplotlib.rcParams["axes.axisbelow"] is True
        # 한글 폰트를 찾았다면 font.family 맨 앞에 있어야 한다
        if font is not None:
            assert matplotlib.rcParams["font.family"][0] == font
    finally:
        matplotlib.rcParams.update(before)


def test_setup_plots_font_covers_hangul_and_minus_when_possible():
    """고른 폰트가 한글과 U+2212 를 모두 가지면 unicode_minus 가 켜져야 한다.

    이 둘이 어긋나면 로그 눈금에서 글리프 경고가 쏟아진다.
    """
    import matplotlib
    from matplotlib.ft2font import FT2Font
    import matplotlib.font_manager as fm
    from src.utils import setup_plots

    before = dict(matplotlib.rcParams)
    try:
        font = setup_plots()
        if font is None:
            pytest.skip("이 환경에 한글 폰트가 없습니다")
        path = fm.findfont(font, fallback_to_default=False)
        face = FT2Font(path)
        covers_both = all(face.get_char_index(cp) != 0 for cp in (ord("가"), 0x2212))
        assert matplotlib.rcParams["axes.unicode_minus"] == covers_both
    finally:
        matplotlib.rcParams.update(before)
