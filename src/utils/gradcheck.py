"""수치미분으로 손유도한 backward를 검증한다.

이 프로젝트에는 자동미분이 없으므로, 유도한 식이 맞는지 확인해 주는 것은
오직 이 파일뿐이다. 새 레이어를 만들면 학습을 돌리기 전에 반드시 여기를 통과시킨다.
학습이 잘 안 될 때 원인을 backward 버그와 하이퍼파라미터 문제로 나누는 기준선이기도 하다.

원리는 중심차분이다.

    ∂L/∂x_i ≈ (L(x + εe_i) - L(x - εe_i)) / 2ε + O(ε²)

레이어 출력은 벡터이므로 스칼라 손실이 필요하다. 임의의 상류 기울기 v를 하나
뽑아서 L = <forward(x), v> 로 두면, 정의상 ∂L/∂x = backward(v)가 된다.
v를 무작위로 잡는 이유는, 특정 v(예: 전부 1)에서만 우연히 맞는 잘못된 유도를
걸러내기 위해서다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """f를 x의 각 원소에 대해 중심차분한다.

    f는 인자를 받지 않는다. 대신 이 함수가 x를 제자리에서 흔들고 f를 다시 부르므로,
    f는 그 x를 들여다보는 클로저여야 한다.
    """
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError(f"실수 배열이어야 합니다 (받은 dtype: {x.dtype})")

    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig  # 원상복구
        grad[idx] = (fp - fm) / (2.0 * eps)
    return grad


def rel_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-12,
              elementwise: bool = False) -> float:
    """두 기울기 텐서 사이의 상대오차.

        ‖a - b‖∞ / max(‖a‖∞, ‖b‖∞)

    즉 최대 절대오차를 그 텐서에서 실제로 나타나는 기울기 크기로 나눈다.
    절대오차와 달리 스케일에 무관해서, 레이어가 달라도 같은 기준(예: 1e-7)으로
    판단할 수 있다.

    왜 원소별 |a_i-b_i|/max(|a_i|,|b_i|) 가 아닌가
    ----------------------------------------------
    고전적인 정의는 원소별로 나누지만, 기울기가 0에 가까운 원소에서 무너진다.
    중심차분의 절대 잡음 바닥은 ε² + 2⁻⁵²/ε ≈ 1e-10 수준이다. 텐서 스케일이 1인데
    어떤 원소의 참값이 1e-9라면 그 원소에는 애초에 유효숫자가 없고, 자기 자신으로
    나누는 순간 상대오차가 1e-1까지 튄다. 유도는 멀쩡한데 검사만 실패한다.

    전체 스케일로 나누면 그 잡음에 흔들리지 않으면서도 실제 유도 오류는 잡아낸다.
    전치 실수, 빠뜨린 축 합, 틀린 상수 같은 오류는 기울기와 같은 차수의 차이를
    만들기 때문이다. tests/test_gradcheck.py가 일부러 틀린 backward들로 이를 확인한다.

    진단할 때 어느 원소가 얼마나 틀렸는지 보고 싶으면 elementwise=True를 준다.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"모양이 다릅니다: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    if elementwise:
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), eps)
        return float(np.max(np.abs(a - b) / denom))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), eps)
    return float(np.max(np.abs(a - b)) / scale)


class GradCheckError(AssertionError):
    """해석적 기울기와 수치 기울기가 허용오차 밖으로 벌어졌을 때."""


def check_layer(
    layer,
    *inputs: np.ndarray,
    dout: np.ndarray | None = None,
    seed: int = 0,
    eps: float = 1e-6,
    tol: float = 1e-7,
    verbose: bool = False,
    raise_on_fail: bool = True,
    **forward_kwargs,
) -> dict[str, float]:
    """레이어의 입력 기울기와 파라미터 기울기를 모두 검증한다.

    Args:
        layer: forward/backward를 구현한 Layer.
        *inputs: forward에 넘길 인자들. 실수 배열만 수치검증하고
            정수 배열(예: 토큰 인덱스)은 미분 대상이 아니므로 건너뛴다.
        dout: 상류 기울기. 생략하면 seed로 무작위 생성한다.
        eps: 차분 폭. 너무 작으면 부동소수점 상쇄오차가, 너무 크면 절단오차가 커진다.
            float64에서 1e-6 부근이 최적이다.
        tol: 허용 상대오차. 올바른 유도라면 보통 1e-9 언저리가 나온다.
            중심차분의 잡음 바닥(절단오차 O(ε²) + 상쇄오차 O(2⁻⁵²/ε))이 그 정도라
            1e-7이면 여유가 충분하면서도 실제 유도 오류는 놓치지 않는다.
        raise_on_fail: False면 예외 대신 결과 dict만 돌려준다.

    Returns:
        {이름: 상대오차} 딕셔너리.
    """
    rng = np.random.default_rng(seed)
    # 호출자의 배열을 흔들면 안 되므로 복사한다. forward는 이 사본을 본다.
    xs = [np.array(a, dtype=np.float64) if np.issubdtype(np.asarray(a).dtype, np.floating)
          else np.asarray(a) for a in inputs]

    out = layer.forward(*xs, **forward_kwargs)
    out = np.asarray(out, dtype=np.float64)
    if dout is None:
        dout = rng.standard_normal(out.shape)
    dout = np.asarray(dout, dtype=np.float64)
    if dout.shape != out.shape:
        raise ValueError(f"dout 모양 불일치: {dout.shape} != 출력 {out.shape}")

    def scalar_loss() -> float:
        return float(np.sum(np.asarray(layer.forward(*xs, **forward_kwargs)) * dout))

    # --- 해석적 기울기: 위의 forward가 남긴 캐시를 그대로 쓴다 --------------
    layer.zero_grad()
    dx = layer.backward(dout)
    dxs = dx if isinstance(dx, tuple) else (dx,)
    analytic: dict[str, np.ndarray] = {}
    for i, g in enumerate(dxs):
        if g is not None:
            analytic[f"dx[{i}]" if len(xs) > 1 else "dx"] = np.array(g, dtype=np.float64)
    for name, p in layer.named_parameters():
        analytic[f"d{name}"] = p.grad.copy()

    # --- 수치 기울기 -------------------------------------------------------
    numeric: dict[str, np.ndarray] = {}
    for i, x in enumerate(xs):
        key = f"dx[{i}]" if len(xs) > 1 else "dx"
        if key not in analytic:
            continue  # backward가 이 입력에 대한 기울기를 내놓지 않음(예: 정수 인덱스)
        if not np.issubdtype(x.dtype, np.floating):
            continue
        numeric[key] = numerical_gradient(scalar_loss, x, eps)
    for name, p in layer.named_parameters():
        numeric[f"d{name}"] = numerical_gradient(scalar_loss, p.data, eps)

    # --- 비교 -------------------------------------------------------------
    errors: dict[str, float] = {}
    failed: list[str] = []
    for key, num in numeric.items():
        ana = analytic.get(key)
        if ana is None:
            failed.append(f"{key}: backward가 기울기를 내놓지 않았습니다")
            continue
        if ana.shape != num.shape:
            failed.append(f"{key}: 모양 불일치 해석 {ana.shape} vs 수치 {num.shape}")
            continue
        err = rel_error(ana, num)
        errors[key] = err
        if not np.isfinite(err) or err > tol:
            failed.append(f"{key}: 상대오차 {err:.3e} > tol {tol:.1e}")

    if verbose:
        title = type(layer).__name__
        print(f"[gradcheck] {title}")
        for key, err in errors.items():
            mark = "ok  " if err <= tol else "FAIL"
            print(f"  {mark} {key:<24s} rel_err = {err:.3e}")

    if failed and raise_on_fail:
        raise GradCheckError(
            f"{type(layer).__name__} gradient check 실패:\n  " + "\n  ".join(failed)
        )
    return errors


def check_function(
    forward,
    backward,
    x: np.ndarray,
    *,
    dout: np.ndarray | None = None,
    seed: int = 0,
    eps: float = 1e-6,
    tol: float = 1e-7,
    verbose: bool = False,
    raise_on_fail: bool = True,
    name: str | None = None,
) -> float:
    """Layer 클래스로 만들기 전, 함수 쌍으로 짜 본 forward/backward를 검증한다.

    노트북에서 먼저 손으로 짜 보고 나중에 라이브러리로 옮기는 흐름을 위한 것이다.
    캐시를 들고 다닐 필요 없이 backward가 x를 다시 받는다.

        forward(x) -> y
        backward(dout, x) -> dx

    파라미터가 없는 레이어(활성화 함수 등)에 맞는 형태다. 파라미터가 생기면
    Layer로 옮기고 check_layer를 쓴다.

    Returns:
        상대오차.

    Example:
        >>> f  = lambda x: np.maximum(x, 0)
        >>> df = lambda dout, x: dout * (x > 0)
        >>> check_function(f, df, np.array([1.0, -2.0, 3.0]))   # doctest: +SKIP
    """
    x = np.array(x, dtype=np.float64)  # 호출자 배열을 흔들지 않는다
    y = np.asarray(forward(x), dtype=np.float64)
    if dout is None:
        dout = np.random.default_rng(seed).standard_normal(y.shape)
    dout = np.asarray(dout, dtype=np.float64)
    if dout.shape != y.shape:
        raise ValueError(f"dout 모양 불일치: {dout.shape} != 출력 {y.shape}")

    analytic = np.asarray(backward(dout, x), dtype=np.float64)
    if analytic.shape != x.shape:
        raise ValueError(f"backward가 입력과 다른 모양을 반환했습니다: "
                         f"{analytic.shape} != {x.shape}")

    numeric = numerical_gradient(lambda: float(np.sum(np.asarray(forward(x)) * dout)), x, eps)
    err = rel_error(analytic, numeric)

    label = name or getattr(forward, "__name__", "function")
    if verbose:
        mark = "ok  " if err <= tol else "FAIL"
        print(f"[gradcheck] {label}\n  {mark} dx  rel_err = {err:.3e}")
    if (not np.isfinite(err) or err > tol) and raise_on_fail:
        raise GradCheckError(
            f"{label} gradient check 실패: 상대오차 {err:.3e} > tol {tol:.1e}")
    return err


def check_loss(
    loss_layer,
    y_pred: np.ndarray,
    y_true,
    *,
    eps: float = 1e-6,
    tol: float = 1e-7,
    verbose: bool = False,
    raise_on_fail: bool = True,
) -> dict[str, float]:
    """스칼라를 반환하는 손실 레이어 전용 검증.

    손실은 이미 스칼라이므로 상류 기울기 v가 필요 없다. dout=1.0으로 둔다.
    """
    x = np.array(y_pred, dtype=np.float64)
    loss = float(loss_layer.forward(x, y_true))
    if not np.isscalar(loss) and np.asarray(loss).ndim != 0:
        raise ValueError("손실 레이어는 스칼라를 반환해야 합니다")

    loss_layer.zero_grad()
    analytic = np.array(loss_layer.backward(1.0), dtype=np.float64)
    numeric = numerical_gradient(lambda: float(loss_layer.forward(x, y_true)), x, eps)

    err = rel_error(analytic, numeric)
    if verbose:
        mark = "ok  " if err <= tol else "FAIL"
        print(f"[gradcheck] {type(loss_layer).__name__}\n  {mark} dx  rel_err = {err:.3e}")
    if err > tol and raise_on_fail:
        raise GradCheckError(
            f"{type(loss_layer).__name__} gradient check 실패: 상대오차 {err:.3e} > tol {tol:.1e}"
        )
    return {"dx": err}
