"""run_check: 미구현·불일치·통과를 예외 없이 구분해 알린다."""
import numpy as np

from src.utils import run_check


def _check_double(candidate):
    np.testing.assert_allclose(candidate(np.array([1.0, 2.0])), [2.0, 4.0])
    print("통과")


def test_run_check_reports_not_implemented(capsys):
    def skeleton(x):
        raise NotImplementedError
    assert run_check(_check_double, skeleton, hint="x * 2") is False
    out = capsys.readouterr().out
    assert "아직 채우지" in out and "x * 2" in out


def test_run_check_reports_mismatch(capsys):
    assert run_check(_check_double, lambda x: x * 3) is False
    assert "손계산과 다릅니다" in capsys.readouterr().out


def test_run_check_passes(capsys):
    assert run_check(_check_double, lambda x: x * 2) is True
    assert "통과" in capsys.readouterr().out
