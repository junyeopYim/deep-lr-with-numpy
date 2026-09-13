"""연습 문제 검사 도우미.

연습은 코딩 테스트처럼 함수 뼈대를 주고 핵심 한두 줄만 채우게 합니다. 아직 채우지 않은
자리에는 ``raise NotImplementedError`` 가 있으므로, 검사 함수를 그대로 부르면 노트북이
멈춥니다. ``run_check`` 는 그 경우를 잡아 "아직 채우지 않았습니다"라고만 알리고 계속 진행하게 합니다.

사용법::

    run_check(check_three_outputs, exercise_affine, hint="00번 4절의 한 줄")

검사 함수(check_*)는 후보 함수를 받아 손계산 값과 비교하고, 맞으면 메시지를 출력합니다.
"""

from __future__ import annotations


def run_check(check, candidate, *, hint: str | None = None) -> bool:
    """검사 함수를 후보에 적용한다. 미구현·불일치는 예외 대신 안내문으로 알린다."""
    name = getattr(candidate, "__name__", "함수")
    try:
        check(candidate)
    except NotImplementedError:
        message = f"☐ {name}: 아직 채우지 않았습니다. `raise NotImplementedError` 줄을 지우고 그 자리에 코드를 쓰세요."
        if hint:
            message += f"\n   힌트: {hint}"
        print(message)
        return False
    except AssertionError as error:
        detail = str(error).strip().splitlines()
        first = detail[0] if detail else ""
        print(f"✗ {name}: 값이 손계산과 다릅니다. 채운 줄을 다시 보세요. {first}")
        return False
    except Exception as error:  # 모양 오류 등 다른 실행 오류도 멈추지 않고 알린다
        print(f"✗ {name}: 실행 중 오류가 났습니다 → {type(error).__name__}: {error}")
        return False
    return True
