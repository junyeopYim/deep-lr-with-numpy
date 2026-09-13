# 2026-09-13 · 코드 주석 규칙(WRITING.md 7절) 적용: 01 · 00b · 02 · 03 · 04

D-013의 시범(00·01b) 뒤 HANDOFF 절차대로 기초 노트북 다섯 개의 코드 셀을 재작성했습니다. 구현·수식·검산·실험 설정은 그대로 두고,
출력이 같은 가독성 재작성만 했습니다. 재작성 전 노트북에 저장돼 있던 출력(stream 문장, PNG md5)과 새 커널 실행 결과를 비교했습니다.
아래 실행 시간은 16개를 한 번에 재실행한 마지막 확인의 값입니다.

| 노트북 | 셀 | 그림 | 오류 | 실행 | 그림 바이트 동일 | 출력 문장 차이 |
|---|---|---|---|---|---|---|
| 01 | 75 (변동 없음) | 10 | 0 | 4.4초 | 10 / 10 | 검산만 있던 8절 손실 격자 셀에 통과 print 1줄 추가. 나머지 51줄 동일 |
| 00b | 48 → 52 | 6 | 0 | 2.8초 | 4 / 6 (시간 측정 그림 2장만 다름) | 반복문 대비 배열 연산 배율 수치(302배 → 386배, 실행마다 다름). 나머지 동일 |
| 02 | 75 → 76 | 11 | 0 | 3.5초 | 11 / 11 | 없음 (46줄 동일) |
| 03 | 79 → 81 | 12 | 0 | 15.1초 | 12 / 12 | 검산만 있던 2절 결정 경계 셀에 통과 print 1줄 추가. 나머지 80줄 동일 |
| 04 | 78 → 79 | 10 | 0 | 3.8초 | 10 / 10 | 없음 (44줄 동일) |

- verify_exercises: 01 3/3, 00b 4/4, 02 3/3, 03 3/3, 04 3/3 통과. check_code_comments: 다섯 노트북 통과. pytest 22 통과.
- 셀 증가: 00b 4개(시간 재기·데이터 셀을 정의/실행으로, 1절·2절 center_loop·5절 정의/실행 분리), 02 1개(BCE 손계산과 극단 logit 셀 분리),
  03 2개(27줄 셀을 bce_with_logits / mlp_backward / mlp_loss_and_gradients 세 셀로), 04 1개(9절 초기화 셀을 완전열거 / 6층 실험으로).
- 재작성 종류: docstring(def 01 20, 00b 18, 02 22, 03 28, 04 20개 — 검사기용 `lambda`를 이름 있는 def로 바꾼 것 포함), `# 쓰는 것:` 출처 줄(19·10·7·23·11셀),
  코드 상자(6·7·8·10·12개), `lambda` → `def`(00b 4, 02 6, 03 6, 04 2), 한 줄의 두 문장 분리(01 4, 00b 1, 02 3, 03 7, 04 11곳), `assert a and b` 분리,
  assert 메시지와 `# 검산:` 주석 전부.
- 비교 절차: 재작성 전 사본과 출력 스냅샷(`scripts/compare_notebook_outputs.py snapshot`)을 세션 scratchpad와 `figures/check/snapshots/`에 두고,
  `scripts/nbedit.py`로 고유 문자열이 가리키는 셀을 통째로 바꾼 뒤 `check_code_comments.py` → `check_notebook.py`(새 커널) → `verify_exercises.py` →
  `compare_notebook_outputs.py compare`(stream 텍스트는 시간 수치를 마스킹한 뒤 diff, 그림은 PNG md5). `run_check` 셀의 `False` 표시와 `src/utils/*_plots.py`는 손대지 않았습니다.
- 같은 날 이어서 D-014로 `run_check`의 반환값을 없애 검사 셀의 `False` 표시를 정리했습니다. 그 재실행·비교 결과는 `results/foundation-validation/2026-09-13-run-check-no-return.md`에 있습니다.
