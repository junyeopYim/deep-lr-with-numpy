# HANDOFF · 코드 주석 규칙(WRITING.md 7절)을 나머지 16개 노트북에 적용

이 파일은 단발성입니다. 읽은 세션이 바로 삭제하고 `memory/CHECKPOINT.md`를 갱신하며 진행합니다.

## 먼저 읽을 것 (순서대로)

1. `AGENTS.md` 전체, `memory/00-INDEX.md`, `memory/DECISIONS.md`의 D-013, `memory/CHECKPOINT.md`.
2. `docs/WRITING.md` 7절(코드 주석 규칙 여섯 가지)과 6절 표(적용 현황).
3. **완성 기준 두 개**: `notebooks/00_기초/00_math_to_numpy.ipynb`, `notebooks/00_기초/01b_minibatch.ipynb`. 이 둘의 코드 셀이 주석 밀도·표현의 기준입니다. 새 노트북을 고치기 전에 두 노트북의 코드 셀을 처음부터 끝까지 훑어 눈에 익히세요. 특히 01b의 `gradient_on`·`epoch_batches`·`train_minibatch` 셀과 00의 2절 검산 셀.
4. `git status`. 00·01b, WRITING.md, AGENTS.md, scripts 4개(check_code_comments·compare_notebook_outputs·nbedit·기존 파일), memory, results 1개가 **미커밋** 상태여야 합니다. 커밋·푸시는 사용자 지시가 있을 때만 합니다.

## 할 일

`01_gradients_and_learning` → `00b_vectorization` → `02` → `03` → `04` → `05` → `06` → … → `15` 순서로 **한 노트북씩** 코드 셀을 재작성합니다. 한 노트북이 끝날 때마다 짧게 보고하고 다음으로 갑니다.

노트북 하나의 절차:

1. **재작성 전 스냅샷**: `python scripts/compare_notebook_outputs.py snapshot <노트북>` (노트북에 저장된 출력을 `figures/check/snapshots/`에 JSON으로 둡니다).
2. **재작성**: scratchpad에 스크립트를 쓰고 `scripts/nbedit.py`의 `replace`/`edit`/`append`/`split`으로 셀을 **고유 문자열로 찾아** 바꿉니다. 노트북 JSON을 손으로 고치지 않습니다. 00·01b의 재작성 스크립트는 남아 있지 않으므로 같은 방식으로 새로 씁니다.
3. **실행·검증** 세 가지가 모두 오류 0·통과여야 합니다.
   ```bash
   python scripts/check_notebook.py <노트북>
   python scripts/verify_exercises.py <노트북>
   python scripts/check_code_comments.py <노트북>
   ```
4. **출력 비교**: `python scripts/compare_notebook_outputs.py compare figures/check/snapshots/<이름>.outputs.json <노트북>`. 출력 문장 diff가 비고 그림 md5가 같아야 합니다. 허용되는 차이는 세 가지뿐입니다: 시간 측정 수치, 검산만 있던 셀에 추가한 통과 `print`, 힌트·정답의 관용구 교체. 남은 차이는 보고에서 하나씩 설명합니다.
5. **PNG 확인**: `figures/check/<노트북>/`의 그림을 눈으로 봅니다. 바이트가 같은 그림은 생략해도 됩니다.
6. **기록**: `docs/WRITING.md` 6절 표의 그 노트북 행에 "코드 주석 규칙(7절) 적용 (날짜): 무엇을 했는지"를 덧붙입니다.

## 지킬 것

- **구현·수식·검산·실험 설정은 바꾸지 않습니다.** 허용되는 것은 출력이 같은 재작성뿐입니다: `-(-a // b)` → `math.ceil(a / b)`(준비 셀에 `import math`), 한 줄의 `a; b` → 두 줄, `lambda` → 이름 있는 `def`, `assert a and b` → 주장별 `assert`, 정의와 실행·검산이 한 셀 → 두 셀, 25줄 초과 셀 → 함수별 셀.
- **여섯 규칙**(WRITING.md 7절): 모든 `def`에 목적·수식·모양 docstring(연습 뼈대·검사 함수도 한 줄), 다른 절이나 네 셀 이상 위에서 온 이름은 셀 첫 줄 `# 쓰는 것: X, T, N (📎 데이터 셀) · w0, b0 (1절)`, 관용구는 쉬운 형태로 바꾸거나 첫 등장 마크다운의 코드 상자 `> **코드** · ...`로, `assert`에는 "…이어야 합니다" 메시지와 `np.testing.assert_*`에는 줄 끝 `# 검산: 무엇 = 무엇`, 한 셀 한 역할(코드 25줄 이하, 필요하면 `# 1. 예측과 잔차` 단계 번호), 주석 내용은 모양 → 수식 항 → 이유.
- 주석 줄은 코드 줄의 1/3 이하. 앞 마크다운 셀이 이미 말한 것을 코드에서 되풀이하지 않습니다. `# 반복문 시작` 같은 낭독형 주석은 두지 않습니다.
- 코드 상자는 그 관용구가 **그 노트북에서** 처음 나오는 마크다운 셀에 둡니다. 00·01b에서 설명한 관용구라도 다른 노트북에서 처음 나오면 한 줄로 다시 둡니다(노트북은 따로 읽힙니다). 마크다운에 물결표를 쓰지 않고, 수식은 mathtext가 지원하는 명령만 씁니다.
- 마크다운 태그 규칙 유지: 새로 넣는 마크다운 셀은 📎 또는 🔍(🔑는 절 수 이하), 제목은 `## 🔑 1. …`처럼 표시를 제목 줄 안에.
- 셀을 나눌 때 계산 순서를 바꾸지 않습니다. 특히 난수 생성기(`rng`) 호출 순서와 `print` 순서가 그대로여야 출력 diff가 빕니다.
- 검산만 있던 셀(아무 출력이 없던 셀)에만 통과 `print` 한 줄을 넣습니다. 이미 print가 있는 셀에는 추가하지 않습니다.
- 그림 보조 파일 `src/utils/*_plots.py`는 손대지 않습니다. 노트북 코드 셀만 대상입니다.

## 하지 않을 것 (사용자가 나중으로 미룸)

- `run_check(...)` 셀 아래에 반환값 `False`가 표시되는 문제. 18개 노트북 공통 형식이라 이번 범위 밖입니다. 그대로 둡니다.

## 알려진 함정 (00·01b에서 겪은 것)

- `check_code_comments`의 "멀리서 온 이름" 검사는 그림 호출만 있는 셀과 `exercise_*` 함수 안은 제외합니다. `check_*` 함수가 본문 함수(예: `epoch_batches`, `gradient_on`)를 쓰면 docstring 바로 아래에 `# 쓰는 것:`이 필요합니다.
- 07·09·13에는 46–54줄 셀이 있습니다(07 `gated_forward`·`gated_backward`, 09 `transformer_forward`·`transformer_backward`·`token_ce`, 13 자동미분). 함수마다 셀을 나누고, backward의 기울기 줄에는 "어느 수식의 어느 항"인지 주석을 답니다. `np.add.at`, `[:, None]`, `keepdims`, `caches[0][-1]` 같은 관용구는 코드 상자 대상입니다.
- `verify_exercises`는 `raise NotImplementedError` 줄을 정답 블록으로 바꿔 실행합니다. 연습 뼈대에 docstring을 넣어도 영향이 없지만, 힌트에서 관용구를 바꾸면(`-(-a // b)` → `math.ceil`) 접힌 정답 블록도 함께 바꿔야 합니다.
- 시간 측정(`perf_counter`) 셀과 그 그림은 실행마다 달라집니다. compare 스크립트가 시간 수치를 가리므로 그 그림만 "시간 측정 그림"으로 보고에 적습니다.
- 마크다운 정답 블록의 `~`는 check_notebook 경고를 냅니다. `np.float64`가 f-string에 섞이면 `f"{v:g}"`로.

## 끝났을 때

- 16개 모두 세 검증 통과 + 출력 비교 결과를 표로 `results/foundation-validation/`(00b–04), `results/architecture-validation/`(05–10), `results/bridge-validation/`(11–15)에 날짜 파일로 저장합니다. 형식은 `results/foundation-validation/2026-09-13-code-comments-00-01b.md`를 따릅니다.
- `docs/WRITING.md` 6절 표 갱신, `memory/CHECKPOINT.md` 갱신(이전 버전은 `memory/checkpoints/`에 보존), `memory/SESSION-LOG.md`에 추가. `DECISIONS.md`는 사용자가 새로 정한 것이 있을 때만 추가.
- 최종 보고는 노트북 × (셀 수 변화·그림 동일 수·출력 차이·정적 점검) 표 하나로 시작하고, 남은 일을 에이전트가 끝낼 수 있는 것 / 사용자 결정이 필요한 것으로 나눠 적습니다. 커밋은 하지 않은 상태로 보고합니다.
