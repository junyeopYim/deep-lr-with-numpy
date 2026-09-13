# HANDOFF — 05–15 노트북에 글·그림·연습 규칙 적용 (2026-09-13 작성, 읽은 뒤 삭제)

이 파일은 다음 세션이 한 번 읽고 지웁니다. 상태는 CHECKPOINT.md, 결정은 DECISIONS.md에 있습니다.

## 먼저 읽을 것 (순서대로, 15분)
1. `AGENTS.md` — 절 구성 순서와 노트북 작성 규칙.
2. `docs/WRITING.md` — 표시(🔑/🔍/📎/✏️)·용어 상자·상황 도입·연습 형식(5절)·적용 현황.
3. `docs/DESIGN.md` — 그림 원칙, 부품(`concept_plots`·`schematic_plots`), 노트북별 그림 계획(05–15는 "계획" 상태).
4. 완성본 두 개를 열어 형식을 눈에 익힐 것: `notebooks/00_기초/01b_minibatch.ipynb`(새로 만든 것), `notebooks/00_기초/03_perceptron_to_mlp.ipynb`(기존 노트북을 고친 것).

## 할 일: 05–15 (11개) 를 00–04와 같은 형식으로
`notebooks/01_아키텍처/05–10`, `notebooks/02_연결/11–15`. 05부터 번호 순서로, **한 노트북씩 끝내고 짧게 보고**한다.

노트북마다 체크리스트:
- [ ] 맨 위 셀: 이번 질문 / 준비물·도착점·다음 연결 / "읽는 법" 표 / "반드시 가져갈 세 가지" / 링크(글 규칙·그림 설계·보조 코드). 00–04의 첫 셀을 그대로 본뜬다.
- [ ] 📎 예시 셀: 그 노트북의 작은 데이터(기존 것 유지) + MNIST 고정 예시 `load_examples()`(필요할 때만). 새 변수는 `mnist_` 접두어. 본문 변수와 이름이 겹치지 않게 할 것(04에서 `g1`, 03에서 `corners`가 겹쳐 오류 났음).
- [ ] 모든 마크다운 셀에 표시. 제목 셀은 `## 🔑 1. …`처럼 제목 줄 안에 넣는다(제목 앞 같은 줄에 두면 제목이 깨진다). 🔑는 절 수 이하.
- [ ] 🔑 절: "**이 절에서 가져갈 것:**" 한 문장 → 용어 상자(첫 등장 용어, 뜻 + 코드에서의 모습) → 손에 잡히는 상황 → 수식 → 예시 그림. 셀 순서는 🔑 설명 → 계산 셀(NumPy만, 뒤의 구현 함수 사용 금지) → 그림 셀 → 🔍 해석 → 구현 → 검산. 학습 결과 그림만 학습 셀 뒤(예외).
- [ ] 그림: `src/utils/<노트북>_plots.py`에 조합 함수를 추가하고 `concept_plots`·`schematic_plots` 부품만 쓴다. DESIGN.md 3절의 계획을 따른다. 05는 `draw_cells` 커널 창·풀링·`draw_blocks` 블록 흐름 + 7 이미지 위의 합성곱, 06·07은 7을 28행 시퀀스로, 08·09는 4×4 패치 attention, 10·13은 MNIST 없이 도식, 11은 평균·고유 이미지·복원, 12는 클래스 평균·entropy, 14는 dropout 마스크·L2, 15는 축 없는 혼동 행렬.
- [ ] 연습: WRITING.md 5절 형식. 노트북당 3–4개, 채울 줄은 한두 줄, `run_check` 검사, `<details>` 정답. 기존 "my_xxx를 직접 작성" 연습을 뼈대로 바꾼다. 준비도 질문은 `### ✏️ 준비도 질문`으로 유지.
- [ ] 검증: `python scripts/check_notebook.py <노트북>`(새 커널 실행, 그림 PNG 추출, 표시 통계) → 추출한 PNG를 **직접 눈으로** 확인(화살표 위치, 제목·라벨 겹침, 색 눈금) → `python scripts/verify_exercises.py <노트북>`(정답을 채워 실행, 실패 0이어야 함) → `python -m pytest -q tests`.
- [ ] 기록: `docs/DESIGN.md` 3절과 `docs/WRITING.md` 적용 현황을 "완료"로, `results/<suite>-validation/`에 검증 표, `memory/SESSION-LOG.md`에 세션 기록, `memory/CHECKPOINT.md` 갱신(이전본은 `checkpoints/`에 보존). DECISIONS.md는 추가만.

## 하지 말 것
- 커밋·푸시는 사용자가 시키기 전에는 하지 않는다.
- 공부 중인 수식·구현·검산·실험 설정을 바꾸지 않는다. 글·그림·연습 형식만 바꾼다.
- 그림 제목·라벨의 mathtext에 `\tfrac`, `\frac1N`, `\ge`, `\text`를 쓰지 않는다(`\frac{1}{N}`, `\geq`).
- 마크다운에서 물결표 `~`를 범위 표기로 쓰지 않는다(`1–4`).
- MNIST 학습은 5,000장 부분집합, 배치 계산은 CPU 몇 초 안에 끝나게 한다.

## 이미 해결한 문제 (같은 자리에서 헤매지 말 것)
- 화살표가 짧거나 어긋남: `concept_plots.new_figure`가 전역 autolayout을 끄고, `connect`가 tight bbox를 쓴다. 제목이 패널보다 넓으면 제목을 `connect` 뒤에 붙인다.
- 학습 전 난수 가중치가 남아 은닉 유닛 이미지가 잡음처럼 보임 → 초기값을 뺀 변화량 `W − W⁽⁰⁾`을 그린다.
- 공통 색 눈금에서 한 판이 다른 판을 가림 → 행/판마다 `vmax`를 따로 둔다.
- `figure.autolayout` 경고 → `cp.new_figure` 사용.

## 완료 기준
05–15 모두 check_notebook 오류 0·표시 없는 셀 0, verify_exercises 실패 0, pytest 통과, DESIGN·WRITING 상태 갱신, SESSION-LOG·CHECKPOINT 기록. 끝나면 노트북별 셀·그림·연습 수와 실행 시간을 표로 보고한다.
