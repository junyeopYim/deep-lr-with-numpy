# Checkpoint — D-012 완료·main 반영, 05–15 글·그림·연습 규칙 적용 — 2026-09-13 15:15

## The story so far
사용자가 정한 글 규칙(🔑/🔍/📎/✏️ 표시, 첫 등장 용어 상자, 상황 도입, 🔑 절마다 예시 그림)과 코딩 테스트형 연습(run_check + 접힌 정답)을 00–04에 이어 05–15 열한 개 노트북에 모두 적용했습니다. 원본 코드 셀은 조립 스크립트로 그대로 복사했으므로 수식·구현·검산·실험 설정은 바뀌지 않았습니다. 새 그림 보조 파일 11개(`src/utils/cnn_plots.py` … `evaluation_plots.py`)와 공용 부품 `schematic_plots.draw_unrolled_chain`이 생겼습니다. 사용자 요청으로 741f730(00–04)과 4798abf(05–15)를 origin/main에 푸시했고 원격 SHA가 로컬과 같음을 확인했습니다(`results/delivery/2026-09-13-main.json`).

## Decided
D-009 개념 그림 문법, D-010 00–04 적용, D-011 코딩 테스트형 연습·00b·01b, D-012 05–15 적용(그림 파일은 노트북별 `<이름>_plots.py`, 옛 `my_xxx` 연습은 뼈대 연습 4개 + 🔍 완성 예제로 교체).

## Verified
11개 노트북을 마지막에 한 번에 새 커널로 재실행: 셀 735, 그림 94, 연습 44, 오류·stderr·표시 없는 셀 0, 정답 실패 0, 실행 38.7초. pytest 22개 통과. PNG 94장 직접 확인. 기록은 `results/architecture-validation/2026-09-13-writing-rules-exercises.md`, `results/bridge-validation/2026-09-13-writing-rules-exercises.md`, `figures/check/<노트북>/cellNN.png`(gitignore).

## Waiting on the user
05–15 결과 검토(특히 그림과 연습 난이도)와 다음 범위 선택(16 정규화 층부터 구조 조합, 또는 22 Autoencoder부터 생성 경로). 조립 스크립트(scratchpad의 `nbbuild.py`, `build_05.py`–`build_15.py`)는 세션 임시 폴더에만 있으므로 노트북 자체가 산출물입니다.

## Next first action
사용자 피드백을 반영합니다. 노트북 하나를 다시 손볼 때는 `notebooks/...ipynb`를 직접 고친 뒤 `python scripts/check_notebook.py <노트북>`과 `python scripts/verify_exercises.py <노트북>`을 돌리고 `figures/check/<노트북>/`의 PNG를 눈으로 봅니다. 새 노트북(16 이후)을 만들 때는 05–15의 첫 셀·📎 예시 셀·🔑 절 구성을 본뜨고 DESIGN.md 3절에 그림 표를 먼저 적습니다.

## Tried
- 그림 함수의 `connect` 화살표 글자는 짧게(한두 단어). 길면 옆 판의 눈금 라벨과 겹칩니다.
- `draw_cells` 판끼리 칸 크기를 맞추려면 `xlim/ylim`을 같은 크기의 틀로 다시 잡습니다(`cnn_plots._frame`).
- 막대 그림에서 `ax.text(i, v, …)`로 값 라벨을 붙일 때 `ylim`이 값보다 작으면 tight bbox가 라벨까지 포함해 그림이 세로로 수천 픽셀 늘어납니다(15 분할 그림에서 겪음).
- `np.trapezoid`로 KL을 적분할 때 격자를 꼬리까지 넓게(−10부터 12까지) 잡아야 닫힌 식과 1e-5 안에서 맞습니다.
- 마크다운 정답 블록의 `~`는 check_notebook 경고를 냅니다. `np.isin(..., invert=True)`처럼 물결표 없는 표현을 씁니다.
- `np.float64`/`np.int64`가 f-string에 섞이지 않게 `f"{v:g}"`나 `int(v)`로 적습니다.
