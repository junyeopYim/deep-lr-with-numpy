# 2026-09-13 · 05–10 글 규칙·그림 규칙·코딩 테스트형 연습 적용 검증

기본 아키텍처 여섯 노트북을 `docs/WRITING.md`·`docs/DESIGN.md` 규칙(🔑/🔍/📎/✏️ 표시, 첫 등장 용어 상자, 상황 도입, 🔑 절마다 예시 그림, `run_check` 연습과 접힌 정답)에 맞게 고친 뒤,
`python scripts/check_notebook.py`(새 커널 실행·그림 추출·표시 통계)와 `python scripts/verify_exercises.py`(정답을 채워 실행)로 확인한 결과입니다.
수식·구현·검산·실험 설정 코드는 원본 셀을 그대로 옮겼고(그림 함수 호출과 import 줄만 교체), 그림 계산 셀과 일치 검산 줄만 추가했습니다.

| 노트북 | 셀 | 그림 | 오류 | 표시(🔑/🔍/📎/✏️) | 실행 시간 | 정답 검증 |
|---|---:|---:|---:|---|---:|---|
| 05_cnn | 72 | 9 | 0 | 🔑 5 / 🔍 16 / 📎 5 / ✏️ 5 | 3.4s | 채운 연습 4개 · 실패 0개 |
| 06_rnn | 53 | 7 | 0 | 🔑 4 / 🔍 8 / 📎 6 / ✏️ 5 | 2.6s | 채운 연습 4개 · 실패 0개 |
| 07_lstm_gru | 64 | 8 | 0 | 🔑 6 / 🔍 12 / 📎 5 / ✏️ 5 | 4.1s | 채운 연습 4개 · 실패 0개 |
| 08_attention | 63 | 8 | 0 | 🔑 5 / 🔍 13 / 📎 5 / ✏️ 5 | 3.2s | 채운 연습 4개 · 실패 0개 |
| 09_transformer | 61 | 7 | 0 | 🔑 5 / 🔍 13 / 📎 5 / ✏️ 5 | 3.2s | 채운 연습 4개 · 실패 0개 |
| 10_gnn | 64 | 7 | 0 | 🔑 5 / 🔍 13 / 📎 6 / ✏️ 5 | 2.5s | 채운 연습 4개 · 실패 0개 |

합계 셀 377개, 그림 46장, 연습 24개, 실행 19.0초(Miniconda Python 3.13.12, NumPy 2.3.5, CPU). 표시 없는 마크다운 셀 0, 제목 줄 밖의 표시 0, stderr 0.
pytest 22개 통과.

## 새 그림 보조 파일

`src/utils/cnn_plots.py`, `rnn_plots.py`, `lstm_plots.py`, `attention_plots.py`, `transformer_plots.py`, `gnn_plots.py`. 모두 `concept_plots`·`schematic_plots` 부품만 쓰고 받은 배열만 그립니다.
`schematic_plots.draw_unrolled_chain`(시간축으로 펼친 순환 셀)을 공용 부품으로 추가했습니다. 기존 `architecture_plots`의 `plot_curves`·`plot_images`는 각 파일이 다시 내보냅니다.

## 그림 확인

추출한 PNG 46장을 직접 열어 화살표 위치·제목과 라벨 겹침·색 눈금을 확인했고, 다음을 고쳤습니다.
합성곱 장면의 `-0` 표기, RNN 문장 예시의 화살표 글자 겹침, LSTM backward 도식의 라벨 겹침과 표 간격, attention 손계산의 막대 높이·softmax 오답 라벨·mask 화살표 글자,
Transformer LayerNorm·head 분리의 화살표 글자, GNN 집계·순열 그림의 `np.float64`/`np.int64` 표기와 한 층 그림의 배치.

## 한계

새 커널 실행과 정답 채움 실행은 자동화했지만, 브라우저에서 MathJax 수식과 `<details>` 접힘을 직접 클릭해 확인한 검사는 아닙니다. 학습 성능 수치는 원본과 같은 설정이며 이번 변경으로 바뀌지 않았습니다.
