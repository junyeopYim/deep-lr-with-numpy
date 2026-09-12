# Checkpoint — 연결 기초 작성·실행, 독립 검증과 문서 마무리 — 2026-09-12

## The story so far
D-007에 따라 커리큘럼을 64개 학습 단위(00–63)로 펼쳤고 notebooks/02_연결/11–15 다섯 노트북을 작성했습니다. 첫 전체 실행은 코드104개·PNG25개·셀 오류/경고0개입니다. 전체 자료는16개·360코드·86그림입니다.
새 파일: 11_linear_algebra,12_gaussian_information,13_autodiff,14_losses_regularization,15_experiments_evaluation. 핵심 계산은 본문, 그림은 src/utils/bridge_plots.py. scripts/verify_notebooks.py에 bridge 묶음, pyproject에 framework(PyTorch) 선택 의존성 추가. README/CURRICULUM/LEARNING_MAP 연결 수정.
독립 점검: 11 완료; 12의 대각 공분산·KL 증명·재매개화 shape를 보완하고 새 독자 재점검 완료. 13은 공유 분기·부모 누락·반복 backward 오답 검사를 강화했고 새 독자 점검 중. 14는 ridge 일반식의 NK 계수와 K=2 해석해 검산을 추가. 15는 데이터 seed151/분할152를 분리하고 새 사람 점수의 양성 비율 해석 추가.
최근 14·15가 갱신되어 최종 새 커널 실행과 수치 보고를 이어갈 차례입니다. 이전 execution.json의 15 수치는 변경 전입니다.

## Decided
D-007: 보강 및 제작 자율 진행. 이번 범위는 경로 확장과 연결 기초 다섯 단원입니다. 16–63은 후속 학습 설계로 명시했습니다. D-004의 완성 예제→직접 변형 방식을 유지합니다.

## Waiting on the user
추가 입력 필요 없음. 사용자는 완성 자료를 보고 이후 피드백을 주기로 했습니다.

## Next first action
python scripts/verify_notebooks.py --suite bridge 로 최종 11–15를 실행하고 results/bridge-validation/bridges.md와 검증 source manifest를 작성합니다.

## Tried
임시 작성 원본 /tmp/dlfs-bridges/{11..15}.cells, 생성기 /tmp/dlfs-common-foundations/build_notebook.py. 실행 결과는 ipynb가 기준입니다. 독립 실습 /tmp/dlfs-bridges/rehearsal11,12,12-final,13,13-final,14,15.
에이전트 common02_03_final_rehearsal=13 최종 재점검 중, common02_rehearsal=14 1차 점검 중, common04_rehearsal=15 1차 점검 중. 이후 14 수정본과 문서의 새 독자 점검이 필요합니다.
그림 25개는 root가 PNG 모아보기로 직접 확인했습니다. 15 데이터 생성 분리 후의 변경 그림은 최종 실행에서 다시 확인합니다. 기존 00–10 핵심 셀 source는 유지합니다.
