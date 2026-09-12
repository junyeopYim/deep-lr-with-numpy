# Checkpoint — 커리큘럼과 연결 기초 보강 진행 — 2026-09-12

## The story so far
기초·아키텍처 00–10 총 11개·256코드·61그림이 실행 검증되어 있습니다. 사용자가 커리큘럼 보강과 후속 제작을 자율 위임했습니다(D-007). 기존 결과는 results/foundation-validation/, results/architecture-validation/에 있습니다.
이번 구현 범위는 전체 학습 경로 확장과 연결 기초 11–15 다섯 노트북입니다. 선형대수·다변량 확률/정보이론·자동미분·손실/정규화·실험/평가를 작은 완성 예제와 직접 변형으로 연결합니다.

## Decided
D-004의 수학 복습과 완성 예제→직접 변형 방식을 유지합니다. D-007에 따라 중간 승인 없이 산출물과 실행 검증까지 진행합니다. 11–15 배치와 다섯 단원 분할은 구현 선택입니다.

## Waiting on the user
진행에 필요한 추가 입력 없음. 사용자는 결과물을 본 뒤 학습 피드백을 주기로 했습니다.

## Next first action
notebooks/02_연결/11_linear_algebra.ipynb부터 투영·공분산·SVD·PCA의 수식과 구현·검산·변형을 작성합니다.

## Tried
현재 환경에 NumPy, PyTorch, nbformat, nbclient, matplotlib가 있습니다. scripts/verify_notebooks.py에 bridge 묶음을 추가해 새 단원의 결과를 별도로 기록합니다. 기존 00–10의 핵심 코드와 실행 기록을 학습 경로의 출발점으로 사용합니다.
