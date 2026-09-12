# Checkpoint — 기본 아키텍처 05–10 완료 — 2026-09-12

## The story so far
D-006의 CNN, RNN, LSTM·GRU, Attention, Transformer, GNN 여섯 노트북을 모두 작성·실행했습니다.
새 코드 109개·그림 30개, 오류·한글 글리프 경고 0개. 공통 기초 포함 11개·256코드·61그림, pytest 19개 통과.
핵심 수학·모델·역전파·학습·변형은 본문, 긴 그림만 src/utils/architecture_plots.py에 있습니다.
각 단원 독립 실행·직접 backward·변형·오답 검출 완료. 09·10 최종 재점검에서 학습 차단 0건.
근거: results/architecture-validation/architectures.md, execution.json, source-manifest.json, rehearsal/.
README와 CURRICULUM에 실제 05–10 링크가 있으며 11 이후는 후속 계획입니다. 커밋·푸시는 하지 않았습니다.

## Decided
D-006: 기본 아키텍처 전체를 개별 노트북으로 제작했습니다.
D-004: 필요한 수학을 코드와 함께 확인하고 작은 완성 예제 뒤 직접 변형합니다.

## Waiting on the user
이번 제작 완료에 필요한 추가 결정 없음. 실제 학습에서 막히는 위치가 생기면 설명 깊이를 조정합니다.

## Next first action
후속 제작 요청이 오면 docs/CURRICULUM.md의 11 Autoencoder를 병목 표현·복원 목적·전체 backward·작은 학습부터 작성합니다.

## Tried
CNN의 zip 검사기가 빈/부분 반환을 놓쳐 세 반환값 분해와 shape 검사를 추가했습니다.
MHA의 (N,T,T) mask는 N=head일 때 배치/head 축이 섞여 명시적 head 축과 입력 검사를 추가했습니다.
GNN 선형 readout의 구분 불가능성에 상수 특성 조건을 명시하고 실제 고립 노드를 검산했습니다.
일렬 그래프의 자동 y축에서 노드가 잘려 좌표 여백을 명시했습니다.
임시 작성 원본은 /tmp/dlfs-architectures/*.cells이며 최종 실행 결과는 저장소의 ipynb가 기준입니다.
