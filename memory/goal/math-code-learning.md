# 수학을 실행되는 코드로 연결하는 학습 환경

## 현재 완료 지점 — D-007

64개 학습 단위의 경로와00–15 총16개 실행 노트북을 제공합니다. 이번 보강11–15는104코드·25그림과 독립 실습·변형 검증을 마쳤습니다. 전체360코드·86그림·pytest19개 통과.
현재 근거는 results/bridge-validation/bridges.md와 memory/CHECKPOINT.md에 있습니다.16–63은 이어 만들 학습 설계이며, 사용자 피드백에 맞춰 설명 깊이를 조정합니다.

## 초기 재설계의 목표와 완료 기준

목표: 사용자가 개념을 수식으로 구성하고 NumPy로 구현하며 자신의 이해를 검산할 수 있게 학습 경로를 구성합니다.

초기 재설계의 완료 기준: 전체 경로·학습 방식·시작 동선을 맞추고, 새 첫 두 노트북을 독립 커널에서 실행한 뒤 계산·그림·한글 표시와 처음 읽는 사람의 실행을 확인합니다.

## 지형도 — 2026-09-12

| 가지 | 필요한 것 | 이미 가진 것 | 빈틈과 첫 행동 |
|---|---|---|---|
| 사용자에게 맞는 설명 | 수학·구현 출발점과 실습 선호 | D-002·D-004 | 작은 완성 예제와 변형 문제를 첫 두 단원에 구현 |
| 학습 경로 | 구조·목적·최적화·사용의 관계 | LEARNING_MAP 초안, 기존 커리큘럼, 원 논문 | 공통 기초를 연결하고 주제별 전제 명시 |
| 수학→코드 연결 | 기호·shape·연산 축·역전파 검산 | 기존 gradcheck, 00–03 | 공통 기초를 명시적인 손계산부터 다시 작성 |
| 실제 학습 가능성 | 독립 실행, 정확한 그림, 변형 문제 | nbclient·NumPy·matplotlib 설치 관찰 | 첫 두 단원 전체 실행과 rehearsal |

관찰됨: 기존 네 노트북은 저장된 출력이 있으나 현재의 개인 학습 요구와 일치하는지는 새 구성을 통해 다시 확인합니다.
확정됨(사용자): 행렬·미분·확률도 코드와 함께 다시 확인하고, 작은 완성 예제 뒤에 직접 바꾸는 문제를 풉니다.
가정: 아래 설계가 사용자에게 맞을 것이라는 기대는 첫 단원의 실제 학습 반응으로 조정합니다. 개인 학습 효과 자체는 파일 실행만으로 판정하지 않습니다.

## 구조 v1

- 학습 경로가 사용자의 목표를 다룹니다. [filled; docs/CURRICULUM.md]
  - 각 기본 아키텍처의 전제를 명시합니다. [filled; docs/CURRICULUM.md 전제 열]
  - 손실·최적화·생성·RL의 관계를 구분합니다. [filled; docs/LEARNING_MAP.md와 CURRICULUM.md]
  - 준비도 질문으로 다음 복습 지점을 찾습니다. [filled; notebooks/00_기초/00_math_to_numpy.ipynb 8절, 01_gradients_and_learning.ipynb 9절]
- 수식과 배열 연산의 대응이 검산됩니다. [filled; notebooks/00_기초/00_math_to_numpy.ipynb]
  - 내적 한 번을 손으로 계산할 수 있습니다. [filled; 00 1절 손계산 -0.3과 assert]
  - 배치와 다중 출력의 축을 구분할 수 있습니다. [filled; 00 3·4절 인덱스 식과 반복문]
  - 벡터화 결과가 손계산과 일치합니다. [filled; 00 손계산·반복문·벡터화 assert]
  - reshape와 transpose의 차이를 판별합니다. [filled; 00 6절 배열 반례와 그림]
- 미분과 파라미터 갱신의 대응이 검산됩니다. [filled; notebooks/00_기초/01_gradients_and_learning.ipynb]
  - 중심차분과 도함수를 대조합니다. [filled; 01 2·3절 3차식과 중심차분]
  - MSE의 축 평균과 기울기가 일치합니다. [filled; 01 5·9절 (4,2)·(2,2) 출력 검산]
  - affine backward의 입력·가중치·편향을 대조합니다. [filled; results/foundation-validation/execution.json, 정상 최대 상대오차 1.72e-10]
  - 작은 회귀 문제가 학습됩니다. [filled; 01 7절, 파라미터 최대 차이 1.20e-9]
- 학습 자료를 바로 사용할 수 있습니다. [filled; README.md, 독립 설치·실행 관찰]
  - 각 노트북을 새 커널에서 끝까지 실행합니다. [filled; results/foundation-validation/execution.json, 코드 52개]
  - 저장된 그림을 디코딩하고 직접 봅니다. [filled; figures/foundation-validation/, 11개 직접 관찰]
  - 배경 설명 없이 예제와 변형 검산을 실행합니다. [filled; results/foundation-validation/rehearsal.md, 2차 원본 52개+직접 실습 2개 실행]

## 다음 한 작업

D-007의 커리큘럼 보강에 따라 00–15 총 16개 노트북을 제공했습니다. 이어 제작할 기본 순서는 16 정규화 층과 구조 조합이며, 생성 모델 경로는 22 Autoencoder입니다. 사용자의 실제 학습 반응에 따라 설명 깊이와 변형 난도를 조정합니다.

## 알려진 확장

확률 기초·우도, 분류 목적, optimizer, CNN, RNN/LSTM/GRU, attention, Transformer, GNN, VAE/GAN/Score, 자기회귀·GPT, 표현 학습, RL. 전체 경로에는 이름과 전제를 남기고 실제 단원을 만들 때 한 학습 목표씩 나눕니다.

## 완료 점검

관찰됨(2026-09-12):

- pass: 새 00·01의 코드 셀 52개를 독립 커널에서 실행했습니다.
- pass: 그림 11개를 디코딩하고 직접 확인했습니다. 00 평면의 축 제목 잘림을 수정했습니다.
- pass: 현재 공통 검산기 테스트 19개가 통과했습니다.
- pass: 회귀 손실 9.961568→0.008696, 독립 최소제곱 해와 파라미터 차이 1.20e-9.
- pass: 두 차례 독립 rehearsal 완료. 조회 오기·서버 루트 문제를 수정하고 설명을 보충한 뒤 2차에서 차단 0건. 코드 54개, HTTP 17개 경로, 그림 11개 확인. 근거: results/foundation-validation/rehearsal.md. GUI의 실제 클릭은 환경상 미확인이고 개인 학습 효과는 평가 범위 밖입니다.

sub-foundations exposed: 배열 축·브로드캐스팅 — 00 5절; 평균의 분모 — 01 4·5·9절; 미분과 갱신의 구분 — 01 6절; 확률·우도 — 다음 02 단원의 명명된 전제.

## 이전 구상

기존 코세라 주차 순서를 따르는 04 정규화 / 05 최적화와, 잠정 04 CNN / 05 RNN 구성은 D-002·D-003에 따라 재설계합니다.

## 공통 기초 확장 완료 — D-005, 2026-09-12

관찰됨: 02 확률·손실, 03 퍼셉트론·MLP, 04 optimizer를 각각 작성하고 00–04 전체를 실행했습니다.
코드 셀 147개·그림 31개, pytest 19개 통과. 새 단원은 독립 실행과 직접 변형·오답 검출을 거쳤습니다.
증거는 results/foundation-validation/common-foundations.md에 있습니다. 다음 단위 05 CNN은 아직 설계 상태입니다.

## 기본 아키텍처 확장 완료 — D-006, 2026-09-12

관찰됨: 05 CNN, 06 RNN, 07 LSTM·GRU, 08 Attention, 09 Transformer, 10 GNN을 각각 작성·실행했습니다.
수식·배열·모든 핵심 forward/backward·학습·직접 변형은 본문에 있고 긴 그림은 architecture_plots.py에 둡니다.
최신 코드 109개·그림 30개, 오류·stderr·한글 글리프 경고 0개. 공통 기초 재실행 147개·31개와 pytest 19개도 통과했습니다.
각 단원은 독립 실행·자체 backward·변형 학습·오답 검출을 거쳤습니다. 09·10 최종 재점검에서 학습 차단 0건입니다.
실행 근거: results/architecture-validation/architectures.md, execution.json, source-manifest.json, rehearsal/.
CNN 미완성 반환 계약, MHA mask의 배치/head 축, GNN의 상수 특성 조건·실제 고립 노드 및 그래프 잘림을 보완했습니다.
학습용 작은 문제의 성능이며 실제 사용자의 학습 효과를 단정하지 않습니다. 11번 이후는 후속 제작 계획이고 커밋·푸시는 하지 않았습니다.

## 커리큘럼과 연결 기초 보강 — D-007, 2026-09-12

사용자가 커리큘럼 보강과 제작을 자율 위임했습니다. 구현 범위는 64개 학습 단위의 경로 설계와 연결 기초 11–15 다섯 노트북입니다.
선형대수/PCA, Gaussian/정보이론/추정, 작은 자동미분 엔진과 PyTorch 대조, 손실/정규화/AdamW/dropout, 분할/평가/반복 실험을 작성·실행했습니다.
새104코드·25PNG, 전체360코드·86PNG가 새 커널에서 실행됐고 pytest19개가 통과했습니다. 기존00–10의 셀 source는 기존 manifest와 일치합니다.
독립 실습은 자신이 작성한 핵심 함수를 실제 checker와 학습 loop에 연결합니다. Gaussian의 공분산·shape 설명, 자동미분의 공유 분기·부모 연결, 다중 출력 ridge의 NK 규약, 지표의 임계값·F1·단일 클래스 검사를 보강했습니다.
16–63은 후속 제작 설계로 명시했습니다. README·CURRICULUM·LEARNING_MAP에는 실행 자료와 관심 경로의 선수 순서가 연결되어 있습니다.
근거: results/bridge-validation/bridges.md, execution.json, source-manifest.json, render-and-links.json, rehearsal/.

### D-007 완료 확인

관찰됨: 11–15 최종 독립 실습·현재 실행 사본 대조5/5, 문서의 선수 경로·44개 로컬 링크·실행 명령 및 직접 함수 검사 완료. 검증 보고서와 원본 source 해시를 저장소에 보존했습니다.
