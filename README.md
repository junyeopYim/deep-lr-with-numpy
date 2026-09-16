# 개념을 수식과 코드로

NumPy만으로 딥러닝을 공부하는 노트북 모음입니다. 노트북과 보조 코드는 GPT-6 Astra와 Claude Fable 5.1로 구현했습니다.

모델의 이름을 아는 데서 출발해, **왜 그 수식인지 설명하고 NumPy로 직접 구현하는 것**을 목표로 합니다.
행렬·미분·확률은 실제 계산에 쓰이는 자리에서 함께 확인합니다.

현재 **00–15에 00b·01b를 더한 18개 노트북**을 실행하며 공부할 수 있습니다.
[전체 커리큘럼](docs/CURRICULUM.md)은 64개 학습 단위의 선수 개념·구현 과제·검산 기준을 담고 있습니다.

## 여기서 시작합니다

| 순서 | 노트북 | 끝나면 직접 해 볼 것 |
|---|---|---|
| 00 | [수식을 NumPy로 옮기기](notebooks/00_기초/00_math_to_numpy.ipynb) | 기호의 shape와 합하는 축을 정하고, 반복문을 행렬곱으로 옮기기 |
| 00b | [벡터화: 반복문을 배열 연산으로](notebooks/00_기초/00b_vectorization.ipynb) | 원소별·축 합·브로드캐스팅·인덱싱으로 반복문을 없애고 시간 차이를 재기 |
| 01 | [미분으로 학습시키기](notebooks/00_기초/01_gradients_and_learning.ipynb) | 손실의 미분을 유도하고, 검산한 기울기로 회귀 모델 학습시키기 |
| 01b | [미니배치: 왜 데이터를 나누어 학습하나](notebooks/00_기초/01b_minibatch.ipynb) | 배치 기울기의 잡음과 비용을 재고 에폭·스텝·셔플로 미니배치 루프 쓰기 |
| 02 | [확률에서 손실 만들기](notebooks/00_기초/02_probability_and_losses.ipynb) | 관측 모형에서 MSE·BCE·softmax CE를 유도하고 안정적으로 구현하기 |
| 03 | [퍼셉트론에서 MLP까지](notebooks/00_기초/03_perceptron_to_mlp.ipynb) | XOR의 선형 분리 한계를 설명하고 다층 forward·backward를 직접 검산하기 |
| 04 | [optimizer의 계산](notebooks/00_기초/04_optimizers.ipynb) | SGD·Momentum·RMSProp·Adam의 상태와 갱신을 손계산하고 미니배치로 학습하기 |

## 기본 아키텍처

각 노트북은 필요한 수식·shape를 설명하고, NumPy 구현·역전파 검산·작은 학습·직접 변형으로 이어집니다.
모든 노트북의 마크다운에는 🔑 핵심 / 🔍 확인 / 📎 참고 / ✏️ 연습 표시가 있고, 🔑 절마다 끝에 절 연습 하나와 노트북 끝에 연습 A–D가 있으며, 연습은 함수 뼈대에서 핵심 한두 줄만 채우는 형식입니다([글 규칙](docs/WRITING.md)).
공통 기초를 마치셨다면 05부터 시작하시면 됩니다. 데이터는 본문에서 생성하며 CPU에서 실행합니다.

| 순서 | 노트북 | 완성 예제와 직접 확인할 것 |
|---|---|---|
| 05 | [CNN](notebooks/01_아키텍처/05_cnn.ipynb) | 숫자 도안 분류, 패치·합성곱·풀링 미분과 겹침 누적 |
| 06 | [RNN](notebooks/01_아키텍처/06_rnn.ipynb) | 기호 순서 분류, 시간별 캐시·BPTT·길이 변경 |
| 07 | [LSTM·GRU](notebooks/01_아키텍처/07_lstm_gru.ipynb) | 지연 신호 기억, 게이트·상태별 미분·forget bias 변경 |
| 08 | [Attention](notebooks/01_아키텍처/08_attention.ipynb) | key의 값 검색, Q·K·V·softmax 미분·mask |
| 09 | [Transformer](notebooks/01_아키텍처/09_transformer.ipynb) | 토큰열 역순 변환, MHA·LayerNorm·잔차·FFN·embedding |
| 10 | [GNN](notebooks/01_아키텍처/10_gnn.ipynb) | 그래프 분류, 이웃 집계·readout·순열 성질과 표현 한계 |

## 구조와 목적을 잇는 기초

00–10에서 사용한 수학을 확률 모델·자동미분·실험 설계로 넓힙니다. **지금 이어 읽을 순서는 11 → 12 → 13 → 14 → 15입니다.**

| 순서 | 노트북 | 완성 예제와 직접 확인할 것 |
|---|---|---|
| 11 | [선형대수와 PCA](notebooks/02_연결/11_linear_algebra.ipynb) | 투영·공분산·SVD·PCA, 3차원 압축과 복원 오차 |
| 12 | [Gaussian·정보이론·추정](notebooks/02_연결/12_gaussian_information.ipynb) | 샘플·로그 밀도·entropy·KL·MLE·MAP·재매개화 미분 |
| 13 | [작은 자동미분 엔진](notebooks/02_연결/13_autodiff.ipynb) | VJP·공유 그래프·broadcast, 직접 미분과 PyTorch 대조, MLP 학습 |
| 14 | [손실과 정규화](notebooks/02_연결/14_losses_regularization.ipynb) | MSE·MAE·Huber, 가중 손실·L2·ridge·AdamW·dropout |
| 15 | [실험과 평가](notebooks/02_연결/15_experiments_evaluation.ipynb) | 그룹 분할·전처리·학습 진단·지표·calibration·seed 반복 |

## 실행 환경

저장소 최상위 폴더에서 실행합니다. `root_dir`을 지정하면 노트북 안의 문서·보조 코드 링크도 열 수 있습니다.

```bash
python -m jupyter lab --ServerApp.root_dir=. notebooks/00_기초/00_math_to_numpy.ipynb
```

새 Python 환경에서는 다음과 같이 준비합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m jupyter lab
```

13번의 PyTorch 비교에는 다음 추가 환경을 사용합니다. NumPy 구현과 PyTorch 비교 모두 CPU에서 실행합니다.

```bash
python -m pip install -e ".[dev,framework]"
```

### Jupyter에서 셀 실행하기

1. 저장된 출력은 지난 실행의 기록입니다. 노트북을 열면 첫 셀부터 선택하고 **Shift+Enter**로 실행하며 내려갑니다. 코드를 실행해야 그 셀의 변수와 함수를 현재 커널에서 쓸 수 있습니다.
2. 연습은 🔑 절마다 끝에 하나(연습 1, 2, …)와 노트북 끝(연습 A–D)에 있습니다. 함수 뼈대가 미리 있고 핵심 한두 줄만 비어 있습니다. `raise NotImplementedError` 줄을 지우고 그 자리에 코드를 쓴 뒤 셀을 실행하고, 바로 아래 `run_check(...)` 셀을 실행하면 손계산 값과 비교해 줍니다. 막히면 그 아래 "정답 보기"를 펼칩니다.
3. 처음부터 다시 확인할 때는 **Run → Restart Kernel and Run All**을 선택합니다. 커널은 실행 중인 Python의 변수와 함수를 보관하며, 재시작하면 이 상태가 초기화됩니다. 모든 셀이 위에서부터 실행되는지 확인한 뒤 저장합니다.

조작 이름과 단축키는 [JupyterLab 공식 명령 안내](https://jupyterlab.readthedocs.io/en/stable/user/commands.html)를 기준으로 합니다.

## 공부하는 방식

**작은 완성 예제 → 수식과 코드의 대응 → 검산 → 직접 변형** 순서로 공부합니다.

1. 문제와 가정을 읽고 출력값이나 변화 방향을 예상합니다.
2. 숫자가 적은 예제를 손으로 계산한 뒤 실행합니다.
3. 기호마다 배열의 shape와 의미를 쓰고, 반복문과 벡터화를 대조합니다.
4. 미분을 손으로 유도하고 중심차분으로 검산합니다.
5. 차원·계수·목적을 바꾸어 자신의 구현을 확인합니다.

각 노트북 마지막의 준비도 질문으로 다음에 복습할 절을 찾습니다.
자세한 방식은 [학습 설계](docs/LEARNING_MAP.md), 주제와 전제의 연결은 [전체 학습 경로](docs/CURRICULUM.md)에 있습니다.

## 이어지는 경로

**공통 기초:** 배열 → 미분과 학습 → 확률과 손실 → 퍼셉트론·MLP → optimizer

**기본 아키텍처:** CNN → RNN → LSTM·GRU → attention → Transformer → GNN

**연결 기초:** 선형대수·정보이론 → 자동미분 → 손실·정규화 → 실험·평가

**구조 조합:** 정규화 층 → ResNet·U-Net·ViT → encoder–decoder·집합·graph attention

**확률 모델과 생성:** AE → 잠재변수·ELBO·VAE, GAN, flow·에너지·score·diffusion·flow matching → 생성 평가

**표현과 언어:** 대조학습·교사·공분산 제약·mask·JEPA, 토큰화 → GPT·생성 → SFT·LoRA·선호·다중모달

**강화학습:** Bandit·MDP → MC·TD·Q·DQN → 정책 기울기·Actor–Critic·PPO → 연속 행동·계획·오프라인·RLHF

주제마다 필요한 전제와 이해를 확인할 작은 문제를 명시합니다. 예를 들어 GPT는 Transformer와
조건부 확률을 연결한 다음, VAE는 확률·우도와 잠재변수를 연결한 다음 공부합니다.

## 프로젝트 구성

```text
notebooks/00_기초/       배열·벡터화·미분·미니배치·확률·MLP·optimizer
notebooks/01_아키텍처/  CNN·RNN·LSTM/GRU·Attention·Transformer·GNN
notebooks/02_연결/      선형대수·정보이론·자동미분·정규화·평가
src/utils/         수치미분 검산기와 그림 보조 코드
src/data/          이후 실험에서 사용할 작은 데이터와 MNIST 로더
docs/              학습 방식, 전체 경로, 표기 규약
scripts/           노트북 실행 검증
results/           실행 검증 기록
memory/            학습 방향과 작업 상태
```

모델·손실·역전파·갱신·실험 계산은 노트북 본문에 둡니다.
긴 그림 코드는 계산 결과를 인자로 받는 보조 함수로 나눕니다.

## 실행 검증

```bash
python -m pytest
python scripts/verify_notebooks.py --suite foundation
python scripts/verify_notebooks.py --suite architecture
python scripts/verify_notebooks.py --suite bridge
```

검증기는 각 노트북을 별도 커널에서 실행하고, 저장된 그림을 디코딩해 확인합니다.
11–15의 수치와 독립 실습은 [연결 기초 검증](results/bridge-validation/bridges.md), 실행 출력은 [실행 기록](results/bridge-validation/execution.json)에 있습니다.
실행 결과는 [공통 기초 실행 기록](results/foundation-validation/execution.json)과 [아키텍처 실행 기록](results/architecture-validation/execution.json)에 각각 남습니다.
아키텍처의 수치·직접 변형·독립 점검은 [아키텍처 검증](results/architecture-validation/architectures.md)에 있습니다.
공통 기초 다섯 단원의 계산 결과와 독립 실습 점검은 [공통 기초 검증](results/foundation-validation/common-foundations.md)에 있습니다.
초기 00·01의 설치·실습 점검은 [첫 두 단원 점검](results/foundation-validation/rehearsal.md)에 남겨 두었습니다.

## 라이선스

코드·노트북·문서 모두 [MIT 라이선스](LICENSE)입니다. 출처를 남기면 수업 자료나 다른 저장소에 자유롭게 복사·수정해 쓸 수 있습니다.
