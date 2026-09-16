# 개념을 수식과 코드로

NumPy만으로 딥러닝을 공부하는 노트북 모음입니다. 노트북과 보조 코드는 GPT-6 Astra와 Claude Fable 5.1로 구현했습니다.

모델의 이름을 아는 데서 출발해, **왜 그 수식인지 설명하고 NumPy로 직접 구현하는 것**을 목표로 합니다.
행렬·미분·확률은 실제 계산에 쓰이는 자리에서 함께 확인합니다.

현재 **00–15에 00b·01b, 그리고 2026-09-16에 더한 11권을 합친 29개 노트북**을 실행하며 공부할 수 있습니다.
[전체 커리큘럼](docs/CURRICULUM.md)은 69개 학습 단위(번호 00–63의 64개 주제와 b 단위 5개)의 선수 개념·구현 과제·검산 기준을 담고 있습니다.

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
| 05b | [LeNet](notebooks/01_아키텍처/05b_lenet.ipynb) | 실제 MNIST 분류, 5×5 conv 두 단·padding·수용 영역, 6만 파라미터 gradcheck, 이동 반응 검산 |
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

## 구조 조합과 생성 모델 (2026-09-16 추가)

05b의 conv 백본과 11의 PCA 좌표, 12의 Gaussian·KL을 바탕으로 물체 검출과 생성 모델을 붙입니다. 생성 단원의 공통 예시는 **두 봉우리 혼합**(밀도·책임확률·score·최적 denoiser가 닫힌식)과 고정 MNIST 예시이며, GAN·score·diffusion의 MNIST는 PCA 좌표 32개에서 학습해 기저로 복원합니다. 읽는 순서는 05 → 05b → 18b, 그리고 12·15 → 23 → 24 → 25 → 28 → 32 → 33 → 34 → 35 → 35b입니다.

| 순서 | 노트북 | 완성 예제와 직접 확인할 것 |
|---|---|---|
| 18b | [물체 검출 (YOLO)](notebooks/03_구조조합/18b_detection_yolo.ipynb) | 캔버스 위 숫자 검출, IoU·격자 타깃·네 묶음 손실의 mask 고정 미분·NMS·AP |
| 23 | [잠재변수·혼합 모형·EM](notebooks/04_생성/23_latent_mixture_em.ipynb) | 두 봉우리 혼합의 주변 밀도·책임확률·score·EM·최적 denoiser 닫힌식 |
| 24 | [Jensen 부등식과 ELBO](notebooks/04_생성/24_elbo.ipynb) | Jensen 부등식, log p = ELBO + KL 열거 확인, q 학습, Gaussian q의 재매개화 ELBO |
| 25 | [VAE](notebooks/04_생성/25_vae.ipynb) | encoder·재매개화·decoder의 손 backward와 ε 고정 gradcheck, MNIST 잠재 2차원 복원·생성 |
| 28 | [GAN](notebooks/04_생성/28_gan.ipynb) | 두 목적과 교대 갱신, D를 지나 G로 오는 기울기, 최적 판별기·mode 세기, PCA 좌표 MNIST |
| 32 | [score matching](notebooks/04_생성/32_score_matching.ipynb) | score·Hyvärinen 항등식·DSM·Tweedie, MLP score 모델 학습과 닫힌식 대조 |
| 33 | [NCSN](notebooks/04_생성/33_ncsn.ipynb) | σ 사다리·Langevin·σ 조건 score 모델·annealed Langevin, PCA 좌표 MNIST |
| 34 | [Diffusion 전방 과정](notebooks/04_생성/34_diffusion_forward.ipynb) | ᾱ 닫힌식 vs 반복 잡음, posterior 계수, ε·x₀ 타깃과 score 관계, 잡음 예측 신경망 |
| 35 | [Diffusion 역과정·DDIM·guidance](notebooks/04_생성/35_diffusion_reverse.ipynb) | 역과정 한 걸음·전체 루프·DDIM·classifier-free guidance, sampler 비교 |
| 35b | [EDM](notebooks/04_생성/35b_edm.ipynb) | σ-공간 통일·Tweedie·preconditioning·λ(σ)·ρ 격자·Heun, solver 오차와 모델 오차 분리 |

## 실행 환경

Python 3.10 이상이 필요합니다. 쓰는 패키지는 다음과 같고, 전부 `pyproject.toml`에 적혀 있습니다.

| 패키지 | 용도 |
|---|---|
| NumPy | 모든 모델·미분·학습 구현 |
| matplotlib | 그림 |
| Jupyter, ipykernel | 노트북 실행 |
| pytest | 검산기와 연습 검사 테스트 |
| PyTorch (선택) | 13번 자동미분 노트북의 비교 대조에만 사용 |

저장소 최상위 폴더에서 아래 한 줄을 실행하면 가상환경을 만들고 위 패키지를 전부 설치합니다.

```bash
python -m venv .venv && source .venv/bin/activate && python -m pip install -e ".[dev]"
```

Windows PowerShell에서는 `source .venv/bin/activate` 대신 `.venv\Scripts\Activate.ps1`을 실행합니다.
13번의 PyTorch 비교까지 실행하려면 `".[dev]"` 자리에 `".[dev,framework]"`를 넣습니다. NumPy 구현과 PyTorch 비교 모두 CPU에서 실행합니다.

그림의 한글은 시스템에 설치된 한글 폰트(Apple SD Gothic Neo, Malgun Gothic, Noto Sans CJK KR, NanumGothic 등) 중 하나를 자동으로 고릅니다.
한글 폰트가 하나도 없으면 그림의 한글이 깨지니 하나를 설치합니다.

설치가 끝나면 저장소 최상위 폴더에서 JupyterLab을 엽니다. `root_dir`을 지정하면 노트북 안의 보조 코드 링크도 열 수 있습니다.

```bash
python -m jupyter lab --ServerApp.root_dir=. notebooks/00_기초/00_math_to_numpy.ipynb
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
주제와 전제의 연결은 [전체 학습 경로](docs/CURRICULUM.md)에 있습니다.

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
docs/              전체 커리큘럼, 글 규칙, 그림 설계 원칙
scripts/           노트북 실행 검증
```

모델·손실·역전파·갱신·실험 계산은 노트북 본문에 둡니다.
긴 그림 코드는 계산 결과를 인자로 받는 보조 함수로 나눕니다.

## 실행 검증

```bash
python -m pytest
python scripts/verify_notebooks.py --suite foundation
python scripts/verify_notebooks.py --suite architecture
python scripts/verify_notebooks.py --suite bridge
python scripts/verify_notebooks.py --suite composition
python scripts/verify_notebooks.py --suite generative
```

검증기는 각 노트북을 별도 커널에서 실행하고, 저장된 그림을 디코딩해 확인합니다.
실행 기록은 `results/<suite>-validation/`에 남으며, 이 폴더는 저장소에 올리지 않습니다.

## 라이선스

코드·노트북·문서 모두 [MIT 라이선스](LICENSE)입니다. 출처를 남기면 수업 자료나 다른 저장소에 자유롭게 복사·수정해 쓸 수 있습니다.
