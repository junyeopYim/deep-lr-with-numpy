# 전체 학습 경로: 개념을 수식과 코드로

학습 목적은 **문제 → 가정·수식 → 배열 연산 → 미분·갱신 → 실험 → 직접 변형**의 연결입니다.
행렬·미분·확률을 실제 계산과 함께 다시 확인하고, 작은 완성 예제를 재구성한 뒤 조건을 바꿉니다.

**현재 실행 자료는 00–15에 00b·01b를 더한 총 18개 노트북입니다.** 16–63은 다음 제작을 위한 학습 설계입니다.
2026-09-16부터 사용자가 요청한 모델을 더합니다: LeNet은 **05b**, 물체 검출(YOLO)은 **18b**, EDM은 **35b**로 기존 번호를 유지한 채 b 번호로 끼우고(00b·01b와 같은 규칙), VAE(23–25)·GAN(28)·NCSN(32–33)·DDPM(34–35)은 아래 표의 자리 그대로 제작합니다. 제작 중인 단위는 표에 **제작 중**으로 표시합니다.
전체 64개 단위는 주제를 찾는 지도이며, 선수 개념을 확인하면서 관심 경로를 선택합니다.
한 단위를 마치면 **설명 / 수식 구성 / 직접 구현 / 변형 검산**을 각각 기록합니다.

## 지금 연결할 순서

00–10을 읽고 계셨다면 **11 → 12 → 13 → 14 → 15**를 이어 보시면 됩니다.
그 뒤 생성 모델에 관심이 크면 22 Autoencoder로, 구조의 조합을 더 보고 싶으면 16 정규화 층으로 이어갑니다.
기초부터 시작할 때는 00–04 다음에 11–15를 읽고 기본 아키텍처로 들어가도 좋습니다.

아래 관심 경로는 **공통 기초 00–04를 바탕으로** 첫 완성 예제에 도달하는 순서를 적었습니다.
이후 확장은 각 단원 표의 선수 개념을 확인하면서 붙입니다.

| 관심 | 첫 완성 예제까지의 경로 | 이후 넓히는 방향 |
|---|---|---|
| 수식을 직접 구현하기 | 00–04 → 11–15 → 05–10 | 63 논문 한 편을 계산으로 재구성 |
| 이미지 구조 | 05 → 05b → 08–09 → 13 → 16–17 | 18–19 U-Net·ViT → 42–43 표현 학습 |
| 물체 검출 | 05 → 05b → 14–15 → 18b | 18 U-Net → 37 평가 |
| VAE | 11–15 → 22–26 | 27 샘플 추정 → 30 flow |
| GAN | 13–15 → 28–29 | 27 샘플 추정·37 생성 모델 평가 |
| score·diffusion | 11–15 → 23 → 32–33 | 34–35 diffusion → 35b EDM → 36 flow matching, 27·31 샘플 추정·에너지 |
| GPT | 08–09 → 13–15 → 44–46 | 47 encoder 목적, 48–50 적응·선호·다중모달 |
| 자기지도 표현 | 05·08·10 → 11–15 → 38–39 | 19–20 구조 조합·22 AE → 40–43 타깃·기울기 경로 |
| 강화학습 | 11–15 → 27 → 51–58 | 59 연속 행동 → 60–62 계획·오프라인·선호 |

준비도는 표의 과제를 작은 입력으로 수행해 확인합니다. 선수 단원의 이름은 기억나지만 수식이 흐릿하다면
해당 노트북의 손계산·shape 표·검산 셀을 먼저 다시 실행합니다.

## 1. 공통 기초

| 번호 | 학습 단위 | 먼저 연결할 개념 | 준비도 확인 | 자료 |
|---|---|---|---|---|
| 00 | 수식을 NumPy로 옮기기 | 숫자·변수·반복문 | 내적을 배치·다중 출력으로 확장하고 shape와 축을 정하기 | [노트북](../notebooks/00_기초/00_math_to_numpy.ipynb) |
| 00b | 벡터화: 반복문을 배열 연산으로 | 00의 shape·축·브로드캐스팅 | 원소별·축 합·브로드캐스팅·인덱싱으로 반복문을 옮기고 결과 모양을 먼저 적기 | [노트북](../notebooks/00_기초/00b_vectorization.ipynb) |
| 01 | 미분으로 학습시키기 | 00의 선형 계산 | MSE의 미분을 유도·검산하고 작은 회귀 문제 학습시키기 | [노트북](../notebooks/00_기초/01_gradients_and_learning.ipynb) |
| 01b | 미니배치: 왜 데이터를 나누어 학습하나 | 01의 학습 루프, 00b의 인덱싱 | 배치 기울기의 기댓값·잡음·비용을 확인하고 에폭·스텝·셔플로 미니배치 루프 쓰기 | [노트북](../notebooks/00_기초/01b_minibatch.ipynb) |
| 02 | 확률에서 손실 만들기 | 합·평균·함수·미분 | 확률·밀도·기댓값·우도를 계산하고 MSE·BCE·softmax CE 유도하기 | [노트북](../notebooks/00_기초/02_probability_and_losses.ipynb) |
| 03 | 퍼셉트론에서 MLP까지 | 00–02 | XOR에서 비선형성이 필요한 이유를 보이고 다층 역전파 구현하기 | [노트북](../notebooks/00_기초/03_perceptron_to_mlp.ipynb) |
| 04 | optimizer의 계산 | 01·02의 기울기·기댓값, 03의 활성화 | 미니배치·SGD·Momentum·RMSProp·Adam의 상태와 한 스텝을 손으로 계산하기 | [노트북](../notebooks/00_기초/04_optimizers.ipynb) |

00b와 01b는 NumPy 활용과 학습 실무의 기초라서 번호를 새로 매기지 않고 00·01 바로 뒤에 두었습니다. 읽는 순서는 00 → 00b → 01 → 01b → 02 → 03 → 04입니다.
02에서는 확률값과 밀도값, 조건부 확률과 우도, 합과 Monte Carlo 평균을 구분합니다.
03에서는 활성화·캐시·가중치 공유를, 04에서는 학습률·초기화·기울기 크기를 실제 실험에 연결합니다.
정규화와 데이터 분할은 일반화 실험이 등장하는 자리에서 함께 다룹니다.

## 2. 기본 아키텍처

각 단위는 구조 자체의 이유를 보여 주는 작은 지도학습 문제를 가집니다.

| 번호 | 학습 단위 | 먼저 연결할 개념 | 수학→코드의 핵심 | 완성 예제 | 자료 |
|---|---|---|---|---|---|
| 05 | CNN | MLP, 다중분류 | 지역 연결·가중치 공유 → 패치·합성곱·풀링과 backward | 작은 숫자 도안 분류 | [노트북](../notebooks/01_아키텍처/05_cnn.ipynb) |
| 05b | LeNet: 실제 MNIST에 CNN 적용 | 05·01b·04·15 | padding·5×5 conv 두 단·FC 120·84 → 약 6만 파라미터, 미니배치·에폭·test 보고 | 실제 손글씨 6만 장 분류, 커널·특성 맵 관찰, 이동 불변성 검산 | 제작 중 |
| 06 | RNN | 공유 파라미터, 역전파 | 시간별 상태 → 시간축 캐시와 BPTT | 같은 기호의 등장 순서 분류 | [노트북](../notebooks/01_아키텍처/06_rnn.ipynb) |
| 07 | LSTM·GRU | RNN과 시간별 기울기 | 게이트·상태의 덧셈 → 각 경로의 미분 | 지연된 신호 기억 | [노트북](../notebooks/01_아키텍처/07_lstm_gru.ipynb) |
| 08 | Attention | 행렬곱, softmax | 점수 → 정규화된 가중치 → 값의 가중합 | key에 붙은 값 검색 | [노트북](../notebooks/01_아키텍처/08_attention.ipynb) |
| 09 | Transformer | attention, MLP | 위치·mask·정규화·잔차·MHA·FFN의 조합 | 작은 토큰열 역순 변환 | [노트북](../notebooks/01_아키텍처/09_transformer.ipynb) |
| 10 | GNN | 공유 가중치, 합·평균 | 이웃 집계 → message passing·readout·역전파 | 그래프 분류와 순열 검산 | [노트북](../notebooks/01_아키텍처/10_gnn.ipynb) |

Attention과 Transformer를 별도 단위로 두어, 계산 부품과 그것들을 조합한 구조를 연결합니다.
GNN에서는 노드 순서를 바꿨을 때 대응하는 출력이 어떻게 바뀌는지도 검산합니다.
CNN의 공유, RNN의 공유, GNN의 공유를 비교하면서 같은 미분 규칙이 어디에 다시 쓰이는지 확인합니다.

## 3. 구조와 목적을 잇는 기초

이 단원들은 생성 모델·표현 학습·RL에서 반복되는 계산을 먼저 완성합니다.

| 번호 | 학습 단위 | 선수 개념 | 수식→구현→검산 | 자료 |
|---|---|---|---|---|
| 11 | 투영·공분산·SVD·PCA | 00–01 | 직교 투영, 중심화, 고유값, 낮은 차원 복원; 버린 고유값 합과 복원 오차 대조 | [노트북](../notebooks/02_연결/11_linear_algebra.ipynb) |
| 12 | 다변량 Gaussian·정보이론·추정 | 02·11 | Cholesky 샘플, log density, entropy·CE·KL, MLE·MAP, Gaussian KL·재매개화 미분 | [노트북](../notebooks/02_연결/12_gaussian_information.ipynb) |
| 13 | 계산 그래프와 자동미분 | 01·03 | VJP·broadcast 역방향·공유 누적, 작은 엔진, 직접 backward·PyTorch·중심차분·SGD 비교 | [노트북](../notebooks/02_연결/13_autodiff.ipynb) |
| 14 | 손실 선택·정규화·AdamW·dropout | 01–04 | MSE·MAE·Huber, 관측 가중치, L2 전체 미분, ridge 해, 감쇠 두 스텝, dropout 기대 손실 | [노트북](../notebooks/02_연결/14_losses_regularization.ipynb) |
| 15 | 분할·학습 진단·평가·반복 실험 | 02·04·14 | 그룹 분할·전처리, 작은 데이터 학습, 설정 선택, NLL·F1·AUC·Brier·calibration, seed 변동 | [노트북](../notebooks/02_연결/15_experiments_evaluation.ipynb) |

11의 PCA는 22의 선형 AE 기준이 됩니다. 12의 KL·Gaussian 샘플은 24–25에서 변분 추론으로,
13의 stop-gradient는 40의 교사 표현과 55의 TD 타깃으로 이어집니다.
14·15에서는 학습 목적의 숫자와 실제 평가 지표를 함께 기록하는 습관을 만듭니다.

## 4. 기본 부품을 조합하는 아키텍처

16–21은 작은 블록의 전체 backward와 구조의 성질을 검산하는 후속 제작 단위입니다.
09의 LayerNorm·잔차, 04의 초기화·clipping을 출발점으로 사용합니다.

| 번호 | 학습 단위 | 선수 개념 | 직접 구현할 계산 | 완성 예제·변형 검산 |
|---|---|---|---|---|
| 16 | BatchNorm·LayerNorm·RMSNorm | 03·09·13 | 통계 축, affine 파라미터, running 통계, train/eval, 각 VJP | 같은 입력의 배치 조합을 바꾸고 정규화 출력·기울기 비교 |
| 17 | 잔차 블록과 ResNet | 05·09·16 | identity·projection shortcut, 해상도·채널 변화, 두 경로 누적 | 깊이를 바꾸고 activation·gradient norm·분류 학습 비교 |
| 18 | encoder–decoder와 U-Net | 05·17 | down/up sampling, concatenate skip, pixel CE·Dice | 작은 도형 segmentation, 홀수 해상도·foreground 비율 변경 |
| 18b | 물체 검출: 격자 예측과 YOLO 손실 (제작 중) | 05b·14·15 | 상자·IoU, S×S 격자 출력, 책임 셀 배정, 좌표·신뢰도·클래스 가중 SSE와 mask 고정 미분, NMS | 캔버스 위 숫자 1–3개 검출, λ·격자 크기 변경, IoU≥0.5 precision·recall·AP |
| 19 | patch embedding과 ViT | 05·09·16 | 이미지→patch→token, 위치·분류 token·pooling | 같은 도안에서 patch 크기·위치 정보 변경, parameter·token 수 계산 |
| 20 | sequence encoder–decoder | 06–09 | cross-attention, encoder/decoder mask, teacher forcing, 위치 방식 | 가변 길이 기호 변환, padding·미래 정보의 영향 검산 |
| 21 | 집합·graph attention·구조적 대칭 | 08·10 | Deep Sets의 합, edge mask의 attention, GAT 집계 | 노드/집합 순열에 대한 등변·불변 성질, 이웃 수 변경 |

16의 정규화는 배열 통계를, 14의 regularization은 목적·학습 방식에 주는 제약을 다룹니다.
18의 복원 경로와 skip은 34의 이미지 잡음 예측기로 다시 사용하고,
20의 cross-attention은 조건부 생성과 다중모달 구조로 연결합니다.

<a id="3-학습-목적과-생성"></a>

## 5. 확률 모델·학습 목적·생성

아키텍처, 목적, 파라미터 갱신, 샘플 생성 절차를 매번 따로 적습니다.
22–37은 한 모델의 이름 안에 묶여 있던 중간 계산을 실제 학습 단위로 펼친 설계입니다.

| 번호 | 학습 단위 | 선수 개념 | 직접 구현할 계산 | 완성 예제·변형 검산 |
|---|---|---|---|---|
| 22 | Autoencoder와 압축 표현 | 03·11·13–15 | encoder/decoder, 복원 MSE, 병목의 전체 backward | PCA와 선형 AE의 부분공간·복원 비교, 비선형·denoising AE 변형 |
| 23 | 잠재변수·주변화·혼합 모형 (제작 중) | 02·12 | 결합 p(x,z), 합으로 p(x), Bayes posterior, 작은 EM | 두 Gaussian 혼합에서 책임 확률과 로그우도, 잠재 label 교환 |
| 24 | Jensen 부등식과 ELBO 유도 (제작 중) | 12·23 | q로 기대값 만들기, log p(x)=ELBO+posterior KL | 잠재 상태를 열거해 증거·하한·차이를 각각 수치 확인 |
| 25 | 재매개화 VAE의 전체 학습 (제작 중) | 13·22·24 | Gaussian encoder, 관측 likelihood, reconstruction+KL, 샘플 VJP | 2차원 잠재공간에서 복원·prior 샘플, 두 항·전체 미분 확인 |
| 26 | VAE가 배운 분포 읽기 | 15·25 | posterior/prior, beta 가중치, KL warmup, 조건부 입력 | 잠재 사용량·복원·생성, posterior collapse와 과한 압축 관찰 |
| 27 | Monte Carlo·importance sampling·샘플 기울기 | 02·12–13 | 기대값 추정, 분산·표준오차, importance weight, pathwise/score-function | 알려진 적분·미분과 추정치 비교, 샘플 수·제안 분포 변경 |
| 28 | GAN의 두 목적과 교대 갱신 (제작 중) | 03–04·13–15 | D의 분류, G의 minimax/non-saturating 목적, gradient 경로 고정 | 1·2차원 혼합 분포 생성, D/G를 따로 갱신하고 mode별 샘플 수 기록 |
| 29 | GAN 목적과 안정화 비교 | 28 | 분포 거리·critic, Wasserstein 목적, gradient penalty·spectral 제약 | 작은 고정 실험에서 critic·G gradient·coverage·seed 비교 |
| 30 | 변수 변환과 normalizing flow | 11–13·27 | determinant·log Jacobian, 가역 affine coupling, 정확한 density | forward/inverse 왕복, density 적분, 2차원 flow의 likelihood·샘플 |
| 31 | 에너지 기반 모델과 MCMC | 12·27 | exp(-E)/Z, log-partition 미분, positive/negative phase, Langevin | 1차원 격자 적분과 샘플 추정 비교, chain·step size 변화 |
| 32 | score와 denoising score matching (제작 중) | 12–13·27 | score=입력의 log density 기울기, Gaussian 잡음의 조건부 score | 해석 가능한 혼합 분포의 score와 학습 벡터장·샘플 비교 |
| 33 | 잡음 수준을 잇는 score 모델 (제작 중) | 31–32 | noise-conditioned score, annealed Langevin, SDE/ODE의 작은 시간 스텝 | 잡음 크기·적분 간격·초깃값을 바꾸고 분포 이동 확인 |
| 34 | Diffusion의 전방 과정과 학습 타깃 (제작 중) | 12·24·32 | beta/alpha 누적곱, q(x_t\|x_0), posterior 계수, epsilon/x0/v 타깃 | 닫힌식과 반복 잡음의 평균·분산 대조, 작은 denoiser 학습 |
| 35 | Diffusion 역과정·DDIM·조건부 생성 (제작 중) | 20·33–34 | reverse mean/variance, timestep indexing, DDIM, guidance | 같은 모델의 스텝 수·sampler·guidance 변경, 생성 경로·비용 비교 |
| 35b | EDM: σ-공간에서 통일한 diffusion 설계 (제작 중) | 32–35 | VE/VP를 σ(t)·s(t)로 통일, Tweedie의 score–denoiser 관계, c_skip·c_out·c_in·c_noise 유도, λ(σ)·σ 분포, ρ 시간 격자, Heun 2차 샘플러 | 혼합 분포의 정확한 denoiser로 solver 오차와 모델 오차 분리, 같은 학습 모델을 DDPM·DDIM·Heun으로 샘플링 비교 |
| 36 | Flow matching과 ODE 생성 | 27·30·33 | 조건부 경로, 목표 velocity, 회귀 목적, Euler/Heun 적분 | 2차원 분포 수송, 벡터장 오차와 solver 오차를 나누어 확인 |
| 37 | 생성 모델 평가와 비교 실험 | 15·22–36 중 관심 모델 | likelihood·복원·품질·coverage, MMD·특징 통계·샘플 비용 | 알려진 mode 분포와 복제 샘플을 사용해 지표가 읽는 성질 대조 |

### VAE를 공부할 때의 다섯 연결

1. **분포 배열화:** 12에서 평균·로그 분산·샘플·로그 밀도를 계산합니다.
2. **잠재변수 모형:** 23에서 결합·주변·사후분포를 작은 열거로 연결합니다.
3. **학습 가능한 목적:** 24에서 하한과 posterior KL의 관계를 식과 숫자로 확인합니다.
4. **샘플을 통과하는 미분:** 25에서 고정 잡음의 미분과 기대값의 샘플 추정을 연결합니다.
5. **학습한 모형의 사용:** 26에서 입력 복원·prior 생성·잠재 사용을 각각 관찰합니다.

같은 방식으로 GAN은 28–29, score·diffusion은 31–35와 35b, flow는 30·36으로 나누어 공부합니다.
생성 단원의 공통 예시는 **두 봉우리 혼합**(1·2차원 Gaussian 혼합, 23에서 정의)과 고정 MNIST 예시입니다. 혼합 분포는 밀도·책임확률·score·최적 denoiser가 모두 닫힌식이라 학습 결과의 정답지가 됩니다. MNIST는 VAE(25)에서 784 화소 Bernoulli로, GAN·score·diffusion(28·33–35b)에서는 11의 PCA 좌표 32개에서 학습한 뒤 고정 기저로 복원합니다.
37은 관심 모델 하나의 첫 구현을 마친 뒤 바로 붙여도 좋습니다.

## 6. 표현 학습과 자기지도 목적

같은 encoder에 어떤 입력 쌍·타깃·손실을 주는지 비교합니다.
표현의 평가는 15의 분할을 따르는 선형 probe·최근접 이웃·downstream 과제로 연결합니다.

| 번호 | 학습 단위 | 선수 개념 | 직접 구현할 계산 | 완성 예제·변형 검산 |
|---|---|---|---|---|
| 38 | 데이터 변형과 불변성 | 05·10·15 | augmentation 분포, 양성 쌍, 보존하려는 정보 | 회전·색·순서 변형이 라벨과 표현에 주는 영향, 변형 강도 비교 |
| 39 | 대조학습과 InfoNCE | 08·12·38 | 정규화 embedding, 유사도 행렬, temperature, 양성 인덱스·CE | 쌍 매칭, batch 크기·음성 구성, gradient·linear probe |
| 40 | 교사·학생·stop-gradient | 13·38–39 | online/target encoder, predictor, EMA 갱신, 기울기 경로 | 작은 표현 예측, 교사 갱신 속도·stop-gradient 위치와 붕괴 관찰 |
| 41 | 분산·공분산으로 표현 제약하기 | 11–12·40 | alignment, 분산 하한, 비대각 공분산 벌점 | 상수 표현·복제 차원에서 목적·기울기를 검산하고 표현 통계 추적 |
| 42 | masked reconstruction | 18–19·22·38 | mask 샘플, visible token encoder, decoder, masked-only loss | 작은 이미지 복원, mask 비율·손실 평균 분모·표현 평가 |
| 43 | JEPA와 표현 타깃 예측 | 19–20·40–42 | context/target 영역, 교사 표현, predictor·position, latent loss | pixel 타깃과 표현 타깃을 같은 데이터에서 비교, 교사·학생 갱신 검산 |

39–43에서는 목적값, 표현의 평균·분산, 샘플 구분, 분할된 평가 성능을 함께 관찰합니다.
복원 손실과 downstream 성능이 각각 무엇을 측정하는지 설명하는 것이 준비도 기준입니다.

## 7. 언어모델·GPT·적응 학습

Transformer라는 구조에 데이터의 확률 분해와 토큰 타깃을 붙입니다.
44–46이 작은 GPT의 학습부터 생성까지 이어지는 첫 경로입니다.

| 번호 | 학습 단위 | 선수 개념 | 직접 구현할 계산 | 완성 예제·변형 검산 |
|---|---|---|---|---|
| 44 | 토큰화·언어 데이터·자기회귀 목적 | 02·09·15 | 문자/byte/BPE, vocab·embedding, 입력/타깃 한 칸 이동, 문서 경계·padding | 문장별 조건부 확률의 곱과 token CE 합 대조, 토큰 수·perplexity 규약 |
| 45 | 작은 decoder GPT 전체 학습 | 09·13–15·44 | causal block, tied/untied head, token NLL, 전체 backward·학습 loop | 작은 문자 언어모델, causal·padding mask, 학습 문장 복원·held-out NLL |
| 46 | 생성 규칙과 KV cache | 45 | temperature·top-k·top-p, categorical sampling, prefill/한 토큰 cache | 같은 prefix의 logits를 전체 재계산과 대조, 길이·seed·속도·메모리 |
| 47 | encoder 목적과 BERT식 MLM | 20·42·44 | bidirectional mask, 선택 위치 CE, corruption 방식 | 같은 토큰열의 MLM·next-token 정보 접근 범위와 예측 비교 |
| 48 | SFT·전이 학습·LoRA | 13–15·45 | trainable 파라미터, response-only loss, 저랭크 delta-W, scaling | 같은 작은 과제의 full/frozen/LoRA 학습, loss mask·gradient·파라미터 수 |
| 49 | 선호 쌍·Bradley–Terry·DPO | 12·44–48 | sequence log-prob 합, 기준 정책, 선호 확률·log-ratio 목적 | 같은 prompt의 두 답변, 부호·padding·reference 고정과 길이 영향 |
| 50 | 다중모달 표현과 조건부 모델 | 19–20·39·45 | image/text encoder 정렬, projection, cross-attention·조건 token | 도형-설명 검색과 조건부 토큰 생성, modality 교환·목적 비교 |

GPT의 next-token 학습, SFT의 타깃 선택, DPO의 선호 목적을 각각 써 보고 같은 decoder에 적용합니다.
학습 데이터의 단위·평균 축·문서 경계·생성 루프까지 포함해 한 모델을 설명합니다.

## 8. 강화학습: 기대값에서 정책 갱신까지

작은 표 기반 환경에서 값을 손으로 계산한 다음 신경망으로 확장합니다.
환경의 randomness, 행동 샘플링, 데이터 수집 정책과 갱신 정책을 구분해 기록합니다.

| 번호 | 학습 단위 | 선수 개념 | 직접 구현할 계산 | 완성 예제·변형 검산 |
|---|---|---|---|---|
| 51 | Bandit·기대 보상·탐색 | 02·27 | 행동 가치, sample average, epsilon-greedy, regret | 2–5개 행동의 확률 보상, 탐색 계수·seed별 return 비교 |
| 52 | MDP·return·Bellman·동적 계획 | 11·51 | 전이 행렬, discounted return, 정책 가치 방정식, value/policy iteration | 작은 격자의 정확한 V·Q와 rollout 평균 비교 |
| 53 | Monte Carlo·TD·다단계 타깃 | 27·52 | return 추정, TD error, bootstrapping, n-step·lambda return | 같은 정책의 가치 추정, 편향·분산·종료 상태 처리 |
| 54 | SARSA·Q-learning과 데이터 정책 | 52–53 | on/off-policy 타깃, epsilon 행동, importance ratio의 역할 | 표 기반 제어, 수집 정책을 바꾸며 Q·행동 변화 확인 |
| 55 | DQN·replay·target network | 03·13·15·54 | Q 신경망, TD loss, replay sampling, 고정 target·Double DQN | 작은 환경 학습, terminal mask·target gradient·업데이트 주기 검산 |
| 56 | REINFORCE와 정책 기울기 유도 | 12·27·51–53 | trajectory 확률, log-derivative trick, reward-to-go·baseline | 열거 가능한 정책의 정확한 기대 보상 미분과 샘플 추정 대조 |
| 57 | Actor–Critic·advantage·GAE | 13·53·56 | actor/critic 목적, TD residual, GAE 재귀, 고정 타깃 | rollout의 경계·bootstrap·advantage 계산, 두 모델 기울기 분리 |
| 58 | PPO의 ratio·clip·업데이트 | 15·57 | old log-prob, 확률비, clipped surrogate, value·entropy 항 | 작은 한 배치의 손계산, 여러 epoch 재사용과 KL·return 추적 |
| 59 | 연속 행동·DDPG/TD3·SAC | 12·25·55·57 | Gaussian/tanh 정책, log-Jacobian 보정, Q와 actor, entropy 목적 | 작은 연속 제어, action 범위·재매개화·두 Q·entropy 계수 검산 |
| 60 | 모델 기반 RL과 계획 | 27·52·59 | 전이/보상 모델, rollout, shooting·MPC, 모델 오차 | 정확한 작은 환경과 학습 모델의 계획을 비교, horizon 변경 |
| 61 | 오프라인 RL과 정책 평가 | 15·27·54·59 | 데이터 행동 분포, importance 평가, Q 외삽·보수적 목적 | 고정 데이터의 행동 범위·coverage·평가 분산 비교 |
| 62 | 선호 보상과 RLHF 연결 | 45·49·56–58 | 선호 reward 모델, KL 기준 정책, sequence 보상·PPO | 작은 토큰 환경에서 SFT·DPO·보상 기반 정책 갱신 비교 |

52의 정확한 값, 56의 정확한 기대 보상 미분을 기준으로 근사 알고리즘을 검산합니다.
58에서는 policy loss만이 아니라 데이터 수집·종료 처리·value target·확률비의 기준까지 재현합니다.

## 9. 한 편의 논문을 스스로 계산으로 재구성하기

| 번호 | 학습 단위 | 선수 개념 | 실제 산출물 | 검산 기준 |
|---|---|---|---|---|
| 63 | 논문→계산 명세→작은 재현 | 13–15와 관심 경로 | 문제·가정·기호표, 연산별 shape, 목적·평균 축, 갱신/사용 loop, 완성 노트북 | 작은 정확해·수치미분·대조 구현, 한 요소씩 바꾼 실험, 실행 설정·한계 해석 |

63에서는 VAE·GAN·GPT·RL 중 하나를 골라 다음 한 장을 먼저 채웁니다.

| 질문 | 직접 적을 내용 |
|---|---|
| 무엇을 관측하고 예측하나요? | 데이터 단위, 입력·출력 shape, 랜덤 변수와 고정 값 |
| 어떤 구조를 사용하나요? | 연산 순서, 공유 파라미터, mask·cache |
| 목적은 어떤 가정에서 나오나요? | likelihood·거리·보상, 기대값의 분포, 합·평균 축 |
| 무엇을 미분하고 갱신하나요? | 파라미터/입력, stop-gradient, optimizer 상태, 타깃 갱신 시점 |
| 학습 후 어떻게 사용하나요? | 예측·복원·샘플링·행동 선택의 루프 |
| 무엇으로 이해를 검산하나요? | 손계산, 중심차분, 독립 구현, 조건 변경, 분리된 평가 |

## 10. 관심이 생겼을 때 붙이는 확장 가지

다음 가지는 연결할 입구와 작은 실습을 갖춘 심화 후보입니다. 관심 경로의 첫 완성 예제 뒤에 붙입니다.

| 가지 | 연결할 단원 | 추가할 수학·구현 | 작은 확인 문제 |
|---|---|---|---|
| 수치 계산과 큰 모델 학습 | 04·11·13·15 | conditioning, Hessian-vector product, float32/16, gradient accumulation, AMP·분산 평균 | 같은 유효 배치의 gradient 일치, precision·메모리·시간 변화 |
| 기하학·등변 구조 | 10·11·21 | 회전·이동 군 작용, 거리/방향 message, E(n) 등변성 | 좌표 회전 전후 출력·벡터 gradient가 같은 규칙을 따르는지 확인 |
| 연속 상태와 선택적 계산 | 06–09·17 | state-space model·scan, S4/Mamba, MoE routing·load balance | 순차/병렬 scan 대조, 라우팅 선택·전문가별 토큰 수·계산량 |
| 이산 잠재표현과 생성 | 23–27·34·44 | VQ·straight-through·Gumbel-Softmax, 이산 diffusion | 열거 가능한 categorical 기댓값과 근사 미분의 차이 |
| 불확실성·일반화 | 12·15·27 | 편향–분산 분해, bootstrap·교차검증, Bayesian posterior, calibration·conformal | 새 데이터 반복과 seed 반복을 구분해 coverage·오차 추정 |
| 다른 학습 기준선 | 01·11–12·15·23 | k-NN·tree, kernel·SVM·Gaussian process, clustering | 같은 작은 데이터의 가정·표현·목적·비용을 신경망 기준선과 비교 |
| 검색과 시스템 결합 | 39·44–50 | retrieval embedding, 검색·생성 분리 평가, tool 입력/출력 | 검색 오차와 생성 오차를 각각 측정하는 작은 질의 집합 |

## 경로를 넓힌 근거

이 순서는 사용자의 수학 복습·직접 구현 목적에 맞춘 프로젝트 설계입니다.
주제 범위와 연결을 검토할 때 다음 원자료를 참조했습니다.

- [Stanford CS231n 일정](https://cs231n.stanford.edu/schedule.html): 최적화·구조 조합·자기지도·생성의 별도 학습 범위.
- [Stanford CS236 강의 범위](https://deepgenerativemodels.github.io/syllabus.html): 잠재변수·변분 추론·flow·에너지·score·생성 평가의 연결.
- [PyTorch 자동미분 안내](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html): 계산 그래프·기울기 누적·VJP의 API 대응.
- [Spinning Up의 RL 알고리즘 분류](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)와 [정책 기울기 유도](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html): 가치·정책·모델 기반 접근과 기대 보상 미분.
- [VAE](https://arxiv.org/abs/1312.6114), [GAN](https://arxiv.org/abs/1406.2661), [AdamW](https://arxiv.org/abs/1711.05101), [Flow Matching](https://arxiv.org/abs/2210.02747): 각 목적·갱신·생성 절차의 원논문.

학습자의 실제 질문이 모이면 절의 길이·순서·변형 난도를 조정합니다.
파일의 실행·수치 검산과 독립 실습 기록은 로컬 `results/bridge-validation/bridges.md`에 있습니다(저장소에는 올리지 않습니다).
