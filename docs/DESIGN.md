# 개념 그림 설계

노트북의 그림은 세 종류입니다. **검산 그림**은 계산이 맞았는지 보여 주고, **개념 그림**은
"왜 이런 수식을 쓰는가"와 "이 모델이 어떻게 학습하나"를 실제 MNIST 배열로 눈에 잡히게 하며,
**구조 도식**은 뉴런·층·커널·블록·손실 곡면처럼 "구조가 어떻게 생겼나"를 원·선·상자로 보여 줍니다.
이 문서는 개념 그림의 원칙, 부품, 그리고 노트북별로 어디에 어떤 그림을 두는지 적습니다.
다음 작업자는 이 문서를 읽고 같은 문법으로 그림을 이어서 만듭니다.

## 1. 원칙

- **그림 하나에 아이디어 하나.** 패널은 3–5개, 왼쪽에서 오른쪽으로 계산이 흐르는 배치를 씁니다.
- **숫자는 물건으로.** 784차원 벡터는 28×28 화소, 가중치 한 열도 화소, 배치는 얇은 행 묶음, 점수는 막대로 그립니다.
- **축·격자·눈금·colorbar를 끕니다.** 필요한 숫자는 그림 안에 직접 적습니다. 제목은 기호와 한글 한 줄입니다.
- **색의 의미를 고정합니다.** 양수 파랑, 음수 빨강, 강조 금색(한 번에 하나), 나머지 회색. 잉크는 `gray_r`(0 흰색, 1 검정).
- **같은 예시를 계속 씁니다.** `src/data/mnist.py`의 `load_examples()`가 숫자 0–9의 고정 이미지 10장을 돌려줍니다.
  00번에서 본 7이 05번의 합성곱, 11번의 PCA에서도 같은 7입니다.
- **흰 배경.** 노트북의 다른 그림, 인쇄, 문서 공유와 어울리게 유지합니다.
- **계산은 본문, 그리기는 보조 파일.** 템플릿·점수·평균·기울기 같은 수학은 노트북 셀에서 계산하고,
  `src/utils/<노트북>_plots.py`의 조합 함수는 받은 배열만 그립니다. 조합 함수는 `concept_plots`의 부품을 씁니다.
- **자리.** 🔑 수식·설명 바로 뒤에 둡니다. 순서는 🔑 설명 → 계산 셀 → 그림 셀 → 🔍 해석 → 구현 → 검산입니다. 그림 계산 셀은 뒤에 나올 구현 함수를 쓰지 않고 NumPy 연산만 씁니다. 구현 함수와의 일치 검산은 구현 뒤 검산 셀에 둡니다.
- **🔑 핵심 절마다 최소 하나.** [WRITING.md](WRITING.md)의 규칙입니다. 절의 "가져갈 것" 문장을 보여 주는 그림이어야 하며, 🔑 소절도 포함합니다.
- **학습 결과를 보여 주는 그림**(XOR 학습, optimizer 비교처럼 본문의 학습 루프가 있어야 나오는 것)은 그 학습 셀 바로 뒤에 둡니다. 이때는 본문의 구현 함수를 써도 됩니다.
- **잘못 그려지는 mathtext.** 그림 제목·라벨의 수식은 matplotlib mathtext라 `\tfrac`, `\frac1N`, `\ge`, `\text`를 지원하지 않습니다. `\frac{1}{2}`, `\geq`처럼 씁니다.
- **MNIST와 맞지 않는 개념은 억지로 엮지 않습니다.** 계산 그래프(13), 그래프 신경망(10), 시퀀스 구조(06–09의 일부)는 작은 도식이 더 맞습니다.

## 2. 부품 — `src/utils/concept_plots.py`

| 부품 | 하는 일 |
|---|---|
| `flow(widths, rows=1)` | 폭 비율로 가로 흐름 패널을 만든다. 축은 모두 꺼져 있다. |
| `new_figure(figsize)` | 전역 autolayout을 끈 흰 그림. 화살표 좌표가 어긋나지 않게 한다. |
| `draw_image(ax, img, kind="gray" \| "signed")` | 28×28 배열을 화소로. signed는 0 중심 빨강·파랑. |
| `draw_strip(ax, v, vertical=True)` | 1차원 벡터를 띠로. 펼침(flatten)을 보여 줄 때. |
| `draw_rows(ax, X)` | (N, D) 배치를 얇은 행 N개로. 행 하나가 샘플 하나. |
| `draw_grid(ax, M, highlight=[(i, j)])` | 작은 행렬을 칸과 숫자로. 금색 테두리로 칸을 강조. |
| `draw_bars(ax, values, highlight=k, marker=k)` | 가로 막대. 파랑은 선택, 금색 테두리는 정답. |
| `draw_formula(ax, tex, sub)` | 패널 사이에 수식 한 줄과 회색 설명. |
| `connect(fig, a, b, text)` | 두 패널 사이 화살표. |
| `label(ax, text)` | 패널 아래 회색 주석. |

새 부품은 "다른 노트북에서도 두 번 이상 쓰일 때"만 추가합니다. 한 노트북에서만 쓰는 조합은 그 노트북의 `*_plots.py`에 둡니다.

## 2-1. 구조 도식 부품 — `src/utils/schematic_plots.py`

| 부품 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `draw_network(ax, [D, H, K], node_labels, edge_labels, layer_titles)` | 완전 연결 층 도식. 큰 층은 앞 몇 개 + ⋮ + 마지막으로 줄인다. 작은 층에는 선마다 $W_{dk}$를 적을 수 있다. | 00 뉴런·층, 01 기울기가 흐르는 선, 03 MLP |
| `index_labels("x", 784)` | 큰 층에 붙일 $x_0, x_1, x_2, \ldots, x_{783}$ 라벨 | 위와 같음 |
| `draw_blocks(ax, [("conv 3×3, 8", "8×26×26"), …])` | 상자와 화살표의 블록 흐름. 아래에 텐서 모양을 적는다. | 05 CNN, 09 Transformer, 17–19 |
| `draw_cells(ax, M, window=(r, c, h, w), cell_colors, highlight_cells)` | 숫자 격자 위의 금색 창. 합성곱 한 장면, 풀링 구역 색칠. | 05 CNN, 08 attention 점수표 |
| `draw_surface_paths(ax3d, xx, yy, zz, paths)` / `draw_contour_paths(ax, …)` | 손실 곡면과 optimizer 경로. 시작 점, 끝 네모. | 01 경사하강, 04 optimizer 비교 |

구조 도식의 규칙은 개념 그림과 같습니다. 축·격자 없음, 회색 선, 강조는 금색 하나, 출력 노드만 옅은 파랑.
도식은 **본문 수식의 기호를 그대로** 적습니다. 도식에 나온 $w_0$가 코드의 `w[0]`이고, 선 하나가 $W$의 원소 하나입니다.
값이 있는 도식(뉴런의 0.2, −0.3)은 본문 손계산 값을 인자로 받아 적습니다.

## 3. 노트북별 그림 목록

상태: **완료** = 노트북에 들어가 실행 확인됨, **계획** = 자리와 내용만 정함.

### 00 · 수식을 NumPy로 옮기기 — `array_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 2 내적 | `plot_dot_as_overlap` | 왜 곱해서 더하는가: 내적은 이미지와 템플릿의 겹침 | 7의 템플릿 $w$ = (7의 평균 − 전체 평균), $x\odot w$, $z$ |
| 3 배치 | `plot_batch_rows` | 왜 샘플을 행으로 쌓는가: `X @ w` 한 번에 점수 N개 | 이미지 10장을 (10, 784)로, $z = Xw$ |
| 4 다중 출력 | `plot_templates` | 왜 W의 열이 출력인가: 열 10개 = 템플릿 10장, $Z = XW$는 점수표 | 클래스별 템플릿 (784, 10), $Z$, 행별 argmax |
| 5 축 합 | `plot_axis_sums` | axis=0과 axis=1이 남기는 것: 이미지 vs 샘플별 숫자 | `X.mean(axis=0)`, `X.sum(axis=1)` |
| 5 브로드캐스팅 | `plot_broadcast_ways` | 같은 (2,)를 눕히면 열에, 세우면 행에 더해짐 | `base + v`, `base + v[:, None]` |
| 6 reshape·transpose | `plot_reshape_vs_transpose` | 순서 유지 vs 축 교환 vs 잘못된 줄바꿈 | `reshape(784)`, `reshape(28, 28)`, `.T`, `reshape(14, 56)` |

구조 도식 (완료):

| 절 | 그림 | 보여 주는 것 |
|---|---|---|
| 1 숫자 두 개 | `plot_neuron_diagram` | 입력 2개가 $w_0, w_1$ 선을 타고 $\Sigma$로 모여 $z$가 되는 뉴런. 손계산 값을 선 위에 적음 |
| 4 다중 출력 | `plot_layer_diagram` | 왼쪽 $2\to2$에 선마다 $W_{dk}$, 오른쪽 $784\to10$은 ⋮으로 줄인 같은 구조 |

7절의 2입력 평면 그림과 손계산용 작은 행렬 그림은 그대로 둡니다.

### 00b · 벡터화 — `vectorization_plots.py` — 완료

상황: MNIST 5,000장의 잉크 양·중심화·숫자별 평균 이미지와의 거리·one-hot.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 왜 벡터화 🔑 | `plot_timing_bars` | 같은 계산의 시간이 수백 배 차이 | 반복문·`sum(axis=1)` 시간 |
| 2 원소별·축 합 🔑 | `plot_centering` | 이미지 − 평균 이미지 = 중심화 | `X.mean(axis=0)`, 브로드캐스팅 뺄셈 |
| 3 브로드캐스팅 🔑 | `plot_broadcast_shapes`, `plot_distance_grid` | 뒤 축부터 맞추는 규칙, 쌍마다 거리표 | `np.broadcast_shapes`, `(10,1,784)−(1,10,784)` |
| 4 인덱싱 🔑 | `plot_indexing` | 마스크·one-hot·쌍으로 고르기 | `y == 7`, `np.eye(10)[y]`, `P[rows, y]` |
| 5 실전 🔍 | `plot_timing_bars` | 전개식으로 큰 중간 배열 없이 벡터화 | 최근접 평균 분류기 정확도·시간 |

### 01b · 미니배치 — `minibatch_plots.py` — 완료

상황: MNIST 60,000장 전체의 "7이면 1" 회귀.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 비용 🔑 | `plot_step_cost` | 한 스텝 시간은 B에 비례, 1초에 가능한 갱신 수 | 배치 크기별 기울기 시간 |
| 2 추정치 🔑 | `plot_batch_gradient_images`, `plot_batch_noise` | 기댓값은 같고 잡음은 $1/\sqrt B$ | 전체·8장·64장 기울기 이미지, 배치 2,000개의 표준편차 |
| 3 에폭·셔플 🔑 | `plot_epoch_batches` | 섞어서 B개씩 끊기, 마지막 배치 | `epoch_batches(12, 4)` 두 에폭 |
| 4 루프 🔑 | `plot_loss_vs_seen` | 같은 표본 처리량에서 미니배치가 훨씬 더 내려감 | 3에폭 미니배치 vs 전체 배치 3스텝 |
| 5 배치 크기 🔑 | `plot_batch_size_effect` | 잡음·시간·에폭당 진행의 절충 | B=8·64·512·4096 한 에폭 |

### 01 · 미분으로 학습시키기 — `gradient_plots.py` — 완료

상황: 붉기 $x$로 당도 $y$를 맞히는 직선(과일 40개), MNIST "7이면 1" 점수 회귀(고정 예시 10장, 학습 5,000장).

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 손실 🔑 | `plot_loss_from_residuals` | 왜 잔차를 제곱해 평균내는가 | 시작 직선의 잔차 40개, $r^2$, $J$ |
| 2 미분 🔑 | `plot_tangent` | 미분은 접선 기울기, 근처에서만 맞음 | $w^3$의 접선과 폭 0.3 평균 변화율 |
| 4 역전파 🔑 | `plot_gradient_image` | 기울기도 이미지다: 7의 기여 + 나머지의 기여 = $dW$ | $w=0, b=0.5$에서 $dZ, dW, db$ |
| 6 갱신 🔑 | `plot_update_step` | $w-\eta\,dW=w_{\mathrm{new}}$가 점수와 손실을 어떻게 바꾸나 | $\eta=0.02$ 한 스텝의 점수·손실 |
| 7 반복 🔑 | `plot_training_images` | 0에서 시작해 템플릿이 생겨나는 과정 | 네 줄 학습 루프 100스텝, 0·1·5·20·100 스냅숏 |
| 8 지형 🔍 | `plot_descent_landscape` | 등고선과 수직으로 내려가는 경로 | 손실 격자, 50스텝 경로 |

3절 수치미분 오차, 5절 gradcheck, 7절 직선 스냅숏, 8절 학습률 곡선은 기존 검산 그림을 유지합니다.

### 02 · 확률에서 손실 만들기 — `probability_plots.py` — 완료

상황: 과일 상자(흠집 수, 붉음·닮), 당도 직선(01의 데이터), MNIST 템플릿 점수.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 기댓값 🔑 | `plot_pmf` | 확률 가중합이 기댓값 | 흠집 수 확률과 평균 |
| 2 결합·조건부 🔑 | `plot_joint_table` | 축을 더하면 주변, 행을 나누면 조건부 | 결합표, 행·열 합, 행 정규화 |
| 3 밀도 🔑 | `plot_density_area` | 높이가 아니라 넓이가 확률 | 균일 밀도, 구간 넓이 |
| 5 우도 🔑 | `plot_likelihood_curves` | 우도 최대 = NLL 최소 | $\theta$ 격자의 우도와 NLL |
| 6 가우스→MSE 🔑 | `plot_gaussian_observation` | 잔차 제곱은 종 모양의 지수에서 나옴 | 참 직선, 표준편차 0.15의 종 세 개 |
| 7 Bernoulli→BCE 🔑 | `plot_bce_examples` | 틀린 확신에 큰 손실 | 0.4배 템플릿 점수의 sigmoid, 이미지별 $-\log p_y$ |
| 8 softmax CE 🔑 | `plot_softmax_examples` | 점수 10개 → 확률 10개 → 정답 확률의 $-\log$ | 예시 7·3의 템플릿 점수, softmax, CE |

4절 Monte Carlo, 7절 BCE 곡선, 8절 gradcheck, 9절 분류기는 기존 그림을 유지합니다.

### 03 · 퍼셉트론에서 MLP까지 — `mlp_plots.py` — 완료

상황: 과일 판정(붉음·무거움 → 익음, AND/XOR), MNIST "7인가" 퍼셉트론과 MLP `[784, 32, 1]`.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 퍼셉트론 🔑 | `plot_perceptron_decision` | 점수 → 0 기준 판정, 틀린 표본 | 템플릿 점수와 판정 |
| 2 XOR 🔑 | `plot_separability` | 직선 하나로 나뉘는 AND, 안 나뉘는 XOR | AND의 결정 경계 |
| 3 활성화 🔑 | `plot_activation_curves` | 활성화와 역전파에서 곱하는 도함수 | tanh·relu·sigmoid 곡선 |
| 4 forward 🔑 | `plot_mlp_forward_flow` (+ `draw_network` 도식) | 784 → 32 → 1로 줄어드는 순전파 | 학습 전 난수 가중치의 $A_1, z, p$ |
| 5 backward 🔑 | `plot_backward_flow` (금색 경로 도식) | 오차가 선을 거슬러 퍼지고 $dW_1[:,j]=x\cdot dZ_1[j]$ | $dz, dA_1, dZ_1, dW_1$ |
| 7 학습 🔑 | 기존 XOR 손실·결정 영역·은닉 지도 + `plot_hidden_templates` (학습 셀 뒤) | 은닉 유닛이 저마다 다른 획을 봄 | `fit_mlp`로 MNIST 300스텝, $W_1-W_1^{(0)}$ |

### 04 · optimizer — `optimizer_plots.py` — 완료

상황: 이차 목적(곡률 1과 20), 과일 회귀(기존), MNIST 7 판별 회귀 `(5000, 785)`.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 학습률 🔑 | `plot_step_size_parabola` | $\eta<2/a$면 수렴, 넘으면 발산 | $a=20$에서 학습률 셋의 8스텝 |
| 2 미니배치 🔍 | (그림 없음, 01b로 이동) | 표본 기울기의 평균이라는 항등식 복습 | `regression_loss_gradient`, 가중 합 검산 |
| 3 SGD 🔑 | `plot_gd_zigzag` | 가파른 방향은 튀고 완만한 방향은 느림 | 이차 목적 40스텝 경로 |
| 4 Momentum 🔑 | `plot_state_bars` | 같은 방향은 쌓이고 번갈아면 상쇄 | $g_t, v_t$ 8스텝 |
| 5 RMSProp 🔑 | `plot_state_bars` | $\sqrt s$로 나누면 보폭이 같아짐 | $g_t, s_t, g_t/\sqrt{s_t}$ |
| 6 Adam 🔑 | `plot_bias_correction` | 0에서 시작한 이동 평균의 편향과 보정 | 상수 기울기의 $m_t, s_t$ 재귀 |
| 7 경로 🔍 | `plot_optimizer_landscape` | 네 optimizer의 경로(곡면·등고선·손실) | 기존 `run_quadratic` 결과 |
| 8 MNIST 🔍 | `plot_optimizer_templates` (학습 셀 뒤) | 가중치 이미지가 자라나는 속도, Adam의 가장자리 화소 | `train_regression` 1·10·100스텝 |

### 05 · CNN — `architecture_plots.py` — 계획

개념 그림: 3×3 커널이 7 위를 지나는 한 장면(입력 패치 금색 테두리, 커널, 출력 한 칸), 특성 맵, 풀링 전후.
도식: `draw_cells`로 5×5 입력·3×3 커널·3×3 출력의 숫자 격자와 금색 창, 2×2 max pooling의 구역 색칠과 stride 비교, `draw_blocks`로 입력→conv→ReLU→pool→펼침→선형→softmax 블록 흐름과 각 단계의 텐서 모양.

### 06–09 · RNN·LSTM·Attention·Transformer — 계획

06·07은 7을 28행의 시퀀스로 읽으며 은닉 상태를 시간축 띠로 그립니다.
08·09는 7을 4×4 패치 토큰 49개로 자르고, 한 패치가 어느 패치를 보는지 attention 가중치를 이미지 위에 겹칩니다.
도식: 06은 시간축으로 펼친 RNN 셀 도식(같은 $W$가 매 시각 반복), 09는 `draw_blocks`로 embedding→attention→FFN→LayerNorm 블록 흐름.

### 10 GNN · 13 자동미분 — 계획

MNIST를 쓰지 않습니다. 작은 그래프와 계산 그래프 도식을 `concept_plots`의 색·여백 규칙으로 그립니다.

### 11 · 선형대수 — `bridge_plots.py` — 계획

평균 이미지, 상위 고유 이미지 8장, 주성분 1·5·20·50개로 복원한 7과 복원 오차 막대.

### 12 · Gaussian·정보 — 계획

클래스별 평균 이미지 10장과 표준편차 이미지, 예측 분포 세 개(확신·애매·틀림)의 entropy 막대.

### 14 · 손실·정규화 — 계획

7에 dropout 마스크를 씌운 모습 3장, L2 세기별 템플릿 이미지 3장.

### 15 · 실험·평가 — 계획

혼동 행렬을 `draw_grid`로 축 없이, 틀린 예시 이미지 6장과 예측·정답 표시.

## 4. 작업 순서

1. 계산 셀을 본문에 추가합니다. 결과 배열의 shape를 주석으로 적습니다.
2. 조합 함수를 해당 `*_plots.py`에 추가하고 `concept_plots` 부품만 씁니다.
3. 새 커널에서 노트북을 위부터 끝까지 실행하고 그림을 눈으로 확인합니다. 화살표·제목 겹침·글리프 경고를 봅니다.
4. 이 문서의 표에서 상태를 계획 → 완료로 바꿉니다.
