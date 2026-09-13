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
- **자리.** 🔑 수식·설명 바로 뒤에 둡니다. 순서는 🔑 설명 → 계산 셀 → 그림 셀 → 🔍 해석 → 구현 → 검산입니다. 그림 계산 셀은 뒤에 나올 구현 함수를 쓰지 않고 NumPy 연산만 씁니다(앞 절에서 이미 구현한 함수는 써도 됩니다). 구현 함수와의 일치 검산은 구현 뒤 검산 셀에 둡니다.
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
| `draw_unrolled_chain(ax, times, highlight, backward)` | 시간축으로 펼친 순환 셀 도식. 같은 $W_x, W_h$ 라벨을 모든 화살표에 적고, `backward=True`면 직접 기여 $g_t$와 미래 기여 $r_t$를 금색으로 덧그린다. | 06 RNN (07의 LSTM 셀 도식은 `lstm_plots.draw_lstm_cell`) |

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

### 05 · CNN — `cnn_plots.py` — 완료

상황: 5×5 숫자 격자에 3×3 커널, MNIST 7, 본문에서 만드는 8×8 숫자 도안. 학습 곡선·도안 묶음은 `architecture_plots`의 `plot_curves`·`plot_images`를 그대로 씁니다.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 합성곱 🔑 | `plot_conv_scene` (`draw_cells` 창), `plot_conv_on_image` | 창 하나의 내적을 모든 자리에서 반복, 7의 가로·세로 경계 특성 맵 | 5×5 격자의 3×3 출력(반복문), 7의 26×26 특성 맵 두 장 |
| 2 패치 행렬 🔑 | `plot_patch_matrix` | 창 9개를 행으로 쌓으면 행렬곱 하나 | `sliding_window_view`로 (9, 9) $P$, $P@k$ |
| 3 backward 🔑 | `plot_overlap_count` | 겹친 자리로 돌아온 기여를 `+=`로 더함 | 2×2 창 네 자리의 0/1 기여와 그 합 |
| 4 풀링 🔑 | `plot_pooling` | 구역마다 값 하나, 최댓값 자리 | 4×4 → 2×2 최대·평균, 7 특성 맵의 13×13 |
| 5 완성 모델 🔑 | `plot_cnn_blocks` (`draw_blocks`) | 텐서 모양 흐름과 파라미터가 있는 층 | 파라미터 수 60 + 550 |

### 06 · RNN — `rnn_plots.py` — 완료

상황: A/B 순서 문장, MNIST 7을 28행 시퀀스로.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 한 스텝 🔑 | `plot_rnn_unrolled` (`draw_unrolled_chain`) | 같은 $W$를 매 시각 다시 씀, 7을 읽은 상태 띠 | 난수 $W_x, W_h$로 28×8 상태 |
| 2 BPTT 🔑 | `plot_bptt` | 직접 기여 $g_t$와 미래 기여 $r_t$를 더함 | 스칼라 $T=3$의 $\delta_t, r_t, dh_0, dW_h$ |
| 3 완성 실험 🔑 | `plot_order_examples`, `plot_state_paths` (학습 셀 뒤) | 평균은 같고 순서만 다른 문장, 학습된 상태 경로 | one-hot과 시각 평균 |
| 4 시간 미분 🔑 | `plot_gradient_decay` | $w_h^{T-t}$로 사라지거나 커짐 | 기존 temporal_gradients |

### 07 · LSTM·GRU — `lstm_plots.py` — 완료

상황: 손계산 셀($f=i=o=0.5$, $g=0.8$), MNIST 7 시퀀스, 지연 기억 데이터.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 LSTM 셀 🔑 | `plot_lstm_cell` (`draw_lstm_cell` 도식) | 덧셈 경로와 게이트가 곱하는 자리 | 손계산 $c=0.5$, $h$ |
| 2 backward 🔑 | `plot_lstm_backward` | $c$로 오는 두 기울기 경로 | $do, dc, df, dc_{prev}, di, dg$ |
| 3 덧셈 경로 🔑 | `plot_retention` | $\prod f=f^t$ | 기존 retention |
| 4 GRU 🔑 | `plot_gru_mix` | $z$로 옛 상태와 후보를 섞음 | $z\in\{0,\ldots,1\}$의 $h$ |
| 5 시간 펼치기 🔑 | `plot_gated_sequence` | 7 시퀀스의 $h_t, c_t, f_t$ 띠 | 학습 전 LSTM 28스텝 |
| 6 완성 실험 🔑 | `plot_memory_examples`, `plot_gate_traces` (학습 셀 뒤) | 첫 시각의 ±1 신호, 학습된 게이트 | 손으로 만든 예 두 개 |

### 08 · Attention — `attention_plots.py` — 완료

상황: 손계산 검색(항목 2개), MNIST 7의 4×4 패치 49개, 값 검색 문제.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 Q·K·V 🔑 | `plot_attention_lookup`, `plot_patch_attention` | 점수→softmax→가중합, 7의 한 패치가 어느 패치를 읽나 | 손계산 $[3/4, 1/4]$, 화소를 $Q=K=V$로 한 49개 가중치 |
| 2 $\sqrt{d_k}$ 🔑 | `plot_score_variance` | 분산 $d_k$와 softmax 몰림 | 기존 분산 + $d=128$의 softmax 8개 |
| 3 softmax backward 🔑 | `plot_softmax_backward` | Jacobian $p_i(\delta_{ij}-p_j)$와 원소별 오답 | 2×2 Jacobian, $ds$ |
| 4 mask 🔑 | `plot_masks` | causal AND padding, 허용 위치에서만 합 1 | 기존 combined, $P$ |
| 5 검색 학습 🔑 | `plot_retrieval_example`, `plot_attention_weights` (학습 셀 뒤) | query 정체와 같은 key의 값, 학습 전후 가중치 | 손으로 만든 예제 하나 |

### 09 · Transformer — `transformer_plots.py` — 완료

상황: 손계산 토큰(특성 4), 토큰 3·폭 4의 head 나누기, 역순 변환.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 LayerNorm 🔑 | `plot_layernorm` | 토큰마다 D축 정규화, 전체 +3의 차이가 사라짐 | $\mu$, $\sqrt{v+\epsilon}$, $\hat x$ |
| 2 MHA 🔑 | `plot_head_split` | 열을 head로 나눠 $P$를 따로 만들고 합침 | head 2개의 $P$와 merge |
| 3 블록 🔑 | `plot_transformer_block` (도식) | 두 잔차의 항등 경로 | — |
| 4 embedding·위치 🔑 | `plot_embedding_positions` | $E$의 행 고르기 + 위치 표, sinusoidal 표 | $E[\mathrm{tokens}]$, $P[:T]$, 12×12 sin 표 |
| 5 역순 변환 🔑 | `plot_reverse_examples`, `plot_trained_heads` (학습 셀 뒤) | 위치 $t$의 출력은 위치 $T-1-t$의 입력 | 예 세 개 |

### 10 · GNN — `gnn_plots.py` — 완료

MNIST를 쓰지 않습니다. `draw_graph`로 3노드 방향 그래프, 5노드 사슬, 사슬·별을 그립니다.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 인접 행렬 🔑 | `plot_aggregation` | 화살표 방향과 행·열, $AX$와 $SX$ | 3노드 예제 |
| 2 message passing 🔑 | `plot_layer_paths` | self·neighbor 경로를 더해 tanh | 노드 0의 $z, h$ 손계산 |
| 3 층 쌓기 🔑 | `plot_hops` | $(I+A)^k>0$의 범위 | 5노드 사슬 |
| 4 완성 학습 🔑 | `plot_two_graphs` | 사슬과 별의 degree 분포 | 기존 좌표 |
| 5 순열 🔑 | `plot_permutation` | 행과 열을 함께 바꿈 | perm [2, 0, 1] |
| 6 한계 🔍 | `plot_two_graphs` | 고리 하나와 두 고리 | 기존 |

### 11 · 선형대수 — `linalg_plots.py` — 완료

상황: 2차원 점(손계산·타원 구름), MNIST 5,000장.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 투영 🔑 | `plot_projection_2d`, `plot_projection_images` | 가장 가까운 점과 수직 잔차, 7을 템플릿 방향에 투영 | $z, \hat x, r$ |
| 1 최소제곱 🔑 | `plot_least_squares` | 잔차가 모든 열에 직교 | 정규방정식 해 |
| 2 공분산 🔑 | `plot_covariance_small`, `plot_covariance_images` | $X_c^\top X_c/N$, 평균·분산 이미지와 784×784 $C$ | 3×2 예제, 5,000장 |
| 3 고유벡터·SVD 🔑 | `plot_eigen_images`, `plot_eigen_directions` | 고유 이미지 8장과 스펙트럼, 2차원 고유 방향 | `eigh(C)`, 누적 비율 |
| 4 PCA 🔑 | `plot_pca_reconstruction`, `plot_pca_clouds` | $k=1,5,20,50$ 복원과 오차 | 닫힌 해 복원 |
| 5 단위 🔑 | `plot_pca_clouds` | 단위·표준화가 방향을 바꿈 | 기존 |

### 12 · Gaussian·정보 — `gaussian_plots.py` — 완료

상황: 2차원 Gaussian, 예측 분포 세 개(확신·애매·틀림), MNIST 클래스별 Gaussian.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 샘플 🔑 | `plot_sampling_flow` | $\epsilon\to\mu+L\epsilon$ | 600점 |
| 2 로그 밀도 🔑 | `plot_log_density` | 등고선과 세 항 | 격자 $d^2$, 점 A·B |
| 3 entropy·KL 🔑 | `plot_entropy_examples`, `plot_ce_kl_curves` | 확신·애매·틀림의 $H$와 CE, $H(p,q)=H(p)+KL$ | 세 분포 |
| 4 MLE 🔑 | `plot_class_gaussians`, `plot_two_covariances` | 클래스별 $\mu_c, \sigma_c$ 이미지 | 5,000장의 평균·표준편차 |
| 4 MAP 🔑 | `plot_map_shrink` | $\tau^2$에 따라 0 쪽으로 | 닫힌 해 |
| 5 Gaussian KL 🔑 | `plot_gaussian_kl` | $q$ 가중 $\log(q/p)$의 넓이 | 1차원 밀도, 닫힌 식과 적분 |
| 6 재매개화 🔑 | `plot_reparameterization` | 같은 $\epsilon$, 다른 $\mu$ | 잡음 40개 |
| 7 완성 예제 🔍 | `plot_kl_vs_mean` | $p$ 분산과 이동 비용 | 기존 |

### 13 · 자동미분 — `autodiff_plots.py` — 완료

MNIST를 쓰지 않습니다. `draw_compute_graph`로 값 노드(원)·연산(상자)·국소 미분(금색)을 그립니다.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 VJP 🔑 | `plot_shared_path`, `plot_vjp` | 국소 미분의 곱과 경로의 합, $J^\top g$ | $u=x^2, y=u+u$; 2×2 $J$ |
| 2 broadcast 🔑 | `plot_broadcast_backward` | 복사한 만큼 더함 | (2, 3) 합 |
| 3 Node 🔑 | `plot_mlp_graph`, `plot_local_rules` | MLP 손실의 그래프와 위상 순서, pullback 한 줄씩 | 순서 목록, row·column 예제 |
| 5 학습 루프 🔑 | `plot_training_cycle`, `plot_fit` (학습 셀 뒤), `plot_stop_gradient` | 새 그래프→backward→갱신, 한 경로 끊기 | — |

### 14 · 손실·정규화 — `regularization_plots.py` — 완료

상황: 이상치가 있는 직선, 작은 손계산 배열, MNIST 7(판별 가중치·dropout mask).

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 손실 🔑 | `plot_loss_shapes` | 잔차→손실·미분 | 세 곡선 |
| 2 가중치 🔑 | `plot_weighted_mean` | 가중 평균과 상수배 불변 | 관측 3개 |
| 3 L2 🔑 | `plot_l2_templates`, `plot_outlier_fits`, `plot_ridge_norms` | $\lambda$별 7 판별 가중치 이미지, 이상치 직선 | ridge 닫힌 해 3개 |
| 4 AdamW 🔑 | `plot_decay_comparison` | 상태에 무엇을 넣었는지가 방향을 바꿈 | 첫 스텝 세 방식 |
| 5 dropout 🔑 | `plot_dropout_masks`, `plot_dropout_objective` | 7의 mask 3장과 평균, 기대 손실 = 기본 + 분산 항 | 2,000 mask 평균 |

### 15 · 실험·평가 — `evaluation_plots.py` — 완료

상황: 사람 120명의 반복 관측, 특성 8개 분류 실험, MNIST 7 판별(템플릿).

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 분할 단위 🔑 | `plot_split_units` | 행 vs 사람 분할, 겹친 사람 수 | seed 150 분할 |
| 2 전처리 🔑 | `plot_standardize` | 학습 통계로 표준화 | 수직선 |
| 4 검증 선택 🔑 | `plot_selection_table`, `plot_training_curves` | $\lambda\times$seed 표와 평균, 곡선 | runs |
| 5 지표 🔑 | `plot_metrics_hand`, `plot_confusion_examples`, `plot_reliability` | 임계값→혼동 행렬→지표, MNIST 7 판별의 혼동 행렬(축 없음)과 틀린 예 6장, 신뢰도 곡선 | 손계산, 템플릿 분류 |
| 6 test 🔍 | `plot_confusion` | 축 없는 혼동 행렬 | 기존 |

## 4. 작업 순서

1. 계산 셀을 본문에 추가합니다. 결과 배열의 shape를 주석으로 적습니다.
2. 조합 함수를 해당 `*_plots.py`에 추가하고 `concept_plots` 부품만 씁니다.
3. 새 커널에서 노트북을 위부터 끝까지 실행하고 그림을 눈으로 확인합니다. 화살표·제목 겹침·글리프 경고를 봅니다.
4. 이 문서의 표에서 상태를 계획 → 완료로 바꿉니다.
