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

### 05b · LeNet: 05의 합성곱을 실제 손글씨에 적용하기 — `lenet_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 padding·두 단 🔑 | `plot_padding_scene` | 왜 p=(k−1)/2를 두르면 출력이 입력과 같은 크기인가: 회색 0 칸 위의 금색 창 | 4×4 격자에 np.pad 한 겹, 3×3 창의 반복문 출력 4×4 |
| 1 padding·두 단 🔑 | `plot_feature_flow` | conv→pool→conv→pool에서 크기가 28→28→14→10→5로 줄고, 마지막 한 칸이 입력 16×16을 본다 | 난수 커널로 7의 네 층 특성 맵, 커널 1·평균 풀링의 backward로 잰 수용 영역 상자 |
| 2 조립 🔑 | `plot_lenet_blocks (draw_blocks 두 줄)` | 텐서 모양 흐름과 파라미터가 있는 다섯 층 | 층별 (이름, 모양) 목록, 파라미터 수 61,706 |
| 2 조립 🔑 | `plot_lenet_trace` | 7 한 장이 conv1 6장 → pool2 16장 → A3·A4 → logit 10개로 바뀌는 값 | 05 함수와 행렬곱으로 손으로 이은 forward(학습 전), lenet_forward와 일치 검산 |
| 3 학습 루프 🔑 | `plot_data_plan` | 6만 장을 학습·검증·안 씀으로 나누고 test는 따로; 한 에폭 313스텝, 마지막 배치 32장, 첫 배치 64장 | math.ceil(20000/64), 마지막 배치 크기, rng.permutation의 첫 배치 |
| 3 학습 루프 🔑 (학습 셀 뒤) | `plot_training_result`, `plot_kernels_before_after (전/뒤/변화)`, `plot_feature_maps` | 배치 CE와 에폭별 검증 정확도, 학습이 conv1 커널에 더한 변화, 학습된 커널이 7에서 본 것 | train_lenet history, K1_before/lenet_p['K1']/K1_delta, conv_forward(mnist_7, 학습된 K1) |
| 4 이동 반응 🔑 | `plot_shift_response` | 입력을 1·2·3칸 옮기면 conv1 특성 맵은 정확히 따라 옮겨지고(금색 칸) softmax 확률은 조금 변한다 | np.roll 입력의 학습된 conv1 특성 맵·logit·softmax, 원본 최댓값 자리, 등변 차이 0 |
| 5 test 보고 🔍 | `plot_confusion_matrix`, `plot_wrong_examples` | 어느 숫자끼리 헷갈리나, 자신 있게 틀린 8장 | np.add.at 혼동 행렬(대각선 합/N = 정확도 검산), 틀린 장 중 예측 확률 상위 8 |

### 18b · 물체 검출: 격자마다 상자·신뢰도·클래스를 예측하기 (YOLO) — `detection_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 📎 데이터 | `plot_canvases` | 검출 문제의 입력과 답은 무엇인가: 캔버스 한 장에 숫자 여러 개와 상자 여러 개 | make_canvases가 만든 학습 캔버스 4장, 물체별 (cx, cy, w, h) 캔버스 비율과 클래스 |
| 1 상자와 IoU 🔑 | `plot_iou_boxes` | 겹침을 왜 넓이의 비로 재는가: 교집합 ÷ 합집합 | 두 상자의 모서리, 겹친 가로·세로 6×6, 넓이 144·72, 교집합 36, 합집합 180, IoU 0.2 |
| 2 격자 출력과 타깃 인코딩 🔑 | `plot_grid_targets` | 어느 셀이 어느 물체를 책임지고 그 셀의 15칸에 무엇이 들어가나 | 고정 예시 7·3을 놓은 캔버스, 중심 × S의 내림으로 얻은 책임 셀 (1,1)·(3,2), obj mask 4×4, 금색 셀의 타깃 15칸 |
| 3 YOLO 손실 🔑 | `plot_loss_terms` | mask가 어느 셀에 어느 항을 걸고, 네 항의 크기는 어떻게 다른가 | 2×2 손계산의 신뢰도 격자와 네 항 0.4 / 0.04 / 0.03 / 0.05 (합 0.52) |
| 4 완성 모델 🔑 | `plot_detector_blocks` | 텐서 모양이 (N,1,32,32)에서 (N,4,4,15)로 어떻게 가고 파라미터는 어느 층에 있나 | 블록별 텐서 모양 목록, 파라미터 수 백본 5,888 + head 495 = 6,383, 완전 연결 head면 123,120 |
| 4 학습 결과 | `plot_curves`(bridge_plots 재수출), `plot_detections` | 네 항이 각각 어떻게 줄고, 학습한 셀 16개는 무엇을 내놓는가 | 700스텝 동안 50스텝마다 평균낸 네 항, decode_boxes로 푼 검증 상자·신뢰도·클래스 argmax |
| 5 NMS 🔑 | `plot_nms_hand` | 겹친 상자 중 무엇을 남기고 무엇을 지우는가 | 상자 셋의 IoU 표(A–B 0.667), 점수 내림차순, 손계산 keep {A, C} |
| 5 NMS 결과 | `plot_nms_result` | 학습한 모델의 상자가 실제로 몇 개로 줄어드는가 | 검증 캔버스 0의 신뢰도 0.3 통과 6개와 NMS 뒤 5개, 정답 3개 |
| 5 precision–recall과 AP 🔑 | `plot_pr_curve` | 임계값을 낮추면 precision과 recall이 어떻게 맞바뀌고 그 전체를 어떻게 한 숫자로 요약하나 | 손계산 tp [1,1,0,1]·정답 5개의 누적 precision·recall·오른쪽 최댓값 상한과 AP 0.55, 검증 검출 639개의 같은 곡선과 AP 0.4761 |

### 23 · 잠재변수: 보이지 않는 선택을 더해서 없애기 (혼합 모형과 EM) — `mixture_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 주변 밀도 🔑 | `plot_marginal_sum` | 왜 성분 밀도를 더하면 관측의 밀도인가, 그리고 로그에서도 작은 항은 왜 보이지 않는가 | 격자 (701,)의 가중 성분 밀도 π_k N(x;μ_k,σ_k²) (701,2)와 그 합 (701,), x=0의 두 항 (2,)와 log p(0) |
| 1 2차원 🔍 소절 | `plot_mixture2_density` | 성분이 2차원 Gaussian이어도 더해서 없애는 구조는 같은가 | 161×161 격자의 성분별 π_k N(x;μ_k,Σ_k) 두 판과 합, 격자 합 × 칸 넓이 |
| 2 책임확률 🔑 | `plot_responsibility` | 거꾸로 물으면 답이 어떻게 나뉘고, 경계는 어디인가. 784차원에서는 왜 0 아니면 1인가 | r_k(x) 곡선 (701,2), 이차식 계수와 두 근, x=0의 r, MNIST 10성분 로그 항(최댓값 대비)과 책임확률 (10,) |
| 3 미분 🔑 | `plot_score_field` | score는 어느 쪽을 가리키고, 파라미터 기울기는 왜 책임확률로 가중되는가 | 자리 21곳의 score (21,), x=0의 s(0)·∂log p/∂μ (2,), 2차원 log p 등고선 격자와 벡터장 (143,2) |
| 4 EM 🔑 | `plot_em_progress` | E·M 두 단계를 반복하면 성분이 어떻게 갈라지고 로그우도는 어떻게 움직이는가 | 관측 800개, 반복 0·2·10의 가중 성분 곡선 (701,2)와 합, 반복 40회의 평균 로그우도 (40,) |
| 4 2차원 🔍 소절 | `plot_em2_progress` | 2차원에서 책임확률이 어떻게 갈리고 공분산 타원이 어떻게 기우는가 | 관측 600개, 반복 0·2·10·30의 μ (2,2)·Σ (2,2,2)·책임확률 (600,2), z를 훔쳐본 MLE와의 대조 |
| 5 샘플링 🔑 | `plot_ancestral_sampling` | 밀도를 뒤집지 않고 어떻게 점을 만드는가, 성분별 히스토그램을 쌓으면 왜 p(x)인가 | z 4,000개와 x=μ_z+σ_z ε (4000,), 1절의 밀도 곡선, 표본 평균·분산과 닫힌식 0.8·4.135 |
| 5 최적 복원 🔑 소절 | `plot_denoiser_curves` | 잡음이 커질수록 최선의 추측이 왜 전체 평균으로 수축하는가 | σ=0.25·1·3의 E[x\|x̃] 곡선 (701,3)과 잡음 낀 밀도 q_σ (701,3), 손계산 점 (0, 0.403833) |
| 6 완성 예제 🔍 | `plot_em_variants` | 같은 EM인데 시작값이 다르면 왜 다른 곳에 도착하는가 | 성분 3개 관측 900개, 시작값 두 가지의 학습된 성분 곡선 (701,3)과 60반복 평균 로그우도 |

### 24 · 하한으로 학습하기: Jensen 부등식과 ELBO — `elbo_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 Jensen 🔑 | `plot_jensen_chord` | 왜 평균을 먼저 취한 로그가 로그의 평균보다 큰가 (오목 함수와 현) | log 곡선 격자, 두 값 (1, 4)와 그 가중 평균 2.5, log E[f]·E[log f]·차이, 가중치 w_1을 0에서 1까지 바꾼 두 곡선 |
| 2 ELBO 🔑 | `plot_elbo_gap` | 하한은 증거를 넘지 않고 어디서 정확히 닿나, 남은 차이는 무엇인가 | q_1 격자 999개의 ELBO 곡선과 KL 곡선, 증거 log p(0) = −3.273491, posterior r(0), q=(0.5,0.5)에서의 세 숫자 |
| 3 두 항 🔑 | `plot_two_terms` | 재구성 항과 사전분포 KL이 q에 따라 어떻게 절충되나 | q 세 가지(균등·π·posterior)의 E_q[log p(x\|z)], KL(q‖π), 그 차이 = ELBO |
| 4 φ 경사 상승 🔑 | `plot_phi_ascent` | 기울기가 0이 되는 자리가 왜 정확히 posterior인가 | x = −1에서 q_1 격자의 ELBO 곡선과 ∂ELBO/∂φ_1 곡선, 경사 상승 61스텝의 q_1·ELBO·KL 기록 |
| 5 연속 잠재 🔑 | `plot_gaussian_elbo_landscape` | 연속 잠재에서 최적 q가 왜 닫힌식 posterior이고 그 높이가 왜 증거인가 | prior N(0,1)·닫힌식 posterior N(0.4706, 0.0588)·시작 q의 밀도, μ 축과 ℓ 축을 따라 본 정확한 ELBO(각 361점) |
| 5 🔍 소절 (샘플 수) | `plot_elbo_sample_spread` | 샘플 수 K는 추정의 무엇을 바꾸고 무엇을 바꾸지 않나 | K = 1·4·16·64에서 4,000번 되풀이한 ELBO 추정값과 표준편차 (2.041 / 0.993 / 0.482 / 0.246) |
| 5 🔍 소절 (학습 셀 뒤) | `plot_gaussian_q_learning` | 샘플로만 계산한 잡음 섞인 기울기가 닫힌식 posterior로 데려가나 | 600스텝의 정확한 ELBO 기록, 학습 전후 q 밀도, (μ, σ) 경로와 닫힌식 목표 (0.4706, 0.2425) |
| 🔍 완성 예제 | `plot_posterior_variants` | a와 s가 posterior의 폭을 어떻게 정하나 (a²/s²가 정보의 양) | (a, s) = (2, 0.5)·(2, 1.5)·(0.5, 0.5)의 닫힌식 posterior 밀도와 같은 하이퍼파라미터로 학습한 q 밀도 |

### 25 · VAE: encoder가 q를, decoder가 p(x|z)를 계산하는 ELBO 학습 — `vae_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 구조 🔑 | `plot_vae_flow` | 이미지 한 장이 어떻게 숫자 두 개가 되었다가 다시 784개가 되나 (그리고 그 사이의 '뽑기'는 어디 있나) | 학습 전 파라미터로 고정 예시 7을 직접 통과시킨 H·μ·ℓ·σ·ε·z·logit·확률 (NumPy 인라인, 1절 함수 정의 전) |
| 2 두 항 🔑 | `plot_loss_terms` | 재구성 항이 무엇을 재는가, 그리고 KL 항과 크기가 얼마나 다른가 | 전체 평균 이미지의 logit, 후보 세 개(평균 이미지 156.73 / 학습 전 559.00 / 거의 맞힘 68.75)의 화소별 BCE와 그 합, 학습 전 KL 0.1562 |
| 3 backward 🔑 | `plot_reparam_graph` | 샘플 z를 지나 기울기가 어떻게 μ와 ℓ 두 갈래로 갈리나, ε은 왜 되돌아가지 않나 | dz=0.30, ε=−0.80, ℓ=−0.40에서 σ=0.819, z 경로 dμ 0.3000·dℓ −0.0982, KL 경로 dμ 0.5000·dℓ −0.1648 |
| 4 완성 학습 🔑 | `plot_eps_draws` | ε을 다시 뽑을 때마다 z와 −ELBO가 얼마나 달라지나, 그래도 왜 평균으로 학습할 수 있나 | 고정 예시 7의 (μ, σ)에서 뽑은 z 200개, 같은 배치 32장에 ε만 바꿔 잰 −ELBO 12개 (569.98–580.16, 평균 575.00) |
| 4 학습 셀 뒤 | `plot_loss_history` | 재구성 항과 KL 항은 학습 동안 각각 어느 쪽으로 가나 | 3,000스텝 중 50스텝마다 기록한 미니배치 재구성(575.5 → 155.3)과 KL(0.25 → 6.10) |
| 4 학습 셀 뒤 | `plot_recon_pairs` | 잠재 2차원으로 좁힌 목을 지나면 고정 예시 10장이 어떻게 돌아오나 | encoder_forward의 μ만 decoder에 넣은 복원 확률 (10, 28, 28), 화소 평균 절대오차 10개 |
| 4 학습 셀 뒤 | `plot_latent_map` | 잠재 평면의 자리마다 어떤 그림이 나오고, 학습 데이터의 μ는 어디에 놓이나 | μ의 2–98% 분위수 8×8 격자를 decoder에 넣은 (224, 224) 타일, 5,000장의 μ (5000, 2)와 숫자 라벨, μ 평균 (0.62, −0.83)·표준편차 (1.35, 1.21) |
| 4 🔍 소절 (혼합) | `plot_mixture_vae` | 밀도를 아는 데이터에서 학습한 하한이 정말 진짜 평균 log 밀도 아래에 있나 | mix2_sample 2,000점, 잠재 1차원 VAE의 decoder 곡선 m(z) 200점과 모델 샘플 600점, ELBO 평균 −3.0823, mix2_logpdf 평균 −2.8674 |
| 5 사용 🔑 | `plot_vae_uses` | 복원·생성·보간이 어떻게 모두 같은 decoder 하나에서 나오나 | 고정 예시 네 장의 복원, 사전분포에서 뽑은 z 8개의 화소 확률과 잉크 평균, 7의 μ에서 1의 μ로 가는 보간 z (8, 2)와 그 그림 |
| 🔍 완성 예제 | `plot_latent_dim_variants` | 잠재 차원을 넓히면 두 항이 어느 쪽으로 움직이고, 차원마다 실제로 쓰이고 있나 | 잠재 2/8의 전체 데이터 재구성(147.2 / 93.1)·KL(6.31 / 16.48), 차원별 μ 표준편차 |
| 🔍 완성 예제 | `plot_recon_pairs` | 잠재 8차원이면 4절에서 뭉개졌던 숫자가 돌아오나 | wide_p로 vae_reconstruct한 고정 예시 10장 (10, 28, 28) |

### 28 · GAN: 판별기의 분류 손실을 생성기가 거꾸로 타고 오르기 — `gan_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 두 손실 🔑 | `plot_generator_losses` | 왜 minimax 대신 non-saturating을 쓰나: 학습 초기 p≈0에서 기울기 크기가 p와 1−p로 갈린다 | p 격자 999개의 판별기 두 항, 생성기 두 손실 log(1−p)·−log p, logit 기울기 −p·−(1−p), p=0.01 손계산 자리 |
| 2 기울기 경로 🔑 | `plot_gan_paths` | 생성기 갱신과 판별기 갱신에서 어떤 기울기가 흐르고 무엇을 버리나 | 작은 두 층 G·D의 raw NumPy forward(z (8,16) → x̂ (8,32) → a (8,1))와 ∂L_G/∂a, 각 상자에 적을 shape |
| 3 최적 판별기 🔑 | `plot_optimal_discriminator` | 생성기를 고정하면 판별기의 정답이 두 밀도의 비율로 정해진다 | 격자 4001개의 p_data, 선형 G의 p_g = N(0.5, 1.5²), logit* = log p_data − log p_g, D*, x=−2의 손계산 0.78312 |
| 3 판별기 학습 결과 🔍 | `plot_discriminator_fit` | 학습한 판별기가 닫힌식 D*에 얼마나 가까운가, 어디서 어긋나는가 | D만 3,000스텝 학습한 D(x) 곡선, 격자 적분 L_D 1.155538 vs 최적 1.152090, 밀도 가중 RMS 0.0266, 두 밀도 합이 최댓값 5%를 넘는 구간 |
| 3 1차원 GAN 결과 🔍 | `plot_gan_1d` | 교대 갱신이 진행되면 생성 히스토그램이 진짜 밀도를 덮고 판별기는 1/2에서 포기한다 | 학습 전·200스텝·2,000스텝의 생성 샘플 4,000개, 학습이 끝난 D(x) 곡선, 진짜 밀도 격자 |
| 4 봉우리 세기 🔑 | `plot_mode_counting` | 책임확률 argmax로 배정해 세면 mode 붕괴가 왜 숫자로 보이나 | 손으로 만든 세 구름(진짜·덮음·몰림) 2,000개씩, 배정 라벨과 개수 표 [[770,1230],[800,1200],[0,2000]], 책임확률 1/2 경계 격자 161×161 |
| 4 coverage 🔍 | `plot_mode_coverage` | 판별기가 약하면 같은 코드가 seed에 따라 붕괴하기도 하고 안 하기도 한다 | 균형·약한 D 두 설정 × seed 3개의 1,200스텝 학습 샘플 4,000개, 봉우리별 개수 6쌍, 기대 개수 4000·π |
| 5 PCA 왕복 🔑 | `plot_pca_roundtrip` | 생성기가 만들 것은 이미지가 아니라 좌표 32개다 | 앞 5,000장의 공분산·eigh, mnist_Q (784,32)·규모 s·mnist_Z, 고정 예시 7의 좌표 32개와 복원 이미지, 기저 이미지 6장, 남긴 분산 74.9% |
| 5 생성 결과 🔍 | `plot_generated_images · plot_digit_counts · plot_curves` | 생성한 좌표를 되돌린 숫자는 어떤 모습이고 열 가지 숫자를 모두 덮는가 | 3,000스텝 학습한 좌표 4,000개, 디코딩 16장(0–1로 자름), 클래스 평균 최근접 개수(생성·진짜), 50스텝 평균 손실 곡선 60점 |
| 🔍 완성 예제 | `plot_variant_bars` | k_D와 학습률 비대칭이 두 손실과 coverage를 어떻게 바꾸나 | 세 설정 800스텝 학습의 마지막 100스텝 평균 L_D·L_G와 봉우리 1 비율, 균형점 기준선 2log2·log2·0.4 |

### 32 · score: 밀도 대신 밀도의 기울기를 배우기 — `score_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 score 🔑 | `plot_score_overview` | 밀도 곡선 위 화살표가 봉우리를 향하고 봉우리에서 score가 0을 지나는가, 2차원에서는 등고선을 가로질러 봉우리로 향하는 벡터장인가 | 격자 p(x), 책임확률 r, s(x)=Σ r_k(μ_k−x)/σ_k², 밀도의 국소 최대(봉우리), 화살표 자리 7개의 s, 2차원 17×12 격자의 Σ r_k Σ_k⁻¹(μ_k−x), 등고선용 mix2_logpdf |
| 2 score matching 🔑 | `plot_hyvarinen` | 정답을 아는 식과 정답 없는 식의 차이가 s_θ와 무관한 상수 ½E[s²]인가 | 시험 함수 셋(−x, −(x−1)/2, 2 sin x)의 ½E_p[(s_θ−s)²]와 E_p[½s_θ²+s_θ'] 격자 적분(np.trapezoid, np.gradient), ½E_p[s²] |
| 3 DSM 🔑 | `plot_noisy_scores` | 타깃 −ε/σ가 x로 돌아가는 방향인가, σ가 커질수록 q_σ와 그 score가 봉우리 사이에서 매끈해지는가 | x=1의 잡음 6개 x̃와 −(x̃−x)/σ², σ=0/0.3/1의 q_σ 밀도와 score Σ r_k^σ(μ_k−x)/(σ_k²+σ²) |
| 4 회귀 데이터 🔑 | `plot_dsm_regression` | 점 하나만 보면 정답이 없어 보이는 시끄러운 타깃의 구간 평균이 q_σ score 곡선 위에 놓이는가 | 학습 데이터 4,000개의 (x̃, −ε/σ) 쌍, np.digitize 구간별 타깃 평균·개수, q_σ score·밀도 |
| 4 학습 결과 (학습 셀 뒤) | `plot_learned_score_1d`, `plot_learned_field_2d`, `plot_curves`(bridge_plots 재수출) | 학습한 s_θ가 봉우리 근처에서 닫힌식과 겹치고 봉우리 사이·바깥(데이터 없는 곳)에서 틀리는가 | 학습 전후 s_θ 격자값, \|오차\|, 구간별 RMS, 2차원 닫힌식·학습 벡터장, 촘촘한 격자의 오차 크기와 q_σ 밀도 |
| 5 오르막·Langevin 🔑 | `plot_dynamics` | 오르막은 가까운 봉우리에 멈추고 Langevin은 봉우리 주변을 흔들리며 분포 p를 따라 퍼지는가 | 출발점 7개의 200걸음 오르막 경로, 4,000입자 Langevin의 처음 150걸음 경로와 3,000걸음 뒤 히스토그램, p(x)Δx, TV 거리 |
| 완성 예제 🔍 | `plot_sigma_comparison` | σ가 클수록 정답 q_σ score가 매끈해지고 학습 오차(특히 봉우리 사이)가 작아지는가 | σ=0.1/0.3/1.0의 닫힌식·학습 score, 구간별 RMS 오차 |

### 33 · 잡음 수준을 잇는 score 모델: NCSN과 annealed Langevin — `ncsn_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 왜 잡음 수준이 여러 개인가 🔑 | `plot_sigma_ladder` | σ가 커지면 봉우리 사이 골이 메워지고 score가 완만해지는가, 사다리가 로그 축에서 등간격인가, 그리고 '원래 분포와의 거리'와 '데이터가 닿지 않는 구간'이 서로 반대 방향인가 | ladder = 3.0(0.1/3)^(i/9), 수준별 q_σ 밀도와 ∇log q_σ (격자 1601점), TV(q_σ,p) 사다리꼴 적분, 폭 0.25 구간의 기대 표본 수 < 1인 구간의 비율과 그 구간의 score RMS |
| 2 Langevin 🔑 | `plot_langevin_walk` | 경로가 봉우리 주변을 흔들리며 등고선을 채우는가, α가 작으면 아직 못 퍼지고 크면 치우침이 남는가 | 2차원 정확한 score로 800걸음(입자 2,500, 처음 150걸음 경로 4개), 등고선용 111×121 격자의 mix2_logpdf, α=0.01/0.1/0.6의 1,000걸음 히스토그램과 TV 거리 |
| 3 σ 조건 신경망 🔑 | `plot_loss_weighting` | 타깃 −ε/σ의 크기가 정확히 1/σ이고 최적 score는 그 아래인가, σ² 가중 전에는 작은 σ가 손실을 독차지하고 가중 뒤에는 모든 수준이 0.3–1로 모이는가 | 표본 20,000개에 수준별 잡음을 더해 닫힌식 최적 score를 넣은 E‖s+ε/σ‖²와 E‖σs+ε‖², 최적 score의 RMS √E[s²], 타깃의 제곱 평균 E[(ε/σ)²] |
| 3 학습 결과 (학습 셀 뒤) | `plot_curves`(bridge_plots 재수출), `plot_learned_ladder` | 하나의 모델이 σ=3부터 σ=0.1까지 닫힌식과 겹치는가, 작은 수준일수록 상대 오차가 커지는가 | 1차원 NCSN 6,000스텝 학습의 100스텝 평균 손실, 수준마다 격자 1601점의 학습 전·후 s_θ와 닫힌식 q_σ score, q_σ 가중 상대 오차 10개 |
| 4 annealed Langevin 🔑 | `plot_annealed_snapshots` | 왼쪽 봉우리 한 점에서 출발한 구름이 큰 σ에서 퍼졌다가 작은 σ로 내려오며 두 봉우리로 갈라지는가, 비율은 어느 수준에서 정해지는가 | α_i = ε σ_i²/σ_L² (ε=0.01), 정확한 q_σ score로 수준마다 100걸음씩 돈 입자 2,000개의 수준별 스냅숏과 책임확률 평균(오른쪽 봉우리 비율) |
| 4 coverage 비교 🔍 (학습 셀 뒤) | `plot_coverage_compare` | 같은 출발점·같은 걸음 수에서 단일 σ_L Langevin은 한 봉우리에 갇히고 annealed는 (정확한 score든 학습한 s_θ든) 두 봉우리를 채우는가 | annealed(정확) / 단일 σ_L 1,000걸음(정확) / annealed(3절의 ncsn_p2)의 최종 입자와 mix2_responsibility 평균, 닫힌식 평균·공분산과의 오차 |
| 5 MNIST PCA-32 🔑 | `plot_pca_ladder` | 좌표 32개만으로 고정 예시 7이 되살아나는가, σ가 커질수록 되돌린 그림에서 무엇이 먼저 사라지는가 | 5,000장의 공분산과 eigh 상위 32 고유벡터 mnist_Q, 좌표 mnist_Z와 규모 c, 7의 좌표·복원, 사다리 네 수준의 잡음을 좌표에 더해 되돌린 28×28 네 장, 앞 두 좌표의 산점 |
| 5 샘플 (학습 셀 뒤) | `plot_curves`(bridge_plots 재수출), `plot_sample_grid` | 잡음에서 출발한 좌표가 숫자 모양이 되는가, 숫자별로 고르게 나오는가 | 8,000스텝 학습의 100스텝 평균 손실, annealed Langevin(ε=0.05, 수준마다 100걸음)으로 만든 좌표 16개와 mnist_decode 복원, 클래스 평균 최근접으로 센 숫자별 개수 |
| 완성 예제 🔍 | `plot_variant_bars` | 사다리를 없애거나(L=1) 걸음을 1/400로 줄이면 봉우리 비율·평균·공분산이 얼마나 나빠지는가 | 총 1,000걸음 고정으로 L=1·3·10과 ε/400의 annealed 샘플 1,000개, 오른쪽 봉우리 비율·닫힌식 평균과의 거리·공분산 최대 오차 |

### 34 · Diffusion 전방 과정: 잡음을 조금씩 더하고 그 잡음을 맞히기 — `diffusion_forward_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 전방 과정 🔑 | `plot_forward_band` | 한 걸음 규칙을 t번 이으면 정말 한 줄로 건너뛸 수 있나: 고정 예시 7이 다섯 시각에 지워지는 띠와 ᾱ_t·1−ᾱ_t 곡선 | 선형 β 스케줄(T=200, 1e−4→0.02)과 누적곱 alpha_bar, 화소를 [−1,1]로 옮긴 7에 닫힌식 한 줄로 만든 t=0·25·50·100·200 이미지 |
| 1-1 혼합 전체 🔍 | `plot_mixture_forward` | 분포 전체는 시각 t에서 어떤 모양인가: 두 봉우리가 N(0,1) 하나로 뭉개지는 과정 | q_t_logpdf(성분 평균 √ᾱ_t μ_k, 분산 ᾱ_t σ_k²+1−ᾱ_t)의 밀도와 q_sample 표본 10,000개의 히스토그램, TV 거리 |
| 2 되묻기 🔑 | `plot_posterior_coefficients` | 되묻기의 평균은 x₀와 x_t를 어떻게 섞고, 한 걸음 잡음보다 얼마나 확실한가 | t별 계수 c₀(t)·c_t(t), 비율 β̃_t/β_t, t=50·x₀=1.5·x_t=0.9에서 격자 120,001점의 Bayes 적분 posterior(평균·분산) |
| 3 학습 타깃 🔑 | `plot_targets` | x₀ 예측과 ε 예측은 어떻게 생겼고 왜 서로 한 줄로 바뀌는가 | t=20·80·160의 q(x_t) 밀도, mix_denoiser(x_t/√ᾱ, σ_t)로 만든 E[x₀\|x_t], ε* = (x_t − √ᾱ x̂₀)/√(1−ᾱ), 잡음 무시선 x_t/√ᾱ |
| 4 신경망 🔑 | `plot_denoiser_blocks` | 모델은 지금이 몇 번째 시각인지를 어떻게 입력으로 받나: sinusoidal 시간 표와 블록 흐름 | t=0…200의 시간 표 (201, 16)을 09번 4절 식으로 직접 계산, 텐서 모양 블록 목록과 파라미터가 있는 층 두 개 |
| 4 학습 결과 🔍 | `plot_learned_eps` | 학습한 ε̂가 닫힌식 ε*와 얼마나 같은가, 어느 시각에서 덜 맞나 | t=20·80·160에서 학습 전·후 ε̂ 곡선, 시각 50개(4,8,…,200)의 E‖ε̂ − ε*‖² |
| 4 학습 결과 🔍 | `plot_eps_field_2d` | 2차원에서도 같은 코드가 되는가: 화살표장 비교 | t=80 격자 221점의 optimal_eps2(mix2_denoiser)와 학습한 ε̂, 좌표 변환·야코비안까지 넣은 q(x_t) 등고선 |
| 4 학습 결과 🔍 | `plot_denoised_examples` | 한 번에 되돌리면 왜 흐려지는가 (35번 역과정의 동기) | MNIST PCA-32 좌표에서 q_sample로 만든 x_t, ε̂로 되돌린 x̂₀ = (x_t − √(1−ᾱ)ε̂)/√ᾱ, decode_pca로 28×28 복원 |
| 5 SNR 🔍 | `plot_loss_by_t` | 같은 예측을 ε 자와 x₀ 자로 재면 왜 정반대로 보이는가 | 시각 100개의 ε 손실(모델·최적 바닥), x₀ 손실, SNR(t)=ᾱ_t/(1−ᾱ_t), t 네 구간의 평균 막대 |
| 완성 예제 🔍 | `plot_schedule_comparison` | 코사인 스케줄로 바꾸면 무엇이 달라지는가 | cosine_schedule의 ᾱ_t·SNR과 선형 스케줄의 그것, 같은 데이터·seed로 다시 학습한 시각별 ε 손실(모델·바닥) |

### 35 · Diffusion 역과정: 잡음 예측으로 $T$걸음 되돌리기, DDIM, 조건부 생성 — `diffusion_reverse_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 🔑 1 | `plot_reverse_step` | 역과정 한 걸음은 지금 자리 x_t를 어디로, 얼마나 움직이나? 두 분산 선택은 얼마나 다른가? | 세 시각(t=160·100·40)에서 x̂₀(x_t)·μ_θ(x_t)·σ_t 곡선과 x_t=1.2의 값, q(x_t) 밀도, √β_t와 √β̃_t의 t별 값 |
| 🔍 1-1 | `plot_reverse_histograms` | 정답 ε*를 알면 200걸음으로 원래 분포가 정말 나오나? 34번 스케줄로는 왜 틀어지나? | 두 스케줄에서 20,000개 역과정 표본의 히스토그램, 원래 밀도, TV 거리·평균·봉우리 비율 |
| 🔑 2 | `plot_reverse_paths` | 잡음 하나가 T걸음 동안 어떻게 두 봉우리로 갈라지나? | 60개 경로의 x_t(t) 기록, 도착한 봉우리 mask, 끝점 20,000개의 히스토그램 |
| 🔑 2 | `plot_samples_2d` | 되돌린 2차원 샘플이 원래 밀도와 같은가? 닫힌식 ε*와 학습한 ε̂의 차이는? | 원래 데이터 4,000점, 닫힌식·학습 모델로 되돌린 샘플 각 4,000점, mix2_logpdf 등고선 |
| 🔑 2 | `plot_curves`(bridge_plots 재수출) | MNIST PCA-32 좌표의 잡음 예측 학습이 수렴하나? | 미니배치 MSE 손실 3,000스텝 기록 (16.881 → 10.103) |
| 🔑 2 | `plot_sample_grid` | 좌표 샘플을 되돌려 디코딩하면 숫자로 보이나? 숫자별로 고르게 나오나? | DDPM 200걸음 샘플 16장의 decode_pca 이미지, nearest_class 숫자, 숫자별 개수 |
| 🔑 2 | `plot_snapshot_strip` | 200걸음 동안 한 장이 어떻게 만들어지나? | 표본 3개의 t=200·150·100·50·20·0 스냅숏을 decode_pca로 디코딩한 이미지와 최종 최근접 숫자 |
| 🔑 3 | `plot_ddim_paths` | 같은 x_T에서 걸음 수를 줄여도 같은 곳에 도착하나? 비용은 얼마나 줄어드나? | 같은 출발점 6개의 S=10·20·50·200 DDIM 경로와 도착점, 설정별 ε̂ 호출 수 |
| 🔑 3 | `plot_image_rows` | 같은 x_T에서 걸음 수만 바꾸면 그림이 눈에 띄게 달라지나? | 같은 x_T 8개를 S=200·50·20·10으로 되돌린 좌표의 decode_pca 이미지와 최근접 숫자 |
| 🔑 4 | `plot_guidance_concept` | guidance는 ε를 어떻게 바꾸고, w를 키우면 분포가 어떻게 왜곡되나? | 1차원 혼합의 닫힌식 ε(x_t,t,y)·ε(x_t,t,∅)·w=1·3 결합 곡선, w별 6,000개 표본 히스토그램과 평균·표준편차 |
| 🔑 4 | `plot_image_rows` | 숫자를 지정하면 실제로 그 숫자가 나오나? w를 키우면 무엇이 좋아지고 무엇이 나빠지나? | label 0–9 × 3줄 30장 격자(w=1), 같은 x_T 8개에서 label 3을 w=0·1·3으로 바꾼 24장, 맞은 비율과 좌표 평균 길이 |
| 🔍 5 | `plot_sampler_comparison` | 비용·조건 충실도·규모의 절충을 한눈에 보면 무엇을 골라야 하나? | 7개 설정(DDPM 200 w=0·1·3, DDIM 50·20·10)의 ε̂ 호출 수, 200장의 nearest_class 맞은 비율, 평균 좌표 길이 |
| ✏️ 6 (🔍 완성 예제) | `plot_reverse_histograms` | 걸음의 분산을 β̃_t 대신 β_t로 두면 표본이 달라지나? | 같은 ε*로 분산만 바꾼 20,000개 표본의 히스토그램, TV 거리·평균·표준편차·봉우리 비율 |

### 35b · EDM: score·DDPM·DDIM을 σ 하나로 다시 쓰기 — `edm_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 🔑 1 | `plot_unified_sigma(families`, `x`, `pdf`, `sigma_show`, `points`, `denoised`, `curve_sigmas`, `curves)` | NCSN·DDPM/DDIM·EDM이 정말 같은 σ 축 위의 서로 다른 일정인가? x에서 D(x;σ)로 가는 화살표가 곧 score인가? | 본문: vp_alpha_bar와 vp_to_sigma로 만든 세 일정의 (진행도, σ) 곡선 sigma_families, mix_logpdf로 만든 q_σ 밀도 tweedie_pdf, 화살표 점 tweedie_points와 mix_denoiser로 만든 tweedie_denoised, σ=0.2·1·4의 denoiser 곡선 denoiser_curves |
| 🔑 2 | `plot_preconditioning(image_sigmas`, `raw_images`, `scaled_images`, `vmax`, `coef_sigmas`, `coef_curves`, `scale_sigmas`, `scale_rows`, `sigma_data)` | 전처리 없이 신경망에 넣으면 입력의 규모가 얼마나 오가고, c_in을 곱하면 무엇이 같아지는가? 네 계수는 σ에 따라 어떻게 움직이는가? | 본문: 고정 예시 7을 [−1,1]로 옮긴 band_x0에 σ=0.05·0.3·1·4의 같은 잡음을 더한 band_raw와 c_in을 곱한 band_scaled, 닫힌식으로 만든 coef_curves(c_skip·c_out·c_in·c_noise), MNIST 좌표 1,000개로 잰 scale_rows(x·c_in x·F_target의 표준편차) |
| 🔑 3 | `plot_loss_weighting(log_sigma_samples`, `sigmas`, `weights`, `grid_range`, `bin_labels`, `raw_losses`, `weighted_losses) · plot_training_loss(steps`, `raw`, `smooth_steps`, `smooth`, `title)` | 학습용 σ는 어디에 몰려 있고 샘플러가 지나는 구간과 어떻게 다른가? λ(σ)를 곱하면 구간별 손실이 정말 평평해지는가? 학습은 어디서 멈추는가? | 본문: sample_sigma의 ln σ 표본 weight_log_samples, precondition으로 만든 λ(σ) 곡선 weight_curve, 정확한 mix2_denoiser로 σ 구간마다 잰 bin_raw와 bin_weighted, train_edm의 손실 기록 mixture_history와 100스텝 이동평균 mixture_smooth |
| 🔑 4 | `plot_ode_paths(rho_grids`, `contour`, `paths) · plot_solver_error(step_counts`, `error_series`, `slopes`, `call_labels`, `call_counts`, `call_errors)` | ρ는 걸음을 어디에 몰아 주는가? 같은 출발점에서 Euler·Heun·기준 해의 궤적은 얼마나 벌어지고, 걸음을 늘리면 오차가 어떤 기울기로 줄어드는가? 모델 오차는 어디서 바닥을 치는가? | 본문: sigma_steps로 만든 ρ=1·3·7 격자 rho_grids, mix2_logpdf 등고선 contour_pdf, euler_sample·heun_sample로 푼 궤적 path_traces, 기준 해 converge_reference(정확한 D로 Heun 512걸음)와 N=5·10·20·40·80의 오차 converge_errors, np.polyfit으로 맞춘 converge_slopes, 같은 D 호출 수 비교 call_errors |
| 🔍 5 | `plot_sampler_grid(rows`, `labels`, `captions) · plot_sampler_metrics(labels`, `calls`, `distances`, `real_distance`, `norms`, `real_norm)` | 같은 학습 모델을 네 샘플러로 돌리면 같은 잡음에서 같은 숫자가 나오는가? 비용과 지표는 실제 데이터와 얼마나 다른가? | 본문: train_edm으로 학습한 mnist_p와 mnist_denoiser, heun_sample·euler_sample·stochastic_heun으로 만든 sampler_coords(각 200장), decode_pca로 되돌린 sampler_rows, nearest_class로 잰 sampler_distances·sampler_classes와 좌표 노름 sampler_norms, 실제 데이터의 real_distance·real_norm |
| 🔍 완성 예제 | `plot_variants(step_counts`, `rho_errors`, `sigma_data_labels`, `grid_errors`, `sigmas`, `coefficient_sets)` | ρ와 σ_data를 바꾸면 무엇이 달라지는가? 전처리 계수를 틀리게 잡으면 얼마나 나빠지는가? | 본문: ρ=1·3·7로 다시 푼 Heun 오차 variant_rho_errors, σ_data를 1.62·0.5·4.0으로 두고 train_edm을 다시 돌려 mix2_denoiser와 비교한 variant_grid_errors, precondition으로 만든 σ_data별 c_skip 곡선 variant_skip_curves |


<!-- 2026-09-17 추가 노트북: 07b SNN · 10b CVNN -->

### 07b · SNN: 막전위가 문턱을 넘을 때만 스파이크를 내는 뉴런 — `snn_plots.py` — 완료

상황: 새는 물통(일정 전류 0.3·0.08의 LIF 손계산), MNIST 7의 rate coding, 실제 MNIST 20,000장(05b와 같은 분할). 학습 곡선은 `snn_plots.plot_training_result`로 따로 그립니다. 색은 파랑(선택·양수)·회색(나머지)·금색(강조 하나)만 쓰고, 범주는 선 모양으로 구분합니다.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 LIF 한 스텝 🔑 | `plot_lif_trace` | 막전위는 어떻게 새고·모이고·문턱을 넘으면 쏜 뒤 문턱만큼 빠지나, 어느 전류부터 쏘기 시작하나 | 손계산 I=0.3·0.08의 12시각 U_t·S_t(스파이크 4·9시각, U_5=0.2285), 정상 상태 I/(1−β), β=0.8·0.9·0.95 × 전류 121개의 200시각 발화율(문턱 전류 (1−β)θ 아래 0) |
| 2 부호화 🔑 | `plot_rate_coding`, `plot_coding_noise` | 밝기를 켜질 확률로 뽑은 0/1 프레임의 시간 평균이 왜 밝기로 돌아오고, 흔들림(RMS)은 왜 1/√T로 주나 | 7의 64시각 Bernoulli 프레임과 T=4·16·64 평균, T=1–256의 RMS(0.0546 → 0.028 → 0.0158)와 이론 √(mean x(1−x)/T), latency 발화 시각 |
| 3 층 펼치기 🔑 | `plot_row_sum`, `plot_snn_flow` | 0/1 입력의 행렬곱은 왜 켜진 행의 합(덧셈만)인가, 입력 래스터가 은닉 래스터와 시간 평균 logit으로 어떻게 바뀌나 | 6×4 손계산 [1, 2, −1, 4], 학습 전 난수 W1·W2로 7의 16시각(입력 스파이크 805·은닉 142), 덧셈 104,460번과 MLP 곱셈(0까지 101,632 / 0을 건너뛰면 11,904) |
| 4 대리 도함수 🔑 | `plot_step_vs_smooth` | 계단 LIF의 손실은 왜 평평한 계단이라 미분이 0이고, 부드러운 σ(kv)와 대리 도함수 σ_k'은 어떻게 생겼나 | 3–10번 쏜 은닉 뉴런 하나의 편향 601값에서 계단·부드러운 LIF의 스파이크 수와 7의 CE(b=0 기울기 0 vs −0.0797), k=2·5·10의 σ(kv)·σ_k'(꼭대기 k/4, 넓이 1) |
| 4 대리 도함수 🔑 | `plot_lif_bptt` (도식 + 표) | U_t로 오는 β 경로·σ_k' 경로와 S_t로 오는 직접 기울기·리셋 경로 −θ dU_{t+1}을 어떻게 더하나 | 스칼라 뉴런 T=3(전류 0.6)의 손계산 표(dU_3, dU_2, dU_1 = 0.5785, 0.9879, 0.8955) |
| 5 완성 실험 🔑 | `plot_snn_blocks`, `plot_epoch_encodings` | 한 스텝의 forward(계단 LIF)·backward 텐서 모양은 어떻게 흐르고, 같은 이미지가 에폭마다 어떻게 다른 입력이 되나 | forward·backward 블록 목록, 7을 에폭 1·2·3으로 부호화한 첫 프레임(25·23화소 차이, 기댓값 20.7) |
| 5 완성 실험 🔑 (학습 셀 뒤) | `plot_training_result`, `plot_trained_spikes`, `plot_time_steps` | 계단 forward·대리 도함수 backward로 학습되나, 뉴런별 발화율은 어떻게 퍼져 있나, 추론 시각 수 T가 정확도와 덧셈 수를 어떻게 함께 정하나 | train_snn history(3에폭, 검증 0.9545→0.9700, 발화율 0.43대), test 0.9460(한 번), 검증 뉴런별 발화율(0.000–0.819)과 학습 뒤 7의 래스터·logit, 검증 2,000장·seed 5개의 T=1–32 정확도(0.9524–0.9688)·장당 덧셈 수(13,676–441,027)와 MLP 곱셈 두 기준선(101,632 / 20,765) |

### 10b · CVNN: 크기와 위상을 함께 나르는 뉴런 — `cvnn_plots.py` — 완료

상황: 한 줄로 반 파장 간격인 마이크 8개와 30°에서 온 소리($z_m=s\,e^{i\pi m\sin\alpha}$), 방향 네 개(−30·0·30·50°)의 신호, MNIST 고정 예시 7·3의 푸리에 계수, 여섯 방향 맞히기(학습 12–3,000개, test 3,000개). 곡선 그림은 `gradient_plots._clean_axes`로 축을 정리합니다.

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 곱은 회전·확대 🔑 | `plot_complex_multiply` | 복소수 $w$ 하나를 곱하면 모든 화살표가 같은 각도로 돌고 같은 비율로 늘어난다; 마이크 신호는 길이가 같고 각도만 일정하게 돈다 | 손계산 $(1+i)(\sqrt3+i)=0.7321+2.7321i$(길이 2.8284, 75°), 화살표 4개의 길이 비 2·각도 차 30°, 📎 데이터 셀의 $z_m$ 8개 |
| 1 곱은 회전·확대 🔑 | `plot_real_block` | 복소층을 실수부·허수부 번갈아 펼치면 가중치마다 2×2 블록만 허용: 파라미터 $2DK$ vs $4DK$ | 난수 $W$ (4, 3)의 실수부·허수부를 `0::2`·`1::2` 슬라이스로 채운 $R$ (8, 6) |
| 2 위상의 정보 🔑 | `plot_magnitude_vs_phase` | 크기 표는 방향과 무관, 위상 표는 이웃마다 $\pi\sin\alpha$씩 늘어나고 이웃 곱의 각도로 방향을 되찾는다 | (4, 8) 크기·위상(도), 이웃 위상 차이 −90·0·90·137.9°, arcsin으로 −30·0·30·50° |
| 2 위상의 정보 🔑 | `plot_phase_swap` | 7의 위상 + 3의 크기로 되돌리면 7을 닮는다: 무늬의 자리는 위상이 정한다 | `np.fft.fft2` 계수의 log(1 + 크기)·위상, 교환 복원 두 장, 상관 0.775 / 0.276과 0.045 / 0.775 |
| 3 복소 기울기 🔑 | `plot_complex_descent` | $g=\partial L/\partial x+i\,\partial L/\partial y$로 내려가면 $w^*=c/z_1$에 수렴하고, 켤레를 잘못 둔 방향은 멀어진다; $w_tz_1$이 기준 화살표 1로 돌아온다 | $L(w)$ = ($wz_1-c$의 제곱 크기) 격자, 점마다 $-g$ 화살표, 12스텝 경로(잔차 0.0046)와 4스텝 오답 경로($w^*$까지 거리 1.939 → 4.031) |
| 4 활성화 🔑 | `plot_tanh_blowup` | 실수축에서 1 이하인 tanh가 허수축 $\pm i\pi/2$에서 발산한다(Liouville) | 복소 격자의 `np.abs(np.tanh(z))`, 실수축·허수축 단면(최댓값 0.9951, $y$=1.5707에서 10381.3) |
| 4 활성화 🔑 | `plot_activation_fields` | modReLU는 방향을 두고 길이만 바꾸며 짧은 화살표는 0, CReLU는 사분면에 따라 꺾인다; 입력을 돌리면 modReLU만 같이 돈다 | 격자점 48개의 modReLU($b=-0.5$, 12개가 0)·CReLU, 돌린 각도별 등변 차이 최댓값(4.0e−16 / 1.500) |
| 5 방향 맞히기 🔑 | `plot_doa_samples` | 전체 위상만 돌린 같은 소리: 화살표와 실수 MLP 입력 16개는 달라지고 빔포머 응답은 그대로 | 30° 표본 A와 $e^{2.4i}A$, `complex_to_real`, 각도 361개의 빔포머 응답($a(\alpha)^Hz$의 제곱 크기 / 8, 최대 응답 31°, 두 곡선 차이 4.4e−16) |
| 5 학습 결과 (학습 셀 뒤) | `plot_sample_curves` | 학습 표본 수별 test 정확도: 복소 모델(103개)은 30개에서 0.878, 실수 MLP(742개)는 0.531이고 300개에서 0.871, 크기만 MLP는 우연 수준, 빔포머 0.934 | 표본 수 6개 × seed 3개 × 모델 3개의 full batch Adam 400스텝 학습 |
| 5 학습 결과 (학습 셀 뒤) | `plot_phase_invariance` | 입력 위상을 돌려도 복소 모델의 정답 확률은 한 값이고 실수 MLP는 0.420–1.000으로 흔들린다; 예측이 바뀐 비율 0% vs 16.3% | test 표본 0번의 돌린 각도 181개 softmax, 표본마다 무작위 위상의 argmax 비교 |


## 4. 작업 순서

1. 계산 셀을 본문에 추가합니다. 결과 배열의 shape를 주석으로 적습니다.
2. 조합 함수를 해당 `*_plots.py`에 추가하고 `concept_plots` 부품만 씁니다.
3. 새 커널에서 노트북을 위부터 끝까지 실행하고 그림을 눈으로 확인합니다. 화살표·제목 겹침·글리프 경고를 봅니다.
4. 이 문서의 표에서 상태를 계획 → 완료로 바꿉니다.
