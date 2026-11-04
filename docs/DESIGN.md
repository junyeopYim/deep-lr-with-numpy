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
| 4 🔍 소절 (혼합) | `plot_mixture_vae` | 데이터 로그 밀도의 표본 평균과 모델 ELBO 추정치는 무엇이 다른가 | 학습과 독립인 평가 1,000점, 잠재 32회 평균 ELBO와 잠재 MC 표준오차, 데이터 로그 밀도 평균과 데이터 표준오차; 부등식 자체는 별도 선형 Gaussian 닫힌식으로 검산 |
| 5 사용 🔑 | `plot_vae_uses` | 복원·생성·보간이 어떻게 모두 같은 decoder 하나에서 나오나 | 고정 예시 네 장의 복원, 사전분포에서 뽑은 z 8개의 화소 확률과 잉크 평균, 7의 μ에서 1의 μ로 가는 보간 z (8, 2)와 그 그림 |
| 🔍 완성 예제 | `plot_latent_dim_variants` | 같은 평가 데이터에서 기대 재구성 항·KL과 잠재 MC 오차는 어떻게 달라지는가 | 학습에서 제외한 500장, 잠재 16회 평균의 재구성 항·정확한 KL·MC 표준오차, 학습 데이터의 차원별 μ 표준편차; 손실 방향과 잠재 사용 여부를 차원 수만으로 단정하지 않음 |
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


### 16 · 정규화 층: 어느 축의 통계를 사용하는가 — `normalization_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 통계와 축 🔑 | `plot_axis_normalization` | 같은 열의 값에서 평균을 빼고 크기를 나누면 무엇이 남나 | 작은 배열의 평균·중심화·분산·정규화, 손계산·반복문·벡터화, ε를 포함한 정확한 출력 분산 |
| 2 BatchNorm 🔑 | `plot_bn_modes` | 현재 배치 통계를 쓸 때와 저장 통계를 쓸 때 무엇이 다른가 | train 출력·현재 배치 평균/분산, 별도로 갱신한 running 통계·eval 출력, 두 모드의 입력·affine 파라미터 기울기 |
| 3 LayerNorm 🔑 | `plot_batch_independence` | 같은 표본의 동료만 바꾸면 어떤 정규화가 변하나 | 두 배치의 고정 첫 표본에 대한 LN·BN 출력, 표본별 특징 축 통계와 전체 backward |
| 4 RMSNorm 🔑 | `plot_norm_comparison` | 평균을 빼는 계산과 제곱 평균의 크기만 나누는 계산은 어떻게 다른가 | 같은 행렬의 BN·LN·RMS 출력, 이동·배율·상수 입력·배치 하나의 경계 검산 |
| 5 MNIST 🔑 | `plot_mlp_norm_flow` | 선형 출력과 활성화 사이의 정규화가 어느 배열을 바꾸나 | 고정 MNIST 예시의 선형 출력·LN 출력·tanh·logit |
| 5 학습 결과 🔍 | `plot_norm_training`, `plot_norm_predictions` | 같은 조건에서 네 방식의 학습과 독립 평가가 어떻게 달라지나 | 정규화 없음·BN·LN·RMS MLP의 학습 기록, 고정 validation·test 예측, 학습 파라미터 수 |

### 17 · ResNet: 입력과 기울기가 두 길로 흐르는 블록 — `resnet_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 두 경로 🔑 | `plot_two_paths` | 입력에 수정량을 더하면 기울기는 어떻게 합쳐지나 | 작은 선형 잔차의 출력·직접 기울기·수정 기울기, 손계산·반복문·벡터화 |
| 2 블록 🔑 | `plot_dense_block` | 비선형 수정량을 만들며 어떤 중간 배열을 지나나 | 고정 MNIST 예시의 두 특징·tanh 은닉 값·수정량·합, 모든 파라미터와 입력의 미분 |
| 3 투영 🔑 | `plot_projection` | 채널·해상도가 달라진 입력을 어떻게 더하나 | 3×3 주경로·1×1 stride 투영의 출력 shape, 홀수 크기의 두 경로와 기울기 합 |
| 4 깊이 🔑 | `plot_depth_flow`, `plot_depth_statistics` | 잔차 경로도 신호와 기울기를 상쇄하거나 키울 수 있나 | 스칼라 반복의 정확한 미분, 같은 가중치의 plain·잔차 모델 깊이별 활성 RMS·기울기 비율, 개별 seed 곡선·평균 |
| 5 분류기 🔑 | `plot_classifier_flow` | 잔차 블록을 분류 손실과 어떻게 연결하나 | 고정 MNIST 예시의 축소 입력·stem·블록 출력·분류 확률 |
| 5 학습 결과 🔍 | `plot_training_comparison`, `plot_test_predictions` | 같은 파라미터 수·초기화·학습 예산에서 결과가 어떻게 달라지나 | 세 seed의 학습·검증 곡선·stem 기울기 노름·독립 test 예측, 대응하는 seed별 정확도 차이 |

### 18 · U-Net: 해상도를 되돌리며 화소마다 예측하기 — `unet_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 화소별 예측 🔑 | `plot_pixel_task` | 입력과 같은 좌표에서 무엇을 정답으로 예측하나 | 작은 도형 입력·binary mask·logit·sigmoid 확률 |
| 2 해상도 🔑 | `plot_resize_coordinates` | 홀수 격자를 줄였다 늘릴 때 어느 화소가 어디로 가나 | 우하단 0 padding·분모 4의 평균 pooling·nearest 확대·원래 해상도로 crop, 역방향 합산 개수 |
| 3 skip 연결 🔑 | `plot_skip_channels`, `plot_unet_trace` | 서로 다른 경로의 채널을 붙이고 기울기를 어떻게 다시 나누나 | encoder·확대된 bottleneck·concat과 채널별 gradient, 실제 U-Net 중간 특성 맵·logit, 전체 backward |
| 4 손실 🔑 | `plot_pixel_losses` | 화소 BCE와 soft Dice가 각각 무엇에 반응하나 | 타깃·확률·화소별 BCE·결합 목적의 logit 기울기, 빈 mask와 hard 지표의 별도 규약 |
| 5 비교 조건 🔑 | `plot_segmentation_plan` | 어떤 자료·학습 예산·파라미터를 고정해 비교하나 | train/validation/test 개수·갱신 예산, skip 사용/0 대조의 전체·활성 파라미터 수 |
| 5 학습·변형 🔍 | `plot_training_curves`, `plot_segmentation_predictions`, `plot_resolution_variants` | 학습한 확률은 도형 경계를 찾으며 해상도·전경 비율 변경에도 계산이 이어지나 | 같은 조건의 학습 기록·독립 test 확률·hard IoU/Dice, 홀수 해상도와 전경 비율 변형의 실제 forward·backward |

### 19 · ViT: 이미지를 패치 토큰으로 읽기 — `vit_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 패치 🔑 | `plot_patch_flow` | 화소의 어느 축을 옮겨 패치 한 행을 만드는가 | 4×4 숫자판의 손계산·반복문·벡터화 patchify와 역변환 |
| 2 토큰 🔑 | `plot_token_embedding` | 같은 패치 투영에 CLS와 위치를 어떻게 더하나 | 고정 MNIST 7의 16×49 패치·공유 투영·CLS·위치 표·합 |
| 3 블록 🔑 | `plot_attention_block` | 토큰별 정규화와 attention 수정량이 잔차 경로에 어떻게 합쳐지나 | LN·QKV·행 softmax·출력 투영·잔차, MLP와 전체 backward |
| 4 위치·비용 🔑 | `plot_permutation`, `plot_patch_budget` | 패치 순서와 위치를 함께 옮길 때와 내용만 옮길 때 무엇이 다른가 | 위치 유무의 순열 출력, patch 크기별 실제 파라미터 수·토큰 수·attention 원소 수 |
| 5 분류기 🔑 | `plot_classifier_flow` | 같은 encoder 출력을 CLS 또는 패치 평균으로 어떻게 읽나 | 고정 MNIST 예시의 패치·encoder·최종 LN·readout·학습 전 확률 |
| 5 학습 결과 🔍 | `plot_vit_training`, `plot_vit_predictions`, `plot_cls_attention` | 동일한 초기값·학습 예산의 두 readout은 어떻게 학습하며 어느 패치를 읽나 | 학습 기록·독립 test 예측·CLS의 원래 attention 확률과 자기 자신에 대한 비중 |

### 20 · 시퀀스 encoder–decoder: 보고, 한 칸씩 이어 쓰기 — `seq2seq_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 한 칸 이동 🔑 | `plot_teacher_shift` | 정답을 직접 읽지 않으려면 입력과 target을 어떻게 옮기나 | source·BOS로 시작하는 decoder 입력·EOS를 포함한 target·padding mask |
| 2 읽는 범위 🔑 | `plot_attention_masks` | 미래와 PAD를 함께 가리고 빈 행을 어떻게 다루나 | causal·query/key padding mask, 허용 key에서만 정규화한 확률·가중합 |
| 3 기억·질문 🔑 | `plot_encoder_decoder_flow` | decoder 질문과 encoder 기억은 어느 배열이며 기울기가 어디로 돌아가나 | 작은 항등 투영의 self/cross attention·잔차·중간 특징, 세 attention과 embedding 전체 backward |
| 4 유효 손실 🔑 | `plot_masked_loss` | 길이가 다른 문자열의 어느 위치를 손실 분모에 넣나 | 위치별 NLL·valid target 평균 CE·logit 미분, 파라미터·연속 특징 중심차분 |
| 5 자유 생성 🔑 | `plot_generation_protocol` | 정답 prefix와 모델이 생성한 prefix는 어떻게 다른가 | 미학습 모델의 실제 greedy 단계·teacher prefix·EOS 종료 |
| 5 학습 결과 🔍 | `plot_sequence_history`, `plot_generated_examples` | teacher-forcing 학습 뒤 정답 없이도 끝까지 생성하나 | 고정 학습의 validation 기록·독립 test 자유 생성·토큰/전체 일치·EOS 지표, PAD 폭과 미학습 길이 변형 |

### 21 · 집합과 graph attention — `sets_graph_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 순열 🔑 | `plot_permutation` | 원소 순서를 바꾸면 무엇이 함께 움직이고 무엇이 그대로인가 | 과일 특징·공유 함수·순열 결과·합, 손계산·반복문·벡터화 |
| 2 집합 크기 🔑 | `plot_pool_cardinality` | 같은 꾸러미를 복제하면 합과 평균이 어떻게 다른가 | 공유 φ 특징·복제된 원소·sum/mean, Deep Sets의 전체 forward·backward |
| 3 이웃 mask 🔑 | `plot_neighbor_mask` | 고립 노드와 padding을 어떻게 다르게 처리하나 | 인접행렬·valid/self-loop mask·안정적인 이웃 softmax, 차단된 점수의 기울기 |
| 4 GAT 🔑 | `plot_gat_aggregation` | 공유 특징을 읽는 이웃 비중도 어떻게 학습하나 | 투영 특징·additive 점수·LeakyReLU·이웃 비중·가중합, 값과 점수의 두 미분 경로 |
| 5 학습 계획 🔑 | `plot_task_plan` | 꾸러미 회귀와 노드 회귀의 정답·분모·예산은 무엇인가 | 과제별 타깃·train/validation/test 개수·갱신 예산 |
| 5 결과·변형 🔍 | `plot_learning_curves`, `plot_heldout_predictions`, `plot_size_degree_metrics` | 학습과 고정 평가, 순열 성질과 크기·차수 변경은 어떻게 구분하나 | 두 과제의 학습 기록·test 예측·별도 MSE, 고정 모델의 집합 크기·노드 수·이웃 수 변형 |

### 22 · Autoencoder: 압축 좌표에서 복원과 잡음 제거까지 — `autoencoder_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 병목 🔑 | `plot_linear_flow` | 좌표 하나만 남기면 어떤 정보가 사라지고 복원 손실은 어떻게 생기나 | 작은 입력·encoder·decoder의 행렬곱, 잠재 좌표·복원·잔차, 손계산·반복문·벡터화한 half-MSE |
| 2 선형 AE 🔑 | `plot_pca_projection` | PCA 방향으로 접었다 펼친 결과와 버린 방향은 무엇인가 | 학습 평균으로 중심화한 배열, 상위 주성분 좌표·투영·잔차, 기저를 바꾸어도 같은 복원인지 검산 |
| 3 비선형 AE 🔑 | `plot_nonlinear_flow` | 화소가 은닉층과 병목을 지나 다시 화소로 돌아오는가 | 고정 MNIST 예시의 은닉 출력·잠재 좌표·초기 복원; 본문의 전체 forward·backward와 작은 모델 중심차분 |
| 4 비교 조건 🔑 | `plot_comparison_setup` | 같은 잠재 차원에서 평균 복원·PCA·미학습 AE는 무엇을 돌려주나 | train에서 구한 평균·PCA 기저, 고정 예시의 세 복원 |
| 4 학습 결과 🔍 | `plot_training_history`, `plot_reconstruction_rows` | 학습한 AE가 독립 표본도 복원하며 어떤 획을 잃나 | epoch별 train·validation MSE, 같은 분할·잠재 차원의 기준선과 AE 복원, 별도 test 평가 |
| 5 잡음 제거 🔑 | `plot_denoising_flow` | 잡음은 어디에 넣고 손실의 타깃은 무엇으로 두나 | 깨끗한 예시·고정 잡음·오염 입력·깨끗한 타깃, 잡음을 고정한 기울기 검산 |
| 5 학습 결과 🔍 | `plot_training_history`, `plot_reconstruction_rows` | 같은 잡음 입력에 일반 AE와 denoising AE가 어떻게 반응하나 | 두 학습 기록과 같은 오염 test 입력의 복원·MSE; 복원 목적과 잠재 사전분포의 차이 |

### 26 · VAE 분포 읽기: posterior·prior와 조건부 생성 — `vae_distribution_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 분포 역할 🔑 | `plot_distribution_roles` | 관측별 posterior·합친 posterior·prior는 어떻게 다른가 | Gaussian 밀도·혼합 밀도, 같은 평균·분산이어도 다른 분포인 예 |
| 2 목적·일정 🔑 | `plot_beta_schedule` | β와 warmup은 복원·KL의 비중을 어떻게 바꾸나 | β=0·1·4와 200스텝 warmup 일정, 공통 평가 R+KL과 학습 목적의 구별 |
| 3 조건부 경로 🔑 | `plot_conditional_flow` | 조건은 encoder와 decoder의 어느 입력에 들어가나 | 두 경로의 연결 배열·posterior·재매개화·decoder 평균, 전체 파라미터·입력·조건의 미분 |
| 4 평가 축 🔑 | `plot_evaluation_flow` | 관측·MC 반복·학습 seed를 어느 축에서 평균내나 | 독립 분할·고정 평가 잡음·평균 축, 관측별 R·KL |
| 4 실제 학습 🔍 | `plot_learning`, `plot_test_metrics` | 같은 예산의 네 설정은 공통 평가에서 어떻게 다른가 | 3 seed × 4설정의 400스텝 학습, validation 기록·test R/KL/−ELBO·seed SD·MC SE |
| 5 생성 경로 🔑 | `plot_generation_path` | decoder 평균과 조건부 관측 표본은 무엇이 다른가 | 공통 prior z·조건·decoder 평균·독립 관측 잡음 |
| 5 생성·진단 🔍 | `plot_generated_comparison`, `plot_latent_diagnostics` | 조건별 생성과 β 설정별 잠재 사용량은 어떻게 다른가 | 고정 seed의 짝지은 표본, 좌표별 KL·조건 내 평균 분산·decoder 민감도, 의도적으로 만든 collapse 대조 |

### 27 · Monte Carlo: 기대값에서 샘플 기울기까지 — `monte_carlo_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 표본 평균 🔑 | `plot_mean_flow` | 몇 개 뽑은 평균과 분포 전체의 기대값은 어떻게 다른가 | 잡음 네 개의 제곱값·손계산·반복문·벡터화 평균, 닫힌식 기대값 |
| 2 오차 🔑 | `plot_replication_flow`, `plot_mc_precision` | 한 실행의 SE와 여러 독립 실행 사이의 SD는 어떻게 연결되나 | 반복·표본 축을 나눈 배열, 표본 수별 평균·표본분산·SE와 정확한 SD |
| 3 중요도 추출 🔑 | `plot_importance_flow`, `plot_importance_results` | 표본을 다르게 뽑은 대가를 p/q로 고칠 수 있으며 무엇을 확인해야 하나 | 유한 상태의 정확한 가중합, Gaussian 제안분포의 log weight, raw·self-normalized 추정·반복 SD·ESS |
| 4 경로 미분 🔑 | `plot_pathwise_flow`, `plot_optimization` | 고정한 ε에서 계산한 미분과 기대값의 미분을 어떻게 각각 검산하나 | Gaussian 위치 재매개화, 표본 손실·pathwise 미분·고정 잡음 중심차분, 새 잡음으로 갱신한 경로와 정확한 목적값 |
| 5 score-function 🔑 | `plot_score_flow`, `plot_baselines` | 이산 표본의 확률을 미분하면 어떤 기여가 생기며 baseline은 언제 편향을 만드나 | Bernoulli 두 상태의 확률·보상·score, 작은 배치 전체 열거, 고정·같은 배치 평균·leave-one-out baseline의 정확한 평균·분산 |

### 29 · GAN 목적과 안정화: critic 기울기와 생성 품질 구분 — `gan_objectives_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 분포 차이 🔑 | `plot_distribution_distances` | 완벽히 구별되는 두 분포도 이동 거리가 다른가 | 유한 질량·최적 판별 확률·JS·1D CDF의 W₁ 손계산·반복문·벡터화 |
| 2 G 목적 🔑 | `plot_generator_objectives` | 같은 fake 점수에서 minimax·NS·Wasserstein의 미분은 어떻게 다른가 | 안정적인 목적값·점수 도함수, tanh G/critic의 전체 backward·수치미분 |
| 3 GP 🔑 | `plot_gp_path` | 입력 기울기의 벌점은 어떤 지점에서 어떻게 계산하나 | 고정 보간점·입력 기울기·norm·표본별 벌점, 직접·간접 경로의 혼합 미분 |
| 4 투영 🔑 | `plot_spectral_projection` | 큰 특잇값만 자르면 무엇이 보장되고 무엇이 달라지나 | SVD·특잇값 자르기·투영 행렬·층별 norm 상한, 작은 특잇값 보존과 재투영 검산 |
| 5 평가 지표 🔑 | `plot_quality_coverage` | 네 mode를 다 덮어도 대부분의 점이 틀릴 수 있나 | 의도적 진단 후보 세 개·최근접 거리·반경 내 판정·전체 표본 분모의 품질과 최소 개수 coverage |
| 5 실제 학습 🔍 | `plot_training_diagnostics`, `plot_seed_samples` | 제약의 진단값과 최종 생성 품질이 어떻게 다른가 | 3방법 × 3seed의 공통 초기값·표본 계획·160 G/480 D 갱신, critic gap·평균 입력 norm·G 기울기, 모든 생성 표본·지표·시간 |

### 30 · Normalizing Flow: 부피 보정으로 학습하는 밀도 — `flow_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 부피 🔑 | `plot_affine_volume` | 같은 질량의 구간·면적을 늘리면 밀도가 어떻게 바뀌나 | 1D affine 밀도의 손계산·반복문·벡터화, 2D 꼭짓점 변환·행렬식 |
| 2 coupling 🔑 | `plot_coupling_map` | 한 좌표를 유지하면 어떻게 역변환과 logdet를 계산하나 | conditioner 출력 s,t·변환 좌표·한 점의 삼각 Jacobian, 순·역 logdet |
| 3 합성 🔑 | `plot_flow_stages` | coupling과 swap을 합쳐 두 축을 모두 바꾸려면 | 층별 좌표·logdet, 전체 역전파와 파라미터·입력 중심차분, 홀수 swap 부호 검산 |
| 4 학습 계획 🔑 | `plot_training_plan` | 곡선 자료를 어떤 목적·분할·예산으로 학습하나 | 2D 합성 자료, 고정 train/validation/test 개수·500스텝·배치64 |
| 4 학습 결과 🔍 | `plot_nll_history`, `plot_fitted_density` | 같은 raw 좌표에서 Gaussian 기준선과 어떻게 다른가 | train만으로 적합한 Gaussian, 200개 파라미터 flow의 validation NLL·독립 test·밀도 격자 |
| 5 적분 🔑 | `plot_quadrature_cells` | 셀 중앙의 밀도를 어떤 면적으로 곱해 질량을 얻나 | 밀도 격자·셀 면적·질량 기여, 알려진 Gaussian 상자 질량 검산 |
| 5 생성·질량 🔍 | `plot_generation_diagnostics`, `plot_mass_checks` | 역변환 생성과 밀도 적분이 일치하며 오차는 어떻게 나뉘나 | heldout 잠재·독립 생성·왕복, 고정 상자 해상도·고정 간격 영역 확장·20,000개 생성 표본의 MC 질량과 SE |

### 37 · 생성 모델 평가: 지표와 실험 조건을 함께 고정하기 — `generative_evaluation_plots.py` — 완료

| 절 | 그림 | 답하는 질문 | 본문에서 계산하는 것 |
|---|---|---|---|
| 1 평가 조건 🔑 | `plot_evaluation_protocol` | 어떤 자료에서 통계를 구하고 어떤 자료를 마지막에 평가하나 | train만으로 구한 평균·규모, 고정 test 변환, train·test·생성 표본 수 |
| 2 밀도·mode 🔑 | `plot_density_diagnostics`, `plot_em_learning` | 높은 정답 밀도에 모인 생성점이 두 mode를 다 덮는가, 학습한 밀도 자체는 얼마나 맞나 | 격자 밀도·영역별 비율·정답 밀도의 표본 진단, EM 전후 밀도·학습 로그우도와 자기 모델의 held-out NLL |
| 3 MMD 🔑 | `plot_mmd_terms`, `plot_mmd_sensitivity` | 두 표본 집합의 쌍별 유사도를 어떻게 합하며 대역폭과 표본 수가 수치에 어떻게 작용하나 | RBF 커널 세 행렬·대각항을 포함한 V통계, 반복문·벡터화 대조, 대역폭 변화·독립 반복 |
| 4 이미지 특징 🔑 | `plot_feature_summary`, `plot_copy_diagnostics` | 특징 평균·공분산이 같아도 복제가 숨어 있을 수 있나 | 고정 MNIST 예시의 직접 정의한 특징, 공분산, 고유 그림 비율, train·test 화소 최근접 거리와 대응 이미지 |
| 5 비교 실험 🔑 | `plot_paired_protocol`, `plot_comparison_summary` | 같은 표본 예산과 평가 집합에서 후보 차이·seed 변동·비용을 어떻게 함께 기록하나 | oracle·학습 GMM·단일 Gaussian·붕괴 진단·복제 진단의 동일 표본 수 비교, 지표별 평균·seed SD·짝지은 차이·측정 비용 |

### 31 · 에너지 기반 모델과 MCMC — `energy_plots.py` — 완료

log-partition과 positive/negative phase, 격자 우도 학습, Langevin·MALA·chain 의존성 비교. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 낮은 에너지의 상태에 더 큰 확률을 줍니다 | `plot_energy_table` | `energy_three, weights_three, masses_three` |
| 2. 자료와 모형의 평균을 서로 당깁니다 | `plot_phase_terms` | `phase_grid, phase_density, phase_positive, phase_negative, phase_gradient` |
| 3. 정규화된 우도로 에너지 모형을 실제 학습합니다 | `plot_training_plan` | `grid, true_density, train[:4], training_settings` |
| 3. 정규화된 우도로 에너지 모형을 실제 학습합니다 | `plot_energy_fit` | `history, grid, [initial_density, learned_density, true_density], ['초기', '학습', '자료 생성']` |
| 4. 기울기로 제안하고 수락 비율로 보정합니다 | `plot_langevin_proposal` | `proposal_states, proposal_parts, proposal_terms` |
| 5. chain의 길이와 독립 표본 수를 구별합니다 | `plot_dependent_blocks` | `block_example, block_averages` |
| 5. chain의 길이와 독립 표본 수를 구별합니다 | `plot_chains` | `chain_paths, sampling_labels` |
| 5. chain의 길이와 독립 표본 수를 구별합니다 | `plot_chain_density` | `grid, learned_density, hist_centers, histograms, sampling_labels, np.asarray(chain_summary)` |

### 38 · 데이터 변형과 불변성 — `augmentation_plots.py` — 완료

라벨 보존·양성 쌍·순열, 전체 미분과 변형 강도별 분류 학습·배경 이동 평가. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 같은 과일이라는 조건부터 정합니다 | `plot_panels` | `[example, flip_example, rotate_example, label_table], ['원본: 상태·배경', '배경 부호 반전', '90도 회전', '원본·반전·회전 라벨']` |
| 2. 두 번 뽑은 변형이 양성 쌍을 만듭니다 | `plot_panels` | `[view_one, view_two, pair_gap], ['같은 원본의 뷰 1', '같은 원본의 뷰 2', '좌표별 제곱 차이']` |
| 3. 순서를 바꾸면 출력은 어떻게 바뀌어야 하나요? | `plot_panels` | `[features[0], reordered[0], box_means], ['원소별 특징', '같은 특징의 순서 변경', '원본·재정렬 평균']` |
| 4. 변형한 입력에서 실제 손실을 미분합니다 | `plot_panels` | `[small_x, small_view, small_h, small_logits], ['입력 X', '고정 변형 V', 'tanh 은닉 H', '익음 logit']` |
| 5. 강도를 미리 정하고 다른 배경에서 평가합니다 | `plot_panels` | `[preview, plan, strength_plan], ['train 과일: 상태·배경', 'train·valid·test: N·seed', '조건: 잡음 σ·부호 파괴']` |
| 5. 강도를 미리 정하고 다른 배경에서 평가합니다 | `plot_curves` | `histories, [s[0] for s in settings], 'validation 표본 평균 BCE'` |
| 5. 강도를 미리 정하고 다른 배경에서 평가합니다 | `plot_panels` | `[train_scores[:, 1:2], scores], ['조건별 train 정확도', 'test: CE·정확도·특징분산'], [None, ['CE', '정확도', '분산']]` |

### 44 · 토큰화·언어 데이터·자기회귀 목적 — `tokenization_plots.py` — 완료

UTF-8·byte BPE, 문서 경계·shift·padding, bigram 전체 학습과 held-out NLL·생성. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 같은 글자도 문자 수와 byte 수가 다릅니다 | `plot_token_units` | `unit_text, unit_characters, unit_bytes` |
| 2. 자주 붙는 byte 쌍을 train 문서에서만 합칩니다 | `plot_pair_merge` | `merge_before, merge_counts, ['(0,1)', '(1,0)'], merge_after_hand` |
| 3. 문서마다 한 칸 이동하고 유효 정답만 셉니다 | `plot_shift` | `shift_inputs_hand, shift_targets_hand, shift_valid_hand` |
| 4. token 번호로 고른 행을 다음 확률로 바꿉니다 | `plot_embedding_path` | `embedding_ids, embedding_rows, embedding_probabilities` |
| 5. 같은 tokenizer의 held-out 문장과 생성 결과를 읽습니다 | `plot_token_budget` | `budget_counts, ['문자', 'byte', 'toy BPE'], budget_costs` |
| 5. 같은 tokenizer의 held-out 문장과 생성 결과를 읽습니다 | `plot_learning` | `training_history` |

### 39 · 대조학습과 InfoNCE — `contrastive_plots.py` — 완료

대칭 InfoNCE·정규화 전체 미분, 실제 무라벨 encoder 학습과 동결 선형 probe 평가. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 길이를 줄이고 방향을 비교합니다 | `plot_panels` | `[vectors, unit, similarity], ['정규화 전 Z', '행별 방향 U', '모든 쌍 내적']` |
| 2. 자기 자신을 빼고 짝의 확률을 높입니다 | `plot_panels` | `[demo_u, demo_map, demo_p], ['A0,A1,B0,B1', 'anchor·양성 인덱스', 'τ=1의 후보 확률']` |
| 3. 점수표의 양쪽 경로를 모두 미분합니다 | `plot_panels` | `[anchor_path, candidate_path, both_paths], ['anchor 행 경로', '후보 열 경로', 'τ=1에서 합한 dU']` |
| 4. 라벨 없이 실제 표현기를 학습합니다 | `plot_panels` | `[view_preview_a, view_preview_b, plan], ['품종 상태·배경: 뷰 A', '같은 상태·새 배경: 뷰 B', 'train·valid·test: N·seed']` |
| 4. 라벨 없이 실제 표현기를 학습합니다 | `plot_curves` | `[history], ['고정 valid 32쌍'], '대칭 InfoNCE · τ=.2'` |
| 5. 고정 표현 위의 선형 평가를 따로 합니다 | `plot_panels` | `[probe_features, probe_weights, probe_scores, probe_prob], ['고정 표현', '학습할 선형 Wq', '품종 점수', '품종 확률']` |
| 5. 고정 표현 위의 선형 평가를 따로 합니다 | `plot_panels` | `[summary], ['초기·학습 후 독립 평가'], [['정확도', '평균', '분산', '쌍 거리']]` |
| 5. 고정 표현 위의 선형 평가를 따로 합니다 | `plot_points` | `[u[:, :2] for u in representations], [test_y, test_y], ['초기 표현 앞 두 축', '학습 표현 앞 두 축']` |

### 51 · Bandit·기대 보상·탐색 — `bandit_plots.py` — 완료

표본평균·epsilon-greedy·pseudo-regret, 48개 seed의 실제 탐색 학습과 고정 정책 독립 평가. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 선택 확률과 보상 확률을 구분합니다 | `plot_reward_expectation` | `true_means, example_policy, reward_contributions` |
| 2. 선택한 행동의 표본평균만 갱신합니다 | `plot_average_update` | `small_rewards, small_averages, small_table` |
| 3. 동률까지 포함한 ε-greedy 확률을 씁니다 | `plot_epsilon_policy` | `tie_values, policy_rows, ['ε=0', 'ε=.3', 'ε=1']` |
| 4. 관측 보상과 pseudo-regret를 분리합니다 | `plot_regret_paths` | `regret_steps, np.cumsum(regret_rewards), regret_pseudo, regret_realized` |
| 5. 같은 예산으로 배우고 정책을 멈춘 뒤 평가합니다 | `plot_bandit_protocol` | `protocol_settings, protocol_table, protocol_policy` |
| 5. 같은 예산으로 배우고 정책을 멈춘 뒤 평가합니다 | `plot_bandit_learning` | `learning_steps, learning_reward_mean, learning_regret_mean, learning_regret_se, experiment_names` |
| 5. 같은 예산으로 배우고 정책을 멈춘 뒤 평가합니다 | `plot_bandit_evaluation` | `experiment_names, exact_evaluation, sampled_evaluation, pseudo_regrets[:, :, -1]` |

### 45 · 작은 decoder GPT 전체 학습 — `decoder_gpt_plots.py` — 완료

causal decoder의 tied·untied 전체 backward, 실제 문자 학습·held-out NLL·greedy 생성·padding 검산. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 앞부분으로 다음 문자 하나를 예측합니다 | `plot_decoder_contract` | `contract_inputs, contract_targets, contract_valid` |
| 2. 미래와 padding을 동시에 막는 attention | `plot_causal_attention` | `attention_scores, attention_allowed, attention_probabilities` |
| 3. 정규화·FFN·잔차의 모든 경로를 미분합니다 | `plot_residual_block` | `residual_x, residual_normalized, residual_output` |
| 4. 전체 decoder를 고정 예산으로 실제 학습합니다 | `plot_loss_path` | `loss_probabilities, loss_values, loss_gradients` |
| 4. 전체 decoder를 고정 예산으로 실제 학습합니다 | `plot_decoder_learning` | `history` |
| 5. held-out NLL과 정답 없는 생성을 구분합니다 | `plot_next_token` | `prefix_ids, prefix_prob[shown], shown_labels, selected_id` |

### 40 · 교사·학생·stop-gradient — `teacher_student_plots.py` — 완료

online·predictor 전체 미분과 고정 타깃, optimizer 뒤 EMA, 상수 표현 대조·교사 속도별 학습·probe. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 타깃의 값은 쓰고 미분 경로는 멈춥니다 | `plot_panels` | `[scalar_paths, pu, tu, row_losses], ['p=2,t=1: L·dp·dt', '정규화 예측', '고정 정규화 타깃', '관측별 반 제곱거리']` |
| 2. 예측기와 학생의 모든 경로를 연결합니다 | `plot_panels` | `[demo_x, demo_h, demo_z, demo_pred], ['두 촬영 입력', '공유 tanh 특징', 'encoder 표현 Z', 'predictor 출력 P']` |
| 3. 학생을 갱신한 뒤 교사를 천천히 옮깁니다 | `plot_panels` | `[old_teacher, new_student, ema_cases], ['이전 교사', 'optimizer 뒤 학생', 'm=0, .9, 1의 새 교사']` |
| 4. 손실이 작아도 모든 과일이 같아질 수 있습니다 | `plot_panels` | `[constant, separate, statistics], ['상수 표현', '구분되는 표현', '각 경우: 평균·평균분산']` |
| 5. 교사 속도별로 실제 학습하고 독립 평가합니다 | `plot_panels` | `[preview_x, preview_pair, plan], ['뷰 A: 상태·배경', '뷰 B: 같은 상태·새 배경', 'train·valid·test: N·seed']` |
| 5. 교사 속도별로 실제 학습하고 독립 평가합니다 | `plot_curves` | `histories, [f'EMA m={m}' for m in momenta], '고정 valid alignment'` |
| 5. 교사 속도별로 실제 학습하고 독립 평가합니다 | `plot_panels` | `[report], ['초기와 교사 속도별 독립 평가'], [['정확도', '평균', '분산', '다른 거리']]` |

### 52 · MDP·return·Bellman·동적 계획 — `mdp_plots.py` — 완료

전이표·정확한 정책 가치·value/policy iteration, 정책 전수열거와 종료·수집 제한별 rollout 비교. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 전이와 보상을 같은 세 축에 적습니다 | `plot_mdp_transition` | `start_transition, start_rewards, start_expected_reward` |
| 2. 보상을 할인해 더하고 마지막 경계를 표시합니다 | `plot_return_path` | `return_rewards, return_powers, return_contributions, return_totals` |
| 3. 정책의 가치를 Bellman 방정식으로 풉니다 | `plot_bellman_backup` | `backup_contributions, backup_q, policy[0], backup_v` |
| 3. 정책의 가치를 Bellman 방정식으로 풉니다 | `plot_policy_solution` | `policy_transition, policy_immediate, exact_policy_value` |
| 4. 행동 평균을 최댓값으로 바꾸어 계획합니다 | `plot_policy_improvement` | `exact_policy_value, improvement_q, policy, improvement_policy` |
| 4. 행동 평균을 최댓값으로 바꾸어 계획합니다 | `plot_value_iterations` | `value_history, enumerated_optimum, pi_history` |
| 5. rollout을 정확한 유한·무한 목적과 비교합니다 | `plot_rollout_boundary` | `boundary_states, boundary_rewards, boundary_terminal, boundary_bootstrap` |
| 5. rollout을 정확한 유한·무한 목적과 비교합니다 | `plot_rollout_comparison` | `horizons, finite_means, finite_ses, finite_exact, bootstrap_means, bootstrap_ses, exact_policy_value[0]` |

### 46 · 생성 규칙과 KV cache — `generation_kv_cache_plots.py` — 완료

temperature·top-k·top-p, 실제 학습 모델의 prefix별 full/cache 동등성·같은 난수 생성·시간·저장량. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 같은 logits에서 다른 생성 분포를 만듭니다 | `plot_temperature` | `temperature_logits, temperature_rows, temperatures` |
| 2. top-k와 top-p는 남길 후보를 다르게 정합니다 | `plot_truncation` | `truncation_original, truncation_keep, truncation_filtered` |
| 3. prefix 전체를 처리해 K와 V를 저장합니다 | `plot_prefill` | `prefill_features, prefill_k, prefill_v` |
| 4. 새 token의 query 하나만 과거를 읽습니다 | `plot_cached_step` | `cache_keys, new_query, new_weights, new_mixed` |
| 5. 같은 난수로 생성하고 비용을 따로 측정합니다 | `plot_cache_budget` | `budget_lengths, score_counts, cache_bytes` |
| 5. 같은 난수로 생성하고 비용을 따로 측정합니다 | `plot_cache_timing` | `budget_lengths, timing_results` |

### 41 · 분산·공분산으로 표현 제약하기 — `variance_covariance_plots.py` — 완료

정렬·표준편차 하한·비대각 공분산의 전체 미분, 상수·복제 대조와 목적항별 실제 학습·probe. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 관측 축을 중심화하고 특징끼리 비교합니다 | `plot_panels` | `[stats_z, stats_center, stats_cov], ['복제된 두 특징', '특징별 중심화', '표본 공분산']` |
| 2. 특징마다 표준편차 하한을 둡니다 | `plot_panels` | `[variance_table[:, :2], variance_table[:, 2:]], ['상수·작음·충분: 각 std', '같은 순서: 각 hinge 비용']` |
| 3. 복제 차원은 비대각 공분산에서 드러납니다 | `plot_panels` | `comparison_cov + comparison_off, ['복제 특징 공분산', '비상관 특징 공분산', '복제의 비대각', '비상관의 비대각']` |
| 4. 세 항을 합하고 공유 표현기까지 미분합니다 | `plot_panels` | `[term_a, term_b, term_table, weighted_table], ['뷰 A', '뷰 B', '정렬·분산·공분산 항', '가중치 5·5·1 적용']` |
| 5. 목적항을 빼 보며 실제 표현을 평가합니다 | `plot_panels` | `[weights_table, plan, collapsed_terms], ['조건별 정렬·분산·공분산 가중치', 'train·valid·test: N·seed', '상수 쌍의 세 비용']` |
| 5. 목적항을 빼 보며 실제 표현을 평가합니다 | `plot_curves` | `[h[:, [0, 1]] for h in histories], [s[0] for s in settings], '공통 validation 정렬 항'` |
| 5. 목적항을 빼 보며 실제 표현을 평가합니다 | `plot_panels` | `[report[:, :3], report[:, 3:]], ['초기·정렬·분산추가·전체: 정확도·평균·최소std', '같은 순서: 공분산 비용·다른 거리']` |

### 36 · Flow matching과 ODE 생성 — `flow_matching_plots.py` — 완료

조건부·주변 속도 목적의 연결, 시간 조건부 MLP 전체 미분·실제 학습, 정확 Gaussian 기준과 같은 NFE의 ODE 생성. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 잡음 점과 자료 점을 직선으로 잇습니다 | `plot_straight_paths` | `demo_paths, demo_mid, demo_u` |
| 2. 같은 위치의 여러 속도를 평균합니다 | `plot_conditional_average` | `conditional_targets, conditional_mean, decomposition` |
| 3. 시간까지 입력으로 받아 전체를 미분합니다 | `plot_velocity_network` | `network_inputs, network_hidden, network_prediction, network_upstream` |
| 4. 이동을 시뮬레이션하지 않고 실제 학습합니다 | `plot_flow_data` | `source_preview, target_preview, training_plan` |
| 4. 이동을 시뮬레이션하지 않고 실제 학습합니다 | `plot_flow_learning` | `history` |
| 5. 학습한 속도를 적분하고 오차를 나눕니다 | `plot_ode_steps` | `one_step, step_terms` |
| 5. 학습한 속도를 적분하고 오차를 나눕니다 | `plot_solver_error` | `solver_nfes, solver_errors, ['Euler', 'Heun']` |
| 5. 학습한 속도를 적분하고 오차를 나눕니다 | `plot_flow_fields` | `grid_points, [grid_exact, grid_learned], ['t=.5 정답 주변 속도', 't=.5 학습한 속도']` |
| 5. 학습한 속도를 적분하고 오차를 나눕니다 | `plot_flow_samples` | `[test_data, euler_path[-1], heun_path[-1]], ['독립 test 관측', 'Euler40 · NFE40', 'Heun20 · NFE40'], [None, euler_path[:, :8], heun_path[:, :8]]` |

### 47 · encoder 목적과 BERT식 MLM — `masked_language_model_plots.py` — 완료

양방향 문맥·80/10/10 corruption·선택 위치 CE 전체 미분, 실제 MLM·AR 학습과 held-out 빈칸 복원. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 같은 token열에서도 읽을 수 있는 위치가 다릅니다 | `plot_context_masks` | `context_tokens, bidirectional, causal` |
| 2. 선택한 위치와 바꾼 위치를 구분합니다 | `plot_corruption` | `corruption_clean, corruption_selected, corruption_hand` |
| 3. 선택 CE를 전체 encoder로 역전파합니다 | `plot_selected_loss` | `selected_probabilities, selected_positions, selected_delta` |
| 4. 같은 정답이어도 주는 정보가 다릅니다 | `plot_information_access` | `information_mlm, information_ar, information_answer` |
| 4. 같은 정답이어도 주는 정보가 다릅니다 | `plot_mlm_learning` | `mlm_history` |
| 5. 빈칸 복원은 자유 생성과 다른 사용법입니다 | `plot_fill` | `fill_original, fill_masked, fill_prediction` |

### 53 · Monte Carlo·TD·다단계 타깃 — `mc_td_plots.py` — 완료

완료 return·TD·n-step·유한 lambda, 경로 전수열거의 정확한 편향·분산과 32 seed 실제 가치 학습. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 끝까지 관측한 return으로 가치를 배웁니다 | `plot_mc_path` | `path_states, path_rewards, path_returns` |
| 2. 한 전이와 현재 가치로 TD 오차를 만듭니다 | `plot_td_update` | `td_current, td_next, td_rewards, td_targets, td_errors` |
| 3. n-step과 유한 λ-return을 연결합니다 | `plot_lambda_weights` | `step_targets, lambda_weights, lambda_contributions` |
| 4. 타깃의 편향·분산을 모든 경로에서 계산합니다 | `plot_target_distribution` | `path_probabilities, example_targets, example_moments, method_names` |
| 5. 같은 episode에서 네 가지 가치 학습을 비교합니다 | `plot_prediction_protocol` | `plan_settings, plan_values, plan_targets, plan_updated` |
| 5. 같은 episode에서 네 가지 가치 학습을 비교합니다 | `plot_prediction_learning` | `episode_numbers, mean_rmse, se_rmse, method_names` |
| 5. 같은 episode에서 네 가지 가치 학습을 비교합니다 | `plot_prediction_final` | `method_names, rmse_records[:, :, -1], mean_final_values, truth[:3]` |

### 42 · masked reconstruction — `masked_reconstruction_plots.py` — 완료

visible patch encoder·위치 decoder·masked-only 화소 손실 전체 미분, mask 비율별 실제 복원·평균 이미지 기준·동결 probe. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 이미지를 patch로 나누고 일부 위치만 고릅니다 | `plot_images` | `[small_image[0], small_shown[0], small_mask.reshape(2, 2)], ['원본 4×4 예시', 'visible 위치0·3', 'patch mask · 1=가림']` |
| 2. 보이는 조각을 인코딩하고 위치별로 복원합니다 | `plot_panels` | `[demo_visible[0], demo_h[0], demo_g, demo_r[0]], ['선택한 화소만', '위치를 더한 특징', 'visible 평균 context', '네 위치 decoder 특징']` |
| 3. 가린 화소 수로만 오차를 평균합니다 | `plot_panels` | `[loss_target[0], loss_pred[0], masked_residual[0], hand_grad[0]], ['복원 타깃', '예측', '가린 위치의 오차만', '가린 화소2개로 나눈 미분']` |
| 4. 위치 선택을 거슬러 전체 미분을 검산합니다 | `plot_panels` | `[scatter_ids, scatter_values.reshape(2, 2), scatter_result, mean_distributed[0]], ['두 이미지의 ids', '위치마다 생긴 기여', '위치2는2+3 누적', '평균의 K=3 역전파']` |
| 5. mask 비율과 복원·표현 평가를 분리합니다 | `plot_images` | `[vertical, horizontal, vertical * (1 - mask_pixels), mask_pixels], ['세로 띠: 방향0', '가로 띠: 방향1', '세로 띠의 일부만', '화소로 펼친 mask']` |
| 5. mask 비율과 복원·표현 평가를 분리합니다 | `plot_curves` | `histories, ['train mask25%', 'train mask50%', 'train mask75%'], '공통 50% validation mask 반 MSE'` |
| 5. mask 비율과 복원·표현 평가를 분리합니다 | `plot_panels` | `[report], ['초기·mask25·50·75%의 동일 test 평가'], [['반 MSE', '정확도', '평균', '분산']]` |
| 5. mask 비율과 복원·표현 평가를 분리합니다 | `plot_images` | `[test_images[display_id], masked_pixels, reconstructions[1], completed_image], ['고정 test0 원본', '같은 test mask 입력', '50%학습 모델 원시 예측', 'visible 원본+masked 예측']` |

### 48 · SFT·전이 학습·LoRA — `sft_lora_plots.py` — 완료

답변·EOS 손실, 고정 base를 통과하는 LoRA 전체 미분·merge, full/frozen/LoRA 실제 적응·기존 과제 평가. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. prompt를 읽되 답변 위치에서만 손실을 냅니다 | `plot_response_mask` | `response_inputs_hand, response_targets_hand, response_weights_hand` |
| 2. 큰 행렬을 고정하고 작은 두 행렬을 배웁니다 | `plot_lora_weights` | `lora_base_hand, lora_delta_hand, lora_effective_hand` |
| 3. 고정 가중치도 입력의 미분 경로에 남습니다 | `plot_frozen_gradient` | `gradient_upstream, gradient_base_path, gradient_adapter_path` |
| 4. 같은 base에서 full·frozen·LoRA를 비교합니다 | `plot_trainable_budget` | `trainable_counts, ['frozen', 'full', 'LoRA']` |
| 4. 같은 base에서 full·frozen·LoRA를 비교합니다 | `plot_adaptation_learning` | `[adaptation_histories[method] for method in methods], methods` |
| 5. merge의 동등성과 정답 없는 답변을 확인합니다 | `plot_merge_logits` | `unmerged_logits[0, -2:, :4], merged_logits[0, -2:, :4], merge_difference[0, -2:, :4]` |

### 43 · JEPA와 잠재 타깃 예측 — `jepa_plots.py` — 완료

분리된 context·target과 위치 조건 잠재 예측 전체 미분, optimizer 뒤 EMA·실제 화소/잠재 학습·동결 probe. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. context와 target 영역을 겹치지 않게 고릅니다 | `plot_images` | `[preview, context_image, target_image, pixel_region], ['전체 이미지', '학생 context만', '교사가 읽는 target 영역', 'M=1: 예측할 영역']` |
| 2. 같은 위치를 화소 또는 표현으로 맞힙니다 | `plot_panels` | `[demo_pixels, demo_latent, demo_units], ['두 patch의 원본4화소', '작은 teacher의2특징', '실제 폭·한 target patch의 2D 분모']` |
| 3. 학생의 전 경로만 미분하고 타깃은 고정합니다 | `plot_panels` | `[path_input[0], path_visible[0], path_scatter[0]], ['원래 네 위치', '학생이 본0·3 위치', '학생 입력 미분:1·2 위치는0']` |
| 4. EMA와 위치만 남는 붕괴를 구분합니다 | `plot_panels` | `[position_only[0], position_only[1], position_statistics, ema_example], ['이미지0: 위치만 특징', '이미지1: 같은 위치 특징', '전체 분산·위치별 이미지 분산', '이전교사·새학생·m=.9교사']` |
| 5. 같은 자료에서 두 목적을 실제로 학습합니다 | `plot_panels` | `[budget, view_sizes, same_state], ['train·valid·test: N·seed', 'pixel·latent: context·target·출력폭', '두 조건: 초기·배치·영역 seed']` |
| 5. 같은 자료에서 두 목적을 실제로 학습합니다 | `plot_objectives` | `[pixel_history, latent_history], ['pixel · 원본 화소 단위', 'latent · 현재 teacher 단위']` |
| 5. 같은 자료에서 두 목적을 실제로 학습합니다 | `plot_panels` | `[report], ['초기·pixel·latent의 공통 독립 평가'], [['정확도', '평균', '분산', '다른 거리']]` |
| 5. 같은 자료에서 두 목적을 실제로 학습합니다 | `plot_panels` | `[content_variances], ['teacher의 위치별 이미지 분산'], [['초기 teacher', 'EMA teacher']]` |
| 5. 같은 자료에서 두 목적을 실제로 학습합니다 | `plot_images` | `[test_images[0], shown, pixel_prediction_image, combined_image], ['고정 test0 원본', '두 목적의 같은 context', 'pixel 모델 원시 예측', 'context 원본+target 예측']` |

### 63 · 논문에서 수식·구현·실험으로 — `paper_reproduction_plots.py` — 완료

Flow Matching 논문의 한 주장 재현, 정확 Gaussian 기울기·최소제곱 기준과 3 seed 실제 학습·같은 초기값 ODE 생성. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 논문의 주장과 재현할 범위를 먼저 고정합니다 | `plot_scope_table` | `scope_values` |
| 2. 기호를 배열과 갱신 대상에 대응시킵니다 | `plot_specification` | `np.column_stack([spec_x, spec_t]), spec_design, spec_outputs` |
| 3. 같은 구현을 믿기 전에 독립 기준을 붙입니다 | `plot_independent_checks` | `check_design, check_residual, check_gradient` |
| 4. 한 요소를 바꾼 실제 학습과 기준해를 기록합니다 | `plot_reproduction_plan` | `plan` |
| 4. 한 요소를 바꾼 실제 학습과 기준해를 기록합니다 | `plot_reproduction_learning` | `histories[:2], ['seed6301 시간 없음', 'seed6301 시간2차']` |
| 5. 생성 그림과 근거의 한계를 함께 남깁니다 | `plot_exact_paths` | `path_times, exact_paths` |
| 5. 생성 그림과 근거의 한계를 함께 남깁니다 | `plot_paired_evidence` | `metrics, ['시간 없음', '시간2차']` |
| 5. 생성 그림과 근거의 한계를 함께 남깁니다 | `plot_paired_samples` | `display_initial, display_samples, display_reference, ['seed6301 시간 없음', 'seed6301 시간2차']` |

### 49 · 선호 쌍에서 DPO 목적 만들기 — `dpo_plots.py` — 완료

EOS까지의 sequence 로그확률·고정 reference DPO 전체 미분, 실제 SFT/DPO 학습과 후보 순위·절대 확률·자유 생성 평가. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. 답변 두 개의 점수 차이를 확률로 읽습니다 | `plot_preference_probability` | `score_differences, preference_probabilities` |
| 2. 답변의 로그확률은 token 축으로 더합니다 | `plot_sequence_sum` | `selected_log_hand, response_mask_hand, sequence_sum_hand` |
| 3. 고정 기준 대비 선호 변화에 미분합니다 | `plot_reference_margin` | `policy_hand, reference_hand, margin_hand` |
| 4. 동일한 출발점에서 SFT와 DPO를 실제 학습합니다 | `plot_objective_signals` | `sft_signal_hand, dpo_signal_hand` |
| 4. 동일한 출발점에서 SFT와 DPO를 실제 학습합니다 | `plot_dpo_learning` | `[preference_histories[name] for name in ['SFT', 'DPO']], ['chosen SFT', 'DPO']` |
| 5. 선호 후보의 순위와 자유 생성을 따로 봅니다 | `plot_absolute_probabilities` | `absolute_log_probs, list(models)` |

### 54 · SARSA·Expected SARSA·Q-learning — `control_plots.py` — 완료

실제 다음 행동·기대·max 타깃과 importance support, 32 seed 제어 학습·탐색 유지/제거의 정확한 정책 가치. 수식·학습·평가 및 아래 그림 입력은 본문에서 계산하고 보조 파일은 배치·색·글씨와 그리기만 맡습니다.

| 절의 질문 | 그림 함수 | 본문에서 전달하는 계산 결과 |
|---|---|---|
| 1. Q 표에서 실제 행동정책을 만듭니다 | `plot_control_environment` | `example_next, example_reward, example_policy` |
| 2. SARSA는 실제로 할 다음 행동을 사용합니다 | `plot_sarsa_targets` | `next_q, next_probabilities, sarsa_targets, np.array([expected_target])` |
| 3. Q-learning은 다음 최대 행동을 타깃에 넣습니다 | `plot_q_target_comparison` | `next_q, comparison_targets, comparison_updated` |
| 4. 다른 정책의 경로 평균에는 확률비가 필요합니다 | `plot_importance_paths` | `np.column_stack([behavior_paths, target_paths]), path_ratios, path_returns, path_contributions` |
| 5. 실제 Q 학습과 두 배포 규약을 비교합니다 | `plot_control_protocol` | `protocol_settings, protocol_policies, protocol_values` |
| 5. 실제 Q 학습과 두 배포 규약을 비교합니다 | `plot_control_learning` | `checkpoints, mean_values, se_values, methods` |
| 5. 실제 Q 학습과 두 배포 규약을 비교합니다 | `plot_control_final` | `methods, final_values, observed_returns` |

## 4. 작업 순서

1. 계산 셀을 본문에 추가합니다. 결과 배열의 shape를 주석으로 적습니다.
2. 조합 함수를 해당 `*_plots.py`에 추가하고 `concept_plots` 부품만 씁니다.
3. 새 커널에서 노트북을 위부터 끝까지 실행하고 그림을 눈으로 확인합니다. 화살표·제목 겹침·글리프 경고를 봅니다.
4. 이 문서의 표에서 상태를 계획 → 완료로 바꿉니다.
