# numpy로 밑바닥부터

퍼셉트론에서 Transformer까지 numpy만으로 구현한다. 자동미분을 쓰지 않고
**모든 역전파를 손으로 유도해서** 코드로 옮긴다.

순서는 코세라 [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
(Andrew Ng, deeplearning.ai) 5개 코스를 따라가고, 그 앞에 퍼셉트론 서장을 하나 둔다.

의존성은 numpy뿐이다 (테스트에 pytest, 그림에 matplotlib).

## 시작하기

```bash
.venv/bin/jupyter lab notebooks/00_perceptron.ipynb
```

퍼셉트론으로 논리 게이트를 만들고, XOR이 단층으로 **불가능함을 증명**한 뒤,
층을 쌓아 해결한다. 은닉층이 왜 필요한지에 대한 답이 여기 다 들어 있다.

```bash
.venv/bin/python -m pytest
```

새 환경에서 시작한다면:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m ipykernel install --user --name dl-scratch --display-name "Python (from-scratch)"
```

## 작업 방식 — 노트북에서 이해하고, `src/`에 정착시킨다

라이브러리에 바로 쓰지 않는다. 노트북에서 이해가 끝난 것만 옮긴다.

| 단계 | 하는 일 | 위치 | 도구 |
|---|---|---|---|
| 1 | 미분을 유도한다 | 노트북 마크다운 | |
| 2 | 클래스 없이 함수로 짜 본다 | 노트북 | |
| 3 | **수치미분으로 검증한다** | 노트북 | `check_function` |
| 4 | `Layer`로 정리해 옮긴다 | `src/` | |
| 5 | 회귀 테스트를 붙인다 | `tests/` | `check_layer` |
| 6 | 학습이 되는지 확인한다 | 노트북 → `examples/` | |
| 7 | 유도를 정리해 남긴다 | `docs/derivations/` | |

3번을 건너뛰면, 나중에 학습이 안 될 때 **유도 오류인지 하이퍼파라미터 문제인지**
구분할 방법이 사라진다.

3번과 5번은 같은 검증을 다른 시점에 하는 것이다. `check_function`은 아직
함수 쌍일 때, `check_layer`는 `Layer`가 된 뒤.

```python
import numpy as np
from src import check_function, check_layer

# 노트북에서 — 아직 함수 쌍일 때
check_function(lambda x: np.maximum(x, 0),        # forward(x) -> y
               lambda dout, x: dout * (x > 0),    # backward(dout, x) -> dx
               x)

# src/ 로 옮긴 뒤 — Layer가 되었을 때
check_layer(my_layer, x, verbose=True)
```

올바른 유도는 상대오차 `1e-9` 언저리, 틀린 유도는 `1e-1` 이상이 나온다.
9자릿수가 벌어지므로 애매한 판정이 없다.

## 구조

```
notebooks/         ← 여기서 이해한다
  00_perceptron            퍼셉트론 · 논리 게이트 · XOR과 다층 (완료)
  01_logistic_regression   시그모이드 · 로그 손실 · 계산 그래프 · 벡터화 (완료)
  02_...                   이후 코세라 순서대로
src/               ← 이해가 끝난 것만 여기로
  functional.py    sigmoid                    (노트북 01에서 정착)
  losses.py        binary_cross_entropy       (노트북 01에서 정착)
  utils/
    gradcheck.py   수치미분 검증기 ← 이 프로젝트의 안전망
    plotting.py    setup_plots() — 그림 설정 일괄 적용, 학습 곡선·결정경계·gradcheck 막대
  data/            DataLoader, 장난감 데이터셋, MNIST 로더
tests/             src/ 에 들어간 것들의 회귀 테스트
examples/          end-to-end 학습 스크립트
docs/
  CURRICULUM.md    전체 로드맵 ← 다음에 무엇을 할지는 여기
  derivations/     미분 유도 노트
```

`src/`가 거의 비어 있는 것이 정상이다. 노트북을 진행하며 채워 나간다.
지금까지 정착한 것은 `sigmoid` 와 `binary_cross_entropy` 뿐이다.

## 로드맵

| | 노트북 | 내용 |
|---|---|---|
| 서장 | `00_perceptron` ✅ | 퍼셉트론, 논리 게이트, XOR의 불가능성, 다층 해법 |
| **C1** | `01_logistic_regression` ✅ | 로지스틱 회귀를 신경망으로, 계산 그래프, 벡터화 |
| | `02_shallow_nn` | 은닉층 1개, 활성화 함수, 왜 비선형이어야 하는가 |
| | `03_deep_nn` | L층 일반화 — `Layer` 추상화가 여기서 나온다 |
| **C2** | `04_regularization` | L2, 드롭아웃, 초기화, 기울기 소실, gradient checking |
| | `05_optimization` | 미니배치, Momentum, RMSProp, Adam, 학습률 감쇠 |
| | `06_batchnorm_softmax` | BatchNorm, softmax 다중분류 |
| **C3** | `07_ml_strategy` | 편향·분산 분해, 오차 분석, 학습곡선 (진단 도구를 numpy로) |
| **C4** | `08_cnn_foundations` | im2col, Conv2d, 풀링 |
| | `09_cnn_architectures` | LeNet, VGG, ResNet 잔차 블록, MobileNet |
| | `10_object_detection` | IoU, NMS, YOLO 손실, U-Net |
| | `11_face_and_style` | Triplet loss, Gram 행렬, 입력에 대한 경사하강 |
| **C5** | `12_rnn` | RNN, BPTT, GRU, LSTM, 문자 단위 언어모델 |
| | `13_word_embeddings` | word2vec, GloVe, 유추, 편향 제거 |
| | `14_attention` | seq2seq, 빔 서치, BLEU, Bahdanau 어텐션 |
| | `15_transformer` | self-attention, MHA, LayerNorm, Transformer 블록 |

전부 numpy로 구현한다. C4 후반과 C5W2는 원래 GPU를 전제하는 과제라
각 주차에 CPU에서 끝나는 축소 설정을 명시해 두었다 — 구조와 역전파는 그대로 구현하고
데이터 크기만 줄인다. 자세한 내용은 [docs/CURRICULUM.md](docs/CURRICULUM.md).

## 노트북 규약

모든 노트북의 **첫 셀에서 `setup_plots()` 를 한 번** 부른다. 폰트, 크기, 해상도,
격자, 색 순서, 여백이 전부 잡히므로 이후 셀에서는 그림 내용만 쓰면 된다.
`ax.grid(...)`, `plt.tight_layout()`, 눈금 포매터를 셀마다 부를 필요가 없다.

```python
from src.utils import COLOR_NEG, COLOR_POS, setup_plots
setup_plots()
```

클래스를 색으로 구분할 때는 `COLOR_POS` / `COLOR_NEG` 를 쓴다.
색 코드를 셀마다 적어 두면 나중에 팔레트를 바꿀 때 전부 찾아다녀야 한다.

폰트 선택이 조금 까다롭다. `$...$` 가 섞인 문자열은 **전체가 mathtext 엔진으로**
그려지는데, 이 엔진은 글리프 단위 폴백을 하지 않고 `font.family` 의 첫 폰트만 쓴다.
그래서 첫 폰트가 **한글과 유니코드 마이너스(U+2212)를 모두** 갖고 있어야
`"입력 공간 $(x_1,x_2)$ — 직선으로 못 가름"` 같은 제목과 로그 눈금의 $10^{-10}$ 이
동시에 제대로 나온다. `setup_plots()` 가 설치된 폰트의 글리프 커버리지를
실제로 확인해서 고른다.

## 설계 결정

**`float64`를 쓴다.** `float32`에서는 수치미분의 상대오차가 `1e-3`까지 커져
실제 유도 오류와 구분되지 않는다. 속도보다 검증 가능성을 택했다.

**기울기는 `=`가 아니라 `+=`로 누적한다.** 그래서 매 스텝 `zero_grad()`가 필요하다.
CNN의 가중치 공유, RNN의 시간축 전개, Transformer의 embedding 가중치 공유가
모두 "같은 파라미터를 한 번의 forward에서 여러 번 쓰는" 경우인데,
누적이어야 그 기여들이 제대로 더해진다.

**검증기는 덕 타이핑으로 동작한다.** `check_layer`는 `forward`/`backward`/
`zero_grad`/`named_parameters` 네 가지만 요구한다. 그래서 `Layer` 추상화를
노트북 03번에서 직접 설계해 만들 때까지, 검증기가 먼저 존재할 수 있다.
`tests/test_gradcheck.py`의 `_MiniLayer`가 그 인터페이스의 전부다.
