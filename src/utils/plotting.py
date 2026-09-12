"""노트북에서 결과를 눈으로 확인하기 위한 그림 도구.

학습 곡선과 결정경계는 숫자만 봐서는 놓치는 것을 보여준다.
정확도가 같아도 결정경계 모양은 전혀 다를 수 있다.

노트북 첫 셀에서 `setup_plots()` 를 한 번 부르면 폰트·크기·격자·색이 전부
설정된다. 이후 셀에서는 그림 내용만 쓰면 된다.
"""

from __future__ import annotations

import numpy as np

# 클래스나 대비를 나타낼 때 쓰는 의미 있는 색. 노트북에서 import 해서 쓴다.
COLOR_POS = "#2a9d8f"   # 양성/참/클래스 1
COLOR_NEG = "#e63946"   # 음성/거짓/클래스 0
COLOR_ACCENT = "#1d3557"
COLOR_MUTED = "#8d99ae"

# 선 여러 개를 그릴 때 도는 기본 색 순서
_CYCLE = ["#264653", COLOR_POS, "#e9c46a", "#f4a261", COLOR_NEG, COLOR_ACCENT]

# 한글 폰트 후보. 앞쪽일수록 우선. 실제 선택은 글리프 커버리지를 보고 결정한다.
# NanumSquare Neo 는 글리프는 충분하지만 맥에 Bold(700) 웨이트만 설치되어
# 본문까지 굵게 나오므로 뒤로 뺐다.
_FONT_CANDIDATES = (
    "Arial Unicode MS", "Noto Sans CJK KR", "Noto Sans KR", "Apple SD Gothic Neo",
    "AppleGothic", "NanumGothic", "Malgun Gothic", "NanumSquare Neo",
)

# 한글과 유니코드 마이너스가 있으면 음수 눈금을 정확히 표시할 수 있습니다.
# 아래첨자는 mathtext로 조합하므로 U+2081 지원 여부와 분리합니다.
_NEEDED = (ord("가"), 0x2212)


def _covers(path: str, codepoints) -> bool:
    """폰트 파일이 해당 글리프들을 모두 갖고 있는지. matplotlib 내장 FreeType 사용."""
    from matplotlib.ft2font import FT2Font

    try:
        face = FT2Font(path)
        return all(face.get_char_index(cp) != 0 for cp in codepoints)
    except Exception:
        return False


def _pick_font() -> tuple[list[str], str | None, bool]:
    """쓸 폰트 목록과 고른 이름, 그리고 완전 지원 여부를 돌려준다.

    왜 커버리지를 확인하는가
    ------------------------
    `$...$` 가 섞인 문자열은 **전체가 mathtext 엔진으로** 그려지고, 그 엔진은
    글리프 단위 폴백을 하지 않고 `font.family` 의 **첫 폰트만** 쓴다. 그래서

    - 첫 폰트에 한글이 없으면  → "입력 공간 $(x_1,x_2)$" 의 한글이 두부가 되고
    - 첫 폰트에 U+2212 가 없으면 → 로그 눈금의 $10^{-10}$ 이 경고를 쏟는다

    둘 다 피하려면 **첫 폰트가 한글과 U+2212 를 모두 가져야** 한다.
    맥에서는 Arial Unicode MS 나 NanumSquare Neo 가 해당한다.
    그런 폰트가 없으면 한글 우선으로 물러난다 (두부보다는 경고가 낫다).
    """
    import matplotlib.font_manager as fm

    available: dict[str, str] = {}
    for f in fm.fontManager.ttflist:
        available.setdefault(f.name, f.fname)

    # 기존 캐시에 한글 폰트가 없으면 설치된 폰트를 다시 찾아 등록한다.
    if not any(name in available for name in _FONT_CANDIDATES):
        for path in sorted(fm.findSystemFonts()):
            if _covers(path, (ord("가"),)):
                fm.fontManager.addfont(path)
        for f in fm.fontManager.ttflist:
            available.setdefault(f.name, f.fname)

    # 1순위: 필요한 글리프를 모두 가진 폰트
    for name in _FONT_CANDIDATES:
        if name in available and _covers(available[name], _NEEDED):
            return [name, "DejaVu Sans"], name, True

    # 2순위: 한글만 되는 폰트. mathtext 안의 마이너스는 경고가 날 수 있다.
    for name in _FONT_CANDIDATES:
        if name in available and _covers(available[name], (ord("가"),)):
            return [name, "DejaVu Sans"], name, False

    return ["DejaVu Sans"], None, False


def setup_plots(figsize: tuple[float, float] = (7.0, 4.0), dpi: int = 110,
                grid: bool = True, verbose: bool = False) -> str | None:
    """노트북 첫 셀에서 한 번 부른다. 이후 모든 그림에 적용된다.

    설정하는 것: 한글 폰트, 기본 크기와 해상도, 격자, 색 순서, 테두리, 여백.
    덕분에 개별 셀에서는 `plt.subplots()` 하고 그림 내용만 쓰면 된다.
    `ax.grid(...)`, `plt.tight_layout()`, 눈금 포매터를 따로 부를 필요가 없다.

    폰트 선택 기준은 `_pick_font` 의 설명을 참고.

    Returns:
        고른 한글 폰트 이름 (없으면 None).
    """
    import matplotlib

    family, font, full = _pick_font()

    matplotlib.rcParams.update({
        # 폰트 — 선택 근거는 _pick_font 참고
        "font.family": family,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": full,   # 완전 지원 폰트면 진짜 마이너스를 쓴다
        # 크기와 해상도
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "savefig.bbox": "tight",
        # 셀마다 plt.tight_layout() 을 부르지 않아도 되도록 전역으로 켠다
        "figure.autolayout": True,
        "figure.constrained_layout.use": False,
        # 격자 — 데이터 뒤에 깔리도록
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        # 테두리는 왼쪽·아래만
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 색과 선
        "axes.prop_cycle": matplotlib.cycler(color=_CYCLE),
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "image.cmap": "gray",
    })

    if verbose:
        if font is None:
            print("한글 폰트를 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다.")
        elif not full:
            print(f"폰트: {font} (한글만 지원). 로그 눈금에서 마이너스 경고가 날 수 있습니다.")
        else:
            print(f"폰트: {font}")
    return font


def plot_history(history: dict, metric_name: str = "정확도", ax=None):
    """학습/검증 손실과 지표를 나란히 그린다."""
    import matplotlib.pyplot as plt

    has_val = bool(history.get("val_loss"))
    has_metric = bool(history.get("train_metric")) and not np.isnan(history["train_metric"][0])

    ncols = 2 if has_metric else 1
    if ax is None:
        _, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4))
        axes = np.atleast_1d(axes)  # 단일 축도 배열로 다뤄 아래 코드를 통일
    else:
        axes = np.atleast_1d(ax)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="학습")
    if has_val:
        axes[0].plot(epochs, history["val_loss"], label="검증")
    axes[0].set_xlabel("에폭"); axes[0].set_ylabel("손실"); axes[0].set_title("손실")
    axes[0].legend()

    if has_metric:
        axes[1].plot(epochs, history["train_metric"], label="학습")
        if history.get("val_metric"):
            axes[1].plot(epochs, history["val_metric"], label="검증")
        axes[1].set_xlabel("에폭"); axes[1].set_ylabel(metric_name); axes[1].set_title(metric_name)
        axes[1].legend()

    return axes


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, ax=None,
                           resolution: int = 200, title: str | None = None):
    """2차원 입력 분류기의 결정경계를 그린다.

    격자의 모든 점을 모델에 통과시켜 예측 클래스를 색으로 칠한다.
    입력이 2차원일 때만 쓸 수 있다.

    model은 Layer여도 되고 (X) -> 로짓 형태의 함수여도 된다.
    노트북에서 클래스 없이 손으로 짠 모델도 그대로 그릴 수 있게 하기 위해서다.
    """
    import matplotlib.pyplot as plt

    if X.shape[1] != 2:
        raise ValueError(f"2차원 입력만 그릴 수 있습니다 (받은 차원: {X.shape[1]})")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    pad = 0.15 * (X.max(axis=0) - X.min(axis=0))
    x_min, y_min = X.min(axis=0) - pad
    x_max, y_max = X.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    is_layer = hasattr(model, "eval") and hasattr(model, "training")
    if is_layer:
        was_training = model.training
        model.eval()
    Z = np.argmax(model(grid), axis=1).reshape(xx.shape)
    if is_layer:
        model.train(was_training)

    ax.contourf(xx, yy, Z, alpha=0.25, levels=np.arange(Z.max() + 2) - 0.5, cmap="brg")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=14, cmap="brg", edgecolors="k", linewidths=0.3)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    if title:
        ax.set_title(title)
    return ax


def plot_gradcheck(errors: dict[str, float], tol: float = 1e-7, ax=None):
    """gradcheck 상대오차를 로그 눈금 막대로. 허용선과의 거리가 한눈에 보인다."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.2 * len(errors)), 3.2))
    names = list(errors)
    values = [max(errors[n], 1e-16) for n in names]
    colors = [COLOR_POS if v <= tol else COLOR_NEG for v in values]
    ax.bar(names, values, color=colors)
    ax.axhline(tol, color="k", ls="--", lw=1, label=f"허용치 {tol:.0e}")
    ax.set_yscale("log"); ax.set_ylabel("상대오차"); ax.legend()
    return ax
