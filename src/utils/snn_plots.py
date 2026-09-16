"""07b SNN 노트북의 개념 그림과 구조 도식.

막전위·스파이크·부호화·전류·손실 지형·BPTT 표·학습 결과는 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
"""

import math

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes

ACTIVE_FILL = "#fbe9c6"   # 켜진 입력의 행
LIGHT_GRAY = "#c4c8cc"    # 설정 비교에서 본문이 쓰지 않는 값 (옅은 쪽)
DARK_GRAY = "#6b7178"     # 설정 비교에서 본문이 쓰지 않는 값 (진한 쪽)


def _sweep_styles(values, used):
    """설정값 목록의 선 색·모양. 본문이 쓰는 값(used)은 파랑 실선, 나머지는 회색 두 가지(점선·일점쇄선)로 구분한다."""
    others = iter([(LIGHT_GRAY, ":"), (DARK_GRAY, "-."), (cp.MUTED, (0, (1, 3)))])
    return [(cp.BLUE, "-") if np.isclose(value, used) else next(others) for value in values]


def _mosaic_row(images, gap=2):
    """(n, h, w) 이미지 n장을 가로로 이어 붙인 2차원 배열. 사이는 NaN(흰색)."""
    images = np.asarray(images, float)
    n, h, w = images.shape
    out = np.full((h, n * w + (n - 1) * gap), np.nan)
    for k in range(n):
        out[:, k * (w + gap):k * (w + gap) + w] = images[k]
    return out


def _under_labels(ax, texts, n, w, gap, y):
    """_mosaic_row로 이은 이미지마다 아래에 짧은 설명을 적는다."""
    for k, text in enumerate(texts):
        ax.text(k * (w + gap) + (w - 1) / 2, y, text, ha="center", va="top", fontsize=9, color=cp.INK)


# ---------------------------------------------------------------- 1절
def plot_lif_trace(times, U, S, currents, theta, beta, steady, fi_currents, fi_rates, fi_betas, fi_T):
    """왼쪽: 일정 전류 두 개의 막전위 궤적과 문턱선, 쏘지 않는 뉴런의 정상 상태 steady = I/(1−β), 아래 띠에 스파이크 시각.
    오른쪽: 전류 → 발화율 곡선(β 세 가지, 왼쪽과 같은 β는 파랑)."""
    U, S = np.asarray(U, float), np.asarray(S, float)
    fig = cp.new_figure((13.2, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[3.2, 1], wspace=0.28, hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    _clean_axes(ax, None, "막전위 $U_t$", f"새는 물통: $U_t=\\beta U_{{t-1}}+I-\\theta S_{{t-1}}$, $\\beta={beta:g}$")
    colors = [cp.BLUE, cp.MUTED]
    for j, I in enumerate(currents):
        ax.plot(times, U[:, j], "o-", color=colors[j], ms=4, lw=1.6, label=f"$I={I:g}$")
    ax.axhline(theta, color=cp.GOLD, lw=1.4, ls="--")
    ax.text(times[-1] + 0.4, theta + 0.02, "문턱 $\\theta$", va="bottom", ha="left", fontsize=9, color=cp.GOLD)
    ax.axhline(steady, color=cp.MUTED, lw=0.9, ls=":")
    ax.text(times[-1] + 0.4, steady + 0.02, f"$I/(1-\\beta)={steady:g}$", va="bottom", ha="left", fontsize=8.5, color=cp.MUTED)
    for t in np.flatnonzero(S[:, 0]):
        ax.annotate("딸깍", (times[t], U[t, 0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5,
                    color=cp.BLUE)
    ax.set_xticks([])
    ax.set_xlim(times[0] - 0.5, times[-1] + 2.6)
    ax.set_ylim(-0.05, max(1.3, float(U.max()) + 0.18))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax_s = fig.add_subplot(gs[1, 0])
    _clean_axes(ax_s, "시각 $t$", None, None)
    for j in range(len(currents)):
        spikes = np.flatnonzero(S[:, j])
        ax_s.vlines(np.asarray(times)[spikes], j + 0.1, j + 0.9, color=colors[j], lw=2.5)
    ax_s.set_yticks([0.5, 1.5], [f"$S_t$ ($I={currents[0]:g}$)", f"$S_t$ ($I={currents[1]:g}$)"])
    ax_s.set_ylim(0, 2)
    ax_s.set_xticks(times)
    ax_s.set_xlim(times[0] - 0.5, times[-1] + 2.6)
    ax_s.spines["left"].set_visible(False)
    ax_f = fig.add_subplot(gs[:, 1])
    _clean_axes(ax_f, "일정 전류 $I$", f"발화율 ({fi_T}시각 중 쏜 비율)", "전류 → 발화율: 문턱 전류 $(1-\\beta)\\theta$ 아래는 0")
    for rates, fi_beta, (color, ls) in zip(fi_rates, fi_betas, _sweep_styles(fi_betas, beta)):
        ax_f.plot(fi_currents, rates, color=color, lw=1.8, ls=ls, label=f"$\\beta={fi_beta:g}$")
        ax_f.axvline((1 - fi_beta) * theta, color=color, lw=0.9, ls=":")
    ax_f.legend(frameon=False, fontsize=9, loc="upper left")
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_rate_coding(image, frames, frame_times, means, mean_Ts):
    """밝기 x → 시각마다 Bernoulli(x)로 뽑은 0/1 프레임 → 앞 T시각의 평균. 평균이 밝기로 돌아온다."""
    frames = np.asarray(frames, float).reshape(-1, 28, 28)
    means = np.asarray(means, float).reshape(-1, 28, 28)
    gap = 3
    fig, axes = cp.flow([1, 3.2, 3.2], height=3.3, unit=1.55, wspace=0.28)
    cp.draw_image(axes[0], image, "밝기 $x$ (28×28)")
    cp.label(axes[0], "켜질 확률 = 밝기", y=-0.05)
    cp.draw_image(axes[1], _mosaic_row(frames, gap), "시각마다 뽑은 스파이크 $S^{in}_t$ (0/1)", frame=False)
    _under_labels(axes[1], [f"$t={t}$" for t in frame_times], len(frames), 28, gap, 29.5)
    cp.draw_image(axes[2], _mosaic_row(means, gap), "앞 $T$시각의 평균 $\\bar S$", frame=False)
    _under_labels(axes[2], [f"$T={T}$" for T in mean_Ts], len(means), 28, gap, 29.5)
    cp.connect(fig, axes[0], axes[1], "Bernoulli")
    cp.connect(fig, axes[1], axes[2], "시간 평균")
    cp.show(fig)


def plot_coding_noise(Ts, measured, theory, latency_map, latency_T):
    """왼쪽: 시각 수 T에 따른 평균의 흔들림(화소 전체에 걸친 제곱평균제곱근, RMS)과 이론 √(mean x(1−x)/T). 오른쪽: latency coding의 발화 시각."""
    fig = cp.new_figure((11.5, 3.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "시각 수 $T$ (로그 눈금)", "$\\bar S - x$ 의 RMS (로그 눈금)", "시각을 4배로 늘리면 흔들림은 절반")
    ax.loglog(Ts, theory, color=cp.GOLD, lw=2.0, label="이론 $\\sqrt{\\overline{x(1-x)}/T}$")
    ax.loglog(Ts, measured, "o", color=cp.BLUE, ms=6, label="7 한 장에서 잰 값")
    ax.set_xticks(Ts, [str(t) for t in Ts])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax_l = fig.add_subplot(gs[1])
    cp.blank(ax_l)
    grid = np.asarray(latency_map, float).reshape(28, 28)
    ax_l.imshow(grid, cmap="Blues_r", vmin=1, vmax=latency_T * 1.35, interpolation="nearest")
    ax_l.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, ec=cp.MUTED, lw=0.8))
    ax_l.set_title(f"다른 부호화: 밝을수록 일찍 한 번 ($T={latency_T}$)", color=cp.INK, fontsize=10.5, pad=6)
    cp.label(ax_l, "진한 파랑 = 일찍 쏨 · 옅은 파랑 = 늦게 · 흰색 = 안 쏨", y=-0.04)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_row_sum(s, W, result):
    """0/1 입력 s와 가중치 W의 곱: 켜진 입력의 행(금색 틀)만 골라 더하면 sW와 같다."""
    s, W, result = np.asarray(s, float), np.asarray(W, float), np.asarray(result, float)
    D, H = W.shape
    fig = cp.new_figure((10.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, H + 0.6, H + 0.6], wspace=0.45)
    ax_s = fig.add_subplot(gs[0])
    colors_s = [["white" if v == 0 else ACTIVE_FILL] for v in s]
    sp.draw_cells(ax_s, s[:, None], f"$s$ ({D},)", cell_colors=colors_s)
    ax_w = fig.add_subplot(gs[1])
    colors_w = [[ACTIVE_FILL if s[d] == 1 else "white"] * H for d in range(D)]
    sp.draw_cells(ax_w, W, f"$W_1$ ({D}, {H}): 행 $d$ = 입력 $d$의 가중치", cell_colors=colors_w)
    for d in np.flatnonzero(s):
        ax_w.add_patch(Rectangle((0, D - 1 - d), H, 1, fill=False, edgecolor=cp.GOLD, lw=2.2, zorder=3))
    ax_r = fig.add_subplot(gs[2])
    sp.draw_cells(ax_r, result[None, :], f"$sW_1$ ({H},) = 켜진 행의 합")
    ax_r.set_ylim(-(D - 1) / 2 - 0.15, (D + 1) / 2 + 0.15)
    cp.connect(fig, ax_s, ax_w, "켜진 행 고르기")
    cp.connect(fig, ax_w, ax_r, "열마다 더하기")
    active = ", ".join(str(d) for d in np.flatnonzero(s))
    cp.label(ax_w, f"켜진 입력 {active} → 곱셈 없이 덧셈 {int(s.sum())}×{H} = {int(s.sum()) * H}번", y=-0.03, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_snn_flow(image, input_raster, hidden_raster, logits, label):
    """7 → 입력 스파이크 래스터 (T, 784) → 은닉 LIF 스파이크 래스터 (T, H) → 시간 평균 읽기의 logit 10개."""
    Sin, S = np.asarray(input_raster, float), np.asarray(hidden_raster, float)
    T, D = Sin.shape
    H = S.shape[1]
    fig, axes = cp.flow([1, 2.4, 1.6, 1.05], height=3.5, unit=2.1, wspace=0.42)
    cp.draw_image(axes[0], image, f"$x_{{{label}}}$ (28×28)")
    cp.draw_rows(axes[1], Sin, f"입력 스파이크 $S^{{in}}$ ({T}, {D})")
    cp.label(axes[1], f"행 = 시각 1–{T} · 열 = 화소 번호 · 검정 = 1", y=-0.04)
    cp.draw_rows(axes[2], S, f"은닉 스파이크 $S$ ({T}, {H})")
    cp.label(axes[2], f"열 = 은닉 뉴런 · 전체 발화율 {S.mean():.2f}", y=-0.04)
    for ax in axes[1:3]:
        cp.match_height(ax, axes[0])
    cp.draw_bars(axes[3], logits, "logit (10,)", labels=[str(k) for k in range(10)], highlight=int(np.argmax(logits)),
                 marker=label, signed=True)
    cp.match_height(axes[3], axes[0])
    cp.connect(fig, axes[0], axes[1], "rate coding")
    cp.connect(fig, axes[1], axes[2], "$S^{in}_tW_1+b_1$\n→ LIF")
    cp.connect(fig, axes[2], axes[3], "시간 평균\n읽기")
    cp.label(axes[3], "학습 전 · 파랑 = 가장 큰 logit · 금색 = 정답", y=-0.06)
    cp.show(fig)


# ---------------------------------------------------------------- 4절
def plot_step_vs_smooth(bias, count_step, count_soft, loss_step, loss_soft, T, k, v, step, sigmas, derivs, ks):
    """편향 하나를 움직일 때 계단 LIF는 스파이크 수·손실이 평평한 계단이고 부드러운 LIF(가파름 k)는 매끈하다.
    오른쪽: Θ·σ(kv)와 σ_k' (ks 여러 값, 왼쪽과 같은 k는 검은 파선)."""
    fig = cp.new_figure((15, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.85, 0.85], wspace=0.33)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "은닉 뉴런 하나의 편향 $b_{1,h}$", f"{T}시각 스파이크 수", "스파이크 수: 계단 vs 부드러운 합")
    ax.plot(bias, count_step, color=cp.BLUE, lw=1.8, label="계단: 쏜 횟수")
    ax.plot(bias, count_soft, color=cp.INK, lw=1.3, ls="--", label="부드러운: $\\sigma$의 합")
    ax.axvline(0, color=cp.GOLD, lw=1.2, ls=":")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "은닉 뉴런 하나의 편향 $b_{1,h}$", "7의 CE 손실", "손실: 평평한 구간에서 미분 = 0")
    ax2.plot(bias, loss_step, color=cp.BLUE, lw=1.8, label="계단 LIF")
    ax2.plot(bias, loss_soft, color=cp.INK, lw=1.3, ls="--", label=f"부드러운 ($k={k:g}$)")
    ax2.axvline(0, color=cp.GOLD, lw=1.2, ls=":")
    ax2.text(0.03, 0.6, "금색 점선:\n지금 값 $b=0$\n계단의 미분 = 0", transform=ax2.transAxes, fontsize=8.5, color=cp.GOLD, va="center")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$v = U-\\theta$", None, "계단 $\\Theta(v)$ 와 $\\sigma(kv)$")
    ax3.plot(v, step, color=cp.BLUE, lw=2.0, label="$\\Theta(v)$")
    styles = [(cp.INK, "--") if np.isclose(value, k) else style for value, style in zip(ks, _sweep_styles(ks, k))]
    for sig, value, (color, ls) in zip(sigmas, ks, styles):
        ax3.plot(v, sig, color=color, lw=1.4, ls=ls, label=f"$k={value:g}$")
    ax3.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax4 = fig.add_subplot(gs[3])
    _clean_axes(ax4, "$v = U-\\theta$", None, "대리 도함수 $\\sigma_k'(v)$")
    for der, value, (color, ls) in zip(derivs, ks, styles):
        ax4.plot(v, der, color=color, lw=1.5, ls=ls, label=f"$k={value:g}$, 꼭대기 {value / 4:g}")
    ax4.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax4.set_ylim(0, max(float(np.max(derivs)) * 1.35, 1))
    cp.show(fig)


def _box(ax, xy, text, *, w=0.95, h=0.62, fill=sp.FILL):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=fill, edgecolor=cp.MUTED, lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color=cp.INK)


def _arrow(ax, a, b, *, color=cp.MUTED, lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw))


def plot_lif_bptt(table, row_labels, col_labels, beta, theta, k):
    """왼쪽: 시간으로 펼친 LIF의 forward(회색)와 backward(금색) 경로. U_t 는 β 경로와 S_t 를 거친 −θ 경로로 다음 시각에 쓰인다.
    오른쪽: 스칼라 뉴런 T = 3 의 손계산 표 (마지막 시각부터)."""
    fig = cp.new_figure((15, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.25], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    cp.blank(ax)
    ax.set_aspect("equal")
    step, gold = 3.2, dict(color=cp.GOLD, lw=1.6)
    for i in range(3):
        x = i * step
        t = i + 1
        _box(ax, (x, 0), f"$U_{t}$")
        _box(ax, (x, 1.6), f"$S_{t}$")
        _arrow(ax, (x - 0.12, -1.15), (x - 0.12, -0.33))
        ax.text(x - 0.12, -1.22, f"$I_{t}$", ha="center", va="top", fontsize=10, color=cp.INK)
        _arrow(ax, (x - 0.12, 0.33), (x - 0.12, 1.27))
        ax.text(x - 0.2, 0.8, "$\\Theta$", ha="right", va="center", fontsize=9.5, color=cp.MUTED)
        _arrow(ax, (x + 0.2, 1.27), (x + 0.2, 0.33), **gold)
        ax.text(x + 0.28, 0.8, "$\\sigma_k'$", ha="left", va="center", fontsize=9.5, color=cp.GOLD)
        _arrow(ax, (x - 0.12, 1.93), (x - 0.12, 2.65))
        _arrow(ax, (x + 0.2, 2.65), (x + 0.2, 1.93), **gold)
        ax.text(x + 0.28, 2.3, f"$g_{t}$", ha="left", va="center", fontsize=9.5, color=cp.GOLD)
        if i < 2:
            nx = x + step
            _arrow(ax, (x + 0.49, 0.08), (nx - 0.49, 0.08))
            ax.text(x + step / 2, 0.14, "$\\times\\beta$", ha="center", va="bottom", fontsize=9.5, color=cp.MUTED)
            _arrow(ax, (nx - 0.49, -0.14), (x + 0.49, -0.14), **gold)
            ax.text(x + step / 2, -0.22, f"$\\beta\\,dU_{t + 1}$", ha="center", va="top", fontsize=9.5, color=cp.GOLD)
            _arrow(ax, (x + 0.49, 1.35), (nx - 0.49, 0.2))
            ax.text(x + 0.95, 0.88, "$-\\theta$", ha="center", va="center", fontsize=9.5, color=cp.MUTED)
            _arrow(ax, (nx - 0.3, 0.33), (x + 0.49, 1.75), **gold)
            ax.text(x + 1.62, 1.4, f"$-\\theta\\,dU_{t + 1}$", ha="left", va="bottom", fontsize=9.5, color=cp.GOLD)
    ax.text(step, 2.95, "손실 $L$", ha="center", va="center", fontsize=10.5, color=cp.INK)
    ax.plot([-0.12, 2 * step - 0.12], [2.65, 2.65], color=cp.MUTED, lw=1.0)
    ax.set_xlim(-0.9, 2 * step + 1.5)
    ax.set_ylim(-1.7, 3.2)
    ax.set_title("회색 forward · 금색 backward: $dS_t = g_t - \\theta\\,dU_{t+1}$, $dU_t = dS_t\\,\\sigma_k' + \\beta\\,dU_{t+1}$",
                 color=cp.INK, fontsize=10.5, pad=6)
    ax_t = fig.add_subplot(gs[1])
    cp.blank(ax_t)
    cp.draw_grid(ax_t, table, f"스칼라 뉴런 손계산 ($\\beta={beta:g}$, $\\theta={theta:g}$, $k={k:g}$), 마지막 시각부터", kind="plain", fmt="{:.4f}",
                 fontsize=9, row_labels=row_labels, col_labels=col_labels)
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def _block(ax, x, y, name, sub, *, fill=sp.FILL, width=1.9, height=0.9, fontsize=9.3):
    """(x, y)를 왼쪽 아래로 하는 둥근 상자와 안쪽 이름, 아래 회색 설명(텐서 모양)."""
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fill, edgecolor=cp.MUTED, lw=1.0))
    ax.text(x + width / 2, y + height / 2, name, ha="center", va="center", fontsize=fontsize, color=cp.INK)
    ax.text(x + width / 2, y - 0.12, sub, ha="center", va="top", fontsize=8.5, color=cp.MUTED)


def plot_snn_blocks(forward_blocks, backward_blocks):
    """윗줄 forward(부호화 → 전류 → LIF → 스파이크 수 → logit → CE)는 한 줄 흐름.
    아랫줄 backward는 값이 흐르는 대로 갈래를 그린다: G → (dW2·db2, g_t), g_t → lif_backward → dI → (dW1, db1).
    backward_blocks: 키 "G", "readout", "g", "lif", "W1", "b1" → (상자 안 이름, 아래 모양)."""
    fig = cp.new_figure((14.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.75], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    colors = [ACTIVE_FILL if "LIF" in name.upper() else sp.FILL for name, _ in forward_blocks]
    sp.draw_blocks(ax, forward_blocks, colors=colors, width=1.9, gap=0.5, fontsize=9.3)
    ax.set_title("forward (계단 LIF)", color=cp.INK, fontsize=10.5, pad=4, loc="left")
    xlim = ax.get_xlim()
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_axis_off()
    ax_b.set_aspect("equal")
    w, h, col = 2.3, 1.0, 3.6
    places = {"G": (0, 0), "g": (col, 0), "lif": (2 * col, 0), "W1": (3 * col, 0),
              "readout": (col, -1.75), "b1": (3 * col, -1.75)}
    for key, (x, y) in places.items():
        name, sub = backward_blocks[key]
        _block(ax_b, x, y, name, sub, fill=ACTIVE_FILL if key == "lif" else sp.FILL, width=w, height=h)
    straight = [("G", "g"), ("g", "lif"), ("lif", "W1")]
    for a, b in straight:
        (xa, ya), (xb, yb) = places[a], places[b]
        ax_b.add_patch(FancyArrowPatch((xa + w + 0.05, ya + h / 2), (xb - 0.05, yb + h / 2),
                                       arrowstyle="-|>", mutation_scale=11, color=cp.MUTED, lw=1.0))
    for a, b in [("G", "readout"), ("lif", "b1")]:
        (xa, ya), (xb, yb) = places[a], places[b]
        ax_b.add_patch(FancyArrowPatch((xa + w / 2, ya - 0.45), (xb - 0.05, yb + h / 2), connectionstyle="angle,angleA=-90,angleB=180",
                                       arrowstyle="-|>", mutation_scale=11, color=cp.MUTED, lw=1.0))
    ax_b.text(col * 0.5 + w / 2, h / 2 + 0.12, "$\\times W_2^\\top/T$", ha="center", va="bottom", fontsize=8.5, color=cp.MUTED)
    ax_b.text(2.5 * col + w / 2, h / 2 + 0.12, "$dI$", ha="center", va="bottom", fontsize=8.5, color=cp.MUTED)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(-2.55, h + 0.35)
    ax_b.set_title("backward (대리 도함수): logit의 기울기 G에서 출발해 두 갈래로", color=cp.INK, fontsize=10.5, pad=4, loc="left")
    cp.label(ax_b, "화살표 = 값이 흘러가는 방향 · 옅은 금색 = LIF가 들어가는 단계 · 배치마다 새로 부호화하므로 같은 이미지도 에폭마다 다른 스파이크 열",
             y=-0.02)
    cp.show(fig)


def plot_epoch_encodings(image, frames, means, epochs, T):
    """같은 7을 에폭마다 새로 부호화한 첫 시각 프레임(윗줄)과 T시각 평균(아랫줄)."""
    frames = np.asarray(frames, float).reshape(-1, 28, 28)
    means = np.asarray(means, float).reshape(-1, 28, 28)
    n = len(frames)
    fig = cp.new_figure((2.0 * (n + 1) + 0.8, 4.4))
    gs = fig.add_gridspec(2, n + 1, wspace=0.18, hspace=0.3)
    ax0 = fig.add_subplot(gs[:, 0])
    cp.blank(ax0)
    cp.draw_image(ax0, image, "밝기 $x_7$")
    for k in range(n):
        ax = fig.add_subplot(gs[0, k + 1])
        cp.blank(ax)
        cp.draw_image(ax, frames[k], f"에폭 {epochs[k]}: $S^{{in}}_1$")
        ax = fig.add_subplot(gs[1, k + 1])
        cp.blank(ax)
        cp.draw_image(ax, means[k], f"에폭 {epochs[k]}: {T}시각 평균")
    cp.label(ax0, "같은 이미지, 에폭마다 다른 난수", y=-0.08)
    cp.show(fig)


def plot_training_result(losses, window, epoch_steps, valid_acc, valid_rate, batch_size, n_valid):
    """왼쪽: 스텝별 배치 CE(회색)와 이동평균(파랑). 오른쪽: 에폭 끝마다 잰 검증 정확도와 은닉 발화율."""
    losses = np.asarray(losses, float)
    fig = cp.new_figure((11, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "스텝", "배치 CE", "학습 손실 (계단 forward · 대리 도함수 backward)")
    ax.plot(np.arange(1, len(losses) + 1), losses, color=cp.MUTED, lw=0.6, alpha=0.6, label=f"배치 {batch_size}장의 CE")
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(losses) + 1), smooth, color=cp.BLUE, lw=1.8, label=f"{window}스텝 이동평균")
    for s in epoch_steps[:-1]:
        ax.axvline(s, color=cp.GOLD, lw=1.0, ls=":")
    ax.set_ylim(0, min(2.5, float(losses.max()) * 1.05))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "에폭", "검증 정확도", f"검증 {n_valid:,}장 (갱신에 쓰지 않음)")
    epochs = np.arange(1, len(valid_acc) + 1)
    ax2.plot(epochs, valid_acc, "o-", color=cp.BLUE, lw=1.6, ms=6)
    for e, a, r in zip(epochs, valid_acc, valid_rate):
        ax2.text(e, a + 0.003, f"{a:.3f}\n발화율 {r:.2f}", ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    ax2.set_xticks(epochs)
    ax2.set_xlim(0.6, len(valid_acc) + 0.4)
    ax2.set_ylim(min(valid_acc) - 0.02, max(valid_acc) + 0.025)
    cp.show(fig)


def plot_trained_spikes(image, hidden_raster, neuron_rates, logits, label, n_valid):
    """학습 뒤 7의 은닉 래스터(뉴런을 검증 발화율 순으로 정렬), 검증 집합의 뉴런별 평균 발화율, 7의 logit."""
    S = np.asarray(hidden_raster, float)
    T, H = S.shape
    rates = np.asarray(neuron_rates, float)
    fig = cp.new_figure((14.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 1.6, 1.3, 0.9], wspace=0.35)
    ax0 = fig.add_subplot(gs[0])
    cp.blank(ax0)
    cp.draw_image(ax0, image, f"$x_{{{label}}}$")
    ax1 = fig.add_subplot(gs[1])
    cp.blank(ax1)
    cp.draw_rows(ax1, S, f"학습 뒤 은닉 스파이크 ({T}, {H})")
    cp.label(ax1, "열 = 뉴런 (검증 발화율이 높은 순) · 행 = 시각", y=-0.04)
    ax2 = fig.add_subplot(gs[2])
    _clean_axes(ax2, "뉴런 (발화율 순)", "검증 발화율", f"뉴런별 발화율 (검증 {n_valid:,}장 평균)")
    ax2.bar(np.arange(H), rates, width=1.0, color=cp.MUTED)
    ax2.axhline(rates.mean(), color=cp.GOLD, lw=1.2, ls="--")
    ax2.text(H - 1, rates.mean() + 0.02, f"평균 {rates.mean():.2f}", ha="right", va="bottom", fontsize=8.5, color=cp.GOLD)
    ax2.set_xlim(-1, H)
    ax2.set_ylim(0, 1.02)
    ax3 = fig.add_subplot(gs[3])
    cp.blank(ax3)
    cp.draw_bars(ax3, logits, f"{label}의 logit (10,)", labels=[str(k) for k in range(10)], highlight=int(np.argmax(logits)),
                 marker=label, signed=True)
    cp.match_height(ax3, ax0)
    cp.show(fig)


def plot_time_steps(Ts, acc_mean, acc_std, ops, dense_ops, sparse_ops, train_T, n_valid, n_seeds):
    """같은 학습 가중치로 추론 시각 수 T만 바꿨을 때: 왼쪽 검증 정확도(부호화 seed 평균 ± 표준편차),
    오른쪽 장당 시냅스 연산(덧셈) 수와 같은 크기 MLP의 곱셈 수 두 가지(0을 건너뛰지 않음 / 0인 화소를 건너뜀)."""
    Ts, acc_mean, acc_std = np.asarray(Ts), np.asarray(acc_mean, float), np.asarray(acc_std, float)
    fig = cp.new_figure((12.5, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, f"추론 시각 수 $T$ (로그 눈금, 학습은 $T={train_T}$)", f"검증 정확도 ({n_valid:,}장)",
                f"정확도: seed {n_seeds}개 평균 ± 표준편차")
    ax.set_xscale("log", base=2)
    ax.errorbar(Ts, acc_mean, yerr=acc_std, fmt="o-", color=cp.BLUE, lw=1.6, ms=6, capsize=3)
    for T, a, e in zip(Ts, acc_mean, acc_std):
        ax.text(T, a + e + 0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    ax.set_xticks(Ts, [str(t) for t in Ts])
    ax.minorticks_off()
    ax.set_ylim(float((acc_mean - acc_std).min()) - 0.03, float((acc_mean + acc_std).max()) + 0.012)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "추론 시각 수 $T$ (로그 눈금)", "장당 연산 수 (로그 눈금)", "덧셈 수는 $T$에 비례")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.plot(Ts, ops, "o-", color=cp.BLUE, lw=1.6, ms=6, label="SNN 시냅스 연산 (덧셈)")
    ax2.axhline(dense_ops, color=cp.MUTED, lw=1.2, ls="--", label=f"MLP 곱셈, 0도 모두 곱함 {dense_ops:,.0f}")
    ax2.axhline(sparse_ops, color=cp.GOLD, lw=1.4, ls="--", label=f"MLP 곱셈, 0인 화소를 건너뜀 {sparse_ops:,.0f}")
    for T, o in zip(Ts, ops):
        crosses = any(o < ref < o * 2.2 for ref in (dense_ops, sparse_ops))      # 위에 쓰면 기준선과 겹치는 자리
        ax2.text(T, o / 1.3 if crosses else o * 1.25, f"{o / 1000:.0f}k", ha="center", va="top" if crosses else "bottom",
                 fontsize=8.5, color=cp.INK)
    ax2.set_xticks(Ts, [str(t) for t in Ts])
    ax2.minorticks_off()
    ax2.set_ylim(min(ops) / 2, max(ops) * 4)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.show(fig)
