"""07 LSTM·GRU 노트북의 개념 그림과 구조 도식.

게이트·상태·기울기 값은 노트북 본문에서 계산하고 여기서는 받은 값만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes
from .architecture_plots import plot_curves  # noqa: F401


def _circle(ax, xy, text, r=0.27):
    ax.add_patch(Circle(xy, r, facecolor="white", edgecolor=cp.MUTED, lw=1.0, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=11, color=cp.INK, zorder=4)


def _arrow(ax, a, b, *, color=cp.MUTED, lw=1.2, text=None, dx=0.0, dy=0.1, fontsize=8.5, tcolor=None, va="bottom", ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, zorder=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha=ha, va=va, fontsize=fontsize, color=tcolor or color)


def draw_lstm_cell(ax, v, back=None):
    """LSTM 셀 도식. 위의 굵은 파란 선이 메모리 c의 덧셈 경로, 아래 상자가 게이트를 만드는 affine.

    v    : {"cprev", "f", "i", "o", "g", "c", "h"} 손계산 값.
    back : 주면 {"dh", "dc_future", "dc"} 값으로 c 에 모이는 두 기울기 경로를 금색으로 덧그린다.
    """
    ax.set_axis_off()
    ax.set_aspect("equal")
    yc, yg, yb = 3.0, 1.5, 0.3                                   # c 선, 게이트 높이, 게이트 상자 윗면
    ink = cp.INK
    # ---- c 경로 (덧셈으로 이어짐)
    ax.text(-0.15, yc, f"$c_{{prev}} = {v['cprev']:g}$", ha="right", va="center", fontsize=10, color=ink)
    _arrow(ax, (-0.05, yc), (1.7, yc), color=cp.BLUE, lw=2.2)
    _circle(ax, (2.0, yc), "⊗")
    _arrow(ax, (2.3, yc), (3.7, yc), color=cp.BLUE, lw=2.2, text=f"$f \\odot c_{{prev}} = {v['f'] * v['cprev']:g}$", tcolor=ink, dy=0.14)
    _circle(ax, (4.0, yc), "⊕")
    _arrow(ax, (4.3, yc), (8.2, yc), color=cp.BLUE, lw=2.2, text=f"$c = f \\odot c_{{prev}} + i \\odot g = {v['c']:g}$", tcolor=ink, dy=0.14)
    ax.text(8.35, yc, "$c$", ha="left", va="center", fontsize=11, color=ink)
    # ---- 게이트가 들어오는 자리
    _arrow(ax, (2.0, yb), (2.0, yc - 0.3), text=f"$f = {v['f']:g}$", dx=0.42, dy=0, tcolor=ink, va="center", ha="left")
    _circle(ax, (4.0, yg), "⊗")
    _arrow(ax, (4.0, yg + 0.3), (4.0, yc - 0.3), text=f"$i \\odot g = {v['i'] * v['g']:g}$", dx=0.42, dy=0, tcolor=ink, va="center", ha="left")
    ax.plot([3.0, 3.0], [yb, yg], color=cp.MUTED, lw=1.2, zorder=2)
    _arrow(ax, (3.0, yg), (3.7, yg), text=f"$i = {v['i']:g}$", dy=0.12, tcolor=ink)
    _arrow(ax, (4.0, yb), (4.0, yg - 0.3), text=f"$g = {v['g']:g}$", dx=0.42, dy=0, tcolor=ink, va="center", ha="left")
    # ---- 출력 가지: c → tanh → ⊗ o → h
    _arrow(ax, (5.6, yc - 0.05), (5.6, yg + 0.32), color=cp.BLUE, lw=1.4)
    ax.add_patch(FancyBboxPatch((5.15, yg - 0.25), 0.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=sp.FILL, edgecolor=cp.MUTED, lw=1.0, zorder=3))
    ax.text(5.6, yg, "tanh", ha="center", va="center", fontsize=9.5, color=ink, zorder=4)
    _arrow(ax, (6.05, yg), (6.7, yg), text=f"${np.tanh(v['c']):.2f}$", dy=0.12, tcolor=ink)
    _circle(ax, (7.0, yg), "⊗")
    _arrow(ax, (7.0, yb), (7.0, yg - 0.3), text=f"$o = {v['o']:g}$", dx=0.42, dy=0, tcolor=ink, va="center", ha="left")
    _arrow(ax, (7.3, yg), (8.2, yg), color=cp.BLUE, lw=2.2)
    ax.text(8.35, yg, f"$h = o \\odot \\tanh c = {v['h']:.3f}$", ha="left", va="center", fontsize=10, color=ink)
    # ---- 게이트를 만드는 affine 상자
    ax.add_patch(FancyBboxPatch((0.6, -0.65), 7.2, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=sp.FILL, edgecolor=cp.MUTED, lw=1.0, zorder=1))
    ax.text(4.2, -0.2, "$u = [x, h_{prev}] \\;\\rightarrow\\; uW + b \\;\\rightarrow\\; f, i, o = \\sigma(\\cdot),\\; g = \\tanh(\\cdot)$",
            ha="center", va="center", fontsize=10, color=ink, zorder=2)
    ax.text(0.45, -0.2, "$x, h_{prev}$", ha="right", va="center", fontsize=10, color=ink)
    if back is not None:
        gold = dict(color=cp.GOLD, lw=1.7, tcolor=cp.GOLD)
        _arrow(ax, (10.4, yg - 0.32), (7.35, yg - 0.32), text=f"$dh = {back['dh']:g}$", dy=-0.12, va="top", **gold)
        _arrow(ax, (6.7, yg - 0.32), (6.1, yg - 0.32), **gold)
        _arrow(ax, (5.95, yg + 0.32), (5.95, yc - 0.12), text="$dh \\odot o \\odot (1-\\tanh^2 c)$", dx=0.12, dy=-0.05, va="center", ha="left", **gold)
        _arrow(ax, (8.2, yc - 0.3), (4.35, yc - 0.3), text=f"$dc_{{future}} = {back['dc_future']:g}$", dx=1.35, dy=-0.12, va="top", **gold)
        _arrow(ax, (3.7, yc - 0.3), (2.35, yc - 0.3), text=f"$dc = {back['dc']:.3f}$", dy=-0.12, va="top", **gold)
        _arrow(ax, (1.7, yc - 0.3), (-0.05, yc - 0.3), text="$dc \\odot f$", dy=-0.12, va="top", **gold)
    ax.set_xlim(-2.1, 10.6)
    ax.set_ylim(-0.9, yc + 0.7)
    return ax


def plot_lstm_cell(values):
    """손계산 값을 적은 LSTM 셀 도식."""
    fig = cp.new_figure((12.5, 4.2))
    ax = fig.add_subplot(111)
    draw_lstm_cell(ax, values)
    ax.set_title("굵은 파란 선 = 메모리 $c$의 덧셈 경로 · ⊗ = 게이트가 곱하는 자리 · 값은 손계산 예제", color=cp.INK, fontsize=10.5, pad=4)
    cp.show(fig)


def plot_lstm_backward(values, back, table, row_labels):
    """왼쪽: c 로 모이는 두 기울기 경로(금색). 오른쪽: 손계산 예제의 backward 값 표."""
    fig = cp.new_figure((14, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 0.7], wspace=0.02)
    ax = fig.add_subplot(gs[0])
    draw_lstm_cell(ax, values, back=back)
    ax.set_title("금색: $h$ 에서 온 기울기와 다음 시각에서 온 $dc_{future}$ 가 $c$ 에서 더해짐", color=cp.INK, fontsize=10.5, pad=4)
    ax_t = fig.add_subplot(gs[1]); cp.blank(ax_t)
    cp.draw_grid(ax_t, np.asarray(table)[:, None], "손계산 값", kind="signed", fmt="{:.3f}", fontsize=9.5, row_labels=row_labels)
    cp.show(fig)


def plot_retention(delays, retention):
    """상수 forget gate 에서 이전 메모리가 남는 비율 f^t. 1 에 가까울수록 오래 남습니다."""
    fig = cp.new_figure((7.2, 3.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "지연 스텝 $t$", "$c_t / c_0$ (로그 눈금)", "직접 경로만의 곱 $\\prod f = f^{\\,t}$ : 상수 게이트의 통제 예제")
    for (name, ys), color in zip(retention.items(), [cp.RED, cp.INK, cp.BLUE]):
        ax.semilogy(delays, ys, "o-", color=color, ms=3, lw=1.5, label=name)
    ax.axhline(1, color=cp.GOLD, lw=1.0, ls="--")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    cp.show(fig)


def plot_gru_mix(z_values, hprev, g, h_values):
    """h = z·h_prev + (1−z)·g : z 가 0 이면 후보 g, 1 이면 이전 상태 h_prev. 수직선 위의 점으로."""
    fig = cp.new_figure((8.5, 2.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$h$ 의 값", None, "$h = z \\odot h_{prev} + (1 - z) \\odot g$ : $z$ 가 클수록 이전 상태를 유지")
    ax.axhline(0, color=cp.MUTED, lw=1.0)
    ax.scatter([g], [0], s=90, color=cp.RED, zorder=4)
    ax.text(g, 0.12, f"후보 $g = {g:g}$", ha="center", va="bottom", fontsize=9.5, color=cp.RED)
    ax.scatter([hprev], [0], s=90, color=cp.BLUE, zorder=4)
    ax.text(hprev, 0.12, f"이전 상태 $h_{{prev}} = {hprev:g}$", ha="center", va="bottom", fontsize=9.5, color=cp.BLUE)
    for z, h in zip(z_values, h_values):
        ax.scatter([h], [0], s=40, facecolor="white", edgecolor=cp.GOLD, lw=1.6, zorder=5)
        ax.text(h, -0.12, f"$z={z:g}$", ha="center", va="top", fontsize=8.5, color=cp.INK, rotation=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.45, 0.45)
    lo, hi = min(g, hprev), max(g, hprev)
    ax.set_xlim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    cp.show(fig)


def plot_gated_sequence(image, Hs, Cs, Fs, t_mark):
    """7을 28행 시퀀스로 읽는 학습 전 LSTM: h_t, c_t, f_t 의 시간축 띠(행 = 시각)."""
    T = Hs.shape[0]
    fig = cp.new_figure((11.5, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 0.5, 0.5, 0.5], wspace=0.45)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_image(ax_x, image, f"$x_7$ : {T}행을 위에서부터 읽음")
    ax_x.add_patch(Rectangle((-0.5, t_mark - 1 - 0.5), image.shape[1], 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
    strips = [(Hs, "$h_t$ 출력 상태", 1.0), (Cs, "$c_t$ 메모리", float(np.abs(Cs).max())), (Fs, "$f_t$ forget gate", 1.0)]
    prev = ax_x
    for k, (S, title, vmax) in enumerate(strips):
        ax = fig.add_subplot(gs[1 + k]); cp.blank(ax)
        cp.draw_rows(ax, S, f"{title} : ({T}, {S.shape[1]})", kind="signed", vmax=vmax)
        ax.add_patch(Rectangle((-0.5, t_mark - 1 - 0.5), S.shape[1], 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
        if k == 0:
            cp.connect(fig, prev, ax, "행 하나씩")
    cp.label(fig.axes[1], "행 = 시각 · 파랑 +, 빨강 − · $f$ 는 0–1", y=-0.05)
    cp.show(fig)


def plot_memory_examples(X_rows, labels):
    """지연 기억 데이터 두 개: (2, T) 채널 격자. 첫 시각의 ±1 신호와 쓰기 표시, 이후 잡음."""
    n = len(X_rows)
    T = X_rows[0].shape[1]
    fig = cp.new_figure((11, 1.9 * n + 0.6))
    gs = fig.add_gridspec(n, 1, hspace=0.9)
    for r, (M, y) in enumerate(zip(X_rows, labels)):
        ax = fig.add_subplot(gs[r]); cp.blank(ax)
        cp.draw_grid(ax, M, f"입력 $X[n]^{{\\top}}$ : (2, {T}) · 정답 $y = {y}$ (처음 신호가 {'양수' if y else '음수'})", kind="signed",
                     fmt="{:+.1f}", fontsize=8.5, vmax=1, row_labels=["신호", "쓰기 표시"], col_labels=[str(t) for t in range(1, T + 1)],
                     highlight=[(0, 0)])
    cp.label(fig.axes[-1], "금색 칸의 부호를 12시각 뒤 마지막 상태에서 답해야 함 · 그 사이는 작은 잡음", y=-0.35, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_gate_traces(X_seq, Cs, forget, inputs, y):
    """학습된 LSTM 이 한 시퀀스를 읽는 동안의 입력·메모리·forget·input gate (행 = 시각)."""
    T = X_seq.shape[0]
    fig = cp.new_figure((11, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.35, 0.8, 0.8, 0.8], wspace=0.5)
    panels = [(X_seq, "입력 (신호, 쓰기)", 1.0, "signed"), (Cs, "메모리 $c_t$", float(np.abs(Cs).max()), "signed"),
              (forget, "forget $f_t$", 1.0, "signed"), (inputs, "input $i_t$", 1.0, "signed")]
    for k, (S, title, vmax, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[k]); cp.blank(ax)
        cp.draw_rows(ax, S, f"{title} : ({T}, {S.shape[1]})", kind=kind, vmax=vmax)
        if k == 0:
            for t in range(T):
                ax.text(-0.8, t, f"$t={t + 1}$", ha="right", va="center", fontsize=8, color=cp.MUTED)
    cp.label(fig.axes[0], f"정답 $y = {y}$", y=-0.05, color=cp.INK, fontsize=9.5)
    cp.label(fig.axes[1], "행 = 시각 · 학습된 배열의 관측", y=-0.05)
    cp.show(fig)
