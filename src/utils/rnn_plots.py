"""06 RNN 노트북의 개념 그림과 구조 도식.

상태·기울기·one-hot·평균은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes
from .architecture_plots import plot_curves  # noqa: F401

SYMBOLS = {0: "_", 1: "A", 2: "B"}


def plot_rnn_unrolled(image, states, t_mark):
    """7을 위에서 아래로 한 행씩 읽는 RNN: 이미지의 t번째 행 → 시간축으로 펼친 셀 → 상태 띠(행 = 시각)."""
    T, H = states.shape
    fig = cp.new_figure((13.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.7, 0.75], wspace=0.3)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_image(ax_x, image, f"$x_7$ : {T}행을 위에서부터 읽음")
    ax_x.add_patch(Rectangle((-0.5, t_mark - 1 - 0.5), image.shape[1], 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
    cp.label(ax_x, f"금색 행 = $x_{{{t_mark}}}$ ({image.shape[1]},)", y=-0.05)
    ax_c = fig.add_subplot(gs[1])
    sp.draw_unrolled_chain(ax_c, ["1", "2", "⋯", str(t_mark), "⋯", str(T)], highlight=str(t_mark))
    ax_c.set_title("같은 $W_x, W_h, b$ 를 매 시각 다시 씀 · 상태 $h_t$ 만 시각마다 바뀜", color=cp.INK, fontsize=10.5, pad=4)
    ax_h = fig.add_subplot(gs[2]); cp.blank(ax_h)
    cp.draw_rows(ax_h, states, f"$h_t$ : ({T}, {H})", kind="signed", vmax=1)
    ax_h.add_patch(Rectangle((-0.5, t_mark - 1 - 0.5), H, 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
    cp.label(ax_h, "행 = 시각, 열 = 상태 좌표", y=-0.05)
    cp.connect(fig, ax_x, ax_c, "행 하나씩")
    cp.connect(fig, ax_c, ax_h, "매 시각의 $h_t$")
    cp.show(fig)


def plot_bptt(times, table, row_labels, col_labels):
    """왼쪽: 펼친 셀 위에 직접 기여 g_t 와 미래에서 온 r_t 의 금색 경로. 오른쪽: 스칼라 예제의 값 표."""
    fig = cp.new_figure((13, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.25], wspace=0.2)
    ax_c = fig.add_subplot(gs[0])
    sp.draw_unrolled_chain(ax_c, [str(t) for t in times], backward=True)
    ax_c.set_title("금색: 손실에서 직접 오는 $g_t$ 와 미래에서 돌아오는 $r_t$ 가 $h_t$ 에서 더해짐", color=cp.INK, fontsize=10.5, pad=4)
    ax_t = fig.add_subplot(gs[1]); cp.blank(ax_t)
    cp.draw_grid(ax_t, table, "스칼라 예제 ($W_x=1, W_h=0.5$), 마지막 시각부터", kind="plain", fmt="{:.3f}", fontsize=9,
                 row_labels=row_labels, col_labels=col_labels)
    cp.label(ax_t, "$\\delta_t = (g_t + r_t)(1 - h_t^2)$, $r_{t-1} = \\delta_t W_h$", y=-0.06, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def _draw_tokens(ax, tokens, title=None):
    ax.set_axis_off()
    ax.set_aspect("equal")
    for j, tok in enumerate(tokens):
        ax.add_patch(Rectangle((j, 0), 1, 1, facecolor="white", edgecolor=cp.MUTED, lw=0.8))
        ax.text(j + 0.5, 0.5, SYMBOLS[int(tok)], ha="center", va="center", fontsize=11,
                color=cp.INK if tok else cp.MUTED)   # bold는 한글 폰트에 없어 요청하지 않음(findfont 경고)
    ax.set_xlim(-0.15, len(tokens) + 0.15)
    ax.set_ylim(-0.15, 1.15)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)


def plot_order_examples(token_rows, onehots, means, labels):
    """기호열 → one-hot (3, T) → 시각 평균. 같은 평균, 다른 순서."""
    n = len(token_rows)
    T = len(token_rows[0])
    fig = cp.new_figure((12, 2.8 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[T, T, 2.4], wspace=0.4, hspace=0.55)
    for r in range(n):
        ax_t = fig.add_subplot(gs[r, 0])
        _draw_tokens(ax_t, token_rows[r], f"문장: 시각 1…{T}" if r == 0 else None)
        cp.label(ax_t, f"정답 $y={labels[r]}$ ({'A가 먼저' if labels[r] else 'B가 먼저'})", y=-0.12, color=cp.INK, fontsize=9.5)
        ax_o = fig.add_subplot(gs[r, 1]); cp.blank(ax_o)
        cp.draw_grid(ax_o, onehots[r], "one-hot $X[n]^{\\top}$ : (3, T)" if r == 0 else None, kind="count", fmt="{:g}",
                     fontsize=8.5, row_labels=["빈칸", "A", "B"], col_labels=[str(t) for t in range(1, T + 1)])
        ax_m = fig.add_subplot(gs[r, 2]); cp.blank(ax_m)
        cp.draw_bars(ax_m, means[r], "시각 평균 (3,)" if r == 0 else None, labels=["빈칸", "A", "B"], xlim=(0, 1))
        for i, v in enumerate(means[r]):
            ax_m.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9, color=cp.INK)
        cp.connect(fig, ax_t, ax_o, "one-hot" if r == 0 else None)
        cp.connect(fig, ax_o, ax_m, "평균" if r == 0 else None)
    cp.label(fig.axes[-1], "두 문장의 평균은 같음 → 순서를 읽는 상태가 있어야 구별", y=-0.2, color=cp.RED, fontsize=9.5)
    cp.show(fig)


def plot_state_paths(states_list, token_rows, predictions):
    """학습된 RNN의 상태 경로 두 개. 행 = 시각(왼쪽에 읽은 기호), 열 = 상태 좌표."""
    n = len(states_list)
    fig = cp.new_figure((3.6 * n + 1.5, 4.4))
    gs = fig.add_gridspec(1, n, wspace=0.9)
    for k, (S, toks, pred) in enumerate(zip(states_list, token_rows, predictions)):
        ax = fig.add_subplot(gs[k]); cp.blank(ax)
        cp.draw_rows(ax, S, f"문장 {k + 1}의 상태 경로 $h_t$ : ({S.shape[0]}, {S.shape[1]})", kind="signed", vmax=1)
        for t, tok in enumerate(toks):
            ax.text(-0.8, t, f"$t={t + 1}$  {SYMBOLS[int(tok)]}", ha="right", va="center", fontsize=8.5,
                    color=cp.INK if tok else cp.MUTED)
        cp.label(ax, f"마지막 상태 → 예측 {pred}", y=-0.05, color=cp.INK, fontsize=9.5)
    cp.label(fig.axes[0], "같은 파라미터, 다른 입력 순서 → 다른 경로", y=-0.14)
    cp.show(fig)


def plot_gradient_decay(times, curves):
    """마지막 상태에서 각 입력으로 돌아온 기울기 크기. w_h 가 1보다 작으면 사라지고 크면 커집니다."""
    fig = cp.new_figure((7.2, 3.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "입력 시각 $t$", "$|\\partial h_T / \\partial x_t|$ (로그 눈금)",
                "마지막 상태에서 각 입력으로 돌아온 기울기 $= w_x\\, w_h^{\\,T-t}$")
    for (name, ys), color in zip(curves.items(), [cp.BLUE, cp.INK, cp.RED]):
        ax.semilogy(times, ys, "o-", color=color, ms=3.5, lw=1.5, label=name)
    ax.axhline(1, color=cp.GOLD, lw=1.0, ls="--")
    ax.legend(frameon=False, fontsize=9, loc="center left")
    cp.show(fig)
