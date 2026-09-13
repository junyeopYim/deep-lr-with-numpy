"""00b 벡터화 노트북의 개념 그림. 시간·평균·거리·인덱싱 값은 본문에서 계산하고 여기서는 받은 배열만 그립니다."""

import numpy as np
import matplotlib.pyplot as plt

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_timing_bars(times, title="같은 계산, 다른 시간"):
    """{이름: 초} → 로그 눈금 가로 막대. 가장 느린 것 대비 배율을 적습니다."""
    names = list(times)
    values = np.array([times[n] * 1e3 for n in names])          # ms
    fig = cp.new_figure((7.0, 0.9 + 0.7 * len(names)))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "한 번 실행에 걸린 시간 (ms, 로그 눈금)", None, title)
    colors = [cp.RED if v == values.max() else cp.BLUE for v in values]
    ax.barh(names, values, color=colors, height=0.55)
    ax.set_xscale("log")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v * 1.15, i, f"{v:.3g} ms" + ("" if v == values.max() else f"  · {values.max() / v:.0f}배 빠름"),
                va="center", fontsize=9, color=cp.INK)
    ax.set_xlim(values.min() / 3, values.max() * 40)
    cp.show(fig)


def plot_centering(images, mean_image, centered, labels):
    """행마다: 이미지 − 평균 이미지 = 중심화한 이미지(부호 있음)."""
    n = len(images)
    fig, rows = cp.flow([1, 0.3, 1, 0.3, 1], rows=n, height=2.4, unit=1.7, wspace=0.1, hspace=0.25)
    vmax = float(max(np.abs(c).max() for c in centered))
    for r, (ax_x, ax_m, ax_mean, ax_e, ax_c) in enumerate(rows):
        cp.draw_image(ax_x, images[r], f"$x_{{{labels[r]}}}$" if True else None)
        cp.draw_formula(ax_m, "$-$", None, fontsize=16)
        cp.draw_image(ax_mean, mean_image, "평균 이미지 $\\bar x$" if r == 0 else None)
        cp.draw_formula(ax_e, "$=$", None, fontsize=16)
        cp.draw_image(ax_c, centered[r], "$x - \\bar x$ (파랑 +, 빨강 −)" if r == 0 else None, kind="signed", vmax=vmax)
    cp.label(rows[-1][4], "평균보다 진한 자리는 파랑, 옅은 자리는 빨강", y=-0.08)
    cp.show(fig)


def plot_broadcast_shapes(cases):
    """cases = [(shape_a, op, shape_b, result_or_None, 설명)]. 뒤 축부터 오른쪽 정렬해 규칙을 보여 줍니다."""
    n = len(cases)
    fig = cp.new_figure((10.5, 1.15 * n + 0.6))
    ax = fig.add_subplot(111)
    ax.set_axis_off(); ax.set_xlim(0, 10.5); ax.set_ylim(0, n)
    def fmt(shape):
        return "(" + ", ".join(f"{d:>5}" for d in shape) + ")"
    for i, (a, op, b, result, note) in enumerate(cases):
        y = n - i - 0.5
        width = max(len(a), len(b))
        a_pad = (None,) * (width - len(a)) + tuple(a)
        b_pad = (None,) * (width - len(b)) + tuple(b)
        def line(shape):
            return "  ".join(f"{'':>5}" if d is None else f"{d:>5}" for d in shape)
        ax.text(0.2, y + 0.18, line(a_pad), family="monospace", fontsize=10.5, color=cp.INK, va="center")
        ax.text(0.2, y - 0.18, line(b_pad), family="monospace", fontsize=10.5, color=cp.INK, va="center")
        ax.text(3.9, y, op, fontsize=12, color=cp.MUTED, va="center", ha="center")
        if result is None:
            ax.text(4.4, y, "오류: 같지도 않고 1도 아닌 축", fontsize=10, color=cp.RED, va="center")
        else:
            ax.text(4.4, y, "→ " + line(tuple(result)), family="monospace", fontsize=10.5, color=cp.BLUE, va="center")
        ax.text(7.6, y, note, fontsize=9, color=cp.MUTED, va="center")
        ax.axhline(n - i - 1, color="#eeeeee", lw=0.8)
    ax.set_title("뒤 축부터 맞춰 봅니다: 같으면 통과, 1이면 늘림, 없으면 앞에 1을 붙임", color=cp.INK, fontsize=10.5)
    cp.show(fig)


def plot_distance_grid(D, row_labels, col_labels, title):
    """거리표 (행: 이미지, 열: 평균 이미지). 행마다 가장 가까운 열을 금색으로."""
    fig = cp.new_figure((7.2, 6.0))
    ax = fig.add_subplot(111)
    hi = [(i, int(np.argmin(D[i]))) for i in range(len(D))]
    cp.draw_grid(ax, D, None, kind="blues", fmt="{:.0f}", highlight=hi, fontsize=8.5,
                 row_labels=row_labels, col_labels=col_labels)
    ax.set_title(title, color=cp.INK, fontsize=10.5, pad=20)
    cp.show(fig)


def plot_indexing(labels, mask, onehot, P, rows, targets):
    """세 가지 고르기: 마스크로 행 고르기, one-hot 행렬, (행, 열) 쌍으로 칸 고르기."""
    fig = cp.new_figure((11.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0]); cp.blank(ax)
    cp.draw_grid(ax, np.asarray(labels)[:, None], "마스크 `y == 7`", kind="plain", fmt="{:.0f}",
                 highlight=[(i, 0) for i in np.flatnonzero(mask)], row_labels=[f"y[{i}]" for i in range(len(labels))],
                 col_labels=[""], fontsize=9)
    cp.label(ax, "True인 행만 골라 X[mask]", y=-0.05)
    ax = fig.add_subplot(gs[1]); cp.blank(ax)
    cp.draw_grid(ax, onehot, "one-hot `np.eye(10)[y]`", kind="blues", fmt="{:.0f}", fontsize=7.5,
                 row_labels=[f"y={l}" for l in labels], col_labels=[str(k) for k in range(onehot.shape[1])])
    cp.label(ax, "행 n의 y[n]번째 칸만 1", y=-0.05)
    ax = fig.add_subplot(gs[2]); cp.blank(ax)
    cp.draw_grid(ax, P, "`P[rows, y]` : 쌍으로 고르기", kind="plain", fmt="{:.2f}",
                 highlight=list(zip(rows, targets)), row_labels=[f"행 {r}" for r in rows], col_labels=[str(k) for k in range(P.shape[1])])
    cp.label(ax, "rows[n]행, y[n]열의 칸 N개", y=-0.05)
    cp.show(fig)
