"""05 CNN 노트북의 개념 그림과 구조 도식.

합성곱·패치 행렬·기울기 누적·풀링 값은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)과 도안 묶음(plot_images)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .architecture_plots import plot_curves, plot_images  # noqa: F401  (노트북이 한 파일에서 가져가도록)

REGION_COLORS = ["#cfe0f3", "#f6d5d5", "#d9ecd0", "#fce8c3"]   # 2×2 풀링 구역 네 개의 옅은 색
PARAM_FILL = "#fce8c3"                                          # 파라미터가 있는 블록


def _frame(ax, w, h, size):
    """draw_cells 격자를 size×size 틀 가운데에 두어 옆 판과 칸 크기를 맞춘다."""
    cx, cy = w / 2, h / 2
    ax.set_xlim(cx - size / 2, cx + size / 2)
    ax.set_ylim(cy - size / 2, cy + size / 2)
    return ax


def plot_conv_scene(X, K, Y, positions):
    """숫자 격자 위의 합성곱 장면. 행마다 창을 다른 자리에 대고, 같은 커널로 창 ⊙ 커널 → 합 → 출력 한 칸."""
    X, K, Y = np.asarray(X, float), np.asarray(K, float), np.asarray(Y, float)
    h, w = X.shape
    kh, kw = K.shape
    rows = len(positions)
    size = max(h, w) + 0.3
    fig = cp.new_figure((10.5, 2.75 * rows))
    gs = fig.add_gridspec(rows, 4, width_ratios=[size, kh + 0.3, kh + 0.3, Y.shape[1] + 0.3], wspace=0.5, hspace=0.25)
    for r, (i, j) in enumerate(positions):
        first = r == 0
        ax_x = fig.add_subplot(gs[r, 0])
        sp.draw_cells(ax_x, X, f"입력 $X$ ({h}×{w})" if first else None, window=(i, j, kh, kw), fill=cp.GOLD)
        cp.label(ax_x, f"창의 왼쪽 위 = ({i}, {j})", y=-0.04)
        ax_k = fig.add_subplot(gs[r, 1])
        sp.draw_cells(ax_k, K, f"커널 $K$ ({kh}×{kw})" if first else None)
        _frame(ax_k, kw, kh, size)
        cp.label(ax_k, "모든 자리에서 같음", y=-0.04, color=cp.GOLD)
        patch = X[i:i + kh, j:j + kw]
        ax_m = fig.add_subplot(gs[r, 2])
        sp.draw_cells(ax_m, patch * K + 0.0, "창 ⊙ 커널" if first else None)     # + 0.0: 0×(−1)의 "-0" 표기를 없앰
        _frame(ax_m, kw, kh, size)
        ax_y = fig.add_subplot(gs[r, 3])
        sp.draw_cells(ax_y, Y, f"출력 $Y$ ({Y.shape[0]}×{Y.shape[1]})" if first else None, highlight_cells=[(i, j)])
        _frame(ax_y, Y.shape[1], Y.shape[0], size)
        cp.connect(fig, ax_x, ax_k, "창을 오림")
        cp.connect(fig, ax_k, ax_m, "같은 자리끼리 곱")
        cp.connect(fig, ax_m, ax_y, f"합 = {Y[i, j]:g}")
    cp.show(fig)


def plot_conv_on_image(image, kernels, kernel_titles, feature_maps, peak=None):
    """MNIST 이미지 → 3×3 커널 → 특성 맵. 행마다 커널이 다르고, 첫 행에는 가장 크게 반응한 창과 칸을 금색으로 표시."""
    rows = len(kernels)
    fig = cp.new_figure((10, 3.4 * rows))
    gs = fig.add_gridspec(rows, 3, width_ratios=[1, 0.5, 1], wspace=0.5, hspace=0.25)
    vmax = float(max(np.abs(f).max() for f in feature_maps))
    for r, (K, title, fmap) in enumerate(zip(kernels, kernel_titles, feature_maps)):
        K = np.asarray(K, float)
        ax_x = fig.add_subplot(gs[r, 0]); cp.blank(ax_x)
        cp.draw_image(ax_x, image, f"$x_7$ ({image.shape[0]}×{image.shape[1]})" if r == 0 else None)
        ax_k = fig.add_subplot(gs[r, 1]); cp.blank(ax_k)
        cp.draw_grid(ax_k, K, title, kind="signed", fmt="{:g}", fontsize=9)
        ax_y = fig.add_subplot(gs[r, 2]); cp.blank(ax_y)
        cp.draw_image(ax_y, fmap, f"특성 맵 $Y$ ({fmap.shape[0]}×{fmap.shape[1]})" if r == 0 else None, kind="signed", vmax=vmax)
        if peak is not None and r == 0:
            i, j = peak
            ax_x.add_patch(Rectangle((j - 0.5, i - 0.5), K.shape[1], K.shape[0], fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
            ax_y.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
        cp.connect(fig, ax_x, ax_k, "창 ⊙ 커널" if r == 0 else None)
        cp.connect(fig, ax_k, ax_y, "모든 자리에서 합" if r == 0 else None)
    cp.label(fig.axes[-1], "파랑 = 양수, 빨강 = 음수 · 금색 창의 합이 금색 칸", y=-0.05)
    cp.show(fig)


def plot_patch_matrix(P, k, rows, Y, highlight_row=0):
    """패치 행렬 P(행 = 창의 자리, 열 = 창 안의 칸) @ 펼친 커널 = 출력 열 → (oh, ow)로 접기."""
    P, rows, Y = np.asarray(P, float), np.asarray(rows, float), np.asarray(Y, float)
    n_rows, n_cols = P.shape
    oh, ow = Y.shape
    fig = cp.new_figure((11.5, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[n_cols, 1.4, 1.4, ow + 0.8], wspace=0.6)
    ax_p = fig.add_subplot(gs[0]); cp.blank(ax_p)
    cp.draw_grid(ax_p, P, "패치 행렬 $P$ : 행 = 창의 자리 $(i, j)$, 열 = 창 안의 칸", kind="plain", fmt="{:g}", fontsize=8.5,
                 row_labels=[f"({i},{j})" for i in range(oh) for j in range(ow)], col_labels=[str(c) for c in range(n_cols)])
    ax_p.add_patch(Rectangle((-0.5, highlight_row - 0.5), n_cols, 1, fill=False, ec=cp.GOLD, lw=2.2, zorder=3))
    ax_k = fig.add_subplot(gs[1]); cp.blank(ax_k)
    cp.draw_grid(ax_k, np.asarray(k, float)[:, None], "$K_{flat}^{\\top}$", kind="signed", fmt="{:g}", fontsize=8.5)
    ax_r = fig.add_subplot(gs[2]); cp.blank(ax_r)
    cp.draw_grid(ax_r, rows[:, None], "$Y_{rows}$", kind="signed", fmt="{:g}", fontsize=8.5, highlight=[(highlight_row, 0)])
    ax_y = fig.add_subplot(gs[3]); cp.blank(ax_y)
    cp.draw_grid(ax_y, Y, f"$Y$ : ({oh}, {ow})", kind="signed", fmt="{:g}", fontsize=9.5,
                 highlight=[divmod(highlight_row, ow)], frame_to=(ow, n_rows))
    cp.connect(fig, ax_p, ax_k, "@")
    cp.connect(fig, ax_k, ax_r, "=")
    cp.connect(fig, ax_r, ax_y, "reshape")
    cp.label(ax_p, "금색 행 = 1절에서 (0, 0)에 댄 창의 아홉 값 · 행마다 같은 커널과 내적", y=-0.03)
    cp.show(fig)


def plot_overlap_count(parts, positions, window_shape, count):
    """창 자리마다 돌려주는 기여(0/1 격자와 금색 창)를 같은 칸에 더하면 겹친 횟수 격자가 됩니다."""
    n = len(parts)
    kh, kw = window_shape
    fig = cp.new_figure((2.3 * n + 3.4, 2.9))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.2], wspace=0.45)
    axes = []
    for k, (part, (i, j)) in enumerate(zip(parts, positions)):
        ax = fig.add_subplot(gs[k])
        sp.draw_cells(ax, part, f"창 ({i}, {j})의 기여", window=(i, j, kh, kw), fontsize=9.5)
        axes.append(ax)
    ax_c = fig.add_subplot(gs[n]); cp.blank(ax_c)
    cp.draw_grid(ax_c, count, "겹친 횟수 = 기여의 합", kind="count", fmt="{:g}", fontsize=10.5)
    for a, b in zip(axes, axes[1:]):
        cp.connect(fig, a, b, "+")
    cp.connect(fig, axes[-1], ax_c, "=")
    cp.label(axes[0], "값 1 = 커널과 상류 기울기를 모두 1로 둔 기여", y=-0.05)
    cp.show(fig)


def plot_pooling(X, argmax_cells, Y_max, Y_mean, feature, pooled):
    """왼쪽: 4×4 격자의 2×2 구역(색)과 최댓값 자리(금색) → 최대·평균 pooling 결과. 오른쪽: 7의 특성 맵을 2×2 최대 pooling."""
    X = np.asarray(X, float)
    h, w = X.shape
    colors = [[REGION_COLORS[(i // 2) * 2 + (j // 2)] for j in range(w)] for i in range(h)]
    small = [[REGION_COLORS[a * 2 + b] for b in range(2)] for a in range(2)]
    fig = cp.new_figure((13, 3.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 0.8, 0.8, 1.35, 0.68], wspace=0.4)
    ax_x = fig.add_subplot(gs[0])
    sp.draw_cells(ax_x, X, f"입력 {h}×{w} · 색 = 2×2 구역", cell_colors=colors, highlight_cells=argmax_cells)
    cp.label(ax_x, "금색 = 구역의 최댓값 자리", y=-0.04)
    ax_m = fig.add_subplot(gs[1])
    sp.draw_cells(ax_m, Y_max, "최대 pooling", cell_colors=small, fmt="{:g}")
    _frame(ax_m, 2, 2, h + 0.3)
    ax_a = fig.add_subplot(gs[2])
    sp.draw_cells(ax_a, Y_mean, "평균 pooling", cell_colors=small, fmt="{:g}")
    _frame(ax_a, 2, 2, h + 0.3)
    ax_f = fig.add_subplot(gs[3]); cp.blank(ax_f)
    vmax = float(np.abs(feature).max())
    cp.draw_image(ax_f, feature, f"7의 특성 맵 ({feature.shape[0]}×{feature.shape[1]})", kind="signed", vmax=vmax)
    ax_q = fig.add_subplot(gs[4]); cp.blank(ax_q)
    cp.draw_image(ax_q, pooled, f"{pooled.shape[0]}×{pooled.shape[1]}", kind="signed", vmax=vmax)
    cp.connect(fig, ax_x, ax_m, "구역마다 하나")
    cp.connect(fig, ax_f, ax_q, "max 2×2")
    cp.show(fig)


def plot_cnn_blocks(blocks, param_blocks=(1, 5)):
    """입력에서 손실까지의 블록 흐름. 아래에 텐서 모양, 파라미터가 있는 블록은 옅은 금색."""
    fig = cp.new_figure((13.5, 2.2))
    ax = fig.add_subplot(111)
    colors = [PARAM_FILL if i in param_blocks else sp.FILL for i in range(len(blocks))]
    sp.draw_blocks(ax, blocks, colors=colors, width=1.75, gap=0.5, fontsize=9.5)
    cp.label(ax, "금색 상자 = 학습하는 파라미터가 있는 층 · 상자 아래 = 그 단계의 텐서 모양", y=0.0)
    cp.show(fig)
