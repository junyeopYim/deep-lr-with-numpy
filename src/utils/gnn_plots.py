"""10 GNN 노트북의 개념 그림과 구조 도식.

인접 행렬·집계·층 출력·순열은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
MNIST 대신 작은 그래프 도식을 concept_plots 의 색·여백 규칙으로 그립니다. 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from . import concept_plots as cp
from .architecture_plots import plot_curves  # noqa: F401

LIGHT_BLUE = "#cfe0f3"
LIGHTER_BLUE = "#e9f0f8"


def draw_graph(ax, A, positions, *, labels=None, fills=None, notes=None, highlight_nodes=(), highlight_edges=(),
               directed=False, title=None, r=0.17):
    """작은 그래프. A[i, j] = 1 은 j 에서 i 로 정보를 보낼 수 있다는 뜻(directed 면 j→i 화살표)."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    A = np.asarray(A)
    pos = np.asarray(positions, float)
    M = len(pos)
    for i in range(M):
        for j in range(M):
            if not A[i, j] or (not directed and i > j):
                continue
            hl = (i, j) in highlight_edges
            color, lw = (cp.GOLD, 1.9) if hl else (cp.MUTED, 1.0)
            if directed:                                             # j → i
                start, end = pos[j], pos[i]
                u = (end - start) / np.linalg.norm(end - start)
                ax.add_patch(FancyArrowPatch(start + u * r, end - u * r, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                             connectionstyle="arc3,rad=0.18", zorder=1))
            else:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=color, lw=lw, zorder=1)
    for i in range(M):
        fill = fills[i] if fills is not None else "white"
        hl = i in highlight_nodes
        ax.add_patch(Circle(pos[i], r, facecolor=fill, edgecolor=cp.GOLD if hl else cp.MUTED, lw=2.0 if hl else 1.0, zorder=2))
        ax.text(pos[i, 0], pos[i, 1], str(labels[i]) if labels is not None else str(i), ha="center", va="center", fontsize=9.5, color=cp.INK, zorder=3)
        if notes is not None:
            ax.text(pos[i, 0], pos[i, 1] - r - 0.07, notes[i], ha="center", va="top", fontsize=8, color=cp.MUTED, zorder=3)
    pad = 0.5
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad - (0.15 if notes is not None else 0), pos[:, 1].max() + pad)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)
    return ax


def plot_aggregation(A, positions, X, sum_messages, mean_messages):
    """방향 그래프 → 인접 행렬(행 = 받는 노드) → 노드 특성 → 이웃 합 AX 와 이웃 평균 SX."""
    M = len(A)
    node_labels = [f"노드 {i}" for i in range(M)]
    fig = cp.new_figure((13.5, 3.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 1.1, 0.9, 0.9, 0.9], wspace=0.55)
    ax_g = fig.add_subplot(gs[0])
    draw_graph(ax_g, A, positions, directed=True, title="화살표 = 정보가 가는 방향")
    cp.label(ax_g, "노드 1은 받는 화살표가 없음", y=-0.02)
    ax_a = fig.add_subplot(gs[1]); cp.blank(ax_a)
    cp.draw_grid(ax_a, A, "$A$ : 행 = 받는 노드 $i$", kind="count", fmt="{:g}", fontsize=9.5, row_labels=[f"받는 {i}" for i in range(M)],
                 col_labels=[f"보내는 {j}" for j in range(M)], vmax=1)
    ax_x = fig.add_subplot(gs[2]); cp.blank(ax_x)
    cp.draw_grid(ax_x, X, "$X$ : 노드별 특성", kind="count", fmt="{:g}", fontsize=9.5, row_labels=node_labels)
    ax_s = fig.add_subplot(gs[3]); cp.blank(ax_s)
    cp.draw_grid(ax_s, sum_messages, "$AX$ : 이웃 합", kind="count", fmt="{:g}", fontsize=9.5, row_labels=node_labels)
    ax_m = fig.add_subplot(gs[4]); cp.blank(ax_m)
    cp.draw_grid(ax_m, mean_messages, "$SX$ : 이웃 평균", kind="count", fmt="{:g}", fontsize=9.5, row_labels=node_labels)
    cp.connect(fig, ax_g, ax_a, "연결을 0/1로"); cp.connect(fig, ax_x, ax_s, "$A$ @"); cp.connect(fig, ax_s, ax_m, "degree 로 나눔")
    cp.label(ax_m, "받는 이웃이 없는 노드 1은 0", y=-0.06)
    cp.show(fig)


def plot_layer_paths(A, positions, node, x_i, m_i, z_self, z_neigh, Z_i, H_i, in_edges):
    """노드 하나의 갱신: 자기 특성은 W_s 로, 이웃 합은 W_n 으로 옮겨 더한 뒤 tanh."""
    H = len(Z_i)
    fig = cp.new_figure((13.5, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 0.9, 1.1, 1.1, 1.1], wspace=0.6)
    ax_g = fig.add_subplot(gs[0])
    draw_graph(ax_g, A, positions, directed=True, highlight_nodes=(node,), highlight_edges=in_edges, title=f"노드 {node}이 받는 이웃 (금색)")
    ax_in = fig.add_subplot(gs[1]); cp.blank(ax_in)
    cp.draw_grid(ax_in, np.stack([x_i, m_i]), "자기 특성 / 이웃 합", kind="count", fmt="{:g}", fontsize=9.5,
                 row_labels=[f"$x_{node}$", f"$m_{node}=(SX)_{node}$"])
    ax_two = fig.add_subplot(gs[2]); cp.blank(ax_two)
    cp.draw_grid(ax_two, np.stack([z_self, z_neigh]), "두 경로의 기여", kind="signed", fmt="{:.2f}", fontsize=9,
                 row_labels=["$x_iW_s$", "$m_iW_n$"])
    ax_z = fig.add_subplot(gs[3]); cp.blank(ax_z)
    cp.draw_grid(ax_z, np.asarray(Z_i)[None], "$z_i = x_iW_s + m_iW_n + b$", kind="signed", fmt="{:.2f}", fontsize=9, frame_to=(H, 2))
    ax_h = fig.add_subplot(gs[4]); cp.blank(ax_h)
    cp.draw_grid(ax_h, np.asarray(H_i)[None], "$h_i = \\tanh(z_i)$", kind="signed", fmt="{:.2f}", fontsize=9, vmax=1, frame_to=(H, 2))
    cp.connect(fig, ax_in, ax_two, "$W_s$ / $W_n$"); cp.connect(fig, ax_two, ax_z, "합 $+\\,b$"); cp.connect(fig, ax_z, ax_h, "tanh")
    cp.label(ax_h, "모든 노드가 같은 $W_s, W_n, b$ 를 씀", y=-0.08)
    cp.show(fig)


def plot_hops(A, positions, center, one_hop, two_hop):
    """가운데 노드(금색)가 1층·2층 뒤에 받을 수 있는 노드(파랑 진하기), 그리고 (I+A)>0, (I+A)²>0 표."""
    M = len(A)
    reach1, reach2 = one_hop[center], two_hop[center]
    fills = [LIGHT_BLUE if reach1[j] else (LIGHTER_BLUE if reach2[j] else "white") for j in range(M)]
    fig = cp.new_figure((12.5, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.45)
    ax_g = fig.add_subplot(gs[0])
    draw_graph(ax_g, A, positions, fills=fills, highlight_nodes=(center,), title=f"노드 {center}이 받을 수 있는 범위")
    cp.label(ax_g, "진한 파랑 = 1층(자기+이웃) · 옅은 파랑 = 2층에서 추가", y=-0.02)
    labels = [str(i) for i in range(M)]
    ax_1 = fig.add_subplot(gs[1]); cp.blank(ax_1)
    cp.draw_grid(ax_1, one_hop.astype(float), "1층: $(I + A) > 0$", kind="count", fmt="{:g}", fontsize=9, row_labels=labels, col_labels=labels,
                 highlight=[(center, j) for j in range(M) if reach1[j]], vmax=1)
    ax_2 = fig.add_subplot(gs[2]); cp.blank(ax_2)
    cp.draw_grid(ax_2, two_hop.astype(float), "2층: $(I + A)^2 > 0$", kind="count", fmt="{:g}", fontsize=9, row_labels=labels, col_labels=labels,
                 highlight=[(center, j) for j in range(M) if reach2[j]], vmax=1)
    cp.connect(fig, ax_1, ax_2, "한 층 더")
    cp.label(ax_2, "행 = 받는 노드 · 가능한 연결 범위이지 실제 기여의 크기는 아님", y=-0.05)
    cp.show(fig)


def plot_two_graphs(As, positions_list, degrees, titles, note="노드 안 숫자 = 번호 · 색 = degree"):
    """그래프 두 개를 나란히. 노드 색은 degree 의 진하기, 아래 숫자는 degree."""
    fig = cp.new_figure((9.5, 3.4))
    gs = fig.add_gridspec(1, len(As), wspace=0.3)
    dmax = max(float(np.max(d)) for d in degrees) or 1.0
    for k, (A, pos, deg, title) in enumerate(zip(As, positions_list, degrees, titles)):
        ax = fig.add_subplot(gs[k])
        fills = [plt.cm.Blues(0.15 + 0.55 * float(d) / dmax) for d in deg]
        draw_graph(ax, A, pos, fills=fills, notes=[f"deg {int(d)}" for d in deg], title=title)
    cp.label(fig.axes[0], note, y=-0.04)
    cp.show(fig)


def plot_permutation(A, positions, perm, A_perm):
    """같은 그래프의 노드 번호를 바꾸면 A 의 행과 열이 함께 바뀝니다."""
    M = len(A)
    inverse = np.argsort(perm)                                        # 옛 노드 j 의 새 번호
    fig = cp.new_figure((12.5, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1.2, 1], wspace=0.5)
    ax_g = fig.add_subplot(gs[0])
    draw_graph(ax_g, A, positions, directed=True, title="원래 번호")
    ax_a = fig.add_subplot(gs[1]); cp.blank(ax_a)
    cp.draw_grid(ax_a, A, "$A$", kind="count", fmt="{:g}", fontsize=9.5, row_labels=[str(i) for i in range(M)], col_labels=[str(j) for j in range(M)], vmax=1)
    ax_p = fig.add_subplot(gs[2])
    draw_graph(ax_p, A, positions, directed=True, labels=inverse, title="같은 그래프, 새 번호")
    ax_b = fig.add_subplot(gs[3]); cp.blank(ax_b)
    cp.draw_grid(ax_b, A_perm, "$A' = \\Pi A \\Pi^{\\top}$", kind="count", fmt="{:g}", fontsize=9.5,
                 row_labels=[str(i) for i in range(M)], col_labels=[str(j) for j in range(M)], vmax=1)
    cp.connect(fig, ax_a, ax_p, "perm = [" + ", ".join(str(int(p)) for p in perm) + "]")
    cp.label(ax_b, "행(받는)과 열(보내는)을 함께 바꿈 · 새 i = 옛 perm[i]", y=-0.05)
    cp.show(fig)
