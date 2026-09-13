"""구조 도식 부품: 뉴런·층 연결, 블록 흐름, 격자 위의 창(커널·풀링), 손실 곡면 위 경로.

concept_plots가 "데이터가 어떻게 흐르나"를 실제 배열로 보여 준다면, 이 모듈은
"구조가 어떻게 생겼나"를 원·선·상자로 보여 줍니다. 색·여백 규칙은 concept_plots와 같습니다.

부품
----
- ``draw_network(ax, layers)``      : 층별 노드 수로 완전 연결 도식. 큰 층은 ⋮으로 줄인다.
- ``index_labels(sym, n, max_show)``: 큰 층에 붙일 $x_0, x_1, x_2, \\ldots, x_{n-1}$ 라벨.
- ``draw_blocks(ax, blocks)``       : (이름, 모양) 상자들을 화살표로 이은 블록 흐름.
- ``draw_cells(ax, M, window)``     : 숫자 격자 위에 금색 창을 표시. 커널·풀링 한 장면용.
- ``draw_surface_paths(ax3d, ...)`` : 손실 곡면과 그 위의 optimizer 경로.
- ``draw_contour_paths(ax, ...)``   : 같은 것을 등고선 위에서.

계산(활성 값, 경로 좌표, 곡면 값)은 노트북 본문에서 하고 여기서는 받은 값만 그립니다.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from .concept_plots import BLUE, GOLD, INK, MUTED, RED

FILL = "#eef2f7"       # 상자·노드 배경
FILL_OUT = "#dbe7f5"   # 출력 노드 배경


# ---------------------------------------------------------------- 뉴런·층 연결
def index_labels(sym, n, max_show=5):
    """큰 층의 라벨. n ≤ max_show면 전부, 아니면 앞 max_show-2개 + ⋮ + 마지막."""
    if n <= max_show:
        return [f"${sym}_{{{i}}}$" for i in range(n)]
    head = [f"${sym}_{{{i}}}$" for i in range(max_show - 2)]
    return head + ["⋮", f"${sym}_{{{n - 1}}}$"]


def _shown(n, max_show):
    """실제로 그릴 노드 수와 ⋮ 자리(없으면 None)."""
    if n <= max_show:
        return n, None
    return max_show, max_show - 2


def draw_network(ax, layers, *, node_labels=None, layer_titles=None, edge_labels=None,
                 max_show=5, arrows_out=False, highlight_edges=None, node_r=0.22, y_gap=0.75,
                 x_gap=2.2, edge_color=MUTED, edge_lw=0.8):
    """완전 연결 층 도식.

    layers       : 층별 노드 수, 예 [2, 3, 1] 또는 [784, 10]
    node_labels  : 층별 라벨 목록(그려지는 노드 수만큼). 없으면 비워 둔다. index_labels() 참고.
    layer_titles : 층 위에 적을 이름, 예 ["입력", "은닉", "출력"]
    edge_labels  : {(층, i, j): 텍스트} — 앞 층 i번에서 다음 층 j번으로 가는 선 위에 적는다.
    highlight_edges : [(층, i, j)] — 금색으로 강조할 선.
    """
    ax.set_axis_off()
    ax.set_aspect("equal")
    positions = []
    for li, n in enumerate(layers):
        shown, dots = _shown(n, max_show)
        ys = (np.arange(shown) - (shown - 1) / 2) * -y_gap
        positions.append([(li * x_gap, y, (dots is not None and k == dots)) for k, y in enumerate(ys)])

    # 선 먼저(뒤에 깔리도록)
    for li in range(len(layers) - 1):
        for i, (x0, y0, d0) in enumerate(positions[li]):
            for j, (x1, y1, d1) in enumerate(positions[li + 1]):
                if d0 or d1:
                    continue
                key = (li, i, j)
                hl = highlight_edges and key in highlight_edges
                ax.plot([x0 + node_r, x1 - node_r], [y0, y1], color=GOLD if hl else edge_color,
                        lw=1.8 if hl else edge_lw, zorder=1)
                if edge_labels and key in edge_labels:
                    t = 0.26 + 0.2 * (j % 3)      # 교차하는 선의 라벨이 한 점에 몰리지 않게 위치를 다르게 둔다
                    ax.text(x0 + t * (x1 - x0), y0 + t * (y1 - y0) + 0.09, edge_labels[key],
                            ha="center", va="bottom", fontsize=9.5, color=INK, zorder=3,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    # 노드
    last = len(layers) - 1
    for li, nodes in enumerate(positions):
        labels = (node_labels[li] if node_labels and node_labels[li] is not None else [""] * len(nodes))
        for k, (x, y, is_dots) in enumerate(nodes):
            if is_dots:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=13, color=INK, zorder=3)
                continue
            ax.add_patch(Circle((x, y), node_r, facecolor=FILL_OUT if li == last else "white",
                                edgecolor=MUTED, lw=1.0, zorder=2))
            if k < len(labels) and labels[k] and labels[k] != "⋮":
                ax.text(x, y, labels[k], ha="center", va="center", fontsize=9.5, color=INK, zorder=3)
            if arrows_out and li == last:
                ax.add_patch(FancyArrowPatch((x + node_r, y), (x + node_r + 0.55, y), arrowstyle="-|>",
                                             mutation_scale=11, color=INK, lw=1.0, zorder=2))
        if layer_titles and li < len(layer_titles) and layer_titles[li]:
            top = max(y for _, y, _ in nodes) + node_r + 0.35
            ax.text(nodes[0][0], top, layer_titles[li], ha="center", va="bottom", fontsize=10, color=INK)

    xs = [x for nodes in positions for x, _, _ in nodes]
    ys = [y for nodes in positions for _, y, _ in nodes]
    ax.set_xlim(min(xs) - 0.7, max(xs) + (1.0 if arrows_out else 0.7))
    ax.set_ylim(min(ys) - 0.6, max(ys) + (1.2 if layer_titles else 0.9))
    return positions


# ---------------------------------------------------------------- 블록 흐름
def draw_blocks(ax, blocks, *, colors=None, width=1.7, height=0.9, gap=0.55, fontsize=9.5):
    """(이름, 아래 설명) 상자를 가로로 잇는다. colors는 상자별 배경색 목록(없으면 회청색)."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    for i, block in enumerate(blocks):
        name, sub = (block if isinstance(block, (tuple, list)) else (block, ""))
        x = i * (width + gap)
        ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=(colors[i] if colors else FILL), edgecolor=MUTED, lw=1.0))
        ax.text(x + width / 2, height * 0.5, name, ha="center", va="center", fontsize=fontsize, color=INK)
        if sub:
            ax.text(x + width / 2, -0.12, sub, ha="center", va="top", fontsize=8.5, color=MUTED)
        if i < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.05, height / 2), (x + width + gap - 0.05, height / 2),
                                         arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.0))
    ax.set_xlim(-0.2, len(blocks) * (width + gap) - gap + 0.2)
    ax.set_ylim(-0.75, height + 0.3)
    return ax


# ---------------------------------------------------------------- 시간축으로 펼친 순환 셀
def draw_unrolled_chain(ax, times, *, cell_text="tanh", state="h", inp="x", w_state="$W_h$", w_input="$W_x$",
                        highlight=None, backward=False, width=1.15, height=0.75, gap=1.25, fontsize=9.5):
    """시간축으로 펼친 순환 셀 도식. times는 아래첨자 라벨 목록이고 "⋯"은 점으로 그린다.

    같은 파라미터 라벨(w_state, w_input)을 모든 화살표에 적어 "매 시각 같은 W"를 보여 준다.
    highlight는 금색 테두리로 강조할 시각 라벨. backward=True면 손실에서 직접 오는 g_t(위→아래)와
    미래에서 돌아오는 r_t(오른쪽→왼쪽)를 금색으로 덧그린다. 06 RNN·07 LSTM에서 쓴다.
    """
    ax.set_axis_off()
    ax.set_aspect("equal")
    step = width + gap
    xs = [i * step for i in range(len(times))]
    arrow = dict(arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.0)
    gold = dict(arrowstyle="-|>", mutation_scale=10, color=GOLD, lw=1.6)
    for x, t in zip(xs, times):
        cx = x + width / 2
        if t == "⋯":
            ax.text(cx, height / 2, "⋯", ha="center", va="center", fontsize=14, color=INK)
            continue
        hl = highlight is not None and str(t) == str(highlight)
        ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=FILL, edgecolor=GOLD if hl else MUTED, lw=1.8 if hl else 1.0))
        ax.text(cx, height / 2, cell_text, ha="center", va="center", fontsize=fontsize, color=INK)
        ax.add_patch(FancyArrowPatch((cx, -0.85), (cx, -0.03), **arrow))                      # 입력 x_t (아래에서)
        ax.text(cx, -0.95, f"${inp}_{{{t}}}$", ha="center", va="top", fontsize=fontsize, color=INK)
        ax.text(cx + 0.08, -0.45, w_input, ha="left", va="center", fontsize=8.5, color=INK)
        ax.add_patch(FancyArrowPatch((cx, height + 0.03), (cx, height + 0.7), **arrow))       # 상태 h_t (위로)
        ax.text(cx, height + 0.78, f"${state}_{{{t}}}$", ha="center", va="bottom", fontsize=fontsize, color=INK)
        if backward:
            ax.add_patch(FancyArrowPatch((cx + 0.3, height + 0.7), (cx + 0.3, height + 0.03), **gold))
            ax.text(cx + 0.38, height + 0.42, f"$g_{{{t}}}$", ha="left", va="center", fontsize=8.5, color=GOLD)
    y_mid = height / 2
    x_prev = xs[0] - gap + 0.15
    ax.text(x_prev - 0.05, y_mid, f"${state}_0$", ha="right", va="center", fontsize=fontsize, color=INK)
    for i, (x, t) in enumerate(zip(xs, times)):
        x_in = x + (width * 0.25 if t == "⋯" else 0)
        ax.add_patch(FancyArrowPatch((x_prev + 0.03, y_mid), (x_in - 0.03, y_mid), **arrow))   # 상태를 다음 시각으로
        if t != "⋯" and (i == 0 or times[i - 1] != "⋯"):
            ax.text((x_prev + x_in) / 2, y_mid + 0.08, w_state, ha="center", va="bottom", fontsize=8.5, color=INK)
        if backward and i > 0 and t != "⋯" and times[i - 1] != "⋯":
            ax.add_patch(FancyArrowPatch((x_in - 0.03, y_mid - 0.22), (x_prev + 0.03, y_mid - 0.22), **gold))
            ax.text((x_prev + x_in) / 2, y_mid - 0.3, f"$r_{{{times[i - 1]}}}$", ha="center", va="top", fontsize=8.5, color=GOLD)
        x_prev = x + (width * 0.75 if t == "⋯" else width)
    if backward:
        ax.add_patch(FancyArrowPatch((xs[0] - 0.03, y_mid - 0.22), (xs[0] - gap + 0.18, y_mid - 0.22), **gold))
        ax.text(xs[0] - gap / 2 + 0.07, y_mid - 0.3, f"$d{state}_0$", ha="center", va="top", fontsize=8.5, color=GOLD)
    ax.set_xlim(xs[0] - gap - 0.5, xs[-1] + width + 0.5)
    ax.set_ylim(-1.5, height + 1.4)
    return xs


# ---------------------------------------------------------------- 격자 위의 창
def draw_cells(ax, M, title=None, *, window=None, fill=None, fmt="{:g}", fontsize=9.5,
               cell_colors=None, highlight_cells=None):
    """숫자 격자. window=(행, 열, 높이, 너비)는 금색 테두리 창, fill은 창 안을 옅게 칠할 색.

    cell_colors : M과 같은 모양의 색 배열(문자열) — 풀링 영역별 색 등. 없으면 흰 칸.
    highlight_cells : [(i, j)] 금색 테두리 칸(출력 칸 표시용).
    """
    a = np.atleast_2d(np.asarray(M, dtype=float))
    h, w = a.shape
    ax.set_axis_off()
    ax.set_aspect("equal")
    for (i, j), v in np.ndenumerate(a):
        c = cell_colors[i][j] if cell_colors is not None else "white"
        ax.add_patch(Rectangle((j, h - 1 - i), 1, 1, facecolor=c, edgecolor=MUTED, lw=0.8))
        ax.text(j + 0.5, h - 1 - i + 0.5, fmt.format(v), ha="center", va="center", fontsize=fontsize, color=INK)
    if window is not None:
        r, c, wh, ww = window
        ax.add_patch(Rectangle((c, h - r - wh), ww, wh, facecolor=fill or "none", alpha=0.35 if fill else 1,
                               edgecolor="none" if fill else GOLD, lw=0, zorder=2))
        ax.add_patch(Rectangle((c, h - r - wh), ww, wh, fill=False, edgecolor=GOLD, lw=2.2, zorder=3))
    for (i, j) in (highlight_cells or []):
        ax.add_patch(Rectangle((j, h - 1 - i), 1, 1, fill=False, edgecolor=GOLD, lw=2.2, zorder=3))
    ax.set_xlim(-0.15, w + 0.15)
    ax.set_ylim(-0.15, h + 0.15)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, pad=6)
    return ax


# ---------------------------------------------------------------- 손실 곡면과 경로
PATH_COLORS = [BLUE, RED, GOLD, "#2a9d8f", "#7b5ea7", INK]


def draw_surface_paths(ax, xx, yy, zz, paths, *, elev=32, azim=-55, cmap="Blues", alpha=0.3):
    """3차원 곡면 위에 경로들을 그린다. paths = {"SGD": (T, 3) 배열(x, y, loss)}."""
    ax.plot_surface(xx, yy, zz, cmap=cmap, alpha=alpha, linewidth=0, antialiased=True, rstride=2, cstride=2)
    for (name, pts), color in zip(paths.items(), PATH_COLORS):
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.04 * np.ptp(zz), color=color, lw=2.0, label=name)
        ax.scatter(*pts[0], color=color, s=28, depthshade=False)
        ax.scatter(*pts[-1], color=color, s=40, marker="s", depthshade=False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.set_ticks([])
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$w_0$", color=INK); ax.set_ylabel("$w_1$", color=INK); ax.set_zlabel("손실", color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    return ax


def draw_contour_paths(ax, xx, yy, zz, paths, *, levels=18, cmap="Blues"):
    """등고선 위의 경로. 시작은 점, 끝은 네모."""
    ax.contour(xx, yy, zz, levels=levels, cmap=cmap, linewidths=0.8)
    for (name, pts), color in zip(paths.items(), PATH_COLORS):
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8, label=name)
        ax.scatter(*pts[0, :2], color=color, s=26, zorder=3)
        ax.scatter(*pts[-1, :2], color=color, s=40, marker="s", zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("$w_0$", color=INK); ax.set_ylabel("$w_1$", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    return ax
