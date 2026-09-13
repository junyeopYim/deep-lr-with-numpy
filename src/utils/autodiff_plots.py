"""13 자동미분 노트북의 개념 그림과 계산 그래프 도식.

값·국소 미분·기울기는 노트북 본문에서 계산하고 여기서는 받은 값만 그립니다.
MNIST 대신 작은 계산 그래프 도식을 concept_plots 의 색·여백 규칙으로 그립니다. 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401


def draw_compute_graph(ax, nodes, edges, *, title=None, r=0.3):
    """계산 그래프. nodes = {이름: (x, y, 위 라벨, 아래 값)}, edges = [(from, to, 금색 라벨 또는 None)].

    변수는 원, 연산(이름이 'op:'로 시작)은 둥근 상자. 금색 라벨은 그 edge 의 국소 미분(backward 에서 곱하는 값).
    """
    ax.set_axis_off()
    ax.set_aspect("equal")
    pos = {name: np.array(v[:2], float) for name, v in nodes.items()}
    for src, dst, label in edges:
        a, b = pos[src], pos[dst]
        u = (b - a) / np.linalg.norm(b - a)
        ra = 0.42 if src.startswith("op:") else r
        rb = 0.42 if dst.startswith("op:") else r
        ax.add_patch(FancyArrowPatch(a + u * ra, b - u * rb, arrowstyle="-|>", mutation_scale=11, color=cp.MUTED, lw=1.1, zorder=1))
        if label:
            mid = (a + b) / 2
            normal = np.array([-u[1], u[0]])
            ax.text(*(mid + 0.22 * normal), label, ha="center", va="center", fontsize=8.5, color=cp.GOLD,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=4)
    for name, (x, y, top, bottom) in nodes.items():
        if name.startswith("op:"):
            ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.24), 0.84, 0.48, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        facecolor=sp.FILL, edgecolor=cp.MUTED, lw=1.0, zorder=2))
            ax.text(x, y, top, ha="center", va="center", fontsize=9.5, color=cp.INK, zorder=3)
        else:
            ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=cp.MUTED, lw=1.0, zorder=2))
            ax.text(x, y, top, ha="center", va="center", fontsize=10, color=cp.INK, zorder=3)
        if bottom:
            ax.text(x, y - r - 0.12, bottom, ha="center", va="top", fontsize=8.5, color=cp.INK, zorder=3)
    xs = [v[0] for v in nodes.values()]; ys = [v[1] for v in nodes.values()]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.8)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)
    return ax


def plot_shared_path(x, u, y, du_dx, dy_du_each, dx):
    """u = x², y = u + u : 같은 u 가 두 경로에 쓰여 x 의 기울기는 두 기여의 합."""
    fig = cp.new_figure((9.5, 3.4))
    ax = fig.add_subplot(111)
    nodes = {"x": (0, 0, "$x$", f"$= {x:g}$"), "op:sq": (1.6, 0, "$(\\cdot)^2$", None), "u": (3.2, 0, "$u$", f"$= {u:g}$"),
             "op:add": (5.0, 0, "$+$", None), "y": (6.6, 0, "$y$", f"$= {y:g}$")}
    edges = [("x", "op:sq", None), ("op:sq", "u", f"$du/dx = {du_dx:g}$"),
             ("u", "op:add", f"${dy_du_each[0]:g} + {dy_du_each[1]:g}$ (두 경로)"), ("op:add", "y", None)]
    draw_compute_graph(ax, nodes, edges, title=f"$u = x^2$, $y = u + u$ · 금색 = 국소 미분 · $dy/dx = ({dy_du_each[0]:g} + {dy_du_each[1]:g})\\cdot{du_dx:g} = {dx:g}$")
    cp.show(fig)


def plot_vjp(J, g, vjp, out_labels, in_labels):
    """Jacobian (출력 × 입력) 과 출력 쪽 기울기 g 에서 입력 쪽 기울기 Jᵀg."""
    fig = cp.new_figure((9.5, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.7, 0.7], wspace=0.7)
    ax_j = fig.add_subplot(gs[0]); cp.blank(ax_j)
    cp.draw_grid(ax_j, J, "$J_{ij} = \\partial y_i / \\partial x_j$", kind="signed", fmt="{:g}", fontsize=10, row_labels=out_labels, col_labels=in_labels)
    ax_g = fig.add_subplot(gs[1]); cp.blank(ax_g)
    cp.draw_grid(ax_g, np.asarray(g)[:, None], "$g = \\partial L/\\partial y$", kind="signed", fmt="{:g}", fontsize=10, row_labels=out_labels)
    ax_v = fig.add_subplot(gs[2]); cp.blank(ax_v)
    cp.draw_grid(ax_v, np.asarray(vjp)[:, None], "$dx = J^{\\top} g$", kind="signed", fmt="{:g}", fontsize=10, row_labels=in_labels)
    cp.connect(fig, ax_j, ax_g, "$J^{\\top}$ @"); cp.connect(fig, ax_g, ax_v, "=")
    cp.label(ax_v, "$dx_j = \\sum_i g_i J_{ij}$ : 열마다 가중합", y=-0.08)
    cp.show(fig)


def plot_broadcast_backward(b, expanded, upstream, db):
    """forward: b (D,) 를 (N, D) 로 복사 → backward: 같은 자리의 기울기를 N 개 더해 (D,) 로."""
    N, D = expanded.shape
    fig = cp.new_figure((11.5, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[D + 0.3, D + 0.3, D + 0.3, D + 0.3], wspace=0.55)
    ax_b = fig.add_subplot(gs[0]); cp.blank(ax_b)
    cp.draw_grid(ax_b, np.asarray(b)[None], "$b$ : (D,)", kind="count", fmt="{:g}", fontsize=10, frame_to=(D, N))
    ax_e = fig.add_subplot(gs[1]); cp.blank(ax_e)
    cp.draw_grid(ax_e, expanded, "forward: (N, D) 로 복사", kind="count", fmt="{:g}", fontsize=10, row_labels=[f"n={n}" for n in range(N)])
    ax_u = fig.add_subplot(gs[2]); cp.blank(ax_u)
    cp.draw_grid(ax_u, upstream, "backward: $dY$ : (N, D)", kind="signed", fmt="{:g}", fontsize=10, row_labels=[f"n={n}" for n in range(N)])
    ax_d = fig.add_subplot(gs[3]); cp.blank(ax_d)
    cp.draw_grid(ax_d, np.asarray(db)[None], "$db = \\sum_n dY_{nj}$ : (D,)", kind="signed", fmt="{:g}", fontsize=10, frame_to=(D, N))
    cp.connect(fig, ax_b, ax_e, "N 번 씀"); cp.connect(fig, ax_u, ax_d, "axis=0 합")
    cp.label(ax_d, "복사한 만큼 더함 · 길이 1이던 축은 keepdims 로", y=-0.08)
    cp.show(fig)


def plot_mlp_graph(order):
    """MLP 손실의 계산 그래프. 잎(입력·파라미터)은 아래, 연산은 위 줄. 숫자 = 위상 순서."""
    fig = cp.new_figure((13, 3.6))
    ax = fig.add_subplot(111)
    ops = [("op:mm1", "@"), ("op:add1", "+"), ("op:tanh", "tanh"), ("op:mm2", "@"), ("op:add2", "+"), ("op:sub", "−"), ("op:sq", "$(\\cdot)^2$"), ("op:mean", "mean")]
    nodes = {}
    for i, (name, text) in enumerate(ops):
        nodes[name] = (1.6 * i + 1.6, 1.0, text, None)
    leaves = [("X", 0.0, "$X$"), ("W1", 1.6, "$W_1$"), ("b1", 3.2, "$b_1$"), ("W2", 6.4, "$W_2$"), ("b2", 8.0, "$b_2$"), ("Y", 9.6, "$Y$")]
    for name, x, text in leaves:
        nodes[name] = (x, -0.6, text, None)
    nodes["L"] = (1.6 * len(ops) + 1.6, 1.0, "$L$", "$= \\frac{1}{2}\\,$mean")
    edges = [("X", "op:mm1", None), ("W1", "op:mm1", None), ("op:mm1", "op:add1", None), ("b1", "op:add1", "broadcast"),
             ("op:add1", "op:tanh", None), ("op:tanh", "op:mm2", None), ("W2", "op:mm2", None), ("op:mm2", "op:add2", None),
             ("b2", "op:add2", "broadcast"), ("op:add2", "op:sub", None), ("Y", "op:sub", None), ("op:sub", "op:sq", None),
             ("op:sq", "op:mean", None), ("op:mean", "L", None)]
    draw_compute_graph(ax, nodes, edges, title="$L = \\frac{1}{2}\\,\\mathrm{mean}((\\tanh(XW_1 + b_1)W_2 + b_2 - Y)^2)$ 의 그래프 · 숫자 = 위상 순서, backward 는 거꾸로")
    for k, name in enumerate(order):
        x, y = nodes[name][:2]
        ax.text(x + 0.36, y + 0.34, str(k), ha="center", va="center", fontsize=8, color="white",
                bbox={"boxstyle": "circle,pad=0.15", "facecolor": cp.GOLD, "edgecolor": "none"}, zorder=5)
    cp.label(ax, "원 = 값을 가진 Node (잎은 파라미터·데이터), 상자 = 연산 · 각 상자가 자기 pullback 을 가짐", y=0.0)
    cp.show(fig)


def plot_local_rules(row, column, row_grad, column_grad):
    """(row * column + column).sum() 의 그래프. column 은 두 경로에 쓰이고, 곱셈은 양쪽이 broadcasting."""
    fig = cp.new_figure((12, 4.2))
    ax = fig.add_subplot(111)
    nodes = {"row": (0, 1.3, "$r$", "row (2, 1)"), "column": (0, -1.3, "$c$", "column (3,)"), "op:mul": (2.6, 1.3, "$\\odot$", "(2, 3)"),
             "op:add": (5.2, 0.0, "$+$", "(2, 3)"), "op:sum": (7.4, 0.0, "sum", None), "L": (9.2, 0.0, "$L$", f"$={float(np.sum(row * column + column)):g}$")}
    edges = [("row", "op:mul", "$g\\odot c$ → (2,1)로 합"), ("column", "op:mul", "$g\\odot r$ → (3,)로 합"),
             ("op:mul", "op:add", "$g$"), ("column", "op:add", "$g$ → (3,)로 합"), ("op:add", "op:sum", "$1$"), ("op:sum", "L", None)]
    draw_compute_graph(ax, nodes, edges, title="$L = \\mathrm{sum}(r\\odot c + c)$ · 금색 = 각 pullback 이 부모에게 더하는 것")
    ax.text(0, 2.15, f"r.grad = {np.asarray(row_grad).ravel().tolist()}", ha="center", va="bottom", fontsize=9.5, color=cp.BLUE)
    ax.text(0, -2.15, f"c.grad = {np.asarray(column_grad).ravel().tolist()} (두 경로의 합)", ha="center", va="top", fontsize=9.5, color=cp.BLUE)
    ax.set_xlim(-1.2, 10.2)
    ax.set_ylim(-2.9, 2.8)
    cp.show(fig)


def plot_training_cycle():
    """스텝마다 새 그래프를 만들고 backward 하고 갱신하는 순환."""
    fig = cp.new_figure((10, 2.6))
    ax = fig.add_subplot(111)
    ax.set_axis_off(); ax.set_aspect("equal")
    names = ["파라미터 Node\n$W_1, b_1, W_2, b_2$", "graph_loss\n새 그래프 만들기", "loss.backward()\n기울기 채우기", "SGD 갱신\n$p \\leftarrow p - \\eta\\, p.\\mathrm{grad}$"]
    fills = ["#fce8c3", sp.FILL, sp.FILL, sp.FILL]
    w, h, gap = 2.2, 1.0, 0.6
    xs = [i * (w + gap) for i in range(4)]
    for x, name, fill in zip(xs, names, fills):
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fill, edgecolor=cp.MUTED, lw=1.0))
        ax.text(x + w / 2, h / 2, name, ha="center", va="center", fontsize=9, color=cp.INK)
    for x in xs[:-1]:
        ax.add_patch(FancyArrowPatch((x + w + 0.04, h / 2), (x + w + gap - 0.04, h / 2), arrowstyle="-|>", mutation_scale=11, color=cp.MUTED, lw=1.0))
    ax.add_patch(FancyArrowPatch((xs[-1] + w / 2, -0.05), (xs[0] + w / 2, -0.05), arrowstyle="-|>", mutation_scale=11, color=cp.GOLD, lw=1.6,
                                 connectionstyle="arc3,rad=-0.3"))
    ax.text((xs[0] + xs[-1] + w) / 2, -1.45, "다음 스텝: 갱신된 값으로 그래프를 다시 만듦 (이전 그래프는 버림)", ha="center", va="top", fontsize=9, color=cp.GOLD)
    ax.set_xlim(-0.3, xs[-1] + w + 0.3); ax.set_ylim(-1.9, h + 0.3)
    cp.show(fig)


def plot_fit(x_train, y_train, grid, prediction, truth):
    """학습 관측, 학습된 함수, 잡음 전 함수."""
    fig = cp.new_figure((7, 3.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "입력 $x$", "출력 $y$", "연산 그래프로 학습한 MLP 의 예측")
    ax.scatter(x_train, y_train, s=12, color=cp.MUTED, label="학습 관측")
    ax.plot(grid, truth, color=cp.RED, lw=1.2, ls="--", label="잡음 전 $\\sin(\\pi x)$")
    ax.plot(grid, prediction, color=cp.BLUE, lw=1.8, label="학습 함수")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    cp.show(fig)


def plot_stop_gradient(full, stopped):
    """x·x 와 x·stopgrad(x) 의 그래프: 한쪽 경로를 끊으면 기울기가 절반."""
    fig = cp.new_figure((11, 3.0))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    for k, (ax, title, label_right, note, grad) in enumerate([(fig.add_subplot(gs[0]), "$x \\cdot x$", "$x$", "$=3$", full),
                                                             (fig.add_subplot(gs[1]), "$x \\cdot \\mathrm{stopgrad}(x)$", "$x'$", "값만 복사 $=3$", stopped)]):
        nodes = {"x": (0, 0, "$x$", "$=3$"), "op:mul": (2.0, 0, "$\\times$", None), "y": (4.0, 0, "$y$", "$=9$"),
                 "c": (2.0, -1.4, label_right, note)}
        edges = [("x", "op:mul", "$3$"), ("c", "op:mul", None if k else "$3$"), ("op:mul", "y", None)]
        if k == 0:
            edges.insert(1, ("x", "c", None))
        draw_compute_graph(ax, nodes, edges, title=f"{title} · $dy/dx = {grad:g}$")
        if k == 1:
            ax.text(2.0, -2.1, "경로 없음: $x$ 로 미분이 안 돌아감", ha="center", va="top", fontsize=8.5, color=cp.RED)
    cp.show(fig)
