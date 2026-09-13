"""00번 노트북에서 계산한 배열·내적·평면을 그립니다."""

import numpy as np
import matplotlib.pyplot as plt

from .plotting import COLOR_ACCENT, COLOR_NEG, COLOR_POS


def _matrix(ax, values, title, cmap="Blues"):
    a = np.asarray(values)
    if a.ndim == 1:
        a = a[:, None]
    ax.imshow(a, cmap=cmap, aspect="equal", alpha=0.5)
    for (i, j), v in np.ndenumerate(a):
        ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(a.shape[1]))
    ax.set_yticks(range(a.shape[0]))
    ax.set_xlabel("열 인덱스")
    ax.set_ylabel("행 인덱스")
    ax.set_title(title)
    ax.grid(False)


def plot_weighted_sum(x, w, contributions, bias, output):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    names = [f"특성 {i}" for i in range(len(x))]
    axes[0].bar(names, x, color=COLOR_POS)
    axes[0].set_title("입력 x")
    axes[1].bar(names, w, color=COLOR_ACCENT)
    axes[1].set_title("가중치 w")
    axes[2].bar(names + ["편향"], [*contributions, bias], color=[COLOR_POS] * len(x) + [COLOR_NEG])
    axes[2].set_title(f"각 항의 기여 → 합계 {output:g}")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("값")
    plt.show()


def plot_product(X, W, Z, titles):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, a, title in zip(axes, [X, W, Z], titles):
        _matrix(ax, a, title)
    plt.show()


def plot_layout(original, transposed, reshaped):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax, a, title in zip(axes, [original, transposed, reshaped],
                            ["원본 (2, 3)", "transpose: 축 교환 (3, 2)", "reshape: 순서대로 재배치 (3, 2)"]):
        _matrix(ax, a, title, "YlGn")
    plt.show()


def plot_affine_plane(xx, yy, zz, samples, outputs):
    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, alpha=0.45, color=COLOR_POS, edgecolor="none")
    ax.scatter(samples[:, 0], samples[:, 1], outputs.ravel(), color=COLOR_NEG, s=45, depthshade=False)
    ax.set_xlabel("첫 번째 입력")
    ax.set_ylabel("두 번째 입력")
    ax.set_zlabel("출력")
    ax.set_box_aspect((1, 1, 0.85), zoom=0.8)
    ax.set_title("같은 계산을 좌표로 보면 평면입니다")
    ax.view_init(elev=24, azim=-55)
    plt.show()


# ---------------------------------------------------------------------------
# MNIST 개념 그림. 계산(템플릿·점수·평균·reshape)은 본문에서 하고 여기서는 그리기만 합니다.
# 부품은 concept_plots에 있고, 설계 원칙은 docs/DESIGN.md에 있습니다.
# ---------------------------------------------------------------------------

from . import concept_plots as cp


def plot_dot_as_overlap(images, template, products, scores, labels, template_name="w"):
    """내적을 '겹침'으로 보여 줍니다. 행마다 x · w · x⊙w · 합(점수) 순서입니다."""
    n = len(images)
    fig, rows = cp.flow([1, 1, 1, 0.9], rows=n, height=2.6, unit=1.9, wspace=0.45, hspace=0.25)
    vmax = float(max(np.abs(p).max() for p in products))
    for r, (ax_x, ax_w, ax_p, ax_z) in enumerate(rows):
        cp.draw_image(ax_x, images[r], f"$x$ : 숫자 {labels[r]}" if r == 0 else f"$x$ : 숫자 {labels[r]}")
        cp.draw_image(ax_w, template, f"${template_name}$ : 7의 템플릿" if r == 0 else None, kind="signed")
        cp.draw_image(ax_p, products[r], "$x \\odot w$ : 위치별 곱" if r == 0 else None, kind="signed", vmax=vmax)
        cp.draw_formula(ax_z, f"$z = \\sum_d x_d w_d = {scores[r]:+.2f}$", "모두 더하면 점수 하나")
        cp.connect(fig, ax_x, ax_w, "곱하기" if r == 0 else None)
        cp.connect(fig, ax_p, ax_z, "더하기" if r == 0 else None)
    cp.label(rows[-1][2], "파랑: 같은 자리에 잉크와 +가중치 · 빨강: 잉크가 −가중치 위에", y=-0.1)
    cp.show(fig)


def plot_batch_rows(images, X, scores, labels):
    """샘플을 행으로 쌓으면 X @ w 한 번으로 점수 N개가 나옵니다."""
    n = len(images)
    fig = cp.new_figure((11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 3.0, 1.4], wspace=0.35)
    cols = (n + 1) // 2
    sub = gs[0].subgridspec(2, cols, wspace=0.05, hspace=0.05)
    for i in range(n):          # 위 행 0–4, 아래 행 5–9 (행 우선)
        ax = fig.add_subplot(sub[i // cols, i % cols])
        cp.blank(ax)
        cp.draw_image(ax, images[i], frame=False)
    fig.text(0.5 * (fig.axes[0].get_position().x0 + fig.axes[cols - 1].get_position().x1), 0.93,
             f"이미지 {n}장", ha="center", color=cp.INK, fontsize=10.5)
    ax_rows = fig.add_subplot(gs[1]); cp.blank(ax_rows)
    cp.draw_rows(ax_rows, X, f"$X$ : ({n}, 784) · 행 하나가 이미지 한 장")
    ax_z = fig.add_subplot(gs[2]); cp.blank(ax_z)
    cp.draw_bars(ax_z, scores, "$z = Xw$ : 7 템플릿 점수", labels=[str(l) for l in labels],
                 highlight=int(np.argmax(scores)), signed=True)
    cp.connect(fig, fig.axes[cols - 1], ax_rows, "펼쳐서 쌓기")
    cp.connect(fig, ax_rows, ax_z, "$X@w$")
    cp.show(fig)


def plot_templates(templates, Z, labels):
    """W의 열 하나가 출력 하나의 템플릿이고, Z = XW는 이미지 × 출력의 점수표입니다."""
    K = templates.shape[-1]
    fig = cp.new_figure((11, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.28)
    top = gs[0].subgridspec(1, K, wspace=0.08)
    vmax = float(np.abs(templates).max())
    for k in range(K):
        ax = fig.add_subplot(top[k]); cp.blank(ax)
        cp.draw_image(ax, templates[..., k], f"$W_{{:,{k}}}$", kind="signed", vmax=vmax)
    fig.text(0.5, 0.955, "$W$ : (784, 10) · 열 하나가 출력 하나의 템플릿 (파랑 +, 빨강 −)",
             ha="center", color=cp.INK, fontsize=10.5)
    ax = fig.add_subplot(gs[1]); cp.blank(ax)
    hi = [(i, int(np.argmax(Z[i]))) for i in range(len(Z))]
    cp.draw_grid(ax, Z, None, fmt="{:+.1f}", highlight=hi, fontsize=8.5,
                 row_labels=[f"숫자 {l}" for l in labels], col_labels=[str(k) for k in range(K)])
    ax.set_title("$Z = XW$ : (이미지 수, 10) · 행은 이미지, 열은 출력 $k$ · 금색 = 행에서 가장 큰 점수",
                 color=cp.INK, fontsize=10.5, pad=22)
    cp.show(fig)


def plot_axis_sums(X, mean_image, ink_per_image, labels):
    """같은 X에서 axis=0 합은 이미지가 되고 axis=1 합은 샘플마다 숫자 하나가 됩니다."""
    fig, (ax_x, ax_mean, ax_ink) = cp.flow([2.6, 1.5, 1.4], height=3.3, unit=1.7, wspace=0.45)
    cp.draw_rows(ax_x, X, f"$X$ : ({len(X)}, 784)")
    cp.draw_image(ax_mean, mean_image, "axis=0 평균 : (784,) → 28×28")
    cp.label(ax_mean, "샘플 축을 합치면 남는 것은 화소 → 이미지", y=-0.08)
    cp.draw_bars(ax_ink, ink_per_image, "axis=1 합 : (N,)", labels=[str(l) for l in labels],
                 highlight=int(np.argmax(ink_per_image)), fmt="{:.0f}")
    cp.label(ax_ink, "화소 축을 합치면 남는 것은 샘플 → 잉크 양", y=-0.08)
    cp.connect(fig, ax_x, ax_mean, "↓ 세로로 더함")
    cp.connect(fig, ax_mean, ax_ink, "→ 가로로 더함")
    cp.show(fig)


def plot_reshape_vs_transpose(image, flat, restored, transposed, wrong):
    """reshape는 읽는 순서를 지키고, transpose는 축을 바꾸며, 잘못된 reshape는 그림을 깨뜨립니다."""
    fig, axes = cp.flow([1, 0.35, 1, 1, 1.4], height=3.0, unit=1.9, wspace=0.4)
    cp.draw_image(axes[0], image, "(28, 28)")
    cp.draw_strip(axes[1], flat, every=1)
    cp.match_height(axes[1], axes[0])
    cp.draw_image(axes[2], restored, ".reshape(28, 28)")
    cp.draw_image(axes[3], transposed, ".T : 축 교환")
    cp.draw_image(axes[4], wrong, f".reshape{wrong.shape}")
    cp.connect(fig, axes[0], axes[1]); cp.connect(fig, axes[1], axes[2])
    axes[1].set_title(".reshape(784)", color=cp.INK, fontsize=10.5, pad=6)   # 화살표 계산 뒤에 붙여 겹침을 피합니다
    cp.label(axes[1], "(784,)", y=-0.06)
    cp.label(axes[2], "원소 순서 그대로 → 원본 복원", y=-0.06)
    cp.label(axes[3], "행↔열 → 거울에 비친 대각 반전", y=-0.06)
    cp.label(axes[4], "같은 순서, 다른 줄바꿈 → 원소 수는 같지만 그림이 아님", y=-0.06)
    cp.show(fig)


# ---------------------------------------------------------------------------
# 구조 도식 (schematic_plots 부품)
# ---------------------------------------------------------------------------

from . import schematic_plots as sp


def plot_neuron_diagram(inputs, weights, bias, output):
    """입력 두 개가 가중치를 타고 한 노드로 모이는 도식. 값은 본문의 손계산 값을 그대로 적습니다."""
    fig = cp.new_figure((6.4, 2.9))
    ax = fig.add_subplot(111)
    n = len(inputs)
    labels = [[f"${s}$" for s in inputs], ["$\\Sigma$"]]
    edges = {(0, i, 0): f"${w}$" for i, w in enumerate(weights)}
    pos = sp.draw_network(ax, [n, 1], node_labels=labels, edge_labels=edges, arrows_out=True, x_gap=2.6)
    x_out, y_out, _ = pos[1][0]
    ax.text(x_out + 0.85, y_out, f"${output}$", ha="left", va="center", fontsize=10.5, color=cp.INK)
    ax.text(x_out, y_out - 0.75, f"$+\\,{bias}$ (편향)", ha="center", va="top", fontsize=9, color=cp.MUTED)
    ax.set_xlim(ax.get_xlim()[0], x_out + 2.0)
    cp.show(fig)


def plot_layer_diagram(small_D, small_K, weight_names, big_D, big_K):
    """왼쪽: 작은 D→K 완전 연결(선마다 W_dk). 오른쪽: 784→10처럼 큰 층을 ⋮으로 줄인 같은 구조."""
    fig = cp.new_figure((11, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    labels = [[f"$x_{{{d}}}$" for d in range(small_D)], [f"$z_{{{k}}}$" for k in range(small_K)]]
    edges = {(0, d, k): f"${weight_names[d][k]}$" for d in range(small_D) for k in range(small_K)}
    sp.draw_network(ax, [small_D, small_K], node_labels=labels, edge_labels=edges,
                    layer_titles=[f"입력 $D={small_D}$", f"출력 $K={small_K}$"], x_gap=2.8)
    ax.set_title("$W$의 원소 하나 = 선 하나", color=cp.INK, fontsize=10.5, pad=14)
    ax = fig.add_subplot(gs[1])
    sp.draw_network(ax, [big_D, big_K], node_labels=[sp.index_labels("x", big_D, 6), sp.index_labels("z", big_K, 6)],
                    layer_titles=[f"입력 $D={big_D}$", f"출력 $K={big_K}$"], max_show=6, x_gap=3.0, edge_lw=0.5)
    ax.set_title(f"같은 구조, 선이 ${big_D}\\times{big_K}={big_D * big_K}$개", color=cp.INK, fontsize=10.5, pad=14)
    cp.show(fig)


def plot_broadcast_ways(base, vector, by_output, by_sample):
    """같은 (2,) 값을 (2,)로 더하면 열(출력)에, (2, 1)로 더하면 행(샘플)에 퍼집니다."""
    fig, axes = cp.flow([1, 1, 1, 1, 1], height=2.9, unit=1.5, wspace=0.8)
    v = np.asarray(vector)
    common = {"vmax": 40, "frame_to": (2, 2)}
    cp.draw_grid(axes[0], base, "$B$ : (2, 2)", row_labels=["샘플 0", "샘플 1"], col_labels=["출력 0", "출력 1"], **common)
    cp.draw_grid(axes[1], v[None, :], "$v$ : (2,)", col_labels=["", ""], row_labels=[""], **common)
    cp.label(axes[1], "가로로 눕혀 열에 맞춤", y=-0.1)
    cp.draw_grid(axes[2], by_output, "$B + v$", row_labels=["", ""], col_labels=["+10", "+20"], **common)
    cp.draw_grid(axes[3], v[:, None], "$v$[:, None] : (2, 1)", row_labels=["", ""], col_labels=[""], **common)
    cp.label(axes[3], "세로로 세워 행에 맞춤", y=-0.1)
    cp.draw_grid(axes[4], by_sample, "$B + v$[:, None]", row_labels=["+10", "+20"], col_labels=["", ""], **common)
    cp.connect(fig, axes[1], axes[2]); cp.connect(fig, axes[3], axes[4])
    cp.show(fig)
