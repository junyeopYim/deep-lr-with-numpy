"""03번 본문에서 계산한 활성화·은닉 표현·예측·손실을 그립니다."""

import matplotlib.pyplot as plt


def plot_activations(x, values, derivatives):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, y in values.items():
        axes[0].plot(x, y, label=name)
    for name, y in derivatives.items():
        axes[1].plot(x, y, label=name)
    axes[0].set(xlabel="입력", ylabel="출력", title="비선형 활성화")
    axes[1].set(xlabel="입력", ylabel="도함수", title="역전파에서 곱하는 값")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_representation(inputs, hidden, labels):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, points, title in zip(axes, [inputs, hidden], ["입력 공간의 XOR", "두 은닉값으로 바꾼 XOR"]):
        ax.scatter(points[:, 0], points[:, 1], c=labels.ravel(), cmap="coolwarm",
                   vmin=0, vmax=1, s=100, edgecolors="black")
        for i, point in enumerate(points):
            ax.annotate(str(i), point, xytext=(6, 6), textcoords="offset points")
        ax.set(title=title, xlabel="첫 좌표", ylabel="둘째 좌표")
        ax.margins(0.2)
    plt.show()


def plot_regions(xx, yy, probabilities, inputs, labels, title):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    region = ax.contourf(xx, yy, probabilities, levels=[0, .1, .25, .5, .75, .9, 1],
                        cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
    ax.contour(xx, yy, probabilities, levels=[0.5], colors="black", linewidths=1)
    ax.scatter(inputs[:, 0], inputs[:, 1], c=labels.ravel(), cmap="RdBu_r",
               vmin=0, vmax=1, edgecolors="black", s=34)
    fig.colorbar(region, ax=ax, label="P(Y=1 | x)")
    ax.set(xlabel="첫 번째 입력", ylabel="두 번째 입력", title=title)
    plt.show()


def plot_training(histories, ylabel="평균 BCE", title="같은 목적, 다른 계산 구조"):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for name, history in histories.items():
        ax.plot(range(len(history)), history, label=name)
    ax.set(xlabel="갱신 횟수", ylabel=ylabel, title=title)
    ax.legend()
    plt.show()


def plot_hidden_maps(xx, yy, activations):
    count = activations.shape[-1]
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.1), squeeze=False)
    for k, ax in enumerate(axes[0]):
        ax.contourf(xx, yy, activations[:, :, k], levels=12, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set(xlabel="첫 입력", ylabel="둘째 입력", title=f"은닉 유닛 {k}")
    plt.show()


# ---------------------------------------------------------------------------
# 개념 그림. 점수·판정·활성·순전파·역전파 값은 본문에서 계산하고 여기서는 받은 배열만 그립니다.
# 부품은 concept_plots·schematic_plots, 원칙은 docs/DESIGN.md.
# ---------------------------------------------------------------------------

import numpy as np
from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes


def plot_perceptron_decision(images, scores, decisions, targets, labels):
    """이미지 → 점수 → 0을 기준으로 판정. 정답과 다른 판정은 빨강."""
    n = len(images)
    cols = (n + 1) // 2
    fig = cp.new_figure((11, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.5, 1.0], wspace=0.6)
    sub = gs[0].subgridspec(2, cols, wspace=0.08, hspace=0.35)
    for i in range(n):
        ax = fig.add_subplot(sub[i // cols, i % cols]); cp.blank(ax)
        cp.draw_image(ax, images[i], frame=False)
        ax.set_title(f"{labels[i]}", color=cp.MUTED, fontsize=8.5, pad=2)
    ax_z = fig.add_subplot(gs[1]); cp.blank(ax_z)
    cp.draw_bars(ax_z, scores, "$z = x\\cdot w + b$", labels=[str(l) for l in labels], signed=True, fmt="{:+.1f}")
    cp.label(ax_z, "0을 기준으로 판정", y=-0.06)
    ax_d = fig.add_subplot(gs[2]); cp.blank(ax_d)
    ax_d.set_xlim(0, 1); ax_d.set_ylim(n - 0.5, -0.5)
    for i, (d, t) in enumerate(zip(decisions, targets)):
        ok = d == t
        ax_d.text(0.15, i, f"{int(d)}", ha="center", va="center", fontsize=10, color=cp.INK if ok else cp.RED,
                  fontweight="bold" if not ok else None)
        ax_d.text(0.6, i, "✓" if ok else f"✗ (정답 {int(t)})", ha="left", va="center", fontsize=9, color=cp.INK if ok else cp.RED)
    ax_d.set_title("판정 $\\hat y = 1[z \\geq 0]$", color=cp.INK, fontsize=10.5, pad=6)
    cp.connect(fig, fig.axes[cols - 1], ax_z, "$w$와 내적"); cp.connect(fig, ax_z, ax_d, "$z \\geq 0$?")
    cp.show(fig)


def plot_separability(corners, y_and, y_xor, line_x, line_y):
    """AND는 직선 하나로 나뉘고 XOR은 어떤 직선으로도 나뉘지 않습니다."""
    fig, axes = cp.flow([1, 1], height=3.2, unit=2.4, wspace=0.5)
    for ax, y, title in zip(axes, [y_and, y_xor], ["AND: 붉고 무거우면 익음", "XOR: 둘 중 하나만 1이면 1"]):
        _clean_axes(ax, "$x_0$ (붉음)", "$x_1$ (무거움)", title)
        ax.set_aspect("equal")
        for (a, b), t in zip(corners, np.asarray(y).ravel()):
            ax.scatter([a], [b], s=160, color=cp.BLUE if t == 1 else "white", edgecolor=cp.INK, lw=1.2, zorder=3)
            ax.text(a, b, str(int(t)), ha="center", va="center", fontsize=9, color="white" if t == 1 else cp.INK, zorder=4)
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    axes[0].plot(line_x, line_y, color=cp.GOLD, lw=1.8)
    axes[0].text(1.05, 0.62, "$x_0+x_1=1.5$", color=cp.GOLD, fontsize=9, rotation=-45)
    for (lx, ly) in [((-0.4, 1.4), (1.0, 1.0)), ((1.0, 1.0), (-0.4, 1.4))]:
        axes[1].plot(lx, ly, color=cp.RED, lw=1.2, ls="--", alpha=0.7)
    cp.label(axes[1], "어느 직선을 그어도 한쪽에 1과 0이 섞임", y=-0.32, color=cp.RED)
    cp.show(fig)


def plot_activation_curves(z, values, derivatives):
    """활성화 함수와 그 도함수. 역전파에서 곱하는 값이 도함수."""
    fig, axes = cp.flow([1, 1], height=3.1, unit=2.6, wspace=0.4)
    colors = [cp.INK, cp.BLUE, cp.RED]
    for ax, curves, title, ylabel in zip(axes, [values, derivatives], ["활성화 $a = f(z)$", "도함수 $f'(z)$ : 역전파에서 곱하는 값"], ["", ""]):
        _clean_axes(ax, "$z$", ylabel, title)
        for (name, y), color in zip(curves.items(), colors):
            ax.plot(z, y, color=color, lw=1.8, label=name)
        ax.axhline(0, color=cp.MUTED, lw=0.6)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.show(fig)


def plot_mlp_forward_flow(image, hidden, logit, probability, sizes):
    """이미지 → 은닉 활성 32개 → logit 하나 → 확률. 오른쪽에 층 도식."""
    fig = cp.new_figure((11.5, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.35, 0.9, 0.9, 1.9], wspace=0.5)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x); cp.draw_image(ax_x, image, "$x$ : (784,)")
    ax_h = fig.add_subplot(gs[1]); cp.blank(ax_h)
    cp.draw_strip(ax_h, hidden, f"$A_1$ : ({sizes[1]},)", kind="signed", vmax=1)
    cp.match_height(ax_h, ax_x)
    cp.label(ax_h, "tanh 뒤 −1…1", y=-0.06)
    ax_z = fig.add_subplot(gs[2]); cp.blank(ax_z)
    cp.draw_formula(ax_z, f"$z = {logit:+.2f}$", "$A_1 W_2 + b_2$ (logit)")
    ax_p = fig.add_subplot(gs[3]); cp.blank(ax_p)
    cp.draw_formula(ax_p, f"$p = \\sigma(z) = {probability:.2f}$", "7일 확률 (아직 학습 전)")
    ax_n = fig.add_subplot(gs[4]); cp.blank(ax_n)
    sp.draw_network(ax_n, sizes, node_labels=[sp.index_labels("x", sizes[0], 5), sp.index_labels("h", sizes[1], 5), ["$z$"]],
                    layer_titles=[f"입력 {sizes[0]}", f"은닉 {sizes[1]}", "출력 1"], edge_lw=0.5, arrows_out=True)
    cp.connect(fig, ax_x, ax_h, "$\\tanh(xW_1+b_1)$"); cp.connect(fig, ax_h, ax_z); cp.connect(fig, ax_z, ax_p, "sigmoid")
    cp.show(fig)


def plot_backward_flow(sizes, hidden_indices, dz, dZ1_values, dW1_images, dW2_values):
    """왼쪽: 출력에서 은닉 유닛을 거쳐 입력으로 돌아가는 경로(금색). 오른쪽: 그 은닉 유닛들의 dW1 열을 이미지로."""
    k = len(hidden_indices)
    fig = cp.new_figure((11.5, 3.8))
    gs = fig.add_gridspec(1, 1 + k, width_ratios=[2.0] + [1.0] * k, wspace=0.35)
    ax_n = fig.add_subplot(gs[0]); cp.blank(ax_n)
    shown = [0, 1, 2, None, sizes[1] - 1]        # draw_network가 max_show=5로 보여 주는 은닉 노드
    hl = [(1, j_shown, 0) for j_shown in range(5) if shown[j_shown] is not None] + [(0, i, j_shown) for j_shown in (0, 1, 2) for i in range(3)]
    sp.draw_network(ax_n, sizes, node_labels=[sp.index_labels("x", sizes[0], 5), sp.index_labels("h", sizes[1], 5), ["$z$"]],
                    layer_titles=[f"입력 {sizes[0]}", f"은닉 {sizes[1]}", "출력"], edge_lw=0.4,
                    highlight_edges=[(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 4, 0)])
    ax_n.set_title(f"$dz = p - y = {dz:+.2f}$ 가 선을 거슬러 퍼짐", color=cp.INK, fontsize=10.5, pad=12)
    cp.label(ax_n, "금색: $h_0$을 거쳐 입력으로 가는 경로 · $dW_{1[:,0]} = x \\cdot dZ_{1[0]}$", y=-0.02)
    vmax = float(max(np.abs(img).max() for img in dW1_images))
    for j, (idx, img) in enumerate(zip(hidden_indices, dW1_images)):
        ax = fig.add_subplot(gs[1 + j]); cp.blank(ax)
        cp.draw_image(ax, img, f"$dW_1[:, {idx}]$", kind="signed", vmax=vmax)
        cp.label(ax, f"$dZ_1[{idx}]={dZ1_values[j]:+.3f}$", y=-0.06)
        cp.label(ax, f"$dW_2[{idx}]={dW2_values[j]:+.3f}$", y=-0.17)
    cp.show(fig)


def plot_hidden_templates(images, titles, suptitle):
    """학습된 은닉 유닛의 가중치 열을 28×28로. 유닛마다 다른 획을 봅니다."""
    n = len(images)
    fig = cp.new_figure((1.35 * n + 0.5, 2.3))
    gs = fig.add_gridspec(1, n, wspace=0.08)
    vmax = float(max(np.abs(im).max() for im in images))
    for i, (im, (t, sub)) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[i]); cp.blank(ax)
        cp.draw_image(ax, im, t, kind="signed", vmax=vmax)
        cp.label(ax, sub, y=-0.06)
    fig.suptitle(suptitle, color=cp.INK, fontsize=10.5, y=1.04)
    cp.show(fig)
