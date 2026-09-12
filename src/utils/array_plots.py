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
