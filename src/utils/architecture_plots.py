"""05–10에서 계산한 결과만 그립니다. 모델·미분·학습 계산은 노트북에 둡니다."""

import numpy as np
import matplotlib.pyplot as plt


def plot_matrices(matrices, titles, *, cmap="coolwarm", annotate=True):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.5), squeeze=False)
    for ax, values, title in zip(axes[0], matrices, titles):
        values = np.asarray(values)
        im = ax.imshow(values, cmap=cmap, aspect="auto")
        if annotate:
            for (i, j), value in np.ndenumerate(values):
                ax.text(j, i, f"{value:.2g}", ha="center", va="center", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1})
        ax.set_title(title)
        ax.set_xticks(np.arange(values.shape[1]))
        ax.set_yticks(np.arange(values.shape[0]))
        ax.set_xlabel("열")
        ax.set_ylabel("행")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.show()


def plot_curves(x, series, *, title, xlabel="스텝", ylabel="값", logy=False):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for label, values in series.items():
        ax.plot(x, values, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_images(images, labels, *, title, columns=5):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(str(labels[i]), fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_graphs(adjacencies, positions, node_values, titles):
    fig, axes = plt.subplots(1, len(adjacencies), figsize=(4 * len(adjacencies), 3.6), squeeze=False)
    color_min = min(np.min(values) for values in node_values)
    color_max = max(np.max(values) for values in node_values)
    for ax, adjacency, pos, values, title in zip(axes[0], adjacencies, positions, node_values, titles):
        # 좌표와 색상 값은 본문에서 계산해서 받습니다.
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if adjacency[i, j]:
                    ax.plot(pos[[i, j], 0], pos[[i, j], 1], color="gray", zorder=1)
        ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap="viridis", s=350, zorder=2,
                   vmin=color_min, vmax=color_max)
        for i, point in enumerate(pos):
            text_color = "black" if values[i] > color_min + 0.55 * (color_max - color_min) else "white"
            ax.text(*point, str(i), ha="center", va="center", color=text_color, zorder=3)
        ax.set_title(title)
        padding = 0.2 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), 1.0)
        ax.set_xlim(pos[:, 0].min() - padding, pos[:, 0].max() + padding)
        ax.set_ylim(pos[:, 1].min() - padding, pos[:, 1].max() + padding)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
