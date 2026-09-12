"""11–15 본문에서 계산한 좌표·통계·예측을 그립니다."""

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(x, series, *, title, xlabel="스텝", ylabel="값", logy=False):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, values in series.items():
        ax.plot(x, values, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_clouds(panels, *, xlabel="첫 번째 좌표", ylabel="두 번째 좌표", equal=True):
    """panels: (제목, {범례: (N,2) 좌표}) 목록."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, (title, groups) in zip(axes[0], panels):
        for label, points in groups.items():
            ax.scatter(points[:, 0], points[:, 1], s=13, alpha=0.55, label=label)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if equal:
            ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
    fig.tight_layout()
    plt.show()


def plot_matrices(matrices, titles):
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.8 * len(matrices), 3.3), squeeze=False)
    for ax, matrix, title in zip(axes[0], matrices, titles):
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1})
        ax.set(title=title, xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]))
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plt.show()


def plot_bars(labels, values, *, title, ylabel, errors=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=4, alpha=0.8)
    ax.set(title=title, ylabel=ylabel, xticks=np.arange(len(labels)), xticklabels=labels)
    fig.tight_layout()
    plt.show()
