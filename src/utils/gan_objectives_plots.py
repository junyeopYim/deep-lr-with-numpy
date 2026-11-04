"""29 GAN 목적과 제약의 배열·곡선·실제 생성 표본을 표시합니다.

거리, 모델, 입력 기울기, GP, SVD 투영, 학습, 평가 지표는 노트북에서 계산합니다.
"""

import numpy as np
from matplotlib.patches import Circle

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_distribution_distances(masses, optimal_d, comparison, position_labels):
    """분리된 질량·최적 판별기·거리별 반응의 직접 계산을 표시합니다."""
    fig, axes = cp.flow([3.0, 3.0, 3.6], height=3.3, wspace=0.5)
    cp.draw_grid(axes[0], masses, "실제 p / 생성 q", row_labels=["p", "q"],
                 col_labels=position_labels, kind="plain", fmt="{:.1f}")
    cp.draw_grid(axes[1], optimal_d, "최적 분류 확률 D*", col_labels=position_labels,
                 kind="plain", fmt="{:.1f}")
    cp.draw_grid(axes[2], comparison, "이동 거리의 변화", col_labels=["간격", "JS(nat)", "W₁"],
                 kind="plain", fmt="{:.2f}", fontsize=9)
    cp.label(axes[2], "수치 크기보다 간격 변화에 대한 반응을 봅니다")
    cp.show(fig)


def plot_generator_objectives(scores, losses, gradients):
    """fake 점수에 대한 목적값과 도함수의 차이를 표시합니다."""
    fig, axes = cp.flow([2.5, 3.3, 4.0], height=3.2, wspace=0.55)
    labels = [f"{value:g}" for value in scores]
    cp.draw_grid(axes[0], np.asarray(scores)[None, :], "fake 점수 c", kind="signed", fmt="{:g}")
    cp.draw_grid(axes[1], losses, "생성기 목적 · 한 표본", col_labels=labels,
                 row_labels=["NS", "WGAN"], kind="signed", fmt="{:.2f}", fontsize=9)
    cp.draw_grid(axes[2], gradients, "점수에 대한 미분", col_labels=labels,
                 row_labels=["minimax", "NS", "WGAN"], kind="signed", fmt="{:.3f}", fontsize=9)
    cp.label(axes[2], "실제 G 기울기는 critic 입력 미분도 지나갑니다")
    cp.show(fig)


def plot_gp_path(real, fake, interpolated, input_gradient, norms, penalties):
    """고정 보간점과 해당 위치의 입력 기울기·penalty를 표시합니다."""
    fig, axes = cp.flow([3.2, 2.8, 2.8], height=3.2, wspace=0.55)
    cp.blank(axes[0])
    for r, f, point in zip(real, fake, interpolated):
        axes[0].plot([r[0], f[0]], [r[1], f[1]], color=cp.MUTED, lw=1)
        axes[0].scatter(r[0], r[1], c=cp.BLUE, marker="o", s=32)
        axes[0].scatter(f[0], f[1], c=cp.MUTED, marker="x", s=35)
        axes[0].scatter(point[0], point[1], c=cp.INK, marker="s", s=25)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_title("실제 ● · 가짜 × · 보간 ■", color=cp.INK, fontsize=10)
    cp.draw_grid(axes[1], input_gradient, r"g = ∇ₓf · (N,2)", kind="signed", fmt="{:.2f}")
    table = np.column_stack([norms, penalties])
    cp.draw_grid(axes[2], table, "보간점마다 재는 두 값", col_labels=["rε", "(rε−1)²"],
                 kind="plain", fmt="{:.3f}", fontsize=9)
    cp.show(fig)


def plot_spectral_projection(matrix, singular, clipped, projected):
    """특잇값을 상한에서 잘라내는 사후 투영을 표시합니다."""
    fig, axes = cp.flow([2.4, 2.0, 2.0, 2.4], height=3.0, wspace=0.55)
    for ax, value, title in zip(axes, [matrix, singular[None, :], clipped[None, :], projected],
                                ["갱신 후 W", "특잇값", "상한 1로 자르기", "다음 forward의 W"]):
        cp.draw_grid(ax, value, title, kind="signed", fmt="{:.2f}")
    cp.connect(fig, axes[1], axes[2], "min(σ,1)")
    cp.label(axes[3], "SVD를 loss의 미분 경로에 넣지 않습니다")
    cp.show(fig)


def _mode_scatter(ax, samples, centers, radius, title, limits):
    """전달된 표본과 고정 mode 판정 원을 표시합니다."""
    cp.blank(ax)
    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.35, color=cp.BLUE, rasterized=True)
    for center in centers:
        ax.add_patch(Circle(center, radius, fill=False, ec=cp.INK, lw=1))
        ax.scatter(center[0], center[1], s=18, marker="+", color=cp.INK)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=cp.INK)


def plot_quality_coverage(candidates, names, centers, radius, qualities, coverages, limits):
    """같은 표본 수에서 proximity 품질과 mode coverage의 차이를 보여 줍니다."""
    fig, axes = cp.flow([3.2] * len(candidates), height=3.5, wspace=0.22)
    for ax, points, name, quality, coverage in zip(axes, candidates, names, qualities, coverages):
        _mode_scatter(ax, points, centers, radius, name, limits)
        cp.label(ax, f"mode 근접 비율 {quality:.2f} · coverage {coverage:.2f}")
    cp.show(fig)


def plot_training_diagnostics(histories, methods, seeds):
    """모든 seed의 critic gap·입력 기울기 크기·G 기울기 기록을 그립니다."""
    fig = cp.new_figure((12.0, 8.4))
    axes = fig.subplots(3, len(seeds), squeeze=False)
    colors = [cp.MUTED, cp.BLUE, cp.GOLD]
    labels = ["real−fake score gap", "보간점 입력 기울기 norm", "G 파라미터 기울기 norm"]
    columns = [1, 2, 3]
    for column, seed in enumerate(seeds):
        for row, (label, history_column) in enumerate(zip(labels, columns)):
            ax = axes[row, column]
            _clean_axes(ax, "G 갱신 수", label, f"seed {seed}")
            for method, color in zip(methods, colors):
                history = histories[(method, seed)]
                ax.plot(history[:, 0], history[:, history_column], color=color, label=method)
            if row == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(wspace=0.35, hspace=0.45, left=0.08, right=0.98, bottom=0.08, top=0.95)
    cp.show(fig)


def plot_seed_samples(generated, methods, seeds, centers, radius, metrics, limits):
    """선택하지 않은 모든 방법×seed의 생성 표본과 두 지표를 표시합니다."""
    fig = cp.new_figure((10.5, 3.1 * len(seeds)))
    axes = fig.subplots(len(seeds), len(methods), squeeze=False)
    for row, seed in enumerate(seeds):
        for column, method in enumerate(methods):
            ax = axes[row, column]
            _mode_scatter(ax, generated[(method, seed)], centers, radius, f"{method} · seed {seed}", limits)
            result = metrics[(method, seed)]
            cp.label(ax, f"근접 {result['quality']:.2f} · coverage {result['coverage']:.2f}")
    fig.subplots_adjust(wspace=0.22, hspace=0.35, left=0.04, right=0.97, top=0.95, bottom=0.06)
    cp.show(fig)
