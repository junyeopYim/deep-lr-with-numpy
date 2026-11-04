"""37 생성 평가 노트북의 그림 배치. 통계·거리·커널·학습은 본문에서 계산합니다."""

import numpy as np

from . import concept_plots as cp


def plot_evaluation_protocol(train, test, transformed, counts):
    """학습 표·새 표·학습 통계로 변환한 표와 고정한 표본 수를 보여 줍니다."""
    fig, axes = cp.flow([2.3, 2.3, 2.7, 2.4], height=3.4, wspace=.45)
    cp.draw_grid(axes[0], train, "학습 표에서만 통계 추정", fmt="{:.1f}")
    cp.draw_grid(axes[1], test, "새 관측: 통계는 고정", fmt="{:.1f}")
    cp.draw_grid(axes[2], transformed, "같은 μ·s로 변환", fmt="{:.1f}")
    cp.draw_grid(axes[3], np.asarray(counts)[None], "실험 전에 고정", kind="plain",
                 col_labels=["학습", "평가", "생성"], fontsize=9)
    cp.connect(fig, axes[1], axes[2], "(x − μ) / s")
    fig.subplots_adjust(bottom=.2, top=.84)
    cp.show(fig)


def plot_density_diagnostics(grid, densities, names, masses, oracle_scores):
    """계산된 밀도 곡선과 영역 질량·정답 밀도 진단을 나란히 그립니다."""
    fig, axes = cp.flow([4.3, 2.8, 2.8], height=3.8, wspace=.65)
    colors = [cp.BLUE, cp.RED, cp.MUTED]
    for density, name, color in zip(densities, names, colors):
        axes[0].plot(grid, density, color=color, label=name, lw=2)
    axes[0].set_title("같은 좌표의 세 밀도", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    cp.draw_grid(axes[1], masses, "왼쪽·오른쪽 영역의 비율", kind="plain", fmt="{:.2f}",
                 row_labels=names, col_labels=["x < 0", "x ≥ 0"], fontsize=9)
    cp.draw_bars(axes[2], oracle_scores, "생성점의 정답 NLL 진단", labels=names, fmt="{:.3f}")
    for i, value in enumerate(oracle_scores):
        axes[2].text(value, i, f"  {value:.3f}", va="center", fontsize=9)
    cp.label(axes[2], "작아도 두 영역을 다 생성한다는 뜻은 아닙니다", y=-.12)
    fig.subplots_adjust(left=.04, right=.96, top=.85, bottom=.21)
    cp.show(fig)


def plot_em_learning(grid, before, after, reference, history):
    """EM 전후의 밀도와 학습 평균 로그우도 기록을 그립니다."""
    fig = cp.new_figure((10.5, 3.5))
    axes = fig.subplots(1, 2)
    for curve, label, color in [(reference, "정답 분포", cp.INK), (before, "학습 전", cp.MUTED), (after, "EM 학습 후", cp.BLUE)]:
        axes[0].plot(grid, curve, label=label, color=color)
    axes[0].set(xlabel="x", ylabel="밀도", title="파라미터를 학습해 밀도를 바꿉니다")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(np.arange(len(history)), history, color=cp.BLUE)
    axes[1].set(xlabel="EM 반복", ylabel="학습 평균 log p", title="학습에 쓴 목적값")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_mmd_terms(kxx, kyy, kxy, terms):
    """손계산 커널 행렬 세 개와 대각항을 포함한 V통계의 세 항을 그립니다."""
    fig, axes = cp.flow([2.5, 2.5, 2.5, 2.8], height=3.6, wspace=.4)
    for ax, matrix, title in zip(axes[:3], [kxx, kyy, kxy], ["KXX: 자기 쌍 포함", "KYY: 자기 쌍 포함", "KXY: 두 표 사이"]):
        cp.draw_grid(ax, matrix, title, fmt="{:.3f}", kind="plain", fontsize=9)
    cp.draw_grid(axes[3], np.asarray(terms)[:, None], "세 평균을 더하기", kind="plain", fmt="{:.4f}",
                 row_labels=["+ mean KXX", "+ mean KYY", "− 2 mean KXY", "MMD² V"], fontsize=9)
    fig.subplots_adjust(left=.04, right=.97, top=.84, bottom=.15)
    cp.show(fig)


def plot_mmd_sensitivity(bandwidths, score_matrix, names, sizes, mean_scores, std_scores):
    """고정 표본의 대역폭 변화와 독립 표본 반복의 크기별 MMD 기록을 그립니다."""
    fig = cp.new_figure((11, 3.6))
    axes = fig.subplots(1, 2)
    for values, name, color in zip(score_matrix, names, [cp.BLUE, cp.RED]):
        axes[0].semilogx(bandwidths, values, marker="o", color=color, label=name)
    axes[0].set(xlabel="RBF 대역폭 h", ylabel="MMD² V", title="같은 표본도 커널이 바뀌면 다른 수치")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].errorbar(sizes, mean_scores, yerr=std_scores, color=cp.BLUE, marker="o", capsize=4)
    axes[1].set(xlabel="각 집합의 표본 수", ylabel="MMD² V: 평균 ± 표준편차", title="같은 분포의 독립 표본도 정확히 0은 아닙니다")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_feature_summary(image, features, covariance, feature_names):
    """고정 MNIST 예시에서 계산한 특징 세 개와 학습 특징 공분산을 보여 줍니다."""
    fig, axes = cp.flow([2.0, 3.0, 3.4], height=3.5, wspace=.55)
    cp.draw_image(axes[0], image, "항상 같은 예시 7")
    cp.draw_bars(axes[1], features, "직접 정한 세 특징", labels=feature_names, signed=True)
    for i, value in enumerate(features):
        axes[1].text(value, i, f"  {value:.3f}", va="center", fontsize=9)
    cp.draw_grid(axes[2], covariance, "학습 특징의 공분산", fmt="{:.3f}", row_labels=feature_names,
                 col_labels=feature_names, fontsize=8)
    cp.label(axes[0], "화소에서 세 특징 계산", y=-.12)
    fig.subplots_adjust(left=.05, right=.95, top=.84, bottom=.16)
    cp.show(fig)


def plot_copy_diagnostics(generated, nearest_train, nearest_test, distances, summary):
    """복제 예시와 train/test 최근접 그림·본문에서 계산한 진단 표를 보여 줍니다."""
    fig, axes = cp.flow([2.1, 2.1, 2.1, 3.1], height=3.6, wspace=.45)
    for ax, image, title in zip(axes[:3], [generated, nearest_train, nearest_test],
                               ["복제해 낸 예시 7", f"가장 가까운 train\nRMS {distances[0]:.4f}", f"가장 가까운 test\nRMS {distances[1]:.4f}"]):
        cp.draw_image(ax, image, title)
    cp.draw_grid(axes[3], np.asarray(summary)[:, None], "200장 묶음의 진단", kind="plain", fmt="{:.4f}",
                 row_labels=["고유 그림 비율", "train 평균 RMS", "test 평균 RMS"], fontsize=9)
    fig.subplots_adjust(top=.8, bottom=.15, left=.03, right=.96)
    cp.show(fig)


def plot_paired_protocol(budgets, names, seeds, first, second, differences):
    """같은 표본 수와 seed별 짝지은 점수 차이를 보여 주는 개념 그림입니다."""
    fig, axes = cp.flow([2.7, 3.4, 2.7], height=3.7, wspace=.7)
    cp.draw_grid(axes[0], np.asarray(budgets)[:, None], "모든 후보의 생성 수", kind="plain", row_labels=names, fontsize=9)
    endpoint_labels = {}
    for i, (a, b) in enumerate(zip(first, second)):
        axes[1].plot([0, 1], [a, b], color=cp.MUTED, lw=1)
        axes[1].scatter([0, 1], [a, b], color=[cp.BLUE, cp.RED], s=25)
        endpoint_labels.setdefault(float(b), []).append(str(seeds[i]))
    for value, labels in endpoint_labels.items():
        axes[1].text(1.08, value, "seed " + " · ".join(labels), fontsize=8, va="center")
    axes[1].set_title("같은 행끼리 차이를 봅니다", fontsize=11)
    axes[1].text(0, -.1, "A", transform=axes[1].transAxes, color=cp.BLUE)
    axes[1].text(.85, -.1, "B", transform=axes[1].transAxes, color=cp.RED)
    axes[1].set_xlim(-.2, 1.5)
    cp.draw_grid(axes[2], np.asarray(differences)[:, None], "B − A", kind="plain", fmt="{:.3f}", row_labels=[str(s) for s in seeds])
    fig.subplots_adjust(left=.06, right=.95, top=.84, bottom=.2)
    cp.show(fig)


def plot_comparison_summary(names, means, standard_deviations, metric_labels):
    """최종 집계값의 평균·seed 표준편차를 축마다 따로 보여 줍니다."""
    fig = cp.new_figure((13.5, 4.0))
    axes = fig.subplots(1, len(metric_labels))
    colors = [cp.INK, cp.BLUE, cp.MUTED, cp.RED, cp.GOLD]
    for j, (ax, metric) in enumerate(zip(axes, metric_labels)):
        ax.barh(np.arange(len(names)), means[:, j], xerr=standard_deviations[:, j], color=colors, capsize=3)
        ax.set_yticks(np.arange(len(names)), names if j == 0 else [""]*len(names), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(metric, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.tight_layout()
    cp.show(fig)
