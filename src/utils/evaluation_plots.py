"""15 실험·평가 노트북의 개념 그림.

분할·표준화·검증 표·지표·혼동 행렬·틀린 예시는 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

TRAIN_COLOR, VALID_COLOR = "#cfe0f3", "#f6d5d5"


def _rgb(hex_color):
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def plot_split_units(group_ids, row_train_mask, group_train_mask, per_person, overlaps):
    """사람 120명 × 관측 4개를 격자로. 행 무작위 분할은 한 사람이 양쪽에, 사람 단위 분할은 사람 전체가 한쪽에."""
    n_people = len(np.unique(group_ids))
    train_rgb, valid_rgb = _rgb(TRAIN_COLOR), _rgb(VALID_COLOR)
    fig = cp.new_figure((13, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 2.4, 1], wspace=0.35)
    for k, (mask, title) in enumerate([(row_train_mask, "행 무작위 분할: 같은 사람이 양쪽에"), (group_train_mask, "사람 단위 분할: 사람 전체가 한쪽에")]):
        ax = fig.add_subplot(gs[k]); cp.blank(ax)
        grid = np.zeros((per_person, n_people, 3))
        slots = np.zeros(n_people, dtype=int)
        for g, is_train in zip(group_ids, mask):
            grid[slots[g], g] = train_rgb if is_train else valid_rgb
            slots[g] += 1
        ax.imshow(grid, aspect="auto", interpolation="nearest")
        ax.add_patch(Rectangle((-0.5, -0.5), n_people, per_person, fill=False, ec=cp.MUTED, lw=0.8))
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)
        cp.label(ax, "열 = 사람, 행 = 그 사람의 관측 · 파랑 = train, 빨강 = validation", y=-0.08)
    ax_b = fig.add_subplot(gs[2])
    _clean_axes(ax_b, None, "사람 수", "양쪽에 모두 나타나는 사람")
    ax_b.bar(["행 분할", "사람 분할"], overlaps, color=[cp.RED, cp.BLUE], width=0.55)
    for i, v in enumerate(overlaps):
        ax_b.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9.5, color=cp.INK)
    ax_b.set_ylim(0, max(overlaps) * 1.2 + 1)
    cp.show(fig)


def plot_standardize(train_col, future_val, mean, scale, all_mean):
    """열 하나의 관측을 수직선에. 위: 원래 값과 학습 평균, 아래: 학습 통계로 표준화한 값. 미래 관측을 합치면 평균이 움직임."""
    fig = cp.new_figure((9.5, 3.4))
    gs = fig.add_gridspec(2, 1, hspace=0.9)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "원래 값", None, "학습 관측의 평균 $\\mu_{train}$ 으로 중심을 잡음")
    ax.axhline(0, color=cp.MUTED, lw=1.0)
    ax.scatter(train_col, np.zeros_like(train_col), s=50, color=cp.BLUE, zorder=3, label="학습 관측")
    ax.scatter([future_val], [0], s=50, color=cp.RED, zorder=3, label="미래 관측")
    ax.scatter([mean], [0], s=80, marker="|", color=cp.GOLD, lw=3, zorder=4, label=f"$\\mu_{{train}} = {mean:g}$")
    ax.scatter([all_mean], [0], s=80, marker="|", color=cp.RED, lw=2, zorder=4, label=f"합친 평균 {all_mean:g}")
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(-0.4, 0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=4)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "표준화한 값 $(x - \\mu_{train}) / s_{train}$", None, "같은 $\\mu_{train}, s_{train}$ 을 미래 관측에도 적용")
    ax2.axhline(0, color=cp.MUTED, lw=1.0)
    z_train, z_future = (train_col - mean) / scale, (future_val - mean) / scale
    ax2.scatter(z_train, np.zeros_like(z_train), s=50, color=cp.BLUE, zorder=3)
    ax2.scatter([z_future], [0], s=50, color=cp.RED, zorder=3)
    ax2.text(z_future, 0.12, f"{z_future:.1f}", ha="center", va="bottom", fontsize=9, color=cp.RED)
    ax2.set_yticks([]); ax2.spines["left"].set_visible(False); ax2.set_ylim(-0.4, 0.6)
    cp.show(fig)


def plot_selection_table(l2_values, seeds, table, means, selected):
    """λ × seed 의 검증 BCE 표와 seed 평균. 금색 = 선택한 λ."""
    fig = cp.new_figure((7.5, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[len(seeds), 1.2], wspace=0.5)
    ax = fig.add_subplot(gs[0]); cp.blank(ax)
    rows = [f"$\\lambda={l:g}$" for l in l2_values]
    cp.draw_grid(ax, table, "검증 BCE (선택한 epoch)", kind="count", fmt="{:.4f}", fontsize=9.5, row_labels=rows, col_labels=[f"seed {s}" for s in seeds])
    ax_m = fig.add_subplot(gs[1]); cp.blank(ax_m)
    k = list(l2_values).index(selected)
    cp.draw_grid(ax_m, np.asarray(means)[:, None], "seed 평균", kind="count", fmt="{:.4f}", fontsize=9.5, row_labels=rows, highlight=[(k, 0)])
    cp.connect(fig, ax, ax_m, "평균")
    cp.label(ax_m, f"금색 = 가장 작은 평균 → $\\lambda = {selected:g}$", y=-0.08, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_training_curves(epochs, history, best_epoch):
    """왼쪽: train·validation BCE 와 정규화 포함 목적. 오른쪽: 전체 기울기 norm 과 epoch 업데이트 norm 합 (로그)."""
    fig = cp.new_figure((11.5, 3.5))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "epoch", "nat / 관측", "같은 파라미터에서 읽는 학습·검증 곡선")
    ax.plot(epochs, history[:, 0], color=cp.BLUE, lw=1.6, label="train BCE")
    ax.plot(epochs, history[:, 1], color=cp.RED, lw=1.6, label="validation BCE")
    ax.plot(epochs, history[:, 2], color=cp.MUTED, lw=1.2, ls="--", label="정규화 포함 train 목적")
    ax.axvline(best_epoch, color=cp.GOLD, lw=1.2, ls=":")
    lo, hi = ax.get_ylim()
    ax.text(best_epoch + 1, hi - 0.08 * (hi - lo), f"선택 epoch {best_epoch}", ha="left", va="top", fontsize=8.5, color=cp.GOLD)
    ax.legend(frameon=False, fontsize=8.5)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "epoch", "norm (로그 눈금)", "기울기와 실제 갱신량을 함께 확인")
    ax2.semilogy(epochs, history[:, 3], color=cp.INK, lw=1.5, label="전체 기울기 norm")
    ax2.semilogy(epochs, history[:, 4], color=cp.BLUE, lw=1.5, label="epoch 업데이트 norm 합")
    ax2.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_metrics_hand(y, p, threshold, confusion, metrics):
    """네 관측의 확률과 임계값 → 혼동 행렬 → 지표."""
    fig = cp.new_figure((12, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.9, 1.3], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "예측 확률 $p$", None, f"임계값 $t = {threshold:g}$ 로 판정")
    ax.axhline(0, color=cp.MUTED, lw=1.0)
    ax.axvline(threshold, color=cp.GOLD, lw=1.4, ls="--")
    for yi, pi in zip(y, p):
        ax.scatter([pi], [0], s=70, color=cp.BLUE if yi == 1 else "white", edgecolor=cp.INK, lw=1.2, zorder=3)
        ax.text(pi, 0.12 if yi == 1 else -0.14, f"$y={int(yi)}$", ha="center", va="bottom" if yi == 1 else "top", fontsize=9,
                color=cp.BLUE if yi == 1 else cp.INK)                                  # 양성은 위, 음성은 아래에 라벨
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(-0.4, 0.6)
    cp.label(ax, "파랑 = 양성 $y=1$ · 오른쪽이 양성 판정", y=-0.3)
    ax_c = fig.add_subplot(gs[1]); cp.blank(ax_c)
    cp.draw_grid(ax_c, confusion, "혼동 행렬", kind="count", fmt="{:g}", fontsize=11, row_labels=["정답 0", "정답 1"], col_labels=["예측 0", "예측 1"])
    cp.label(ax_c, "[[TN, FP], [FN, TP]]", y=-0.08)
    ax_m = fig.add_subplot(gs[2]); cp.blank(ax_m)
    names = ["accuracy", "precision", "recall", "f1", "auc", "brier"]
    cp.draw_bars(ax_m, [metrics[n] for n in names], "지표", labels=names, xlim=(0, 1), color=cp.BLUE)
    for i, n in enumerate(names):
        ax_m.text(metrics[n] + 0.03, i, f"{metrics[n]:.3f}", va="center", fontsize=9, color=cp.INK)
    cp.connect(fig, ax, ax_c, "세어서"); cp.connect(fig, ax_c, ax_m)
    cp.show(fig)


def plot_confusion_examples(confusion, images, titles, accuracy, precision, recall):
    """MNIST 7 판별의 혼동 행렬(축 없이)과 틀린 예시 이미지."""
    n = len(images)
    fig = cp.new_figure((2.0 * n + 3.5, 3.0))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.6] + [1] * n, wspace=0.25)
    ax_c = fig.add_subplot(gs[0]); cp.blank(ax_c)
    cp.draw_grid(ax_c, confusion, "5,000장의 혼동 행렬", kind="count", fmt="{:g}", fontsize=10.5,
                 row_labels=["7 아님", "7"], col_labels=["예측 아님", "예측 7"])
    cp.label(ax_c, f"acc {accuracy:.3f} · precision {precision:.3f} · recall {recall:.3f}", y=-0.08, color=cp.INK, fontsize=9)
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[1 + i]); cp.blank(ax)
        cp.draw_image(ax, img, title)
    cp.label(fig.axes[1], "틀린 예 · FP = 7이 아닌데 7이라 함, FN = 7인데 놓침", y=-0.08)
    cp.show(fig)


def plot_reliability(bin_probabilities, bin_frequencies, bin_counts, ece, thresholds, f1_curve, selected_threshold):
    """왼쪽: 구간별 예측 확률과 실제 양성 비율(신뢰도 곡선). 오른쪽: 임계값에 따른 검증 F1."""
    fig = cp.new_figure((11, 3.5))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "구간 평균 예측 확률", "실제 양성 비율", f"검증 신뢰도 곡선 · ECE = {ece:.3f}")
    ax.plot([0, 1], [0, 1], color=cp.MUTED, lw=1.0, ls="--", label="확률 = 비율")
    ax.plot(bin_probabilities, bin_frequencies, "o-", color=cp.BLUE, lw=1.5, ms=5, label="관측 양성 비율")
    for x_, y_, c_ in zip(bin_probabilities, bin_frequencies, bin_counts):
        ax.text(x_, y_ + 0.04, f"n={c_}", ha="center", fontsize=8, color=cp.MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "양성 판정 임계값", "validation F1", "결정 임계값도 검증에서 선택")
    ax2.plot(thresholds, f1_curve, color=cp.BLUE, lw=1.6)
    ax2.axvline(selected_threshold, color=cp.GOLD, lw=1.2, ls=":")
    ax2.text(selected_threshold, ax2.get_ylim()[0], f"선택 {selected_threshold:.3f}", ha="left", va="bottom", fontsize=8.5, color=cp.GOLD)
    cp.show(fig)


def plot_confusion(confusion, title):
    """혼동 행렬 하나를 축 없이."""
    fig = cp.new_figure((3.6, 3.0))
    ax = fig.add_subplot(111); cp.blank(ax)
    cp.draw_grid(ax, confusion, title, kind="count", fmt="{:g}", fontsize=11, row_labels=["정답 0", "정답 1"], col_labels=["예측 0", "예측 1"])
    cp.show(fig)
