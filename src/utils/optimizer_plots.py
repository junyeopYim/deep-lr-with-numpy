"""04번 본문에서 계산한 기울기·상태·궤적·초기화 통계를 그립니다."""

import matplotlib.pyplot as plt


def plot_batch_gradients(batch_sizes, empirical_std, theoretical_std):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(batch_sizes, empirical_std, "o-", label="반복 추출에서 관측한 표준편차")
    ax.plot(batch_sizes, theoretical_std, "s--", label="표본 평균의 이론 표준편차")
    ax.set(xlabel="미니배치 크기", ylabel="w 기울기의 표준편차", title="같은 파라미터에서 배치만 바꾼 실험")
    ax.legend()
    plt.show()


def plot_moments(steps, raw_mean, corrected_mean, raw_square, corrected_square):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, raw, corrected, title in zip(axes, [raw_mean, raw_square],
                                        [corrected_mean, corrected_square],
                                        ["1차 모멘트", "2차 원시 모멘트"]):
        ax.plot(steps, raw, "o-", label="0에서 시작한 이동 평균")
        ax.plot(steps, corrected, "--", label="초기값 편향 보정 후")
        ax.set(xlabel="업데이트 횟수 t", ylabel="값", title=title)
        ax.legend()
    plt.show()


def plot_optimizer_paths(xx, yy, objective, histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].contour(xx, yy, objective, levels=18, cmap="Greys", alpha=0.65)
    for name, history in histories.items():
        points = history["theta"]
        axes[0].plot(points[:, 0], points[:, 1], "-", label=name, alpha=0.85)
        axes[1].semilogy(range(len(history["loss"])), history["loss"], label=name)
    axes[0].scatter([0], [0], marker="*", s=100, color="black")
    axes[0].set(xlabel="첫 파라미터", ylabel="둘째 파라미터", title="같은 이차 목적에서의 경로")
    axes[1].set(xlabel="업데이트 횟수", ylabel="목적값", title="선택한 학습률에서의 손실 변화")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_regression_runs(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, history in histories.items():
        axes[0].plot(history["seen"], history["loss"], label=name)
        axes[1].plot(history["seen"], history["gradient_norm"], label=name)
    axes[0].set(xlabel="처리한 학습 표본 수", ylabel="전체 학습셋 손실", title="같은 데이터 순서로 학습")
    axes[1].set(xlabel="처리한 학습 표본 수", ylabel="전체 손실의 기울기 norm", title="같은 시점의 전체 기울기")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_initialization(depths, second_moments, gradient_rms):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name in second_moments:
        axes[0].semilogy(depths, second_moments[name], "o-", label=name)
        axes[1].semilogy(depths, gradient_rms[name], "o-", label=name)
    axes[0].set(xlabel="층 (0은 입력)", ylabel="활성값 제곱의 평균", title="forward 값의 크기")
    axes[1].set(xlabel="층 (0은 입력)", ylabel="전달된 기울기의 RMS", title="backward 값의 크기")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_schedule(steps, learning_rates, norms_before, norms_after):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(steps, learning_rates)
    axes[0].set(xlabel="업데이트 횟수", ylabel="학습률", title="미리 정한 학습률 스케줄")
    axes[1].bar(["원래 기울기", "clipping 후"], [norms_before, norms_after])
    axes[1].set(ylabel="전체 gradient norm", title="크기를 제한하는 별도 연산")
    plt.show()


# ---------------------------------------------------------------------------
# 개념 그림. 갱신 경로·기울기·상태 값은 본문에서 계산하고 여기서는 받은 배열만 그립니다.
# 부품은 concept_plots·schematic_plots, 원칙은 docs/DESIGN.md.
# ---------------------------------------------------------------------------

import numpy as np
from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes


def plot_step_size_parabola(w_grid, J_grid, paths, curvature):
    """1차원 손실 J = a w²/2 위에서 학습률 세 가지의 갱신 자취. η < 2/a 이면 수렴."""
    n = len(paths)
    fig, axes = cp.flow([1] * n, height=3.2, unit=2.3, wspace=0.35)
    colors = [cp.BLUE, cp.GOLD, cp.RED]
    for ax, (name, w_seq), color in zip(axes, paths.items(), colors):
        w_seq = np.asarray(w_seq)
        _clean_axes(ax, "$w$", "$J$" if ax is axes[0] else None, name)
        ax.plot(w_grid, J_grid, color=cp.INK, lw=1.6)
        J_seq = 0.5 * curvature * w_seq ** 2
        ax.plot(w_seq, J_seq, "o-", color=color, ms=4, lw=1.2, alpha=0.9)
        ax.scatter(w_seq[0], J_seq[0], s=50, color=color, zorder=4)
        ax.set_xlim(w_grid.min(), w_grid.max()); ax.set_ylim(0, J_grid.max() * 1.05)
        ax.set_yticks([]); ax.set_xticks([0])
        cp.label(ax, f"{len(w_seq)-1}스텝 뒤 $|w| = {abs(w_seq[-1]):.2g}$", y=-0.16, color=color)
    cp.show(fig)


def plot_gd_zigzag(xx, yy, surface, path, learning_rate):
    """곡률이 다른 두 방향을 가진 이차 목적에서 경사하강이 지그재그로 내려가는 모습."""
    fig = cp.new_figure((6.4, 4.2))
    ax = fig.add_subplot(111)
    sp.draw_contour_paths(ax, xx, yy, surface, {f"경사하강 η={learning_rate:g}": np.column_stack([path, np.zeros(len(path))])}, levels=20)
    ax.scatter([0], [0], marker="*", s=120, color=cp.GOLD, zorder=4)
    ax.set_xlabel("$\\theta_0$ (곡률 1)", color=cp.INK); ax.set_ylabel("$\\theta_1$ (곡률 20)", color=cp.INK)
    ax.set_title("가파른 방향은 튀고, 완만한 방향은 느리게 갑니다", color=cp.INK, fontsize=10.5)
    cp.show(fig)


def plot_state_bars(steps, series, titles, row_titles, colors=None):
    """스텝별 값들을 막대로. series는 행(좌표)마다 [g_t, v_t, …] 목록. 상태가 어떻게 쌓이는지."""
    rows, cols = len(series), len(series[0])
    fig = cp.new_figure((3.0 * cols + 0.6, 2.2 * rows))
    gs = fig.add_gridspec(rows, cols, wspace=0.3, hspace=0.55)
    colors = colors or [cp.MUTED, cp.BLUE, cp.RED]
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            vals = np.asarray(series[r][c])
            _clean_axes(ax, "스텝 $t$" if r == rows - 1 else None, row_titles[r] if c == 0 else None,
                        titles[c] if r == 0 else None)
            ax.bar(steps, vals, color=colors[c % len(colors)], width=0.7)
            ax.axhline(0, color=cp.MUTED, lw=0.7)
            ax.set_xticks(steps[::max(1, len(steps) // 6)])
    cp.show(fig)


def plot_bias_correction(steps, raw_mean, corrected_mean, raw_square, corrected_square, targets):
    """0에서 시작한 이동 평균은 초반에 작고, 보정하면 참값을 복원합니다."""
    fig, axes = cp.flow([1, 1], height=3.0, unit=2.6, wspace=0.4)
    for ax, raw, corrected, target, title in zip(axes, [raw_mean, raw_square], [corrected_mean, corrected_square], targets,
                                                 ["1차 모멘트 $m_t$", "2차 모멘트 $s_t$"]):
        _clean_axes(ax, "업데이트 횟수 $t$", None, title)
        ax.plot(steps, raw, "o-", color=cp.MUTED, ms=3.5, lw=1.2, label="0에서 시작한 이동 평균")
        ax.plot(steps, corrected, "s-", color=cp.BLUE, ms=3.5, lw=1.2, label="편향 보정 후")
        ax.axhline(target, color=cp.GOLD, lw=1.2, ls="--", label=f"참값 {target:g}")
        ax.set_ylim(0, target * 1.3)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    cp.show(fig)


def plot_optimizer_landscape(xx, yy, surface, histories):
    """같은 이차 목적에서 네 optimizer의 경로: 3차원 곡면, 등고선, 손실 곡선."""
    fig = cp.new_figure((13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.3)
    paths = {}
    for name, h in histories.items():
        pts = np.asarray(h["theta"])
        paths[name] = np.column_stack([pts, np.asarray(h["loss"])])
    ax3 = fig.add_subplot(gs[0], projection="3d")
    sp.draw_surface_paths(ax3, xx, yy, surface, paths, elev=38, azim=-50, alpha=0.25)
    ax3.set_xlabel("$\\theta_0$", color=cp.INK); ax3.set_ylabel("$\\theta_1$", color=cp.INK); ax3.set_zlabel("$J$", color=cp.INK)
    ax2 = fig.add_subplot(gs[1])
    sp.draw_contour_paths(ax2, xx, yy, surface, paths, levels=20)
    ax2.set_xlabel("$\\theta_0$", color=cp.INK); ax2.set_ylabel("$\\theta_1$", color=cp.INK)
    ax2.get_legend().remove()
    ax2.set_title("위에서 본 경로", color=cp.INK, fontsize=10.5)
    ax4 = fig.add_subplot(gs[2])
    _clean_axes(ax4, "업데이트 횟수", "$J$ (로그 눈금)", "목적값의 변화")
    for (name, h), color in zip(histories.items(), sp.PATH_COLORS):
        ax4.semilogy(np.arange(len(h["loss"])), np.maximum(h["loss"], 1e-12), color=color, lw=1.6, label=name)
    ax4.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_optimizer_templates(images_by_name, losses_by_name, steps):
    """행: optimizer, 열: 스텝. MNIST 7 판별 가중치 이미지가 자라나는 속도 비교."""
    names = list(images_by_name)
    rows, cols = len(names), len(steps)
    fig = cp.new_figure((2.2 * cols + 1.2, 2.3 * rows))
    gs = fig.add_gridspec(rows, cols, wspace=0.08, hspace=0.35)
    for r, name in enumerate(names):
        vmax = float(max(np.abs(im).max() for im in images_by_name[name]))     # 행(optimizer)마다 색 눈금을 따로 둡니다
        for c, (img, step, loss) in enumerate(zip(images_by_name[name], steps, losses_by_name[name])):
            ax = fig.add_subplot(gs[r, c]); cp.blank(ax)
            cp.draw_image(ax, img, f"{step}스텝" if r == 0 else None, kind="signed", vmax=vmax)
            cp.label(ax, f"$J={loss:.4f}$", y=-0.05, color=cp.INK, fontsize=9)
            if c == 0:
                ax.text(-0.12, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=10, color=cp.INK)
    cp.show(fig)
