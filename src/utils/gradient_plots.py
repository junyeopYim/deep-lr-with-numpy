"""01번 노트북에서 계산한 손실·도함수·학습 경로를 그립니다."""

import numpy as np
import matplotlib.pyplot as plt

from .plotting import COLOR_ACCENT, COLOR_NEG, COLOR_POS


def plot_derivative(grid, values, tangent, point, point_value, secant_points, secant_values):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, values, label="함수 f(w)")
    ax.plot(grid, tangent, "--", label="선택한 점의 접선")
    ax.plot(secant_points, secant_values, "o-", color=COLOR_NEG, label="두 점 사이 변화율")
    ax.scatter([point], [point_value], color=COLOR_ACCENT, s=60, zorder=5)
    ax.set(xlabel="w", ylabel="함수값", title="미분: 작은 입력 변화에 출력이 얼마나 변하는가")
    ax.legend()
    plt.show()


def plot_difference_errors(epsilons, forward_errors, central_errors):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(epsilons, np.maximum(forward_errors, 1e-17), "o-", label="전진차분")
    ax.loglog(epsilons, np.maximum(central_errors, 1e-17), "o-", label="중심차분")
    ax.set(xlabel="차분 폭 ε", ylabel="도함수와의 절대 차이", title="ε를 줄이는 효과와 부동소수점 오차")
    ax.legend()
    plt.show()


def plot_loss_surface(ww, bb, losses, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    contours = ax.contour(ww, bb, losses, levels=16, linewidths=0.9, cmap="Blues")
    ax.clabel(contours, inline=True, fontsize=8)
    ax.plot(path[:, 0], path[:, 1], "o-", ms=3, color=COLOR_NEG, label="경사하강 경로")
    ax.scatter(*path[0], marker="s", s=65, color=COLOR_ACCENT, label="시작")
    ax.scatter(*path[-1], marker="*", s=130, color=COLOR_POS, label="마지막")
    ax.set(xlabel="가중치 w", ylabel="편향 b", title="손실이라는 지형 위에서 파라미터를 움직입니다")
    ax.legend()
    plt.show()


def plot_fit_snapshots(x, y, line_x, predictions, steps):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, pred, step in zip(axes.ravel(), predictions, steps):
        ax.scatter(x.ravel(), y.ravel(), s=16, alpha=0.7, color=COLOR_ACCENT, label="학습 데이터")
        ax.plot(line_x.ravel(), pred.ravel(), color=COLOR_POS, label="현재 예측")
        ax.set_title(f"{step}회 갱신 뒤")
        ax.set(xlabel="붉기 x", ylabel="당도 y")
    axes.ravel()[0].legend()
    plt.show()


def plot_learning_curves(histories):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, losses in histories.items():
        ax.semilogy(np.arange(len(losses)), np.maximum(losses, 1e-17), label=label)
    ax.set(xlabel="파라미터 갱신 횟수", ylabel="평균 손실", title="같은 기울기도 학습률에 따라 다른 경로를 만듭니다")
    ax.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 개념 그림. 계산(잔차·기울기·갱신·학습 루프)은 본문에서 하고 여기서는 받은 배열만 그립니다.
# 부품은 concept_plots·schematic_plots, 원칙은 docs/DESIGN.md.
# ---------------------------------------------------------------------------

from . import concept_plots as cp
from . import schematic_plots as sp


def _clean_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.set_axis_on()
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cp.MUTED)
    ax.tick_params(colors=cp.MUTED, labelsize=8.5, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=cp.INK, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=cp.INK, fontsize=9.5)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)


def plot_loss_from_residuals(x, y, line_x, line_y, residuals, loss):
    """데이터·현재 직선·잔차 → 잔차 막대 → 제곱 막대와 평균. '왜 제곱해서 평균내는가'."""
    fig, axes = cp.flow([1.6, 1, 1], height=3.3, unit=2.0, wspace=0.55)
    ax = axes[0]
    _clean_axes(ax, "붉기 $x$", "당도 $y$", "관측과 현재 직선 $\\hat y = xw + b$")
    ax.scatter(x, y, s=14, color=cp.INK, zorder=3, label="관측")
    ax.plot(line_x, line_y, color=cp.BLUE, lw=1.8, label="현재 예측")
    yhat = np.interp(x, line_x, line_y)
    for xi, yi, pi in zip(x, y, yhat):
        ax.plot([xi, xi], [yi, pi], color=cp.RED, lw=0.9, alpha=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    r = np.asarray(residuals).ravel()
    idx = np.arange(len(r))
    ax = axes[1]
    _clean_axes(ax, "샘플 번호", None, "잔차 $r = \\hat y - y$")
    ax.bar(idx, r, color=[cp.BLUE if v >= 0 else cp.RED for v in r], width=0.8)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    ax.set_yticks([0]); ax.set_xticks([])
    cp.label(ax, "부호가 있어 그냥 더하면 상쇄됨", y=-0.08)
    ax = axes[2]
    _clean_axes(ax, "샘플 번호", None, "$r^2$")
    ax.bar(idx, r ** 2, color=cp.MUTED, width=0.8)
    ax.axhline(np.mean(r ** 2), color=cp.GOLD, lw=1.6)
    ax.text(0, np.mean(r ** 2), f"평균 {np.mean(r**2):.2f}", color=cp.GOLD, fontsize=8.5, va="bottom", ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax.set_yticks([0]); ax.set_xticks([])
    cp.label(ax, f"$J = \\frac{{1}}{{2}} \\cdot$ 평균 $= {loss:.2f}$", y=-0.08, color=cp.INK, fontsize=9.5)
    cp.connect(fig, axes[0], axes[1], "빼기"); cp.connect(fig, axes[1], axes[2], "제곱")
    cp.show(fig)


def plot_tangent(grid, values, w0, f0, slope, secant_w, secant_f):
    """함수 곡선, 한 점의 접선, 두 점 사이 평균 변화율. 최소 스타일."""
    fig = cp.new_figure((6.4, 3.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$w$", "$f(w)$", "미분 = 그 점에서의 접선 기울기")
    ax.plot(grid, values, color=cp.INK, lw=1.8, label="$f(w)=w^3$")
    ax.plot(grid, f0 + slope * (grid - w0), color=cp.BLUE, lw=1.6, ls="--", label=f"접선, 기울기 {slope:g}")
    ax.plot(secant_w, secant_f, "o-", color=cp.RED, lw=1.4, ms=5,
            label=f"두 점 사이 평균 변화율 {(secant_f[1]-secant_f[0])/(secant_w[1]-secant_w[0]):.2f}")
    ax.scatter([w0], [f0], color=cp.GOLD, s=60, zorder=5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.show(fig)


def plot_gradient_image(images, residuals, labels, part_images, part_titles, dW_image, db):
    """오차 × 이미지를 모두 더하면 기울기 이미지가 됩니다.

    왼쪽: 이미지와 오차. 가운데: 기여를 나눈 이미지들(part_images, 각각 자기 크기로 색칠). 오른쪽: 합 dW.
    """
    n = len(images)
    cols = (n + 1) // 2
    k = len(part_images)
    fig = cp.new_figure((12, 3.8))
    gs = fig.add_gridspec(1, 2 + k, width_ratios=[2.6] + [1.05] * k + [1.15], wspace=0.5)
    sub = gs[0].subgridspec(2, cols, wspace=0.08, hspace=0.55)
    r = np.asarray(residuals).ravel()
    for i in range(n):
        ax = fig.add_subplot(sub[i // cols, i % cols]); cp.blank(ax)
        cp.draw_image(ax, images[i], frame=False)
        ax.set_title(f"{labels[i]}", color=cp.MUTED, fontsize=8.5, pad=2)
        cp.label(ax, f"$r={r[i]:+.2f}$", y=-0.04, color=cp.BLUE if r[i] >= 0 else cp.RED, fontsize=8.5)
    fig.text(0.5 * (fig.axes[0].get_position().x0 + fig.axes[cols - 1].get_position().x1), 0.97,
             "이미지 $x_n$과 그 오차 $r_n$", ha="center", va="top", color=cp.INK, fontsize=10.5)
    parts = []
    for j, (img, title) in enumerate(zip(part_images, part_titles)):
        ax = fig.add_subplot(gs[1 + j]); cp.blank(ax)
        cp.draw_image(ax, img, title, kind="signed")
        parts.append(ax)
    ax_g = fig.add_subplot(gs[1 + k]); cp.blank(ax_g)
    cp.draw_image(ax_g, dW_image, "$dW = \\sum_n r_n x_n / N$", kind="signed")
    cp.label(ax_g, f"$db = \\sum_n r_n / N = {db:+.2f}$", y=-0.06, color=cp.INK, fontsize=9.5)
    cp.connect(fig, fig.axes[cols - 1], parts[0], "$r_n x_n$")
    for a, b_ in zip(parts, parts[1:] + [ax_g]):
        cp.connect(fig, a, b_, "+" if b_ is not ax_g else "=")
    cp.label(parts[0], "판마다 색 눈금이 다름", y=-0.06)
    cp.show(fig)


def plot_update_step(w_img, step_img, w_new_img, scores_before, scores_after, targets, labels,
                     loss_before, loss_after, learning_rate):
    """한 스텝: w − η·dW = w_new (이미지 세 장), 그 전후의 점수와 정답, 손실."""
    fig = cp.new_figure((11, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.45)
    top = gs[0].subgridspec(1, 5, width_ratios=[1, 0.3, 1, 0.3, 1], wspace=0.05)
    vmax = float(max(np.abs(w_img).max(), np.abs(w_new_img).max()))
    ax = fig.add_subplot(top[0]); cp.blank(ax); cp.draw_image(ax, w_img, "$w$ (지금)", kind="signed", vmax=vmax)
    ax = fig.add_subplot(top[1]); cp.blank(ax); cp.draw_formula(ax, "$-$", None, fontsize=18)
    ax = fig.add_subplot(top[2]); cp.blank(ax); cp.draw_image(ax, step_img, f"$\\eta\\, dW$ ($\\eta={learning_rate:g}$)", kind="signed", vmax=vmax)
    ax = fig.add_subplot(top[3]); cp.blank(ax); cp.draw_formula(ax, "$=$", None, fontsize=18)
    ax = fig.add_subplot(top[4]); cp.blank(ax); cp.draw_image(ax, w_new_img, "$w_{\\mathrm{new}}$", kind="signed", vmax=vmax)
    bottom = gs[1].subgridspec(1, 2, wspace=0.5)
    lo = min(np.min(scores_before), np.min(scores_after), 0) - 0.2
    hi = max(np.max(scores_before), np.max(scores_after), 1) + 0.2
    for k, (scores, loss, title) in enumerate([(scores_before, loss_before, "갱신 전 점수 · $J=%.3f$" % loss_before),
                                                (scores_after, loss_after, "갱신 후 점수 · $J=%.3f$" % loss_after)]):
        ax = fig.add_subplot(bottom[k]); cp.blank(ax)
        cp.draw_bars(ax, scores, title, labels=[str(l) for l in labels], color=cp.MUTED, signed=True, xlim=(lo, hi))
        for i, t in enumerate(np.asarray(targets).ravel()):
            ax.plot([t], [i], marker="|", color=cp.GOLD, ms=14, mew=2.2)
        ax.set_xlim(lo, hi)
    cp.label(fig.axes[-1], "금색 눈금 = 정답 (7이면 1, 아니면 0)", y=-0.08)
    cp.show(fig)


def plot_training_images(w_images, losses, steps):
    """반복 갱신으로 가중치 이미지가 생겨나는 과정. 위: w 이미지, 아래: 손실 막대."""
    n = len(w_images)
    fig = cp.new_figure((11, 3.9))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 0.35], hspace=0.15, wspace=0.12)
    vmax = float(max(np.abs(w).max() for w in w_images))
    for i, (w, loss, step) in enumerate(zip(w_images, losses, steps)):
        ax = fig.add_subplot(gs[0, i]); cp.blank(ax)
        cp.draw_image(ax, w, f"{step}회 갱신 뒤", kind="signed", vmax=vmax)
        ax = fig.add_subplot(gs[1, i]); cp.blank(ax)
        ax.barh([0], [loss], color=cp.BLUE if i == n - 1 else cp.MUTED, height=0.5)
        ax.set_xlim(0, max(losses) * 1.05); ax.set_ylim(-1.4, 0.6)
        ax.text(0, -0.55, f"$J={loss:.3f}$", va="top", ha="left", fontsize=9, color=cp.INK)
    cp.show(fig)


def plot_descent_landscape(ww, bb, loss_grid, path, loss_along_path):
    """손실 곡면과 등고선 위의 경사하강 경로. 시작은 점, 끝은 네모."""
    fig = cp.new_figure((11, 4.2))
    pts = np.column_stack([path, loss_along_path])
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    sp.draw_surface_paths(ax3, ww, bb, loss_grid, {"경사하강": pts}, elev=35, azim=-60)
    ax3.set_xlabel("$w$", color=cp.INK); ax3.set_ylabel("$b$", color=cp.INK)
    ax2 = fig.add_subplot(1, 2, 2)
    sp.draw_contour_paths(ax2, ww, bb, loss_grid, {"경사하강": pts}, levels=22)
    ax2.set_xlabel("$w$", color=cp.INK); ax2.set_ylabel("$b$", color=cp.INK)
    ax2.set_title("같은 경로를 위에서 본 것 · 등고선과 수직으로 내려감", color=cp.INK, fontsize=10.5)
    cp.show(fig)
