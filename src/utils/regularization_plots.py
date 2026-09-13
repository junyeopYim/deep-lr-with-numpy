"""14 손실·정규화 노트북의 개념 그림.

손실 곡선·가중 평균·L2 템플릿·AdamW 한 스텝·dropout 마스크는 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

LOSS_COLORS = {"mse": cp.RED, "mae": cp.INK, "huber": cp.BLUE}
LOSS_NAMES = {"mse": "half-MSE $r^2/2$", "mae": "MAE $|r|$", "huber": "Huber ($\\delta=0.5$)"}


def plot_loss_shapes(r, losses, gradients, delta):
    """왼쪽: 잔차 → 손실. 오른쪽: 잔차 → 미분(전달하는 기울기). 큰 잔차에서 세 손실이 갈라집니다."""
    fig = cp.new_figure((11, 3.6))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax_l = fig.add_subplot(gs[0])
    _clean_axes(ax_l, "잔차 $r = \\hat y - y$", "손실 $\\rho(r)$", "같은 잔차에 다른 비용")
    ax_g = fig.add_subplot(gs[1])
    _clean_axes(ax_g, "잔차 $r$", "$\\rho'(r)$", "큰 잔차가 전달하는 기울기")
    for kind in ("mse", "mae", "huber"):
        ax_l.plot(r, losses[kind], color=LOSS_COLORS[kind], lw=1.8, label=LOSS_NAMES[kind])
        ax_g.plot(r, gradients[kind], color=LOSS_COLORS[kind], lw=1.8, label=LOSS_NAMES[kind])
    for a in (ax_l, ax_g):
        a.axvline(delta, color=cp.GOLD, lw=0.9, ls="--"); a.axvline(-delta, color=cp.GOLD, lw=0.9, ls="--")
    ax_l.set_ylim(0, 3.2)
    ax_l.legend(frameon=False, fontsize=8.5, loc="upper center")
    ax_g.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(ax_g, "금색 = $\\pm\\delta$ · MSE 만 잔차에 비례해 커짐", y=-0.2)
    cp.show(fig)


def plot_weighted_mean(values, weights, plain_mean, weighted_mean):
    """관측별 손실 막대 위에 가중치를 표시하고, 단순 평균과 가중 평균을 비교."""
    n = len(values)
    fig = cp.new_figure((8.5, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "관측 $n$", "관측별 손실", "관측마다 손실과 가중치 $a_n$")
    colors = [cp.BLUE if w > 0 else cp.MUTED for w in weights]
    ax.bar(np.arange(n), values, color=colors, width=0.6)
    for i, (v, w) in enumerate(zip(values, weights)):
        ax.text(i, v + 0.02 * max(values), f"$a={w:g}$", ha="center", va="bottom", fontsize=9, color=cp.INK)
    ax.set_xticks(np.arange(n))
    ax_m = fig.add_subplot(gs[1])
    _clean_axes(ax_m, None, None, "평균의 두 가지")
    ax_m.bar(["단순 평균", "가중 평균"], [plain_mean, weighted_mean], color=[cp.MUTED, cp.BLUE], width=0.55)
    for i, v in enumerate([plain_mean, weighted_mean]):
        ax_m.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=cp.INK)
    cp.label(ax_m, "$\\sum_n a_n \\ell_n / \\sum_n a_n$ · 가중치에 상수를 곱해도 같음", y=-0.16)
    cp.show(fig)


def plot_l2_templates(images, lambdas, norms, errors):
    """L2 세기별로 닫힌 해로 구한 7 판별 가중치 이미지. 아래에 가중치 길이와 학습 오차."""
    n = len(images)
    fig = cp.new_figure((2.6 * n + 0.5, 3.4))
    gs = fig.add_gridspec(1, n, wspace=0.15)
    for i, (img, lam, norm, err) in enumerate(zip(images, lambdas, norms, errors)):
        ax = fig.add_subplot(gs[i]); cp.blank(ax)
        cp.draw_image(ax, img, f"$\\lambda = {lam:g}$", kind="signed")
        cp.label(ax, f"$\\|w\\| = {norm:.2f}$ · 학습 half-MSE {err:.4f}", y=-0.06, color=cp.INK, fontsize=9)
    cp.label(fig.axes[0], "판마다 색 눈금이 다름 · λ 가 클수록 짧고 매끈한 $w$", y=-0.18)
    cp.show(fig)


def plot_decay_comparison(W0, sgd, coupled, decoupled):
    """같은 기울기·감쇠 계수에서 SGD+L2, Adam+L2, AdamW 의 첫 스텝 이동량을 좌표별로."""
    fig = cp.new_figure((8.5, 3.4))
    ax = fig.add_subplot(111)
    _clean_axes(ax, None, "$W_1 - W_0$ (첫 스텝 이동)", "상태에 무엇을 넣었는지가 첫 스텝의 방향을 바꿈")
    labels = ["좌표 0 ($W_0=1$)", "좌표 1 ($W_0=10$)"]
    x = np.arange(2)
    for k, (name, W, color) in enumerate([("SGD + L2", sgd, cp.MUTED), ("Adam + L2 (기울기에 $\\lambda W$)", coupled, cp.RED), ("AdamW (별도 감쇠)", decoupled, cp.BLUE)]):
        ax.bar(x + (k - 1) * 0.26, W - W0, width=0.24, color=color, label=name)
        for i, v in enumerate(W - W0):
            ax.text(x[i] + (k - 1) * 0.26, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=cp.INK)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    ax.set_xticks(x, labels)
    lo = min(min(sgd - W0), min(coupled - W0), min(decoupled - W0))
    hi = max(max(sgd - W0), max(coupled - W0), max(decoupled - W0))
    ax.set_ylim(lo - 0.014, hi + 0.02)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.show(fig)


def plot_dropout_masks(image, masked, mean_image, keep_probability, count):
    """7에 inverted dropout 마스크를 씌운 세 장과, 마스크 많이 뽑은 평균."""
    n = len(masked)
    fig = cp.new_figure((2.3 * (n + 2) + 0.5, 3.2))
    gs = fig.add_gridspec(1, n + 2, wspace=0.15)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x); cp.draw_image(ax_x, image, "$H$ = $x_7$")
    vmax = float(max(m.max() for m in masked))
    for i, m in enumerate(masked):
        ax = fig.add_subplot(gs[1 + i]); cp.blank(ax)
        cp.draw_image(ax, m / vmax, f"$H \\odot M_{{{i + 1}}} / q$" + (f" ($q={keep_probability:g}$)" if i == 0 else ""))
    ax_m = fig.add_subplot(gs[n + 1]); cp.blank(ax_m)
    cp.draw_image(ax_m, np.clip(mean_image, 0, 1), f"마스크 {count:,}개의 평균")
    cp.label(fig.axes[1], f"살아남은 화소는 $1/q = {1 / keep_probability:.2f}$ 배 · 평균은 원래 $H$", y=-0.06)
    cp.show(fig)


def plot_dropout_objective(base_loss, extra_loss, expected_loss):
    """모든 mask 를 합한 선형 모델의 기대 손실 = 기본 손실 + 분산 항."""
    fig = cp.new_figure((6.5, 3.2))
    ax = fig.add_subplot(111)
    _clean_axes(ax, None, "half-MSE", "모든 mask 를 확률로 합한 목적")
    ax.bar(["기본 손실", "dropout 분산 항", "기대 손실"], [base_loss, extra_loss, expected_loss], color=[cp.MUTED, cp.RED, cp.BLUE], width=0.55)
    for i, v in enumerate([base_loss, extra_loss, expected_loss]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5, color=cp.INK)
    cp.label(ax, "분산 항 $= \\frac{1-q}{2q}\\sum_j W_j^2\\,\\mathrm{mean}_n X_{nj}^2$ : L2 모양", y=-0.14, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_outlier_fits(X, Y, X_clean, fits, outlier_indices):
    """이상치가 있는 관측과 세 손실로 맞춘 직선."""
    fig = cp.new_figure((7.5, 4.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$x$", "$y$", "같은 관측, 다른 잔차 비용")
    mask = np.zeros(len(X), dtype=bool); mask[outlier_indices] = True
    ax.scatter(X[~mask], Y[~mask], s=12, color=cp.MUTED, label="관측")
    ax.scatter(X[mask], Y[mask], s=22, color=cp.GOLD, label="이상치 (+7)")
    for kind, line in fits.items():
        ax.plot(X_clean, line, color=LOSS_COLORS[kind], lw=1.8, label=kind)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.show(fig)


def plot_ridge_norms(lambdas, norms):
    """L2 계수와 ridge 해의 가중치 길이."""
    fig = cp.new_figure((5.5, 3.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$\\lambda$", "$\\|W\\|_2$", "L2 계수가 클수록 가중치가 짧음")
    ax.bar([str(l) for l in lambdas], norms, color=cp.BLUE, width=0.55)
    for i, v in enumerate(norms):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=cp.INK)
    cp.show(fig)
