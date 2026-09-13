"""12 다변량 Gaussian·정보이론 노트북의 개념 그림.

샘플·밀도·entropy·클래스별 평균/표준편차·KL 값은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401


def _cloud(ax, X, title, color=cp.MUTED, s=7):
    _clean_axes(ax, "$x_0$", "$x_1$", title)
    ax.scatter(X[:, 0], X[:, 1], s=s, color=color, alpha=0.55)
    ax.set_aspect("equal", adjustable="datalim")


def plot_sampling_flow(eps, samples, independent, mu, L):
    """표준정규 잡음 ε → x = μ + Lε (상관 있는 구름) · 같은 좌표별 분산의 독립 Gaussian."""
    fig = cp.new_figure((13, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.45)
    ax_e = fig.add_subplot(gs[0]); _cloud(ax_e, eps, "$\\epsilon \\sim \\mathcal{N}(0, I)$ : 독립 잡음")
    ax_x = fig.add_subplot(gs[1]); _cloud(ax_x, samples, "$x = \\mu + L\\epsilon$ : 기울어진 구름", color=cp.BLUE)
    ax_x.scatter(*mu, color=cp.GOLD, s=50, zorder=4)
    ax_i = fig.add_subplot(gs[2]); _cloud(ax_i, independent, "같은 좌표별 분산, 공분산 0", color=cp.RED)
    ax_i.scatter(*mu, color=cp.GOLD, s=50, zorder=4)
    cp.connect(fig, ax_e, ax_x, "$L$, $+\\mu$")
    cp.label(ax_i, f"금색 = $\\mu$ · 가운데는 왼쪽 점을 $L = [[{L[0, 0]:g}, {L[0, 1]:g}], [{L[1, 0]:g}, {L[1, 1]:g}]]$ 로 옮긴 것", y=-0.2)
    cp.show(fig)


def plot_log_density(xx, yy, logpdf_grid, mu, points, logpdfs, terms):
    """로그 밀도의 등고선과 점 두 개의 값. 오른쪽에 세 항(상수·부피·거리)의 막대."""
    fig = cp.new_figure((11.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.4)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x_0$", "$x_1$", "$\\log p(x)$ 의 등고선 · 평균에서 멀수록 작음")
    ax.contour(xx, yy, logpdf_grid, levels=12, cmap="Blues", linewidths=0.9)
    ax.scatter(*mu, color=cp.GOLD, s=55, zorder=4)
    ax.text(mu[0] + 0.15, mu[1] + 0.15, "$\\mu$", color=cp.GOLD, fontsize=10)
    offsets = [(-0.25, 0.45, "right"), (0.25, -0.55, "left")]
    for (pt, lp), color, name, (dx, dy, ha) in zip(zip(points, logpdfs), [cp.BLUE, cp.RED], ["A", "B"], offsets):
        ax.scatter(*pt, color=color, s=45, zorder=4)
        ax.text(pt[0] + dx, pt[1] + dy, f"{name}: $\\log p = {lp:.2f}$", color=color, fontsize=9, ha=ha)
    ax.set_aspect("equal", adjustable="datalim")
    ax_t = fig.add_subplot(gs[1])
    _clean_axes(ax_t, None, "nat", "$-\\frac{1}{2}[D\\log 2\\pi + \\log\\det\\Sigma + d^2]$ 의 세 항")
    names = ["$D\\log 2\\pi$", "$\\log\\det\\Sigma$", "$d^2$ (A)", "$d^2$ (B)"]
    colors = [cp.MUTED, cp.MUTED, cp.BLUE, cp.RED]
    ax_t.bar(names, terms, color=colors, width=0.6)
    for i, v in enumerate(terms):
        ax_t.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=cp.INK)
    ax_t.tick_params(axis="x", labelsize=9)
    cp.label(ax_t, "$d^2 = (x-\\mu)^{\\top}\\Sigma^{-1}(x-\\mu)$ 만 점에 따라 다름", y=-0.2)
    cp.show(fig)


def plot_entropy_examples(distributions, titles, entropies, ces, target):
    """예측 분포 세 개(확신·애매·틀림)의 확률 막대와 entropy, 정답 $y$ 에 대한 CE."""
    n = len(distributions)
    fig = cp.new_figure((3.4 * n + 1.5, 3.6))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.2], wspace=0.5)
    for k, (p, title) in enumerate(zip(distributions, titles)):
        ax = fig.add_subplot(gs[k]); cp.blank(ax)
        cp.draw_bars(ax, p, title, labels=[str(i) for i in range(len(p))], marker=target, xlim=(0, 1), color=cp.BLUE)
        cp.label(ax, f"$H = {entropies[k]:.2f}$ · $CE = {ces[k]:.2f}$", y=-0.05, color=cp.INK, fontsize=9.5)
    ax_b = fig.add_subplot(gs[n])
    _clean_axes(ax_b, None, "nat", "entropy 와 CE (정답 7)")
    x = np.arange(n)
    ax_b.bar(x - 0.18, entropies, width=0.36, color=cp.MUTED, label="$H(p)$")
    ax_b.bar(x + 0.18, ces, width=0.36, color=cp.RED, label="$CE = -\\log p_y$")
    ax_b.set_xticks(x, ["확신", "애매", "틀림"])
    ax_b.legend(frameon=False, fontsize=8.5)
    cp.label(fig.axes[0], "금색 테두리 = 정답 칸", y=-0.16)
    cp.show(fig)


def plot_ce_kl_curves(probabilities, ce_curve, kl_curve, entropy_value):
    """고정된 관측 분포에서 예측 q 를 바꿀 때의 CE 와 KL. 차이는 항상 H(p)."""
    fig = cp.new_figure((7, 3.5))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "예측 $q$ 의 첫 사건 확률", "nat", "관측 분포 $p=(0.75, 0.25)$ 고정 · $H(p,q) = H(p) + KL(p\\|q)$")
    ax.plot(probabilities, ce_curve, color=cp.RED, lw=1.8, label="$H(p, q)$")
    ax.plot(probabilities, kl_curve, color=cp.BLUE, lw=1.8, label="$KL(p\\|q)$")
    ax.axhline(entropy_value, color=cp.GOLD, lw=1.2, ls="--", label=f"$H(p) = {entropy_value:.3f}$")
    ax.axvline(0.75, color=cp.MUTED, lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=9)
    cp.show(fig)


def plot_class_gaussians(mean_images, std_images, labels):
    """클래스별 대각 Gaussian 의 MLE: 위 = 평균 이미지 μ_c, 아래 = 표준편차 이미지 σ_c."""
    n = len(mean_images)
    fig = cp.new_figure((1.25 * n + 0.8, 3.2))
    gs = fig.add_gridspec(2, n, wspace=0.06, hspace=0.25)
    smax = float(max(s.max() for s in std_images))
    for i in range(n):
        ax = fig.add_subplot(gs[0, i]); cp.blank(ax)
        cp.draw_image(ax, mean_images[i], f"{labels[i]}", frame=False)
        ax = fig.add_subplot(gs[1, i]); cp.blank(ax)
        cp.draw_image(ax, std_images[i] / smax, frame=False)
    fig.axes[0].text(-0.15, 0.5, "$\\mu_c$", transform=fig.axes[0].transAxes, ha="right", va="center", fontsize=11, color=cp.INK)
    fig.axes[1].text(-0.15, 0.5, "$\\sigma_c$", transform=fig.axes[1].transAxes, ha="right", va="center", fontsize=11, color=cp.INK)
    cp.label(fig.axes[-1], "σ 는 획의 가장자리에서 크고, 항상 흰 화소에서 0", y=-0.1)
    cp.show(fig)


def plot_two_covariances(full, diagonal):
    """전체 공분산 추정과 대각 공분산 추정."""
    fig = cp.new_figure((7.5, 3.0))
    gs = fig.add_gridspec(1, 2, wspace=0.6)
    vmax = float(np.abs(full).max())
    for ax, M, title in [(fig.add_subplot(gs[0]), full, "전체 공분산 $\\hat\\Sigma$"), (fig.add_subplot(gs[1]), diagonal, "대각 공분산 $\\mathrm{diag}(e^{\\ell})$")]:
        cp.blank(ax)
        cp.draw_grid(ax, M, title, kind="signed", fmt="{:.2f}", fontsize=10, vmax=vmax, row_labels=["$x_0$", "$x_1$"], col_labels=["$x_0$", "$x_1$"])
    cp.label(fig.axes[-1], "대각 모형은 비대각 성분을 0으로 둠", y=-0.06)
    cp.show(fig)


def plot_map_shrink(observed, theta_mle, tau2_values, theta_maps):
    """수직선 위의 관측, MLE(관측 평균), 사전 평균 0, 그리고 τ² 마다의 MAP."""
    fig = cp.new_figure((8.5, 2.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$\\theta$", None, "MAP 는 MLE 와 사전 평균 0 사이 · 사전 분산 $\\tau^2$ 이 작을수록 0 쪽으로")
    ax.axhline(0, color=cp.MUTED, lw=1.0)
    ax.scatter(observed, np.zeros_like(observed), s=40, color=cp.INK, zorder=3, label="관측 $x_n$")
    ax.scatter([0], [0], s=70, color=cp.RED, zorder=4, label="사전 평균 0")
    ax.scatter([theta_mle], [0], s=70, color=cp.BLUE, zorder=4, label=f"MLE $\\bar x = {theta_mle:g}$")
    for tau2, theta in zip(tau2_values, theta_maps):
        ax.scatter([theta], [0], s=42, facecolor="white", edgecolor=cp.GOLD, lw=1.6, zorder=5)
        ax.text(theta, 0.13, f"$\\tau^2={tau2:g}$", ha="center", va="bottom", fontsize=8.5, color=cp.GOLD, rotation=45)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.4, 0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    cp.label(ax, "금색 = MAP $= \\sum x_n / (N + \\sigma^2/\\tau^2)$", y=-0.28, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_gaussian_kl(x, q_pdf, p_pdf, log_ratio, kl, q_label, p_label):
    """1차원 q 와 p 의 밀도, 그리고 q 로 가중한 log(q/p). KL 은 그 평균."""
    fig = cp.new_figure((11, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$z$", "밀도", "두 Gaussian")
    ax.plot(x, q_pdf, color=cp.BLUE, lw=1.8, label=q_label)
    ax.plot(x, p_pdf, color=cp.RED, lw=1.8, label=p_label)
    ax.fill_between(x, 0, q_pdf, color=cp.BLUE, alpha=0.08)
    ax.legend(frameon=False, fontsize=9)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$z$", None, f"$q(z)\\,[\\log q(z) - \\log p(z)]$ 의 적분 $= KL(q\\|p) = {kl:.3f}$")
    weighted = q_pdf * log_ratio
    ax2.fill_between(x, 0, weighted, where=weighted >= 0, color=cp.BLUE, alpha=0.4)
    ax2.fill_between(x, 0, weighted, where=weighted < 0, color=cp.RED, alpha=0.4)
    ax2.axhline(0, color=cp.MUTED, lw=0.8)
    for a in (ax, ax2):
        a.set_xlim(-5, 7)
    cp.label(ax2, "파랑 넓이 − 빨강 넓이 = KL ≥ 0 · 평균은 $q$ 로 냄", y=-0.2)
    cp.show(fig)


def plot_reparameterization(eps, z_a, z_b, mu_a, mu_b, sigma):
    """같은 잡음 ε 을 두 평균으로 옮긴 샘플. 잡음을 고정하면 μ 를 움직일 때 샘플이 어떻게 움직이는지 미분할 수 있음."""
    fig = cp.new_figure((11, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\epsilon$", None, "고정한 잡음 $\\epsilon \\sim \\mathcal{N}(0, 1)$")
    ax.scatter(eps, np.zeros_like(eps), s=18, color=cp.MUTED)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$z$", None, f"$z = \\mu + \\sigma\\epsilon$ ($\\sigma={sigma:g}$) · 같은 $\\epsilon$, 다른 $\\mu$")
    ax2.scatter(z_a, np.ones_like(z_a), s=18, color=cp.BLUE, label=f"$\\mu = {mu_a:g}$")
    ax2.scatter(z_b, np.zeros_like(z_b), s=18, color=cp.RED, label=f"$\\mu = {mu_b:g}$")
    for a, b in zip(z_a[:10], z_b[:10]):
        ax2.annotate("", xy=(b, 0.08), xytext=(a, 0.92), arrowprops=dict(arrowstyle="-|>", color=cp.GOLD, lw=0.9))
    ax2.set_yticks([0, 1], [f"$\\mu={mu_b:g}$", f"$\\mu={mu_a:g}$"]); ax2.set_ylim(-0.6, 1.6)
    ax2.spines["left"].set_visible(False)
    cp.label(ax2, "각 점이 $\\mu$ 의 변화만큼 그대로 옮겨감 → $\\partial z/\\partial\\mu = 1$, $\\partial z/\\partial\\sigma = \\epsilon$", y=-0.2)
    cp.show(fig)


def plot_kl_vs_mean(means, curves):
    """기준 분포의 분산에 따라 같은 평균 이동이 주는 KL 비용."""
    fig = cp.new_figure((7, 3.4))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$q$ 의 평균 (분산 1)", "$KL(q\\|p)$, nat", "기준 분포 $p$ 의 분산이 작을수록 같은 이동의 비용이 큼")
    for (name, ys), color in zip(curves.items(), [cp.RED, cp.INK, cp.BLUE]):
        ax.plot(means, ys, color=color, lw=1.7, label=name)
    ax.legend(frameon=False, fontsize=9)
    cp.show(fig)
