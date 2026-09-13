"""11 선형대수(투영·공분산·SVD·PCA) 노트북의 개념 그림.

투영·공분산·고유 이미지·복원은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401


def plot_projection_2d(x, q, x_hat):
    """단위 방향 q 위에서 x 에 가장 가까운 점 x̂ = (x·q) q. 잔차는 q 에 수직."""
    x, q, x_hat = (np.asarray(v, float).ravel() for v in (x, q, x_hat))
    fig = cp.new_figure((5.4, 4.4))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$x_0$", "$x_1$", "단위 방향 $q$ 위에서 $x$ 에 가장 가까운 점")
    t = np.linspace(-0.5, 2.3, 50)
    ax.plot(t * q[0], t * q[1], color=cp.MUTED, lw=1.2)
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=cp.GOLD, lw=2.0))
    ax.text(q[0] + 0.08, q[1] - 0.12, "$q$ (길이 1)", color=cp.GOLD, fontsize=9.5)
    ax.plot([x[0], x_hat[0]], [x[1], x_hat[1]], color=cp.RED, lw=1.4, ls="--")
    ax.text((x[0] + x_hat[0]) / 2 + 0.1, (x[1] + x_hat[1]) / 2, "$r = x - \\hat x \\perp q$", color=cp.RED, fontsize=9.5)
    ax.scatter(*x, color=cp.INK, s=55, zorder=4)
    ax.text(x[0] + 0.08, x[1] + 0.08, "$x$", color=cp.INK, fontsize=10.5)
    ax.scatter(*x_hat, color=cp.BLUE, s=55, zorder=4)
    ax.text(x_hat[0] + 0.08, x_hat[1] + 0.12, "$\\hat x = z\\,q$", color=cp.BLUE, fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(-0.4, 1.9)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1])
    cp.show(fig)


def plot_projection_images(image, direction, z, x_hat, residual):
    """같은 투영을 이미지에서: x_7 을 단위 방향 q(템플릿) 에 투영한 x̂ = z q 와 잔차."""
    fig = cp.new_figure((12.5, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.8, 1, 1], wspace=0.35)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x); cp.draw_image(ax_x, image, "$x_7$ (784,)")
    ax_q = fig.add_subplot(gs[1]); cp.blank(ax_q); cp.draw_image(ax_q, direction, "$q$ = 템플릿 / 길이 (784,)", kind="signed")
    ax_z = fig.add_subplot(gs[2]); cp.blank(ax_z); cp.draw_formula(ax_z, f"$z = x\\cdot q = {z:.2f}$", "방향 위의 좌표 하나")
    vmax = float(max(np.abs(x_hat).max(), np.abs(residual).max()))
    ax_h = fig.add_subplot(gs[3]); cp.blank(ax_h); cp.draw_image(ax_h, x_hat, "$\\hat x = z\\,q$", kind="signed", vmax=vmax)
    ax_r = fig.add_subplot(gs[4]); cp.blank(ax_r); cp.draw_image(ax_r, residual, "$r = x - \\hat x$", kind="signed", vmax=vmax)
    cp.connect(fig, ax_x, ax_q, "내적"); cp.connect(fig, ax_q, ax_z); cp.connect(fig, ax_z, ax_h, "$q$ 에 곱함")
    cp.label(ax_r, "$r\\cdot q = 0$ : 잔차는 방향에 수직", y=-0.06)
    cp.show(fig)


def plot_least_squares(x, y, w, residuals):
    """점 4개와 정규방정식으로 맞춘 직선, 그리고 잔차(빨강)."""
    fig = cp.new_figure((5.6, 3.8))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$x$", "$y$", f"최소제곱 직선 $y = {w[0]:.2f} + {w[1]:.2f}x$")
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 20)
    ax.plot(grid, w[0] + w[1] * grid, color=cp.BLUE, lw=1.8, label="$Aw$ (직선)")
    for xi, yi, ri in zip(x, y, residuals):
        ax.plot([xi, xi], [yi, yi + ri], color=cp.RED, lw=1.2, ls="--")
    ax.scatter(x, y, color=cp.INK, s=45, zorder=4, label="관측 $y$")
    ax.text(x[-1] + 0.1, y[-1] + residuals[-1] / 2, "$r = Aw - y$", color=cp.RED, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(ax, "잔차는 $A$ 의 모든 열(상수 1, $x$)에 직교: $A^{\\top} r = 0$", y=-0.2, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_covariance_small(Xc, C, mean):
    """관측 3개 × 좌표 2개의 중심화 → 2×2 공분산."""
    fig = cp.new_figure((8.5, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1], wspace=0.6)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_grid(ax_x, Xc, "$X_c = X - \\mu$ : (N=3, D=2)", kind="signed", fmt="{:g}", fontsize=10, row_labels=["관측 0", "관측 1", "관측 2"], col_labels=["$x_0$", "$x_1$"])
    ax_f = fig.add_subplot(gs[1]); cp.blank(ax_f)
    cp.draw_formula(ax_f, "$C = X_c^{\\top} X_c / N$", f"$\\mu = ({mean[0]:g}, {mean[1]:g})$ · 관측 축 $n$ 을 합함")
    ax_c = fig.add_subplot(gs[2]); cp.blank(ax_c)
    cp.draw_grid(ax_c, C, "$C$ : (D, D)", kind="signed", fmt="{:.2f}", fontsize=10, row_labels=["$x_0$", "$x_1$"], col_labels=["$x_0$", "$x_1$"])
    cp.connect(fig, ax_x, ax_f); cp.connect(fig, ax_f, ax_c)
    cp.label(ax_c, "대각선 = 분산, 대각선 밖 = 같이 커지면 양수", y=-0.06)
    cp.show(fig)


def plot_covariance_images(mean_image, variance_image, C, n):
    """MNIST 5,000장의 평균 이미지, 화소별 분산(C 의 대각선), 784×784 공분산 행렬."""
    fig = cp.new_figure((11.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.4)
    ax_m = fig.add_subplot(gs[0]); cp.blank(ax_m); cp.draw_image(ax_m, mean_image, f"평균 이미지 $\\mu$ ({n:,}장)")
    ax_v = fig.add_subplot(gs[1]); cp.blank(ax_v)
    cp.draw_image(ax_v, variance_image / variance_image.max(), "화소별 분산 = $C$ 의 대각선")
    cp.label(ax_v, "가운데가 크고 가장자리는 0 (항상 흰색)", y=-0.06)
    ax_c = fig.add_subplot(gs[2]); cp.blank(ax_c)
    cp.draw_image(ax_c, C, "$C = X_c^{\\top}X_c/N$ : (784, 784)", kind="signed")
    cp.label(ax_c, "행·열 = 화소 번호 · 파랑 = 같이 진해지는 화소 쌍", y=-0.06)
    cp.connect(fig, ax_m, ax_v, "빼고 제곱 평균"); cp.connect(fig, ax_v, ax_c, "모든 쌍")
    cp.show(fig)


def plot_eigen_images(images, eigenvalues, spectrum, cumulative, marks):
    """위: 공분산의 고유 이미지 8장(분산이 큰 순서). 아래: 고유값 막대와 누적 유지 비율."""
    n = len(images)
    fig = cp.new_figure((1.45 * n + 0.5, 5.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 0.9], hspace=0.55, wspace=0.08)
    vmax = float(max(np.abs(im).max() for im in images))
    for i, (im, lam) in enumerate(zip(images, eigenvalues)):
        ax = fig.add_subplot(gs[0, i]); cp.blank(ax)
        cp.draw_image(ax, im, f"$v_{{{i + 1}}}$ · $\\lambda={lam:.2f}$", kind="signed", vmax=vmax)
    cp.label(fig.axes[0], "고유벡터 = 분산이 큰 방향, 784차원이라 이미지로 보임", y=-0.06)
    ax_s = fig.add_subplot(gs[1, :])
    _clean_axes(ax_s, "고유값 번호 $j$ (큰 순서)", "$\\lambda_j$", "고유값 = 그 방향의 분산 · 금색 선 = 앞에서부터 남긴 분산의 비율 (오른쪽 눈금)")
    ax_s.bar(np.arange(1, len(spectrum) + 1), spectrum, color=cp.BLUE, width=0.8)
    ax2 = ax_s.twinx()
    ax2.plot(np.arange(1, len(cumulative) + 1), cumulative, color=cp.GOLD, lw=1.6)
    ax2.scatter(marks, [cumulative[k - 1] for k in marks], color=cp.GOLD, s=24, zorder=4)
    ax2.text(0.98, 0.06, " · ".join(f"$k={k}$: {cumulative[k - 1]:.0%}" for k in marks), transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=9, color=cp.GOLD)
    ax2.set_ylim(0, 1.1); ax2.set_yticks([0, 0.5, 1.0], ["0", "50%", "100%"]); ax2.tick_params(colors=cp.GOLD, labelsize=8.5, length=3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(cp.GOLD)
    cp.show(fig)


def plot_eigen_directions(X, mean, directions, variances):
    """2차원 관측 구름과 공분산의 두 고유 방향(길이 = 2√분산)."""
    fig = cp.new_figure((5.6, 4.2))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "$x_0$", "$x_1$", "공분산의 고유벡터 = 구름이 퍼진 방향")
    ax.scatter(X[:, 0], X[:, 1], s=8, color=cp.MUTED, alpha=0.6)
    for k, (v, lam, color) in enumerate(zip(directions, variances, [cp.BLUE, cp.RED])):
        tip = mean + 2 * np.sqrt(lam) * v
        ax.annotate("", xy=tip, xytext=mean, arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        ax.text(*(tip + 0.15 * v), f"$v_{k + 1}$, $\\lambda_{k + 1}={lam:.2f}$", color=color, fontsize=9.5)
    ax.scatter(*mean, color=cp.GOLD, s=50, zorder=4)
    ax.set_aspect("equal")
    cp.show(fig)


def plot_pca_reconstruction(image, reconstructions, ks, distances, fractions):
    """7을 주성분 k개로 복원. 아래에 관측별 제곱 거리와 유지 분산."""
    n = len(reconstructions)
    fig = cp.new_figure((2.1 * (n + 1) + 0.5, 3.9))
    gs = fig.add_gridspec(2, n + 1, height_ratios=[1, 0.32], hspace=0.12, wspace=0.12)
    ax_x = fig.add_subplot(gs[0, 0]); cp.blank(ax_x); cp.draw_image(ax_x, image, "원본 $x_7$")
    cp.label(ax_x, "복원 $= \\mu + Z_kQ_k^{\\top}$", y=-0.06, color=cp.INK, fontsize=9.5)
    cp.label(ax_x, "막대 = 이 7 한 장의 $\\|x - \\hat x\\|^2$", y=-0.2)
    dmax = float(max(distances))
    for i, (rec, k, d, f) in enumerate(zip(reconstructions, ks, distances, fractions)):
        ax = fig.add_subplot(gs[0, 1 + i]); cp.blank(ax)
        cp.draw_image(ax, np.clip(rec, 0, 1), f"$k={k}$")
        ax_b = fig.add_subplot(gs[1, 1 + i]); cp.blank(ax_b)
        ax_b.barh([0], [d], color=cp.RED if i < n - 1 else cp.BLUE, height=0.5)
        ax_b.set_xlim(0, dmax * 1.05); ax_b.set_ylim(-1.5, 0.6)
        ax_b.text(0, -0.6, f"오차 {d:.1f} · 유지 분산 {f:.0%}", va="top", ha="left", fontsize=8.5, color=cp.INK)
    cp.show(fig)


def plot_pca_clouds(panels):
    """(제목, 관측 (N,2), 복원 (N,2)) 목록을 나란히. 관측 회색, 복원 파랑."""
    fig = cp.new_figure((5.2 * len(panels), 4.0))
    gs = fig.add_gridspec(1, len(panels), wspace=0.3)
    for k, (title, X, R) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_0$", "$x_1$", title)
        ax.scatter(X[:, 0], X[:, 1], s=9, color=cp.MUTED, alpha=0.6, label="관측")
        ax.scatter(R[:, 0], R[:, 1], s=9, color=cp.BLUE, alpha=0.8, label="복원")
        for a, b in zip(X[:12], R[:12]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cp.RED, lw=0.7, alpha=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(fig.axes[0], "빨간 선 = 복원 오차 (12개만 표시)", y=-0.18)
    cp.show(fig)
