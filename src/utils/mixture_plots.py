"""23 잠재변수·혼합 모형·EM 노트북의 개념 그림.

성분 밀도·주변 밀도·책임확률·score·EM 반복의 파라미터·샘플·denoiser 값은 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.

색: 성분 0 = 파랑, 성분 1 = 빨강, 합(주변 밀도) = 진회색, 강조 = 금색.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

COMPONENT_COLORS = [cp.BLUE, cp.RED, "#6b7280"]   # 성분 0, 1, (완성 예제의) 2
TOTAL_COLOR = cp.INK


def _bars_with_values(ax, names, values, colors, fmt="{:.4f}"):
    """세로 막대와 그 위(또는 아래)의 값. 0 기준선을 둔다."""
    values = np.asarray(values, float)
    ax.bar(names, values, color=colors, width=0.6)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    span = float(np.abs(values).max() or 1.0)
    for i, v in enumerate(values):
        offset = 0.04 * span if v >= 0 else -0.04 * span
        ax.text(i, v + offset, fmt.format(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=cp.INK)
    ax.tick_params(axis="x", labelsize=9)
    ax.margins(y=0.18)          # 막대 위의 값 글자가 제목에 닿지 않게 위아래 여백
    return ax


def plot_marginal_sum(x, weighted, total, x0, terms0, logp0):
    """두 성분 π_k N(x; μ_k, σ_k²)과 그 합 p(x). x0에서 두 항과 합, 그리고 로그 항과 logsumexp를 막대로."""
    weighted, total, terms0 = np.asarray(weighted), np.asarray(total), np.asarray(terms0)
    fig = cp.new_figure((13.5, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.8, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "밀도", "$p(x) = \\pi_0\\,\\mathcal{N}(x;\\mu_0,\\sigma_0^2) + \\pi_1\\,\\mathcal{N}(x;\\mu_1,\\sigma_1^2)$")
    ax.plot(x, weighted[:, 0], color=COMPONENT_COLORS[0], lw=1.6, label="$\\pi_0\\,\\mathcal{N}(x;\\mu_0,\\sigma_0^2)$ (봉우리 0)")
    ax.plot(x, weighted[:, 1], color=COMPONENT_COLORS[1], lw=1.6, label="$\\pi_1\\,\\mathcal{N}(x;\\mu_1,\\sigma_1^2)$ (봉우리 1)")
    ax.plot(x, total, color=TOTAL_COLOR, lw=2.2, label="합 $p(x)$")
    ax.axvline(x0, color=cp.GOLD, lw=1.2, ls="--")
    ax.scatter([x0] * 3, [terms0[0], terms0[1], terms0.sum()], color=[COMPONENT_COLORS[0], COMPONENT_COLORS[1], TOTAL_COLOR],
               edgecolor=cp.GOLD, s=40, zorder=4)
    ax.legend(frameon=False, fontsize=8.5)
    names = ["$\\pi_0\\mathcal{N}_0$", "$\\pi_1\\mathcal{N}_1$", "합 $p$"]
    colors = [COMPONENT_COLORS[0], COMPONENT_COLORS[1], TOTAL_COLOR]
    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "밀도", f"$x = {x0:g}$ 에서 두 항을 더함")
    _bars_with_values(ax_b, names, [terms0[0], terms0[1], terms0.sum()], colors)
    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, None, "nat", "같은 것을 로그로: logsumexp")
    _bars_with_values(ax_c, names, [np.log(terms0[0]), np.log(terms0[1]), logp0], colors, fmt="{:.3f}")
    cp.label(ax_c, "작은 항은 합에서도, 로그의 합에서도 거의 보이지 않음", y=-0.22)
    cp.show(fig)


def plot_mixture2_density(xx, yy, component_grids, total_grid, mu):
    """2차원: 성분 두 개의 가중 밀도(π_k N_k)와 그 합. 금색 점이 성분 평균."""
    mu = np.asarray(mu)
    fig = cp.new_figure((13, 3.9))
    gs = fig.add_gridspec(1, 3, wspace=0.4)
    grids = list(component_grids) + [total_grid]
    titles = ["$\\pi_0\\,\\mathcal{N}(x;\\mu_0,\\Sigma_0)$", "$\\pi_1\\,\\mathcal{N}(x;\\mu_1,\\Sigma_1)$", "합 $p(x)$"]
    cmaps = ["Blues", "Reds", "Greys"]
    top = float(max(g.max() for g in grids))
    axes = []
    for i, (grid, title, cmap) in enumerate(zip(grids, titles, cmaps)):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x_0$", "$x_1$" if i == 0 else None, title)
        ax.contourf(xx, yy, grid, levels=np.linspace(0, top, 11), cmap=cmap, alpha=0.9)
        ax.contour(xx, yy, grid, levels=np.linspace(0.05 * top, top, 6), colors=[cp.MUTED], linewidths=0.5)
        ax.scatter(mu[:, 0], mu[:, 1], color=cp.GOLD, s=45, zorder=4, edgecolor=cp.INK, lw=0.6)
        ax.set_aspect("equal", adjustable="box")
        axes.append(ax)
    cp.connect(fig, axes[0], axes[1], "+")
    cp.connect(fig, axes[1], axes[2], "=")
    cp.label(axes[2], "1차원과 같은 두 식: 성분 밀도에 $\\pi_k$ 를 곱해 더함", y=-0.22)
    cp.show(fig)


def plot_responsibility(x, r, total, boundaries, x0, r0, image, log_relative, resp10, target):
    """왼쪽: x 축을 따라 r_0(x), r_1(x)과 두 경계(좁은 성분이 이기는 구간). 오른쪽: 예시 7 → 10성분 로그 항(최댓값 대비) → 책임확률 막대."""
    r, resp10, log_relative = np.asarray(r), np.asarray(resp10), np.asarray(log_relative)
    fig = cp.new_figure((14.5, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.0, 0.75, 1.05, 1.05], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "확률", "책임확률 $r_k(x) = p(z = k \\mid x)$ · 분모가 $p(x)$")
    ax.fill_between(x, 0, total / total.max() * 0.9, color=cp.MUTED, alpha=0.15, label="$p(x)$ (높이 맞춤)")
    ax.plot(x, r[:, 0], color=COMPONENT_COLORS[0], lw=1.8, label="$r_0(x)$: 봉우리 0")
    ax.plot(x, r[:, 1], color=COMPONENT_COLORS[1], lw=1.8, label="$r_1(x)$: 봉우리 1")
    boundaries = np.atleast_1d(np.asarray(boundaries, float))
    ax.axvspan(boundaries.min(), boundaries.max(), color=COMPONENT_COLORS[0], alpha=0.07)
    for edge in boundaries:
        ax.axvline(edge, color=cp.GOLD, lw=1.2, ls="--")
    ax.text(boundaries.max() + 0.3, 1.06, "경계 $r_0 = r_1$: $x = " + ", ".join(f"{edge:.2f}" for edge in boundaries) + "$",
            color=cp.GOLD, ha="left", va="center", fontsize=8.5)
    ax.scatter([x0, x0], r0, color=[COMPONENT_COLORS[0], COMPONENT_COLORS[1]], edgecolor=cp.GOLD, s=45, zorder=4)
    ax.set_ylim(-0.03, 1.18)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax_i = fig.add_subplot(gs[1])
    cp.blank(ax_i)
    cp.draw_image(ax_i, image, "예시 7")
    labels = [str(i) for i in range(len(resp10))]
    ax_l = fig.add_subplot(gs[2])
    cp.blank(ax_l)
    cp.draw_bars(ax_l, log_relative, "로그 항 − 최댓값 (nat)", labels=labels, highlight=target, signed=True, fmt="{:.0f}")
    ax_r = fig.add_subplot(gs[3])
    cp.blank(ax_r)
    cp.draw_bars(ax_r, resp10, "책임확률 $r_c(x_7)$", labels=labels, highlight=int(np.argmax(resp10)), marker=target,
                 xlim=(0, 1), color=cp.MUTED, fmt="{:.3f}")
    cp.connect(fig, ax_i, ax_l, "10개 성분")
    cp.connect(fig, ax_l, ax_r, "softmax")
    cp.label(ax_r, "금색 테두리 = 정답 7", y=-0.08)
    cp.show(fig)


def plot_score_field(x, density, arrow_x, arrow_score, x0, score0, dmu0, r0, xx, yy, logpdf_grid, qx, qy, U, V, mu2, clip=3.0):
    """왼쪽: 밀도 위의 score 화살표(봉우리 쪽). 가운데: x0에서 μ_k 기울기 막대. 오른쪽: 2차원 score 벡터장."""
    density, arrow_score, dmu0, r0 = np.asarray(density), np.asarray(arrow_score), np.asarray(dmu0), np.asarray(r0)
    fig = cp.new_figure((14.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 0.85, 1.25], wspace=0.5)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "밀도", "score $\\partial_x \\log p(x)$ 는 가까운 봉우리 쪽을 가리킴")
    ax.plot(x, density, color=TOTAL_COLOR, lw=2.0)
    ymax = float(density.max())
    base = -0.14 * ymax
    unit = 0.45 / clip
    for xi, si in zip(arrow_x, arrow_score):
        is_x0 = np.isclose(xi, x0)
        color = cp.GOLD if is_x0 else (cp.BLUE if si > 0 else cp.RED)
        length = unit * np.sign(si) * min(abs(si), clip)
        ax.annotate("", xy=(xi + length, base), xytext=(xi, base),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5 if is_x0 else 1.1, mutation_scale=10))
        ax.plot([xi], [base], marker="|", color=color, ms=7, mew=1.2)
    ax.axhline(base, color=cp.MUTED, lw=0.6, ls=":")
    ax.set_ylim(base - 0.22 * ymax, ymax * 1.12)
    ax.text(x0, base - 0.07 * ymax, f"$x = {x0:g}$: score $= {score0:.3f}$", color=cp.GOLD, ha="center", va="top", fontsize=8.5)
    cp.label(ax, f"파랑 = 오른쪽(양수), 빨강 = 왼쪽(음수) · 길이 ∝ |score| ({clip:g} 에서 자름)", y=-0.2)
    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, None, f"$x = {x0:g}$: $\\partial_{{\\mu_k}} \\log p(x) = r_k (x - \\mu_k)/\\sigma_k^2$")
    _bars_with_values(ax_b, ["$\\mu_0$", "$\\mu_1$"], dmu0, [cp.BLUE if v >= 0 else cp.RED for v in dmu0], fmt="{:.4f}")
    cp.label(ax_b, f"$r_0 = {r0[0]:.4f}$, $r_1 = {r0[1]:.4f}$ 로 가중", y=-0.2, color=cp.INK, fontsize=9)
    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "$x_0$", "$x_1$", "2차원: $\\nabla_x \\log p(x)$ 벡터장 (32번 미리보기)")
    top = float(np.max(logpdf_grid))
    ax_c.contour(xx, yy, logpdf_grid, levels=np.linspace(top - 7, top, 8), cmap="Greys", linewidths=0.7)
    norm = np.hypot(U, V)
    factor = np.minimum(norm, clip) / np.maximum(norm, 1e-12)
    ax_c.quiver(qx, qy, U * factor, V * factor, color=cp.BLUE, angles="xy", scale_units="xy", scale=6, width=0.004)
    ax_c.scatter(mu2[:, 0], mu2[:, 1], color=cp.GOLD, s=45, zorder=4, edgecolor=cp.INK, lw=0.6)
    ax_c.set_aspect("equal", adjustable="box")
    cp.show(fig)


def plot_em_progress(x, data, snapshots, loglik):
    """EM 반복별 성분 곡선(히스토그램 위)과 평균 로그우도 곡선. snapshots: [(반복, 가중 성분 (G, K), 합 (G,)), …]."""
    n = len(snapshots)
    fig = cp.new_figure((3.3 * n + 4.2, 3.4))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.2], wspace=0.35)
    ymax = float(max(np.max(total) for _, _, total in snapshots)) * 1.15
    for i, (it, weighted, total) in enumerate(snapshots):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x$", "밀도" if i == 0 else None, f"반복 {it}")
        ax.hist(data, bins=40, density=True, color=cp.MUTED, alpha=0.3)
        for k in range(np.shape(weighted)[1]):
            ax.plot(x, np.asarray(weighted)[:, k], color=COMPONENT_COLORS[k], lw=1.4)
        ax.plot(x, total, color=TOTAL_COLOR, lw=1.8)
        ax.set_ylim(0, ymax)
    ax_l = fig.add_subplot(gs[n])
    _clean_axes(ax_l, "반복", "평균 로그우도 (nat)", "반복마다 내려가지 않음")
    ax_l.plot(np.arange(len(loglik)), loglik, color=TOTAL_COLOR, lw=1.6)
    for it, _, _ in snapshots:
        ax_l.scatter([it], [loglik[it]], color=cp.GOLD, s=38, zorder=4, edgecolor=cp.INK, lw=0.6)
    cp.label(ax_l, "금색 = 왼쪽 판의 반복", y=-0.2)
    cp.show(fig)


def _ellipse(ax, mu, cov, color, n_sigma=2.0):
    """공분산 cov의 n_sigma 타원. 고유벡터 방향과 고유값 제곱근으로 축을 정한다."""
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(Ellipse(mu, 2 * n_sigma * np.sqrt(values[0]), 2 * n_sigma * np.sqrt(values[1]), angle=angle,
                         fill=False, ec=color, lw=1.6, zorder=3))


def plot_em2_progress(X, snapshots):
    """2차원 EM: 반복마다 점의 색 = 책임확률(파랑 r_0 ↔ 빨강 r_1), 2σ 타원 = 성분 공분산, 금색 = 평균."""
    n = len(snapshots)
    blue, red = np.array(to_rgb(cp.BLUE)), np.array(to_rgb(cp.RED))
    fig = cp.new_figure((4.1 * n, 3.7))
    gs = fig.add_gridspec(1, n, wspace=0.3)
    for i, (it, mu, cov, R) in enumerate(snapshots):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x_0$", "$x_1$" if i == 0 else None, f"반복 {it}")
        colors = np.asarray(R)[:, :1] * blue + np.asarray(R)[:, 1:2] * red
        ax.scatter(X[:, 0], X[:, 1], c=np.clip(colors, 0, 1), s=9, alpha=0.7)
        for k in range(len(mu)):
            _ellipse(ax, mu[k], cov[k], COMPONENT_COLORS[k])
        ax.scatter(np.asarray(mu)[:, 0], np.asarray(mu)[:, 1], color=cp.GOLD, s=45, zorder=4, edgecolor=cp.INK, lw=0.6)
        ax.set_aspect("equal", adjustable="datalim")
    cp.label(fig.axes[-1], "점의 색 = 책임확률 (파랑 $r_0$, 빨강 $r_1$) · 타원 = 2σ · 금색 = $\\mu_k$", y=-0.2)
    cp.show(fig)


def plot_ancestral_sampling(pi, z_strip, samples, z_all, x, density):
    """조상 샘플링: π 막대 → 뽑힌 z 띠 → 성분별로 색칠한 히스토그램과 1절의 p(x)."""
    pi, z_strip, samples, z_all = np.asarray(pi), np.asarray(z_strip), np.asarray(samples), np.asarray(z_all)
    fig = cp.new_figure((13, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.3, 2.0], wspace=0.55)
    ax_p = fig.add_subplot(gs[0])
    _clean_axes(ax_p, None, "확률", "1. 봉우리 고르기 $z \\sim \\pi$")
    ax_p.bar(["$z = 0$", "$z = 1$"], pi, color=COMPONENT_COLORS[:2], width=0.6)
    for i, v in enumerate(pi):
        ax_p.text(i, v + 0.02, f"{v:g}", ha="center", va="bottom", fontsize=9, color=cp.INK)
    ax_p.set_ylim(0, 1)
    ax_z = fig.add_subplot(gs[1])
    cp.blank(ax_z)
    cp.draw_strip(ax_z, 1 - 2 * z_strip, f"뽑힌 $z$ {len(z_strip)}개", kind="signed", vertical=True)
    ax_x = fig.add_subplot(gs[2])
    _clean_axes(ax_x, "$x$", "밀도", "2. 그 봉우리에서 $x \\sim \\mathcal{N}(\\mu_z, \\sigma_z^2)$ · 히스토그램 vs 1절의 $p(x)$")
    ax_x.hist([samples[z_all == 0], samples[z_all == 1]], bins=45, density=True, stacked=True,
              color=COMPONENT_COLORS[:2], alpha=0.45, label=["$z = 0$ 에서 뽑힘", "$z = 1$ 에서 뽑힘"])
    ax_x.plot(x, density, color=TOTAL_COLOR, lw=2.0, label="$p(x)$ (1절)")
    ax_x.legend(frameon=False, fontsize=8.5)
    cp.connect(fig, ax_p, ax_z, "n번")
    cp.connect(fig, ax_z, ax_x, "$\\mu_z + \\sigma_z\\epsilon$")
    cp.label(ax_z, "파랑 0 · 빨강 1", y=-0.06)
    cp.show(fig)


def plot_denoiser_curves(x, curves, sigmas, noisy_densities, x0, y0, sigma0, mean_x):
    """왼쪽: 잡음 크기 σ별 최적 복원 E[x | x̃] 곡선과 손계산 점. 오른쪽: 잡음 낀 관측의 밀도 q_σ(x̃)."""
    curves, noisy_densities = np.asarray(curves), np.asarray(noisy_densities)
    colors = [cp.BLUE, "#7c5cbf", cp.RED, cp.MUTED][: len(sigmas)]
    fig = cp.new_figure((12.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "잡음 낀 관측 $\\tilde{x}$", "$E[x \\mid \\tilde{x}]$", "최적 복원: 잡음이 클수록 전체 평균 쪽으로 수축")
    ax.plot(x, x, color=cp.MUTED, ls="--", lw=1.0, label="항등 ($\\sigma \\to 0$)")
    for k, (sigma, color) in enumerate(zip(sigmas, colors)):
        ax.plot(x, curves[:, k], color=color, lw=1.7, label=f"$\\sigma = {sigma:g}$")
    ax.axhline(mean_x, color=cp.GOLD, lw=1.2, ls=":", label=f"전체 평균 $E[x] = {mean_x:g}$ ($\\sigma \\to \\infty$)")
    ax.scatter([x0], [y0], color=cp.GOLD, s=60, zorder=5, edgecolor=cp.INK, lw=0.7)
    ax.text(x0 + 0.5, y0 - 2.2, f"손계산: $\\tilde{{x}} = {x0:g}$, $\\sigma = {sigma0:g}$ → {y0:.4f}", fontsize=8.5, color=cp.INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$\\tilde{x}$", "밀도", "잡음 낀 관측의 밀도: 성분 분산이 $\\sigma_k^2 + \\sigma^2$ 로 커짐")
    for k, (sigma, color) in enumerate(zip(sigmas, colors)):
        ax_b.plot(x, noisy_densities[:, k], color=color, lw=1.6, label=f"$\\sigma = {sigma:g}$")
    ax_b.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_em_variants(x, data, fits, logliks):
    """완성 예제: 성분 3개 혼합에서 시작값 두 가지의 EM 결과와 평균 로그우도 곡선. fits: [(이름, 가중 성분 (G, 3), 합 (G,)), …]."""
    fig = cp.new_figure((13.5, 3.5))
    gs = fig.add_gridspec(1, len(fits) + 1, width_ratios=[1] * len(fits) + [1.15], wspace=0.35)
    ymax = float(max(np.max(total) for _, _, total in fits)) * 1.15
    for i, (name, weighted, total) in enumerate(fits):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x$", "밀도" if i == 0 else None, name)
        ax.hist(data, bins=45, density=True, color=cp.MUTED, alpha=0.3)
        for k in range(np.shape(weighted)[1]):
            ax.plot(x, np.asarray(weighted)[:, k], color=COMPONENT_COLORS[k], lw=1.4)
        ax.plot(x, total, color=TOTAL_COLOR, lw=1.8)
        ax.set_ylim(0, ymax)
    ax_l = fig.add_subplot(gs[len(fits)])
    _clean_axes(ax_l, "반복", "평균 로그우도 (nat)", "시작값이 다르면 다른 국소해")
    for (name, values), color in zip(logliks.items(), [cp.BLUE, cp.RED]):
        ax_l.plot(np.arange(len(values)), values, color=color, lw=1.6, label=name)
    tail = [value for values in logliks.values() for value in values[3:]]     # 시작값이 나쁜 쪽의 첫 몇 반복은 눈금 밖으로
    low, high = min(tail), max(tail)
    ax_l.set_ylim(low - 0.12 * (high - low), high + 0.12 * (high - low))
    ax_l.legend(frameon=False, fontsize=8.5, loc="center right")
    cp.show(fig)
