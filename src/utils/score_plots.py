"""32 score·denoising score matching 노트북의 개념 그림.

밀도·score·책임확률·목적값·회귀 데이터·학습 결과·Langevin 경로는 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 곡선 축 정리는 gradient_plots._clean_axes, 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

REGION_FILL = "#fce8c3"   # 구간 색칠(옅은 금색)
DENSITY_FILL = 0.15       # 밀도 채움의 투명도


def _density(ax, x, pdf, scale=1.0, color=cp.MUTED):
    """밀도 곡선을 옅게 채운다. scale 은 score 축에 겹칠 때 높이를 맞추는 배율."""
    ax.fill_between(x, 0, np.asarray(pdf) * scale, color=color, alpha=DENSITY_FILL, lw=0)
    ax.plot(x, np.asarray(pdf) * scale, color=color, lw=0.8, alpha=0.7)


def _field(ax, xx, yy, u, v, color, cap=3.0, scale=28):
    """벡터장 화살표. 길이는 cap 에서 자르고 방향은 그대로 둔다."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    mag = np.hypot(u, v)
    factor = np.minimum(mag, cap) / np.maximum(mag, 1e-12)
    ax.quiver(xx, yy, u * factor, v * factor, color=color, angles="xy", scale_units="width", scale=scale,
              width=0.004, headwidth=4, alpha=0.9)


def _box(ax, xx, yy):
    """2차원 패널의 범위를 격자 전체로 고정하고 가로·세로 축척을 같게 한다 (세 패널이 같은 틀에 놓이게)."""
    ax.set_xlim(float(np.min(xx)), float(np.max(xx)))
    ax.set_ylim(float(np.min(yy)), float(np.max(yy)))
    ax.set_aspect("equal", adjustable="box")


def plot_score_overview(x, pdf, score, arrow_x, arrow_score, modes, field_xx, field_yy, field_S, dense_xx, dense_yy, dense_pdf):
    """왼쪽: 밀도 곡선 위 score 화살표(밀도가 높아지는 쪽). 가운데: score 곡선, 0을 지나는 자리가 봉우리. 오른쪽: 2차원 등고선 위 벡터장."""
    fig = cp.new_figure((14.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "밀도", "$p(x)$ 와 score 화살표: 밀도가 높아지는 쪽으로")
    ax.plot(x, pdf, color=cp.INK, lw=1.6)
    _density(ax, x, pdf)
    for xi, si in zip(arrow_x, arrow_score):
        yi = np.interp(xi, x, pdf) + 0.02
        length = 0.45 * np.sign(si) * np.sqrt(np.abs(si))       # 길이 ∝ √|s| (|s| 순서는 그대로), 부호 = 방향
        color = cp.BLUE if si > 0 else cp.RED
        ax.annotate("", xy=(xi + length, yi), xytext=(xi, yi),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8, "mutation_scale": 14})
        ax.scatter([xi], [yi], s=18, color=color, zorder=3)
    k0 = int(np.argmin(np.abs(np.asarray(arrow_x))))
    y0 = np.interp(arrow_x[k0], x, pdf) + 0.02
    ax.scatter([arrow_x[k0]], [y0], s=52, facecolor="none", edgecolor=cp.GOLD, lw=1.6, zorder=5)   # 속을 비우고 작게 두어 화살표를 가리지 않게
    ax.annotate(f"$s(0) = {arrow_score[k0]:.2f}$", xy=(arrow_x[k0], y0), xytext=(arrow_x[k0], max(pdf) * 1.12),
                ha="center", va="bottom", color=cp.GOLD, fontsize=9.5,
                arrowprops={"arrowstyle": "-", "color": cp.GOLD, "lw": 0.9, "shrinkA": 2, "shrinkB": 6})
    ax.set_ylim(0, max(pdf) * 1.35)
    cp.label(ax, "파랑 = 오른쪽(양수), 빨강 = 왼쪽(음수) · 길이 ∝ √|s| (순서는 |s| 순서와 같음)", y=-0.2)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x$", "$s(x)$", "$s(x) = \\frac{d}{dx}\\log p(x)$ : 봉우리에서 0을 지남")
    ax2.axhline(0, color=cp.MUTED, lw=0.8)
    ax2.plot(x, score, color=cp.BLUE, lw=1.8)
    for m in modes:
        ax2.axvline(m, color=cp.GOLD, lw=1.0, ls=":")
    ax2.set_ylim(-12, 12)
    cp.label(ax2, "금색 점선 = 밀도의 봉우리 · 왼쪽은 +, 오른쪽은 −", y=-0.2)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x_0$", "$x_1$", "2차원 혼합: 등고선 위 score 벡터장")
    ax3.contour(dense_xx, dense_yy, dense_pdf, levels=8, cmap="Blues", linewidths=0.8)
    _field(ax3, field_xx, field_yy, field_S[:, 0], field_S[:, 1], cp.INK)
    _box(ax3, dense_xx, dense_yy)
    cp.show(fig)


def plot_hyvarinen(x, pdf, s_true, tests, explicit, implicit, constant):
    """왼쪽: 정답 score 와 시험용 s_θ 들. 오른쪽: 두 목적값(정답을 아는 식 / 모르는 식) 막대. 차이는 s_θ 와 무관한 상수."""
    colors = [cp.BLUE, cp.RED, cp.GOLD]
    fig = cp.new_figure((12.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "score", "정답 $s(x)$ 와 시험용 $s_\\theta(x)$")
    _density(ax, x, pdf, scale=8)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    ax.plot(x, s_true, color=cp.INK, lw=2.0, label="정답 $s = (\\log p)'$")
    for (name, values), color in zip(tests.items(), colors):
        ax.plot(x, values, color=color, lw=1.4, label=name)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-8, 8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, None, "값", "두 목적값: 차이가 항상 $\\frac{1}{2}E_p[s^2]$")
    k = np.arange(len(tests))
    ax2.bar(k - 0.18, explicit, width=0.36, color=cp.MUTED, label="$\\frac{1}{2}E_p[(s_\\theta - s)^2]$ (정답 필요)")
    ax2.bar(k + 0.18, implicit, width=0.36, color=cp.BLUE, label="$E_p[\\frac{1}{2}s_\\theta^2 + s_\\theta']$ (정답 불필요)")
    for i, (e, im) in enumerate(zip(explicit, implicit)):
        ax2.annotate("", xy=(i + 0.18, e), xytext=(i + 0.18, im), arrowprops={"arrowstyle": "<->", "color": cp.GOLD, "lw": 1.3})
        ax2.text(i + 0.42, 0.5 * (e + im), f"{e - im:.3f}", color=cp.GOLD, fontsize=8.5, va="center")
    ax2.axhline(0, color=cp.MUTED, lw=0.8)
    ax2.set_xticks(k, [f"$s_{c}$" for c in "abc"[:len(tests)]])
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.set_ylim(min(0, min(implicit)) - 0.4, max(explicit) + 1.2)
    cp.label(ax2, f"금색 화살표 = 차이 = 상수 {constant:.3f}", y=-0.15, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_noisy_scores(x_clean, x_noisy, targets, sigma_demo, x, sigmas, pdfs, scores):
    """왼쪽: 깨끗한 x(금색)에 잡음을 더한 x̃ 들과 조건부 타깃 −ε/σ 화살표(σ² 배 하면 정확히 x 로 돌아감). 가운데·오른쪽: σ 별 q_σ 밀도와 score."""
    colors = [cp.INK, cp.BLUE, cp.RED, cp.GOLD][:len(sigmas)]
    fig = cp.new_figure((14, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\tilde x$", None, f"$\\tilde x = x + \\sigma\\epsilon$ ($\\sigma = {sigma_demo:g}$) 와 타깃 $-\\epsilon/\\sigma$")
    n = len(x_noisy)
    for i, (xn, t) in enumerate(zip(x_noisy, targets)):
        y = n - i
        ax.scatter([xn], [y], s=28, color=cp.BLUE, zorder=3)
        ax.annotate("", xy=(xn + t * sigma_demo ** 2, y), xytext=(xn, y), arrowprops={"arrowstyle": "-|>", "color": cp.BLUE, "lw": 1.2})
        ax.text(xn + (0.12 if t > 0 else -0.12), y + 0.22, f"$-\\epsilon/\\sigma = {t:.2f}$", fontsize=7.5, color=cp.INK, ha="left" if t > 0 else "right")
    ax.axvline(x_clean, color=cp.GOLD, lw=1.4, ls="--")
    ax.text(x_clean, n + 0.9, f"$x = {x_clean:g}$", color=cp.GOLD, ha="center", fontsize=9.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0.3, n + 1.5)
    cp.label(ax, "화살표 = 타깃 × $\\sigma^2$ = 정확히 $x$ 로 돌아가는 걸음", y=-0.2)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x$", "밀도", "잡음 낀 분포 $q_\\sigma(x) = \\sum_k \\pi_k \\mathcal{N}(\\mu_k, \\sigma_k^2 + \\sigma^2)$")
    for s, p, color in zip(sigmas, pdfs, colors):
        ax2.plot(x, p, color=color, lw=1.6, label=f"$\\sigma = {s:g}$" + (" (원래 $p$)" if s == 0 else ""))
    ax2.legend(frameon=False, fontsize=8.5)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x$", "score", "$q_\\sigma$ 의 score: 봉우리 사이가 매끈해짐")
    ax3.axhline(0, color=cp.MUTED, lw=0.8)
    for s, sc, color in zip(sigmas, scores, colors):
        ax3.plot(x, sc, color=color, lw=1.6, label=f"$\\sigma = {s:g}$")
    ax3.set_ylim(-12, 12)
    ax3.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_dsm_regression(x_noisy, targets, centers, bin_means, counts, x, score_sigma, pdf_sigma, sigma):
    """왼쪽: 회귀 데이터 (x̃, −ε/σ) 산점과 q_σ 의 score 곡선. 오른쪽: x̃ 구간별 타깃 평균(금색)이 그 곡선 위에 놓임."""
    fig = cp.new_figure((12.5, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    cap = 3.5 / sigma
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\tilde x$", "타깃 $-\\epsilon/\\sigma$", f"회귀 데이터: 시끄러운 타깃 ($\\sigma = {sigma:g}$)")
    ax.scatter(x_noisy, targets, s=4, color=cp.MUTED, alpha=0.35)
    ax.plot(x, score_sigma, color=cp.BLUE, lw=2.0, label="$q_\\sigma$ 의 score = 조건부 평균")
    ax.set_ylim(-cap, cap)
    ax.set_xlim(-6, 6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$\\tilde x$", "타깃의 구간 평균", "구간마다 타깃을 평균내면 score 곡선 위")
    _density(ax2, x, pdf_sigma, scale=cap * 0.8)
    ax2.plot(x, score_sigma, color=cp.BLUE, lw=2.0)
    shown = np.asarray(counts) >= 20
    ax2.scatter(np.asarray(centers)[shown], np.asarray(bin_means)[shown], s=34, color=cp.GOLD, zorder=4, edgecolor="white", label="구간 평균 (데이터 20개 이상)")
    ax2.set_ylim(-cap, cap)
    ax2.set_xlim(-6, 6)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    cp.label(ax2, "회색 = $q_\\sigma$ 밀도 · 데이터가 없는 곳에는 구간 평균도 없음", y=-0.2)
    cp.show(fig)


def _shade_regions(ax, regions, ylim):
    """구간 목록 {이름: [(lo, hi), …]} 을 옅은 금색으로 칠하고, 가장 넓은 구간 위에 이름을 번갈아 다른 높이로 적는다."""
    for i, (name, intervals) in enumerate(regions.items()):
        for lo, hi in intervals:
            ax.axvspan(lo, hi, color=REGION_FILL, alpha=0.35 if name == "봉우리 근처" else 0.7, lw=0)
        widths = [hi - lo for lo, hi in intervals]
        lo, hi = intervals[int(np.argmax(widths))]
        ax.text(0.5 * (lo + hi), ylim[1] * (0.95 - 0.09 * (i % 2)), name, ha="center", va="top", fontsize=8.5, color=cp.INK)


def plot_learned_score_1d(x, pdf_sigma, s_true, s_before, s_after, regions, region_errors, sigma):
    """왼쪽: 학습 전·후 s_θ 와 닫힌식 q_σ score. 가운데: |오차| 와 밀도(데이터가 없는 곳에서 큼). 오른쪽: 구간별 RMS 오차."""
    fig = cp.new_figure((14.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.8], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "score", f"학습한 $s_\\theta$ 와 닫힌식 $q_\\sigma$ score ($\\sigma = {sigma:g}$)")
    _density(ax, x, pdf_sigma, scale=10)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    ax.plot(x, s_true, color=cp.INK, lw=2.2, label="닫힌식 (정답)")
    ax.plot(x, s_before, color=cp.MUTED, lw=1.2, ls="--", label="학습 전")
    ax.plot(x, s_after, color=cp.BLUE, lw=1.6, label="학습 후")
    ax.set_ylim(-12, 12)
    ax.set_xlim(-6, 6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    err = np.abs(np.asarray(s_after) - np.asarray(s_true))
    ylim = (0, max(3.0, float(np.percentile(err, 97)) * 1.1))
    _clean_axes(ax2, "$x$", "|오차|", "오차는 데이터가 없는 곳(사이·바깥)에서 큼")
    _shade_regions(ax2, regions, ylim)
    _density(ax2, x, pdf_sigma, scale=ylim[1] * 2.0)
    ax2.plot(x, err, color=cp.RED, lw=1.6)
    ax2.set_ylim(*ylim)
    ax2.set_xlim(-6, 6)
    cp.label(ax2, "회색 = $q_\\sigma$ 밀도(높이 임의) · 금색 = 구간", y=-0.2)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, None, "RMS 오차", "구간별 오차")
    names = list(regions)
    ax3.bar(np.arange(len(names)), region_errors, color=[cp.BLUE, cp.RED, cp.RED][:len(names)], width=0.6)
    for i, v in enumerate(region_errors):
        ax3.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=cp.INK)
    ax3.set_xticks(np.arange(len(names)), names, fontsize=8.5)
    cp.show(fig)


def plot_learned_field_2d(field_xx, field_yy, true_S, learned_S, dense_xx, dense_yy, dense_pdf, err_grid, sigma):
    """왼쪽: 닫힌식 벡터장. 가운데: 학습한 벡터장. 오른쪽: 오차 크기(빨강)와 밀도 등고선 — 데이터가 없는 곳에서 큼."""
    fig = cp.new_figure((14.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    panels = [("닫힌식 $q_\\sigma$ score (정답)", true_S, cp.INK), (f"학습한 $s_\\theta$ ($\\sigma = {sigma:g}$)", learned_S, cp.BLUE)]
    for i, (title, S, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x_0$", "$x_1$", title)
        ax.contour(dense_xx, dense_yy, dense_pdf, levels=8, cmap="Blues", linewidths=0.8)
        _field(ax, field_xx, field_yy, S[:, 0], S[:, 1], color)
        _box(ax, dense_xx, dense_yy)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x_0$", "$x_1$", "$\\|s_\\theta - s\\|$ : 데이터가 없는 곳에서 큼")
    cap = float(np.percentile(err_grid, 97))                     # 색의 상한: 바깥의 큰 오차가 안쪽의 차이를 지우지 않게
    ax3.pcolormesh(dense_xx, dense_yy, np.minimum(err_grid, cap), cmap="Reds", vmin=0, vmax=cap, shading="auto")
    ax3.contour(dense_xx, dense_yy, dense_pdf, levels=6, colors=[cp.INK], linewidths=0.6, alpha=0.6)
    _box(ax3, dense_xx, dense_yy)
    cp.label(ax3, f"흰색 = 오차 0 · 진한 빨강 = {cap:.1f} 이상 · 회색 선 = $q_\\sigma$ 등고선", y=-0.2)
    cp.show(fig)


def plot_dynamics(ascent_path, langevin_path, modes, x, pdf, edges, hist_prob, density_prob, tv, alpha):
    """왼쪽: 오르막 경로(봉우리로 수렴). 가운데: Langevin 경로(봉우리 근처를 흔들리며 돌아다님). 오른쪽: 마지막 입자의 히스토그램 vs 밀도."""
    fig = cp.new_figure((14.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.32)
    steps = np.arange(len(ascent_path))
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "걸음", "$x$", f"오르막 $x \\leftarrow x + \\frac{{\\alpha}}{{2}} s(x)$ ($\\alpha = {alpha:g}$)")
    for m in modes:
        ax.axhline(m, color=cp.GOLD, lw=1.0, ls=":")
    ax.plot(steps, ascent_path, color=cp.BLUE, lw=1.2)
    ax.set_ylim(-6, 6)
    cp.label(ax, "금색 점선 = 봉우리 · 출발점이 어디든 가까운 봉우리로", y=-0.2)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "걸음", "$x$", "Langevin $+\\sqrt{\\alpha}\\,z$: 봉우리 주변을 흔들림")
    for m in modes:
        ax2.axhline(m, color=cp.GOLD, lw=1.0, ls=":")
    ax2.plot(np.arange(len(langevin_path)), langevin_path, color=cp.RED, lw=0.9, alpha=0.8)
    ax2.set_ylim(-6, 6)
    cp.label(ax2, "같은 출발점 7개 · 처음 150 걸음", y=-0.2)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x$", "구간 확률", f"Langevin 입자의 분포 vs $p$ (TV 거리 {tv:.3f})")
    centers = 0.5 * (edges[1:] + edges[:-1])
    ax3.bar(centers, hist_prob, width=edges[1] - edges[0], color=cp.MUTED, alpha=0.6, label="입자 히스토그램")
    ax3.plot(centers, density_prob, color=cp.INK, lw=1.8, label="$p(x)\\,\\Delta x$")
    ax3.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_sigma_comparison(x, sigmas, pdfs, trues, learneds, region_names, errors_by_sigma):
    """σ 마다 학습 score 와 닫힌식(왼쪽 셋), 오른쪽에 σ × 구간 오차 막대. σ 가 클수록 매끈하고 오차가 작음."""
    n = len(sigmas)
    fig = cp.new_figure((4.2 * n + 3.6, 3.7))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.0], wspace=0.35)
    for i, (s, p, t, l) in enumerate(zip(sigmas, pdfs, trues, learneds)):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x$", "score" if i == 0 else None, f"$\\sigma = {s:g}$")
        _density(ax, x, p, scale=10)
        ax.axhline(0, color=cp.MUTED, lw=0.8)
        ax.plot(x, t, color=cp.INK, lw=2.0, label="닫힌식")
        ax.plot(x, l, color=cp.BLUE, lw=1.4, label="학습")
        ax.set_ylim(-14, 14)
        ax.set_xlim(-6, 6)
        if i == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax_b = fig.add_subplot(gs[n])
    _clean_axes(ax_b, None, "RMS 오차", "구간별 오차 (σ 마다)")
    k = np.arange(len(region_names))
    width = 0.8 / n
    colors = [cp.RED, cp.GOLD, cp.BLUE]
    for i, (s, errs) in enumerate(zip(sigmas, errors_by_sigma)):
        ax_b.bar(k + (i - (n - 1) / 2) * width, errs, width=width, color=colors[i % 3], label=f"$\\sigma = {s:g}$")
    ax_b.set_xticks(k, region_names, fontsize=8.5)
    ax_b.set_yscale("log")
    ax_b.legend(frameon=False, fontsize=8.5)
    cp.show(fig)
