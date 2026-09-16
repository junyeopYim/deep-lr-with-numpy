"""33 NCSN(잡음 수준을 잇는 score 모델) 노트북의 개념 그림.

σ 사다리·q_σ 밀도와 score·Langevin 경로·손실 크기·annealed 스냅숏·MNIST 좌표 복원은 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 곡선 축 정리는 gradient_plots._clean_axes,
원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

LADDER_COLORS = [cp.INK, cp.BLUE, cp.RED, cp.GOLD]   # σ 수준을 구분하는 색 (큰 σ → 작은 σ)
DENSITY_FILL = 0.15                                   # 밀도 채움의 투명도


def _density(ax, x, pdf, scale=1.0, color=cp.MUTED):
    """밀도 곡선을 옅게 채운다. scale 은 score 축에 겹칠 때 높이를 맞추는 배율."""
    ax.fill_between(x, 0, np.asarray(pdf) * scale, color=color, alpha=DENSITY_FILL, lw=0)
    ax.plot(x, np.asarray(pdf) * scale, color=color, lw=0.8, alpha=0.7)


def _box(ax, xx, yy):
    """2차원 패널의 범위를 격자 전체로 고정하고 가로·세로 축척을 같게 한다."""
    ax.set_xlim(float(np.min(xx)), float(np.max(xx)))
    ax.set_ylim(float(np.min(yy)), float(np.max(yy)))
    ax.set_aspect("equal", adjustable="box")


def _colors(n):
    """수준 수만큼 색을 돌려 쓴다."""
    return [LADDER_COLORS[i % len(LADDER_COLORS)] for i in range(n)]


def plot_sigma_ladder(x, shown_sigmas, pdfs, scores, ladder, tv_values, empty_fractions):
    """왼쪽 둘: σ 별 q_σ 밀도와 score(σ 가 크면 골이 메워지고 완만해짐).
    셋째: 기하 수열 사다리(로그 축에서 등간격). 넷째: σ 별 '원래 분포와의 거리'와 '데이터가 닿지 않는 구간의 비율'의 맞바꿈."""
    colors = _colors(len(shown_sigmas))
    fig = cp.new_figure((16.5, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 0.9, 1.1], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x$", "밀도", "$q_\\sigma$ : $\\sigma$ 가 크면 봉우리가 뭉개짐")
    for sigma, pdf, color in zip(shown_sigmas, pdfs, colors):
        style = "--" if sigma == 0 else "-"
        ax.plot(x, pdf, color=color, lw=1.7, ls=style, label=f"$\\sigma = {sigma:.3g}$" + (" (원래 $p$)" if sigma == 0 else ""))
    ax.set_xlim(-6, 6)
    ax.legend(frameon=False, fontsize=8.5)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x$", "score", "$\\nabla\\log q_\\sigma$ : $\\sigma$ 가 작으면 가파름")
    ax2.axhline(0, color=cp.MUTED, lw=0.8)
    for sigma, score, color in zip(shown_sigmas, scores, colors):
        style = "--" if sigma == 0 else "-"
        ax2.plot(x, score, color=color, lw=1.7, ls=style, label=f"$\\sigma = {sigma:.3g}$")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-14, 14)
    ax2.legend(frameon=False, fontsize=8.5)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "수준 번호 $i$", "$\\sigma_i$ (로그 축)", "$\\sigma$ 사다리: 로그 축에서 등간격")
    steps = np.arange(1, len(ladder) + 1)
    ax3.plot(steps, ladder, color=cp.MUTED, lw=1.0, zorder=1)
    ax3.scatter(steps, ladder, s=34, color=cp.BLUE, zorder=3)
    ax3.scatter([1, len(ladder)], [ladder[0], ladder[-1]], s=80, facecolor="white", edgecolor=cp.GOLD, lw=1.8, zorder=4)
    ax3.set_yscale("log")
    ratio = ladder[1] / ladder[0]
    cp.label(ax3, f"비 $\\sigma_{{i+1}}/\\sigma_i = {ratio:.3f}$ (일정)", y=-0.2)
    ax4 = fig.add_subplot(gs[3])
    _clean_axes(ax4, "$\\sigma$ (로그 축, 큰 것부터)", None, "맞바꿈: 원래 분포와의 거리 vs 배울 수 있음")
    ax4.plot(ladder, tv_values, color=cp.RED, lw=1.8, marker="o", ms=3.5, label="$q_\\sigma$ 와 $p$ 의 TV 거리")
    ax4.plot(ladder, empty_fractions, color=cp.BLUE, lw=1.8, marker="s", ms=3.5, label="데이터가 닿지 않는 구간의 비율")
    ax4.set_xscale("log")
    ax4.invert_xaxis()
    ax4.set_ylim(-0.03, 0.62)
    ax4.legend(frameon=False, fontsize=8.5, loc="upper center")
    cp.label(ax4, "한 $\\sigma$ 로는 두 선을 함께 낮출 수 없음", y=-0.2)
    cp.show(fig)


def plot_langevin_walk(paths, X_final, dense_xx, dense_yy, dense_pdf, alphas, centers, hists, density_prob, tvs):
    """왼쪽: 2차원 Langevin 경로 몇 개(등고선 위). 가운데: 오래 돌린 입자들의 산점. 오른쪽: α 별 1차원 히스토그램과 $p$."""
    fig = cp.new_figure((14.5, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.34)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x_0$", "$x_1$", "Langevin 경로: 봉우리를 오가며 흔들림")
    for i in range(paths.shape[1]):
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=cp.RED, lw=0.6, alpha=0.55)
    ax.contour(dense_xx, dense_yy, dense_pdf, levels=6, colors=[cp.INK], linewidths=0.8, zorder=5)
    ax.scatter(paths[0, :, 0], paths[0, :, 1], s=30, color=cp.GOLD, zorder=6, edgecolor="white", lw=0.8)
    _box(ax, dense_xx, dense_yy)
    cp.label(ax, "금색 = 출발점 · 빨강 = 처음 150 걸음의 자취", y=-0.18)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x_0$", "$x_1$", "오래 돌린 뒤의 입자들 vs 등고선")
    ax2.scatter(X_final[:, 0], X_final[:, 1], s=4, color=cp.BLUE, alpha=0.25)
    ax2.contour(dense_xx, dense_yy, dense_pdf, levels=7, colors=[cp.INK], linewidths=0.7, alpha=0.7)
    _box(ax2, dense_xx, dense_yy)
    cp.label(ax2, "입자 구름이 등고선을 채움", y=-0.18)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x$", "구간 확률", "걸음 크기 $\\alpha$ 의 맞바꿈 (1차원 혼합)")
    ax3.bar(centers, density_prob, width=centers[1] - centers[0], color=cp.MUTED, alpha=0.45, label="$p(x)\\,\\Delta x$")
    colors = _colors(len(alphas))
    for alpha, hist, tv, color in zip(alphas, hists, tvs, colors):
        ax3.plot(centers, hist, color=color, lw=1.5, label=f"$\\alpha = {alpha:g}$ (TV {tv:.3f})")
    ax3.set_ylim(0, float(np.max(hists)) * 1.45)
    ax3.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    cp.label(ax3, "작으면 아직 못 퍼지고, 크면 치우침이 남음", y=-0.18)
    cp.show(fig)


def plot_loss_weighting(ladder, score_sizes, unweighted, weighted):
    """왼쪽: 타깃 $-\\epsilon/\\sigma$ 의 크기는 정확히 $1/\\sigma$ 이고 최적 score 의 크기는 그 아래. 가운데: 가중 전 손실(로그 축). 오른쪽: $\\sigma^2$ 을 곱한 뒤."""
    fig = cp.new_figure((14.0, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\sigma$ (로그 축)", "크기", "타깃의 크기는 정확히 $1/\\sigma$")
    ax.plot(ladder, 1 / np.asarray(ladder), color=cp.GOLD, lw=1.8, label="타깃 $-\\epsilon/\\sigma$ 의 표준편차 $1/\\sigma$")
    ax.plot(ladder, score_sizes, color=cp.BLUE, lw=1.8, marker="o", ms=3.5, label="최적 score 의 크기 $\\sqrt{E[s^2]}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    cp.label(ax, "score 는 타깃의 조건부 평균이라 항상 금색 선 아래", y=-0.28)
    index = np.arange(len(ladder))
    names = [f"{s:.2f}" for s in ladder]
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$\\sigma$", "손실", "가중 전 $E\\|s + \\epsilon/\\sigma\\|^2$ (로그 축)")
    ax2.bar(index, unweighted, color=cp.RED, width=0.68)
    ax2.set_yscale("log")
    ax2.set_xticks(index, names, fontsize=7.5, rotation=60)
    cp.label(ax2, "작은 $\\sigma$ 의 항이 $1/\\sigma^2$ 로 커져 학습을 독차지", y=-0.28)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$\\sigma$", "손실", "$\\sigma^2$ 을 곱한 뒤 $E\\|\\sigma s + \\epsilon\\|^2$")
    ax3.bar(index, weighted, color=cp.BLUE, width=0.68)
    ax3.axhline(1.0, color=cp.GOLD, lw=1.2, ls=":")
    ax3.set_ylim(0, max(1.3, float(np.max(weighted)) * 1.25))
    ax3.set_xticks(index, names, fontsize=7.5, rotation=60)
    cp.label(ax3, "모든 수준이 0.3 과 1 사이로 모임 (금색 점선 = 1)", y=-0.28)
    cp.show(fig)


def plot_learned_ladder(x, shown_sigmas, pdfs, trues, learneds, ladder, rel_errors):
    """σ 세 값에서 학습한 $s_\\theta(\\cdot,\\sigma)$ 와 닫힌식을 겹쳐 보고, 오른쪽에 σ 별 상대오차 막대."""
    n = len(shown_sigmas)
    fig = cp.new_figure((3.9 * n + 4.0, 3.7))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.15], wspace=0.34)
    for i, (sigma, pdf, true, learned) in enumerate(zip(shown_sigmas, pdfs, trues, learneds)):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x$", "score" if i == 0 else None, f"$\\sigma = {sigma:.2f}$")
        _density(ax, x, pdf, scale=float(np.abs(true).max()) * 1.6 + 1e-9)
        ax.axhline(0, color=cp.MUTED, lw=0.8)
        ax.plot(x, true, color=cp.INK, lw=2.0, label="닫힌식")
        ax.plot(x, learned, color=cp.BLUE, lw=1.4, label="학습한 $s_\\theta$")
        ax.set_xlim(-6, 6)
        if i == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax_b = fig.add_subplot(gs[n])
    _clean_axes(ax_b, "$\\sigma$", "상대 오차", "수준마다 잰 상대 오차")
    index = np.arange(len(ladder))
    ax_b.bar(index, rel_errors, color=cp.BLUE, width=0.68)
    ax_b.set_xticks(index, [f"{s:.2f}" for s in ladder], fontsize=7.5, rotation=60)
    cp.label(ax_b, "하나의 모델이 모든 수준을 함께 맞춤", y=-0.28)
    cp.show(fig)


def plot_annealed_snapshots(snapshots, snapshot_sigmas, dense_xx, dense_yy, dense_pdf, mode_fracs, ladder):
    """앞 네 패널: 큰 σ 에서 작은 σ 로 내려가며 구름이 두 봉우리로 갈라지는 스냅숏. 마지막: 수준마다 잰 오른쪽 봉우리 비율."""
    n = len(snapshots)
    fig = cp.new_figure((3.5 * n + 4.2, 3.9))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.25], wspace=0.3)
    for i, (X, sigma) in enumerate(zip(snapshots, snapshot_sigmas)):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x_0$", "$x_1$" if i == 0 else None, f"$\\sigma_i = {sigma:.2f}$ 를 마친 뒤")
        ax.scatter(X[:, 0], X[:, 1], s=4, color=cp.BLUE, alpha=0.25)
        ax.contour(dense_xx, dense_yy, dense_pdf, levels=6, colors=[cp.INK], linewidths=0.7, alpha=0.6)
        _box(ax, dense_xx, dense_yy)
    ax_b = fig.add_subplot(gs[n])
    _clean_axes(ax_b, "$\\sigma$ (로그 축, 큰 것부터)", "비율", "수준마다 잰 오른쪽 봉우리 비율")
    ax_b.axhline(mode_fracs[-1], color=cp.MUTED, lw=0.8)
    ax_b.plot(ladder, mode_fracs, color=cp.BLUE, lw=1.8, marker="o", ms=4)
    ax_b.axhline(0.6, color=cp.GOLD, lw=1.4, ls="--")
    ax_b.text(ladder[0], 0.62, " 닫힌식 $\\pi_2 = 0.6$", color=cp.GOLD, fontsize=9, va="bottom", ha="left")
    ax_b.set_xscale("log")
    ax_b.invert_xaxis()
    ax_b.set_ylim(0, 1)
    cp.label(ax_b, "큰 $\\sigma$ 에서 넘나들며 비율이 정해지고, 작은 $\\sigma$ 에서 굳음", y=-0.2)
    cp.show(fig)


def plot_coverage_compare(panels, dense_xx, dense_yy, dense_pdf, names, fracs):
    """왼쪽 패널들: 같은 출발점에서 annealed Langevin 과 단일 σ Langevin 의 최종 입자. 오른쪽: 방법마다 봉우리 비율 막대."""
    n = len(panels)
    fig = cp.new_figure((3.6 * n + 4.4, 4.0))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [1.1], wspace=0.3)
    for i, (title, X) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$x_0$", "$x_1$" if i == 0 else None, title)
        ax.scatter(X[:, 0], X[:, 1], s=4, color=cp.BLUE if "annealed" in title else cp.RED, alpha=0.25)
        ax.contour(dense_xx, dense_yy, dense_pdf, levels=6, colors=[cp.INK], linewidths=0.7, alpha=0.6)
        _box(ax, dense_xx, dense_yy)
    ax_b = fig.add_subplot(gs[n])
    _clean_axes(ax_b, None, "봉우리 비율", f"{len(names)}가지 방법의 봉우리 비율")
    index = np.arange(len(names))
    ax_b.bar(index - 0.18, [f[0] for f in fracs], width=0.36, color=cp.MUTED, label="왼쪽 봉우리")
    ax_b.bar(index + 0.18, [f[1] for f in fracs], width=0.36, color=cp.BLUE, label="오른쪽 봉우리")
    ax_b.axhline(0.4, color=cp.GOLD, lw=1.2, ls=":")
    ax_b.axhline(0.6, color=cp.GOLD, lw=1.2, ls="--")
    ax_b.set_xticks(index, names, fontsize=8)
    ax_b.set_ylim(0, 1.3)
    ax_b.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    cp.label(ax_b, "금색 선 = 닫힌식 비율 $\\pi = (0.4, 0.6)$", y=-0.15)
    cp.show(fig)


def plot_pca_ladder(original, reconstructed, noisy_images, sigmas, coordinate_pairs, labels):
    """왼쪽: 고정 예시 7 의 원본·PCA-32 복원과 σ 사다리별 잡음 좌표를 되돌린 이미지. 오른쪽: 앞 두 좌표의 산점(숫자별 색)."""
    n = len(noisy_images)
    fig = cp.new_figure((1.75 * (n + 2) + 4.6, 3.9))
    gs = fig.add_gridspec(1, n + 5, width_ratios=[1] * (n + 2) + [0.12, 2.5, 0.12], wspace=0.14)
    titles = ["원본 7", "PCA-32 복원"] + [f"$\\sigma = {sigma:.2f}$" for sigma in sigmas]
    for i, image in enumerate([original, reconstructed, *noisy_images]):
        ax = cp.blank(fig.add_subplot(gs[i]))
        cp.draw_image(ax, np.clip(image, 0, 1), titles[i])
    cp.blank(fig.add_subplot(gs[n + 2]))
    cp.blank(fig.add_subplot(gs[n + 4]))
    ax = fig.add_subplot(gs[n + 3])
    _clean_axes(ax, "첫 번째 좌표", "두 번째 좌표", "5,000장의 PCA-32 좌표 (앞 두 축)")
    scatter = ax.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], c=labels, cmap="tab10", s=4, alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    handles = scatter.legend_elements(prop="colors", num=None)[0]
    ax.legend(handles[:10], [str(digit) for digit in range(10)], frameon=False, fontsize=7, ncol=2,
              loc="lower left", title="숫자", title_fontsize=7)
    cp.label(ax, "좌표 32개가 784 화소를 대신하고, 되돌리면 숫자가 보임", y=-0.17)
    cp.show(fig)


def plot_sample_grid(images, names, counts, title):
    """왼쪽: 만든 샘플 좌표를 되돌린 이미지 격자(제목은 최근접 클래스 평균이 고른 숫자). 오른쪽: 숫자별 개수 막대."""
    columns = 8
    rows = int(np.ceil(len(images) / columns))
    fig = cp.new_figure((1.35 * columns + 4.6, 1.5 * rows + 1.2))
    gs = fig.add_gridspec(rows, columns + 3, width_ratios=[1] * columns + [0.25, 2.0, 0.25], wspace=0.12, hspace=0.32)
    for i in range(rows * columns):
        ax = fig.add_subplot(gs[i // columns, i % columns])
        cp.blank(ax)
        if i < len(images):
            cp.draw_image(ax, np.clip(images[i], 0, 1), str(names[i]))
    ax_b = fig.add_subplot(gs[:, columns + 1])
    _clean_axes(ax_b, "숫자", "샘플 수", "만든 샘플의 숫자별 개수")
    ax_b.bar(np.arange(10), counts, color=cp.BLUE, width=0.7)
    ax_b.set_xticks(np.arange(10), [str(d) for d in range(10)], fontsize=8.5)
    cp.label(ax_b, "클래스 평균 최근접으로 센 것 (분류기가 아님)", y=-0.16)
    fig.suptitle(title, color=cp.INK, fontsize=11)
    cp.show(fig)


def plot_variant_bars(names, mode_fracs, distances, errors):
    """완성 예제: 사다리 길이 L 과 걸음 크기 ε 을 바꾼 변형들의 봉우리 비율·평균 거리·공분산 오차 막대."""
    fig = cp.new_figure((13.5, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    index = np.arange(len(names))
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, None, "오른쪽 봉우리 비율", "봉우리 비율 (닫힌식 0.6)")
    ax.bar(index, mode_fracs, color=cp.BLUE, width=0.6)
    ax.axhline(0.6, color=cp.GOLD, lw=1.4, ls="--")
    ax.set_xticks(index, names, fontsize=8)
    ax.set_ylim(0, 1)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, None, "거리", "샘플 평균과 닫힌식 평균의 거리")
    ax2.bar(index, distances, color=cp.RED, width=0.6)
    ax2.set_xticks(index, names, fontsize=8)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, None, "최대 오차", "공분산의 최대 오차")
    ax3.bar(index, errors, color=cp.MUTED, width=0.6)
    ax3.set_xticks(index, names, fontsize=8)
    cp.show(fig)
