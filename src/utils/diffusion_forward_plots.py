"""34 diffusion 전방 과정 노트북의 개념 그림.

스케줄·잡음 낀 이미지·posterior 계수·최적 ε·학습 결과·t별 손실은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 곡선 축 정리는 gradient_plots._clean_axes, 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 bridge_plots의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

DENSITY_FILL = 0.15       # 밀도 채움의 투명도
PARAM_FILL = "#fce8c3"    # 파라미터가 있는 블록(옅은 금색)


def _density(ax, x, pdf, scale=1.0, color=cp.MUTED):
    """밀도 곡선을 옅게 채운다. scale 은 다른 축에 겹칠 때 높이를 맞추는 배율."""
    ax.fill_between(x, 0, np.asarray(pdf) * scale, color=color, alpha=DENSITY_FILL, lw=0)
    ax.plot(x, np.asarray(pdf) * scale, color=color, lw=0.8, alpha=0.7)


# ---------------------------------------------------------------- 1절
def plot_forward_band(images, ts, alpha_bar):
    """왼쪽: 고정 예시 7 이 t 마다 흐려지는 띠(같은 ε 하나로 닫힌식). 오른쪽: ᾱ_t 와 1 − ᾱ_t 곡선, 띠의 t 를 금색으로 표시."""
    n = len(images)
    fig = cp.new_figure((2.0 * n + 5.2, 3.0))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [2.4], wspace=0.25)
    axes = []
    for k, (img, t) in enumerate(zip(images, ts)):
        ax = fig.add_subplot(gs[k])
        cp.blank(ax)
        cp.draw_image(ax, img, f"$t = {t}$\n$\\bar\\alpha_t = {alpha_bar[t]:.2f}$")
        axes.append(ax)
    for a, b in zip(axes, axes[1:]):
        cp.connect(fig, a, b)
    cp.label(axes[len(axes) // 2], "$x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$ · 화소를 [−1, 1]로 옮긴 뒤 다섯 시각에 같은 ε 하나", y=-0.06, color=cp.INK, fontsize=9)
    ax = fig.add_subplot(gs[n])
    _clean_axes(ax, "$t$", None, "스케줄: $\\bar\\alpha_t$ (신호의 몫)과 $1 - \\bar\\alpha_t$ (잡음의 몫)")
    t_axis = np.arange(len(alpha_bar))
    ax.plot(t_axis, alpha_bar, color=cp.BLUE, lw=1.8, label="$\\bar\\alpha_t$")
    ax.plot(t_axis, 1 - alpha_bar, color=cp.RED, lw=1.8, label="$1 - \\bar\\alpha_t$")
    for t in ts:
        ax.scatter([t], [alpha_bar[t]], s=34, facecolor="white", edgecolor=cp.GOLD, lw=1.6, zorder=4)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    cp.label(ax, "금색 = 위 띠의 다섯 시각", y=-0.2)
    cp.show(fig)


def plot_mixture_forward(x, ts, alpha_bars, pdfs, centers, hists):
    """t 마다 1차원 두 봉우리 혼합의 닫힌식 q(x_t)(파랑)와 q_sample 로 만든 표본 히스토그램(금색 계단)."""
    fig = cp.new_figure((4.6 * len(ts), 3.3))
    gs = fig.add_gridspec(1, len(ts), wspace=0.26)
    for k, (t, ab, pdf, hist) in enumerate(zip(ts, alpha_bars, pdfs, hists)):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_t$", "밀도" if k == 0 else None, f"$t = {t}$, $\\bar\\alpha_t = {ab:.2f}$")
        ax.fill_between(x, 0, pdf, color=cp.BLUE, alpha=DENSITY_FILL, lw=0)
        ax.plot(x, pdf, color=cp.BLUE, lw=1.8, label="닫힌식 $q(x_t)$")
        ax.step(centers, hist, where="mid", color=cp.GOLD, lw=1.2, label="표본 10,000개")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 0.42)
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(fig.axes[len(ts) // 2], "성분 평균은 $\\sqrt{\\bar\\alpha_t}\\,\\mu_k$, 분산은 $\\bar\\alpha_t\\sigma_k^2 + 1 - \\bar\\alpha_t$ · 시각이 늦어질수록 두 봉우리가 하나로 뭉개짐", y=-0.19)
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_posterior_coefficients(ts, coef_x0, coef_xt, ratio, example):
    """왼쪽: μ̃_t 의 두 계수. 가운데: 되묻기가 한 걸음보다 확실한 정도 β̃_t/β_t. 오른쪽: 한 시각의 예 — x_0(금색)·x_t(검정)와 posterior 종(파랑), 평균 μ̃_t.

    example = {"t", "x0", "xt", "grid", "pdf", "mean", "std"} (노트북 본문의 격자 Bayes 계산)."""
    fig = cp.new_figure((14, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.2], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$t$", "계수", "$\\tilde\\mu_t = c_0(t)\\,x_0 + c_t(t)\\,x_t$ 의 두 계수")
    ax.plot(ts, coef_x0, color=cp.BLUE, lw=1.8, label="$c_0(t) = \\sqrt{\\bar\\alpha_{t-1}}\\beta_t/(1-\\bar\\alpha_t)$")
    ax.plot(ts, coef_xt, color=cp.INK, lw=1.8, label="$c_t(t) = \\sqrt{\\alpha_t}(1-\\bar\\alpha_{t-1})/(1-\\bar\\alpha_t)$")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    cp.label(ax, "$t = 1$ 에서는 $x_0$ 만, $t$ 가 크면 $x_t$ 를 거의 그대로", y=-0.2)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$t$", "비율", "되묻기가 한 걸음보다 확실한 정도 $\\tilde\\beta_t/\\beta_t$")
    ax2.axhline(1, color=cp.GOLD, lw=1.2, ls="--")
    ax2.text(ts[-1], 1.02, "$\\tilde\\beta_t = \\beta_t$ 인 경계", color=cp.GOLD, ha="right", va="bottom", fontsize=8.5)
    ax2.plot(ts, ratio, color=cp.BLUE, lw=1.8)
    ax2.set_ylim(-0.03, 1.12)
    cp.label(ax2, "비율이 1 아래 = $x_0$ 를 알면 되묻기가 더 확실함 · $t$ 가 작을수록 훨씬 확실함", y=-0.2)
    ax3 = fig.add_subplot(gs[2])
    t, x0, xt = example["t"], example["x0"], example["xt"]
    _clean_axes(ax3, "$x_{t-1}$", "밀도", f"$t = {t}$ 에서 $q(x_{{t-1}} \\mid x_t, x_0)$")
    grid, pdf = example["grid"], example["pdf"]
    ax3.fill_between(grid, 0, pdf, color=cp.BLUE, alpha=DENSITY_FILL, lw=0)
    ax3.plot(grid, pdf, color=cp.BLUE, lw=1.6)
    top = float(np.max(pdf))
    ax3.axvline(x0, color=cp.GOLD, lw=1.6, ls="--")
    ax3.text(x0, top * 1.06, f"$x_0 = {x0:g}$", color=cp.GOLD, ha="center", fontsize=9)
    ax3.axvline(xt, color=cp.INK, lw=1.4, ls=":")
    ax3.text(xt, top * 1.20, f"$x_t = {xt:g}$", color=cp.INK, ha="center", fontsize=9)
    ax3.scatter([example["mean"]], [top], s=40, color=cp.BLUE, zorder=4)
    ax3.text(example["mean"] + 0.05, top * 0.78, f"$\\tilde\\mu_t = {example['mean']:.3f}$\n$\\sqrt{{\\tilde\\beta_t}} = {example['std']:.3f}$",
             color=cp.BLUE, ha="left", va="center", fontsize=8.5)
    ax3.set_ylim(0, top * 1.45)
    ax3.set_xlim(min(x0, xt) - 0.6, max(x0, xt) + 0.6)
    ax3.set_yticks([])
    cp.label(ax3, "파랑 = Bayes 규칙으로 격자에서 직접 구한 posterior", y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_targets(x, t_list, alpha_bars, pdfs, x0_hats, eps_stars, naives):
    """t 세 값에서 잡음 낀 분포 q_t(회색 채움) 위에 최적 x₀ 예측 E[x₀|x_t](파랑)와 최적 ε 예측 ε*(빨강), 잡음 무시선 x_t/√ᾱ_t(회색 점선).

    naives: 본문에서 계산한 x_t/√ᾱ_t 곡선들 (t_list 와 같은 길이)."""
    fig = cp.new_figure((14, 3.7))
    gs = fig.add_gridspec(1, len(t_list), wspace=0.28)
    for k, (t, ab, pdf, x0_hat, eps_star, naive) in enumerate(zip(t_list, alpha_bars, pdfs, x0_hats, eps_stars, naives)):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_t$", "예측" if k == 0 else None, f"$t = {t}$, $\\bar\\alpha_t = {ab:.2f}$")
        _density(ax, x, pdf, scale=4.0 / max(float(np.max(pdf)), 1e-9))
        ax.axhline(0, color=cp.MUTED, lw=0.8)
        ax.plot(x, x0_hat, color=cp.BLUE, lw=1.8, label="$\\hat x_0 = E[x_0 \\mid x_t]$")
        ax.plot(x, eps_star, color=cp.RED, lw=1.8, label="$\\epsilon^* = E[\\epsilon \\mid x_t]$")
        ax.plot(x, naive, color=cp.MUTED, lw=0.9, ls="--", label="$x_t/\\sqrt{\\bar\\alpha_t}$ (잡음 무시)")
        ax.set_ylim(-5, 5)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(fig.axes[1], "회색 = $q_t(x_t)$ 의 밀도 · 두 곡선은 $\\epsilon^* = (x_t - \\sqrt{\\bar\\alpha_t}\\hat x_0)/\\sqrt{1-\\bar\\alpha_t}$ 로 서로 변환됨", y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 4절
def plot_denoiser_blocks(embedding_table, blocks, param_blocks=(3, 4)):
    """왼쪽: sinusoidal 시간 표(행 = t, 열 = 좌표; 파랑 +, 빨강 −). 오른쪽: 잡음 예측 신경망의 블록 흐름."""
    fig = cp.new_figure((14, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.2], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    cp.blank(ax)
    table = np.asarray(embedding_table, float)
    ax.imshow(table.T, cmap=cp.SIGNED_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_title(f"시간 표 $\\mathrm{{emb}}(t)$: 열 = $t$ (0…{len(table) - 1}), 행 = 좌표 {table.shape[1]}개", color=cp.INK, fontsize=10)
    cp.label(ax, "위 행은 빠르게, 아래 행은 느리게 진동 (09번 4절)", y=-0.08)
    ax2 = fig.add_subplot(gs[1])
    colors = [PARAM_FILL if i in param_blocks else sp.FILL for i in range(len(blocks))]
    sp.draw_blocks(ax2, blocks, colors=colors, width=1.75, gap=0.5, fontsize=9.5)
    cp.label(ax2, "금색 상자 = 학습하는 파라미터가 있는 층 · 상자 아래 = 텐서 모양", y=0.0)
    cp.show(fig)


def plot_learned_eps(x, t_list, pdfs, eps_stars, eps_before, eps_after, t_grid, err_before, err_after):
    """왼쪽 셋: t 별 최적 ε*(검정)와 학습 전(회색 점선)·후(파랑) ε̂. 오른쪽: t 축의 오차 E‖ε̂ − ε*‖² 학습 전·후."""
    fig = cp.new_figure((16, 3.7))
    gs = fig.add_gridspec(1, len(t_list) + 1, width_ratios=[1] * len(t_list) + [1.15], wspace=0.3)
    for k, (t, pdf, star, before, after) in enumerate(zip(t_list, pdfs, eps_stars, eps_before, eps_after)):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_t$", "$\\hat\\epsilon$" if k == 0 else None, f"$t = {t}$")
        _density(ax, x, pdf, scale=3.0 / max(float(np.max(pdf)), 1e-9))
        ax.axhline(0, color=cp.MUTED, lw=0.8)
        ax.plot(x, star, color=cp.INK, lw=2.2, label="최적 $\\epsilon^*$ (닫힌식)")
        ax.plot(x, before, color=cp.MUTED, lw=1.2, ls="--", label="학습 전 $\\hat\\epsilon$")
        ax.plot(x, after, color=cp.BLUE, lw=1.6, label="학습 후 $\\hat\\epsilon$")
        ax.set_ylim(-4, 4)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[len(t_list)])
    _clean_axes(ax, "$t$", "$E\\|\\hat\\epsilon - \\epsilon^*\\|^2$", "최적 예측과의 거리 (로그 눈금)")
    ax.plot(t_grid, err_before, color=cp.MUTED, lw=1.4, ls="--", label="학습 전")
    ax.plot(t_grid, err_after, color=cp.BLUE, lw=1.8, label="학습 후")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5)
    cp.label(fig.axes[1], "회색 채움 = $q_t$ 의 밀도 · 데이터가 있는 곳에서 잘 맞음", y=-0.2)
    cp.show(fig)


def plot_eps_field_2d(xx, yy, pdf, grid_X, eps_star, eps_hat, t, error_after):
    """2차원 혼합, 한 시각 t: q_t 등고선 위에 최적 ε*(왼쪽, 검정)와 학습한 ε̂(오른쪽, 파랑) 화살표장."""
    fig = cp.new_figure((11, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    for k, (field, color, title) in enumerate([(eps_star, cp.INK, f"최적 $\\epsilon^*(x_t, t = {t})$ (mix2_denoiser)"),
                                               (eps_hat, cp.BLUE, f"학습한 $\\hat\\epsilon(x_t, t = {t})$ · 오차 {error_after:.3f}")]):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_{t,0}$", "$x_{t,1}$" if k == 0 else None, title)
        ax.contour(xx, yy, pdf, levels=7, cmap="Blues", linewidths=0.8)
        f = np.asarray(field, float)
        ax.quiver(grid_X[:, 0], grid_X[:, 1], f[:, 0], f[:, 1], color=color, angles="xy", scale_units="width", scale=40,
                  width=0.004, headwidth=4, alpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")
    cp.label(fig.axes[0], "화살표 = 잡음 방향 ε 의 예측 (봉우리에서 바깥으로) · 등고선 = $q_t$", y=-0.18)
    cp.show(fig)


def plot_denoised_examples(rows, t_list):
    """행마다 (x₀, x_t, x̂₀) 세 이미지: 좌표 공간에서 잡음을 더한 7 과 학습한 ε̂ 로 한 번에 되돌린 x̂₀."""
    n = len(rows)
    fig = cp.new_figure((7.6, 2.5 * n))
    gs = fig.add_gridspec(n, 3, wspace=0.15, hspace=0.35)
    for r, ((x0_img, xt_img, x0hat_img), t) in enumerate(zip(rows, t_list)):
        titles = ["$x_0$ (PCA-32 복원)", f"$x_t$, $t = {t}$", "$\\hat x_0$: 학습한 $\\hat\\epsilon$ 으로 한 번에"]
        for c, (img, title) in enumerate(zip((x0_img, xt_img, x0hat_img), titles)):
            ax = fig.add_subplot(gs[r, c])
            cp.blank(ax)
            cp.draw_image(ax, img, title if r == 0 else (f"$t = {t}$" if c == 1 else None))
    cp.label(fig.axes[-2], "$t$ 가 클수록 되돌린 7 이 \"평균적인 숫자\" 쪽으로 뭉개짐 · 여러 걸음으로 되돌리는 것은 35번", y=-0.1)
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def plot_loss_by_t(t_grid, eps_model, eps_floor, x0_model, x0_floor, snr, bin_labels, bin_model, bin_floor):
    """왼쪽: t 별 ε 손실(모델·최적 바닥). 가운데: 같은 오차를 x₀ 로 잰 손실 = ε 손실/SNR. 오른쪽: SNR(t) 로그 눈금. 맨 오른쪽: t 구간별 ε 손실 막대."""
    fig = cp.new_figure((16.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$t$", "손실", "$E\\|\\epsilon - \\hat\\epsilon\\|^2$: t 가 클수록 작음")
    ax.plot(t_grid, eps_floor, color=cp.INK, lw=2.0, label="최적 $\\epsilon^*$ 의 바닥")
    ax.plot(t_grid, eps_model, color=cp.BLUE, lw=1.4, ls="--", label="학습한 $\\hat\\epsilon$")
    ax.set_ylim(0, max(1.15, float(np.max(eps_model)) * 1.05))
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$t$", "손실", "$E\\|x_0 - \\hat x_0\\|^2 = \\epsilon$ 손실$/\\mathrm{SNR}(t)$: 반대 방향")
    ax2.plot(t_grid, x0_floor, color=cp.INK, lw=2.0, label="최적 바닥")
    ax2.plot(t_grid, x0_model, color=cp.BLUE, lw=1.4, ls="--", label="학습한 모델")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$t$", "SNR", "$\\mathrm{SNR}(t) = \\bar\\alpha_t/(1-\\bar\\alpha_t)$")
    ax3.plot(t_grid, snr, color=cp.BLUE, lw=1.8)
    ax3.axhline(1, color=cp.GOLD, lw=1.2, ls="--")
    ax3.text(t_grid[-1], 2.2, "SNR = 1 (신호 = 잡음)", color=cp.GOLD, fontsize=8.5, ha="right")
    ax3.set_yscale("log")
    ax4 = fig.add_subplot(gs[3])
    _clean_axes(ax4, None, "ε 손실 평균", "t 구간별 ε 손실")
    k = np.arange(len(bin_labels))
    ax4.bar(k - 0.18, bin_floor, width=0.36, color=cp.INK, label="최적 바닥")
    ax4.bar(k + 0.18, bin_model, width=0.36, color=cp.BLUE, label="학습한 모델")
    ax4.set_xticks(k, bin_labels, fontsize=8.5)
    ax4.legend(frameon=False, fontsize=8.5)
    cp.label(ax, "작은 $t$: $\\epsilon$ 은 $x_t$ 와 거의 무관해 최선도 0 근처 예측 (손실 ≈ 1)", y=-0.2)
    cp.label(ax2, "작은 $t$: $x_0 \\approx x_t$ 라 거의 정확 · 큰 $t$: 데이터 분산 쪽으로", y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 완성 예제
def plot_schedule_comparison(ts, ab_linear, ab_cosine, snr_linear, snr_cosine, floor_linear, floor_cosine,
                             model_linear, model_cosine):
    """왼쪽: 선형·코사인 스케줄의 ᾱ_t. 가운데: SNR(t) 로그 눈금. 오른쪽: 두 스케줄에서 t 별 ε 손실(최적 바닥과 학습한 모델).

    snr_linear, snr_cosine: 본문에서 계산한 ᾱ_t/(1 − ᾱ_t)."""
    fig = cp.new_figure((14, 3.7))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$t$", "$\\bar\\alpha_t$", "선형 vs 코사인 스케줄")
    ax.plot(ts, ab_linear, color=cp.BLUE, lw=1.8, label=f"선형 (β 1e−4→0.02), $\\bar\\alpha_T = {ab_linear[-1]:.3f}$")
    ax.plot(ts, ab_cosine, color=cp.RED, lw=1.8, label=f"코사인, $\\bar\\alpha_T = {ab_cosine[-1]:.1e}$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$t$", "SNR", "$\\mathrm{SNR}(t)$ (로그 눈금)")
    ax2.plot(ts, snr_linear, color=cp.BLUE, lw=1.8, label="선형")
    ax2.plot(ts, snr_cosine, color=cp.RED, lw=1.8, label="코사인")
    ax2.axhline(1, color=cp.GOLD, lw=1.2, ls="--")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8.5)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$t$", "ε 손실", "t 별 $E\\|\\epsilon - \\hat\\epsilon\\|^2$")
    ax3.plot(ts, floor_linear, color=cp.BLUE, lw=2.0, label="선형 · 최적 바닥")
    ax3.plot(ts, model_linear, color=cp.BLUE, lw=1.2, ls="--", label="선형 · 학습한 모델")
    ax3.plot(ts, floor_cosine, color=cp.RED, lw=2.0, label="코사인 · 최적 바닥")
    ax3.plot(ts, model_cosine, color=cp.RED, lw=1.2, ls="--", label="코사인 · 학습한 모델")
    ax3.set_ylim(0, 1.15)
    ax3.legend(frameon=False, fontsize=8, loc="lower left")
    cp.label(ax, "코사인은 T 에 상관없이 끝에서 ᾱ ≈ 0", y=-0.2)
    cp.show(fig)
