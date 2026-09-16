"""35b EDM 노트북의 개념 그림.

σ 좌표·Tweedie·preconditioning 계수·손실 가중·ODE 궤적·샘플러 비교는 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 곡선 축 정리는 gradient_plots._clean_axes,
원칙은 docs/DESIGN.md.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes

DENSITY_FILL = 0.15       # 밀도 채움의 투명도


def _density(ax, x, pdf, scale=1.0, color=cp.MUTED):
    """밀도 곡선을 옅게 채운다. scale 은 다른 축에 겹칠 때 높이를 맞추는 배율."""
    ax.fill_between(x, 0, np.asarray(pdf) * scale, color=color, alpha=DENSITY_FILL, lw=0)
    ax.plot(x, np.asarray(pdf) * scale, color=color, lw=0.8, alpha=0.7)


# ---------------------------------------------------------------- 1절
def plot_unified_sigma(families, x, pdf, sigma_show, points, denoised, curve_sigmas, curves):
    """왼쪽: 세 모델의 σ 일정을 한 축에(로그 눈금). 가운데: 한 σ 에서 x → D(x;σ) 화살표(Tweedie).
    오른쪽: σ 세 값의 denoiser 곡선 D(x;σ) 와 대각선 y = x.

    families: {이름: (진행도 (n,), σ (n,))}, points·denoised: 화살표를 그릴 x 와 D(x;σ).
    """
    fig = cp.new_figure((15, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.05], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "진행도 (첫 걸음 0 → 마지막 걸음 1)", "$\\sigma$", "세 모델을 $\\sigma$ 축 하나에 올리면")
    styles = [(cp.BLUE, "-", 1.9), (cp.RED, "--", 1.7), (cp.GOLD, "-.", 1.7), (cp.INK, ":", 1.7)]
    for (name, (frac, sig)), (color, ls, lw) in zip(families.items(), styles):
        ax.plot(frac, sig, color=color, ls=ls, lw=lw, marker="o", ms=2.6, label=name)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    cp.label(ax, "모두 큰 $\\sigma$ 에서 작은 $\\sigma$ 로 내려오는 일정 · 다른 것은 모양뿐", y=-0.22)

    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x$", None, f"$\\sigma = {sigma_show:g}$ 에서 $x \\rightarrow D(x;\\sigma)$")
    top = float(np.max(pdf))
    _density(ax2, x, pdf, scale=1.0 / max(top, 1e-9))
    for k, (x_one, d_one) in enumerate(zip(points, denoised)):
        height = 0.26 + 0.18 * k
        ax2.annotate("", xy=(d_one, height), xytext=(x_one, height),
                     arrowprops={"arrowstyle": "-|>", "color": cp.BLUE, "lw": 1.8, "mutation_scale": 14})
        ax2.scatter([x_one], [height], s=30, color=cp.GOLD, zorder=4)
        ax2.text(x_one, height + 0.05, f"$x = {x_one:g}$", color=cp.GOLD, fontsize=8, ha="center")
        ax2.text(d_one, height - 0.10, f"$D = {d_one:.2f}$", color=cp.BLUE, fontsize=8, ha="center")
    ax2.set_ylim(-0.05, 1.2)
    ax2.set_yticks([])
    cp.label(ax2, "회색 = $q_\\sigma$ · 금색 점에서 파란 화살표 끝이 $D$ · score 는 그 화살표를 $\\sigma^2$ 로 나눈 것", y=-0.22)

    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$x$", "$D(x;\\sigma)$", "$\\sigma$ 가 커지면 $D$ 는 평평해집니다")
    ax3.plot(x, x, color=cp.MUTED, lw=0.9, ls="--", label="$y = x$ (되돌릴 것이 없음)")
    for (sig, curve), color in zip(zip(curve_sigmas, curves), [cp.BLUE, cp.GOLD, cp.RED]):
        ax3.plot(x, curve, color=color, lw=1.8, label=f"$\\sigma = {sig:g}$")
    ax3.set_ylim(float(np.min(x)), float(np.max(x)))
    ax3.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(ax3, "작은 $\\sigma$: 대각선 · 큰 $\\sigma$: 데이터 평균 하나로", y=-0.22)
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_preconditioning(image_sigmas, raw_images, scaled_images, vmax,
                         sigmas, coefficients, scale_sigmas, scale_rows, sigma_data):
    """위 두 줄: 고정 예시 7 에 σ 를 키워 가며 잡음을 더한 입력과, 거기에 $c_{in}$ 을 곱한 입력.
    여덟 장 모두 같은 색 눈금(vmax)으로 그려, 위 줄만 진하기가 크게 달라지는 것을 볼 수 있게 합니다.
    오른쪽 위: 네 계수 곡선. 오른쪽 아래: 입력·타깃의 표준편차가 σ 와 함께 어떻게 변하는가.

    coefficients: {라벨: (len(sigmas),)}, scale_rows: {라벨: (len(scale_sigmas),)}.
    """
    n = len(image_sigmas)
    fig = cp.new_figure((3.0 + 1.5 * n + 8.0, 4.6))
    gs = fig.add_gridspec(2, n + 2, width_ratios=[1] * n + [2.7, 2.7], wspace=0.22, hspace=0.3)
    bottom_axes = []
    for k, (sig, raw, scaled) in enumerate(zip(image_sigmas, raw_images, scaled_images)):
        top = fig.add_subplot(gs[0, k])
        cp.blank(top)
        cp.draw_image(top, raw, f"$\\sigma = {sig:g}$", kind="signed", vmax=vmax)
        bottom = fig.add_subplot(gs[1, k])
        cp.blank(bottom)
        cp.draw_image(bottom, scaled, None, kind="signed", vmax=vmax)
        bottom_axes.append(bottom)
        if k == 0:
            top.text(-0.12, 0.5, "$x$", transform=top.transAxes, ha="right", va="center", color=cp.INK, fontsize=11)
            bottom.text(-0.12, 0.5, "$c_{in}x$", transform=bottom.transAxes, ha="right", va="center", color=cp.INK, fontsize=11)
    cp.label(bottom_axes[len(bottom_axes) // 2], "여덟 장 모두 같은 색 눈금 · 위 줄만 진하기가 크게 달라집니다 (파랑 = 양수, 빨강 = 음수)",
             y=-0.1, fontsize=9)

    ax = fig.add_subplot(gs[:, n])
    _clean_axes(ax, "$\\sigma$ (로그 눈금)", "계수", f"네 계수 ($\\sigma_{{data}} = {sigma_data:g}$)")
    for (name, values), color in zip(coefficients.items(), [cp.BLUE, cp.RED, cp.GOLD, cp.INK]):
        ax.plot(sigmas, values, color=color, lw=1.8, label=name)
    ax.set_xscale("log")
    ax.axvline(sigma_data, color=cp.MUTED, lw=1.0, ls=":")
    ax.text(sigma_data, ax.get_ylim()[1], " $\\sigma = \\sigma_{data}$", color=cp.MUTED, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8)
    cp.label(ax, "$c_{skip}$ 은 1 에서 0 으로, $c_{out}$ 은 0 에서 $\\sigma_{data}$ 로", y=-0.16)

    ax2 = fig.add_subplot(gs[:, n + 1])
    _clean_axes(ax2, "$\\sigma$ (로그 눈금)", "표준편차", "무엇의 규모가 1 로 유지되는가")
    for (name, values), color, style in zip(scale_rows.items(), [cp.MUTED, cp.BLUE, cp.RED], ["-", "-", "--"]):
        ax2.plot(scale_sigmas, values, color=color, lw=1.8, ls=style, label=name)
    ax2.axhline(1.0, color=cp.GOLD, lw=1.1, ls="--")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(ax2, "금색 = 1 · 회색만 $\\sigma$ 를 따라 커짐", y=-0.16)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_loss_weighting(log_sigma_samples, sigmas, weights, grid_range, bin_labels, raw_losses, weighted_losses):
    """왼쪽: 학습용 σ 를 뽑는 분포 $\\ln\\sigma \\sim N(P_{mean}, P_{std}^2)$ 의 히스토그램(로그 축).
    가운데: 가중치 $\\lambda(\\sigma) = 1/c_{out}^2$. 오른쪽: σ 구간별 손실(가중 없음 vs λ 가중)."""
    fig = cp.new_figure((15, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\ln \\sigma$", "표본 수", "학습에 쓰는 $\\sigma$ 는 어디에 몰려 있나")
    ax.hist(log_sigma_samples, bins=40, color=cp.BLUE, alpha=0.75)
    lo, hi = np.log(grid_range[0]), np.log(grid_range[1])
    ax.axvline(lo, color=cp.GOLD, lw=1.2, ls="--")
    ax.axvline(hi, color=cp.GOLD, lw=1.2, ls="--")
    ax.text(lo, ax.get_ylim()[1] * 0.04, " $\\sigma_{min}$", color=cp.GOLD, fontsize=8.5, va="bottom", ha="left")
    ax.text(hi, ax.get_ylim()[1] * 0.04, "$\\sigma_{max}$ ", color=cp.GOLD, fontsize=8.5, va="bottom", ha="right")
    cp.label(ax, "금색 = 샘플러가 지나가는 $[\\sigma_{min}, \\sigma_{max}]$ · 학습은 그 가운데에 집중", y=-0.22)

    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$\\sigma$ (로그 눈금)", "$\\lambda(\\sigma)$", "$\\lambda(\\sigma) = 1/c_{out}(\\sigma)^2$")
    ax2.plot(sigmas, weights, color=cp.BLUE, lw=1.9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    cp.label(ax2, "작은 $\\sigma$ 에서 크게: 거의 맞힌 답의 작은 차이도 크게 셈", y=-0.22)

    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$\\sigma$ 구간", "손실 (로그 눈금)", "구간별 손실: 가중 없음 vs $\\lambda$ 가중")
    k = np.arange(len(bin_labels))
    ax3.bar(k - 0.19, raw_losses, width=0.38, color=cp.MUTED, label="$\\|D - x_0\\|^2$ (가중 없음)")
    ax3.bar(k + 0.19, weighted_losses, width=0.38, color=cp.BLUE, label="$\\lambda(\\sigma)\\|D - x_0\\|^2 = \\|F - F_{target}\\|^2$")
    ax3.set_yscale("log")
    ax3.set_ylim(min(raw_losses) * 0.4, max(max(raw_losses), max(weighted_losses)) * 60)
    ax3.set_xticks(k, bin_labels, fontsize=8)
    ax3.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(ax3, "회색은 구간마다 백 배 넘는 차이 · 파랑은 거의 평평", y=-0.22)
    cp.show(fig)



def plot_training_loss(steps, raw, smooth_steps, smooth, title):
    """학습 곡선: 미니배치마다의 손실(옅은 회색)과 이동평균(파랑)을 겹쳐 그립니다.
    steps·raw: (S,), smooth_steps·smooth: 이동평균의 x 와 값."""
    fig = cp.new_figure((7.5, 3.4))
    ax = fig.add_subplot(1, 1, 1)
    _clean_axes(ax, "스텝", "손실", title)
    ax.plot(steps, raw, color=cp.MUTED, lw=0.6, alpha=0.55, label="미니배치마다")
    ax.plot(smooth_steps, smooth, color=cp.BLUE, lw=1.8, label="이동평균")
    ax.legend(frameon=False, fontsize=8.5)
    cp.show(fig)

# ---------------------------------------------------------------- 4절
def plot_ode_paths(rho_grids, contour, paths):
    """왼쪽: ρ 별 시간 격자 $\\sigma_i$. 오른쪽: 2차원 혼합 위의 ODE 궤적(기준·Euler·Heun).

    rho_grids: {ρ: σ 배열}, contour: (xx, yy, pdf), paths: {이름: (걸음+1, 궤적 수, 2) 배열} (앞에서부터 회색·빨강·파랑).
    """
    fig = cp.new_figure((11.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "걸음 번호 $i$", "$\\sigma_i$ (로그 눈금)", "$\\rho$ 가 걸음을 어디에 몰아 주나")
    for (rho, grid), color in zip(rho_grids.items(), [cp.MUTED, cp.RED, cp.BLUE]):
        ax.plot(np.arange(len(grid)), grid, color=color, lw=1.6, marker="o", ms=3.4, label=f"$\\rho = {rho:g}$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8.5)
    cp.label(ax, "$\\rho = 7$ 은 작은 $\\sigma$ 쪽에 걸음을 더 촘촘히 둡니다", y=-0.2)

    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x_0$", "$x_1$", "같은 출발점에서 푼 ODE 궤적")
    xx, yy, pdf = contour
    ax2.contour(xx, yy, pdf, levels=6, cmap="Blues", linewidths=0.8)
    styles = [(cp.MUTED, "-", 2.2), (cp.RED, "--", 1.4), (cp.BLUE, "-", 1.4)]
    for (name, trace), (color, style, lw) in zip(paths.items(), styles):
        for j in range(np.shape(trace)[1]):
            curve = np.asarray(trace)[:, j, :]
            ax2.plot(curve[:, 0], curve[:, 1], color=color, ls=style, lw=lw, label=name if j == 0 else None)
            ax2.scatter(curve[-1:, 0], curve[-1:, 1], s=22, color=color, zorder=4)
    ax2.set_xlim(float(xx.min()), float(xx.max()))
    ax2.set_ylim(float(yy.min()), float(yy.max()))
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(ax2, "궤적은 그림 밖($\\sigma = \\sigma_{max}$)에서 들어옵니다 · 점 = 도착점 $\\sigma = 0$ · 등고선 = 데이터 분포", y=-0.2)
    cp.show(fig)


def plot_solver_error(step_counts, error_series, slopes, call_labels, call_counts, call_errors):
    """왼쪽: 걸음 수에 따른 최종점 오차(로그-로그)와 맞춘 기울기. 오른쪽: 같은 $D$ 호출 수에서의 오차 막대."""
    fig = cp.new_figure((11.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "걸음 수 $N$ (로그 눈금)", "기준 해와의 거리", "걸음을 늘리면 오차가 어떻게 줄어드나")
    for (name, values), color, marker in zip(error_series.items(), [cp.RED, cp.BLUE, cp.GOLD], ["o", "s", "^"]):
        slope = slopes.get(name)
        tail = f" (기울기 {slope:.2f})" if slope is not None else ""
        ax.plot(step_counts, values, color=color, lw=1.7, marker=marker, ms=4.5, label=name + tail)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(step_counts, [str(n) for n in step_counts], minor=False)
    ax.set_xticks([], [], minor=True)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    cp.label(ax, "기울기 −1 = 걸음을 2배로 하면 오차 절반 · −2 = 4분의 1", y=-0.22)

    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$D$ 호출 수", "기준 해와의 거리", "같은 비용이면 어느 쪽이 정확한가")
    k = np.arange(len(call_labels))
    colors = [cp.RED if label.startswith("Euler") else cp.BLUE for label in call_labels]
    ax2.bar(k, call_errors, width=0.55, color=colors)
    ax2.set_yscale("log")
    ax2.set_xticks(k, [f"{label}\n({calls}회)" for label, calls in zip(call_labels, call_counts)], fontsize=8)
    cp.label(ax2, "빨강 = Euler · 파랑 = Heun (걸음마다 $D$ 를 두 번)", y=-0.28)
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def plot_sampler_grid(rows, labels, captions):
    """샘플러마다 한 줄씩 디코딩한 숫자 이미지. rows: [(n, 28, 28)], labels·captions: 줄마다 한 줄 설명."""
    n = len(rows[0])
    fig = cp.new_figure((1.25 * n + 3.0, 1.45 * len(rows)))
    gs = fig.add_gridspec(len(rows), n, wspace=0.08, hspace=0.42)
    for r, (images, label, caption) in enumerate(zip(rows, labels, captions)):
        for c in range(n):
            ax = fig.add_subplot(gs[r, c])
            cp.blank(ax)
            cp.draw_image(ax, images[c], None)
            if c == 0:
                ax.text(-0.08, 0.5, label, transform=ax.transAxes, ha="right", va="center", color=cp.INK, fontsize=9)
            if c == n - 1:
                ax.text(1.08, 0.5, caption, transform=ax.transAxes, ha="left", va="center", color=cp.MUTED, fontsize=8)
    cp.show(fig)


def plot_sampler_metrics(labels, calls, distances, real_distance, norms, real_norm):
    """왼쪽: 샘플러별 $D$ 호출 수. 가운데: 클래스 평균까지의 평균 거리(실제 데이터 선이 목표).
    오른쪽: 좌표 노름의 평균(실제 데이터 선이 목표)."""
    fig = cp.new_figure((13.5, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    k = np.arange(len(labels))
    panels = [("$D$ 호출 수 (비용)", calls, None), ("클래스 평균까지의 거리", distances, real_distance),
              ("좌표 노름 $\\|z\\|$ 의 평균", norms, real_norm)]
    for slot, (title, values, reference) in enumerate(panels):
        ax = fig.add_subplot(gs[slot])
        _clean_axes(ax, None, None, title)
        ax.bar(k, values, width=0.55, color=cp.BLUE)
        if reference is not None:
            ax.axhline(reference, color=cp.GOLD, lw=1.4, ls="--")
            ax.set_ylim(0, max(max(values), reference) * 1.35)
            ax.text(len(labels) - 0.45, reference * 1.04, f"실제 데이터 {reference:.2f}",
                    color=cp.GOLD, fontsize=8.5, va="bottom", ha="right")
        ax.set_xticks(k, labels, fontsize=8.5)
    cp.label(fig.axes[1], "금색 선에 가까울수록 좋음 · 선보다 작으면 평균 쪽으로 뭉갠 것", y=-0.24)
    cp.show(fig)


# ---------------------------------------------------------------- 완성 예제
def plot_variants(step_counts, rho_errors, sigma_data_labels, grid_errors, sigmas, coefficient_sets):
    """왼쪽: ρ 를 바꿨을 때의 solver 오차. 가운데: $\\sigma_{data}$ 를 바꿔 학습했을 때의 격자 오차.
    오른쪽: $\\sigma_{data}$ 가 바꾸는 $c_{skip}$ 곡선."""
    fig = cp.new_figure((14, 3.6))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "걸음 수 $N$ (로그 눈금)", "기준 해와의 거리", "$\\rho$ 를 바꾸면 (Heun, 정확한 $D$)")
    for (name, values), color in zip(rho_errors.items(), [cp.MUTED, cp.RED, cp.BLUE]):
        ax.plot(step_counts, values, color=color, lw=1.7, marker="o", ms=4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(step_counts, [str(n) for n in step_counts], minor=False)
    ax.set_xticks([], [], minor=True)
    ax.legend(frameon=False, fontsize=8.5)
    cp.label(ax, "$\\rho = 1$ 은 균등 격자 · 작은 $\\sigma$ 에서 걸음이 너무 큽니다", y=-0.22)

    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, None, "격자 RMSE", "$\\sigma_{data}$ 를 바꿔 학습하면")
    k = np.arange(len(sigma_data_labels))
    ax2.bar(k, grid_errors, width=0.5, color=[cp.BLUE if i == 0 else cp.RED for i in k])
    ax2.set_xticks(k, sigma_data_labels, fontsize=8.5)
    cp.label(ax2, "파랑 = 데이터의 실제 표준편차 · 빨강 = 어긋난 값", y=-0.22)

    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$\\sigma$ (로그 눈금)", "$c_{skip}(\\sigma)$", "$\\sigma_{data}$ 가 바꾸는 것")
    for (name, values), color in zip(coefficient_sets.items(), [cp.BLUE, cp.RED, cp.GOLD]):
        ax3.plot(sigmas, values, color=color, lw=1.8, label=name)
    ax3.set_xscale("log")
    ax3.legend(frameon=False, fontsize=8.5)
    cp.label(ax3, "$c_{skip}$ 이 0.5 가 되는 자리가 곧 $\\sigma_{data}$", y=-0.22)
    cp.show(fig)
