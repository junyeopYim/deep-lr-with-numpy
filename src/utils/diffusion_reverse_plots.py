"""35 diffusion 역과정 노트북의 개념 그림.

한 걸음의 평균·분산, 역과정 표본, DDIM 경로, guidance 결합, sampler 비교는 모두 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 곡선 축 정리는 gradient_plots._clean_axes,
원칙은 docs/DESIGN.md. 학습 곡선(plot_curves)은 bridge_plots 의 것을 그대로 내보냅니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .bridge_plots import plot_curves  # noqa: F401

DENSITY_FILL = 0.15       # 밀도 채움의 투명도


def _density(ax, x, pdf, color=cp.MUTED, scale=1.0):
    """밀도 곡선을 옅게 채운다. scale 은 다른 축에 겹칠 때 높이를 맞추는 배율."""
    values = np.asarray(pdf, float) * scale
    ax.fill_between(x, 0, values, color=color, alpha=DENSITY_FILL, lw=0)
    ax.plot(x, values, color=color, lw=0.9, alpha=0.75)


def _image_panel(ax, img, title=None):
    """0–1 잉크 이미지 한 장을 축 없이."""
    cp.blank(ax)
    cp.draw_image(ax, np.clip(np.asarray(img, float), 0, 1), title)


# ---------------------------------------------------------------- 1절
def plot_reverse_step(panels, t_axis, sigma_beta, sigma_tilde):
    """시각마다 한 걸음 장면: 가로축이 지금 자리 x_t, 세로축이 되돌린 값. 회색 대각선 y = x(제자리) 위에
    금색 x̂₀(x_t)(추측한 원본)와 파랑 μ_θ(x_t) ± σ_t(한 걸음 뒤 평균)를 겹쳐 본다. 아래 회색 띠는 q(x_t) 의 밀도.

    panels: [{"t", "ab", "x", "pdf", "x_t", "x0_hat", "mu", "sigma", "x0_curve", "mu_curve"}] (노트북 본문의 계산),
    오른쪽 끝 패널은 두 분산 선택 √β_t 와 √β̃_t.
    """
    fig = cp.new_figure((4.1 * len(panels) + 4.4, 3.9))
    gs = fig.add_gridspec(1, len(panels) + 1, wspace=0.3)
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_t$ (지금 자리)", "되돌린 값" if k == 0 else None,
                    f"$t = {panel['t']}$, $\\bar\\alpha_t = {panel['ab']:.3f}$")
        x = np.asarray(panel["x"], float)
        pdf = np.asarray(panel["pdf"], float)
        height = -5.0 + 1.6 * pdf / max(float(pdf.max()), 1e-9)      # 밀도를 패널 아래쪽 띠로
        ax.fill_between(x, -5.0, height, color=cp.MUTED, alpha=0.18, lw=0)
        ax.plot(x, height, color=cp.MUTED, lw=0.8)
        ax.plot(x, x, color=cp.MUTED, lw=1.0, ls="--", label="$y = x_t$ (제자리)")
        ax.plot(x, panel["x0_curve"], color=cp.GOLD, lw=1.7, label="$\\hat x_0(x_t)$ 추측한 원본")
        ax.fill_between(x, np.asarray(panel["mu_curve"]) - panel["sigma"], np.asarray(panel["mu_curve"]) + panel["sigma"],
                        color=cp.BLUE, alpha=0.35, lw=0)
        ax.plot(x, panel["mu_curve"], color=cp.BLUE, lw=1.7, label="$\\mu_\\theta(x_t) \\pm \\sigma_t$")
        ax.scatter([panel["x_t"]], [panel["x0_hat"]], s=34, color=cp.GOLD, zorder=5)
        ax.scatter([panel["x_t"]], [panel["mu"]], s=34, color=cp.BLUE, zorder=5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.text(0.04, 0.97, f"$x_t = {panel['x_t']:.1f}$ 에서\n$\\hat x_0 = {panel['x0_hat']:.2f}$\n"
                            f"$\\mu_\\theta = {panel['mu']:.2f}$\n$\\sigma_t = {panel['sigma']:.3f}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=cp.INK)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = fig.add_subplot(gs[len(panels)])
    _clean_axes(ax, "$t$", "$\\sigma_t$", "한 걸음에 더하는 잡음의 크기")
    ax.plot(t_axis, sigma_beta, color=cp.RED, lw=1.8, label="$\\sqrt{\\beta_t}$ (위쪽 선택)")
    ax.plot(t_axis, sigma_tilde, color=cp.BLUE, lw=1.8, label="$\\sqrt{\\tilde\\beta_t}$ (아래쪽 선택)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    cp.label(ax, "$\\tilde\\beta_t \\leq \\beta_t$ 이므로 파랑이 언제나 아래", y=-0.2)
    cp.label(fig.axes[0], "아래 회색 띠 = $q(x_t)$ 의 밀도 · 파랑이 대각선에 거의 붙어 있다 = 한 걸음은 아주 조금만 움직인다", y=-0.2)
    cp.show(fig)


def plot_reverse_histograms(x, pdf, centers, hists, titles, caption):
    """역과정으로 만든 표본의 히스토그램(금색 계단)을 원래 밀도 p(x₀)(파랑)와 겹쳐 본다. 패널마다 다른 설정."""
    fig = cp.new_figure((4.7 * len(hists), 3.4))
    gs = fig.add_gridspec(1, len(hists), wspace=0.26)
    for k, (hist, title) in enumerate(zip(hists, titles)):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_0$", "밀도" if k == 0 else None, title)
        ax.fill_between(x, 0, pdf, color=cp.BLUE, alpha=DENSITY_FILL, lw=0)
        ax.plot(x, pdf, color=cp.BLUE, lw=1.8, label="원래 밀도 $p(x_0)$")
        ax.step(centers, hist, where="mid", color=cp.GOLD, lw=1.3, label="역과정 표본")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 0.42)
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(fig.axes[0], caption, y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_reverse_paths(t_axis, paths, x, pdf, centers, hist, right_mask):
    """왼쪽: T 에서 0 까지 점 몇 개가 움직인 길(오른쪽 봉우리로 간 길은 파랑, 왼쪽은 빨강). 오른쪽: 끝점 히스토그램과 원래 밀도."""
    fig = cp.new_figure((13.4, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$t$ (오른쪽으로 갈수록 0 에 가까움)", "$x_t$", "역과정이 지나간 길: 잡음 하나에서 두 봉우리로")
    lines = np.asarray(paths, float)
    for k in range(lines.shape[1]):
        ax.plot(t_axis, lines[:, k], color=(cp.BLUE if right_mask[k] else cp.RED), lw=0.9, alpha=0.7)
    ax.invert_xaxis()
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    cp.label(ax, "파랑 = 오른쪽 봉우리(μ = 2)에 도착한 길 · 빨강 = 왼쪽 봉우리(μ = −2)", y=-0.2)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x_0$", "밀도", "끝점의 분포 vs 원래 밀도")
    ax2.fill_between(x, 0, pdf, color=cp.BLUE, alpha=DENSITY_FILL, lw=0)
    ax2.plot(x, pdf, color=cp.BLUE, lw=1.8, label="원래 밀도")
    ax2.step(centers, hist, where="mid", color=cp.GOLD, lw=1.3, label="역과정 표본")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_ylim(0, 0.42)
    cp.show(fig)


def plot_samples_2d(xx, yy, pdf, panels):
    """2차원 혼합: panels = [(제목, (N, 2) 점)] 을 같은 등고선 위에 나란히 찍는다."""
    fig = cp.new_figure((4.5 * len(panels), 4.0))
    gs = fig.add_gridspec(1, len(panels), wspace=0.22)
    for k, (title, points) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "$x_0$의 첫 좌표", "두 번째 좌표" if k == 0 else None, title)
        ax.contour(xx, yy, pdf, levels=6, cmap="Blues", linewidths=0.8)
        ax.scatter(points[:, 0], points[:, 1], s=5, color=cp.INK if k == 0 else cp.BLUE, alpha=0.3, lw=0)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    cp.label(fig.axes[len(panels) // 2], "등고선 = 원래 밀도 $p(x_0)$ · 점 = 잡음에서 출발해 T걸음 되돌아온 결과", y=-0.17)
    cp.show(fig)


def plot_sample_grid(images, names, counts, title):
    """왼쪽: 만든 좌표를 되돌린 이미지 격자(제목 = 클래스 평균 최근접으로 센 숫자). 오른쪽: 숫자별 개수 막대."""
    columns = 8
    rows = int(np.ceil(len(images) / columns))
    fig = cp.new_figure((1.35 * columns + 4.6, 1.5 * rows + 1.2))
    gs = fig.add_gridspec(rows, columns + 3, width_ratios=[1] * columns + [0.25, 2.0, 0.25], wspace=0.12, hspace=0.32)
    for i in range(rows * columns):
        ax = fig.add_subplot(gs[i // columns, i % columns])
        cp.blank(ax)
        if i < len(images):
            _image_panel(ax, images[i], str(names[i]))
    ax_b = fig.add_subplot(gs[:, columns + 1])
    _clean_axes(ax_b, "숫자", "샘플 수", "만든 샘플의 숫자별 개수")
    ax_b.bar(np.arange(10), counts, color=cp.BLUE, width=0.7)
    ax_b.set_xticks(np.arange(10), [str(d) for d in range(10)], fontsize=8.5)
    cp.label(ax_b, "클래스 평균 최근접으로 센 것 (분류기가 아님)", y=-0.16)
    fig.suptitle(title, color=cp.INK, fontsize=11)
    cp.show(fig)


def plot_snapshot_strip(rows, ts, end_labels):
    """행 하나가 표본 하나. 왼쪽 t = T 의 잡음에서 오른쪽 t = 0 까지, 같은 표본이 숫자로 바뀌는 과정."""
    n_row, n_col = len(rows), len(ts)
    fig = cp.new_figure((1.45 * n_col + 1.8, 1.75 * n_row + 0.9))
    gs = fig.add_gridspec(n_row, n_col, wspace=0.1, hspace=0.3)
    for r, row in enumerate(rows):
        for c, (img, t) in enumerate(zip(row, ts)):
            ax = fig.add_subplot(gs[r, c])
            _image_panel(ax, img, f"$t = {t}$" if r == 0 else None)
            if c == 0:
                ax.text(-0.16, 0.5, f"표본 {r + 1}\n→ {end_labels[r]}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, color=cp.INK)
    cp.label(fig.axes[(n_row - 1) * n_col + n_col // 2],
             "왼쪽은 순수 잡음 $x_T$, 오른쪽으로 갈수록 걸음이 쌓여 하나의 숫자로 모임 (행 왼쪽 = 최근접으로 센 숫자)", y=-0.12)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_ddim_paths(xx, yy, pdf, paths, step_counts, calls):
    """걸음 수 S 마다 한 패널: 같은 x_T 여섯 개에서 출발한 DDIM(η = 0) 경로. 마지막 패널은 ε̂ 호출 수 막대."""
    n = len(step_counts)
    fig = cp.new_figure((3.5 * n + 3.4, 3.8))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.9], wspace=0.28)
    for k, S in enumerate(step_counts):
        ax = fig.add_subplot(gs[k])
        _clean_axes(ax, "첫 좌표", "두 번째 좌표" if k == 0 else None, f"$S = {S}$ 걸음")
        ax.contour(xx, yy, pdf, levels=6, cmap="Blues", linewidths=0.7)
        line = np.asarray(paths[S], float)                     # (S+1, 점 수, 2)
        joints = dict(marker="o", ms=2.4) if S <= 20 else {}    # 걸음이 적을 때만 조각의 이음매를 점으로 보임
        for j in range(line.shape[1]):
            ax.plot(line[:, j, 0], line[:, j, 1], color=cp.MUTED, lw=0.9, **joints)
            ax.scatter(line[0, j, 0], line[0, j, 1], s=18, color=cp.INK, zorder=4)
            ax.scatter(line[-1, j, 0], line[-1, j, 1], s=30, color=cp.GOLD, zorder=5)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    ax = fig.add_subplot(gs[n])
    _clean_axes(ax, None, "$\\hat\\epsilon$ 호출 수", "한 장을 만드는 비용")
    index = np.arange(len(step_counts))
    ax.bar(index, calls, color=cp.BLUE, width=0.6)
    ax.set_xticks(index, [f"S={S}" for S in step_counts], fontsize=8.5)
    cp.label(fig.axes[0], "검정 = 출발점 $x_T$ (네 패널이 같은 여섯 점) · 금색 = 도착점 $x_0$"
                          " · 회색 점 = 걸음 조각의 이음매 ($S \\leq 20$ 패널만)", y=-0.2)
    cp.show(fig)


def plot_image_rows(rows, row_labels, col_labels, title, caption=None):
    """행 = 설정(걸음 수·w·label), 열 = 같은 잡음에서 만든 샘플. 받은 이미지를 그대로 격자로 그린다."""
    n_row, n_col = len(rows), len(rows[0])
    fig = cp.new_figure((1.35 * n_col + 2.0, 1.6 * n_row + 0.9))
    gs = fig.add_gridspec(n_row, n_col, wspace=0.1, hspace=0.34)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            head = col_labels[c] if (r == 0 and col_labels is not None) else None
            _image_panel(ax, img, head)
            if c == 0:
                ax.text(-0.14, 0.5, row_labels[r], transform=ax.transAxes, ha="right", va="center",
                        fontsize=9, color=cp.INK)
    fig.suptitle(title, color=cp.INK, fontsize=11)
    if caption:
        cp.label(fig.axes[(n_row - 1) * n_col + n_col // 2], caption, y=-0.12)
    cp.show(fig)


# ---------------------------------------------------------------- 4절
def plot_guidance_concept(x, eps_cond, eps_uncond, curves, ws, centers, hists, pdf_cond, means, stds):
    """왼쪽: 조건부 ε(검정)·무조건 ε(회색)·결합한 ε̃(색). 가운데: w 별 역과정 표본과 정답 성분 밀도. 오른쪽: 평균·표준편차 막대."""
    colors = [cp.BLUE, cp.GOLD, cp.RED]
    fig = cp.new_figure((14.6, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x_t$", "$\\tilde\\epsilon$", "결합한 잡음 예측 (한 시각에서)")
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    ax.plot(x, eps_cond, color=cp.INK, lw=2.0, label="$\\epsilon(x_t, t, y)$ 조건부 ($w = 0$)")
    ax.plot(x, eps_uncond, color=cp.MUTED, lw=1.3, ls="--", label="$\\epsilon(x_t, t, \\emptyset)$ 무조건")
    for curve, w, color in zip(curves, ws, colors):
        if w == 0:
            continue                                                  # w = 0 은 조건부 곡선과 완전히 같아 겹쳐 그리지 않는다
        ax.plot(x, curve, color=color, lw=1.4, label=f"$w = {w:g}$ 로 결합")
    ax.set_ylim(-4, 4)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x_0$", "밀도", "역과정 결과: $w$ 를 키우면 어떻게 되나")
    ax2.fill_between(x, 0, pdf_cond, color=cp.MUTED, alpha=DENSITY_FILL, lw=0)
    ax2.plot(x, pdf_cond, color=cp.INK, lw=1.6, label="정답 $p(x_0 \\mid y)$")
    for hist, w, color in zip(hists, ws, colors):
        ax2.step(centers, hist, where="mid", color=color, lw=1.3, label=f"$w = {w:g}$")
    ax2.set_xlim(-4.5, 0.5)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "$w$", None, "표본의 평균과 표준편차")
    index = np.arange(len(ws))
    ax3.bar(index - 0.19, means, width=0.36, color=cp.BLUE, label="평균 (정답 −2)")
    ax3.bar(index + 0.19, stds, width=0.36, color=cp.RED, label="표준편차 (정답 0.5)")
    ax3.axhline(-2.0, color=cp.GOLD, lw=1.2, ls="--")
    ax3.axhline(0.5, color=cp.GOLD, lw=1.2, ls="--")
    ax3.set_xticks(index, [f"$w = {w:g}$" for w in ws], fontsize=8.5)
    ax3.legend(frameon=False, fontsize=8, loc="center left")
    cp.label(ax, "$\\tilde\\epsilon = (1+w)\\,\\epsilon(x_t,t,y) - w\\,\\epsilon(x_t,t,\\emptyset)$ · 두 곡선의 차이를 $w$ 배 더 밀어 준 것", y=-0.2)
    cp.label(ax3, "금색 점선 = 정답 성분의 평균 −2 와 표준편차 0.5", y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def plot_sampler_comparison(labels, calls, accuracy, norms, data_norm):
    """sampler·걸음 수·guidance 설정별 막대 셋: ε̂ 호출 수(비용), 지정한 숫자와 맞은 비율, 샘플 좌표의 평균 길이."""
    fig = cp.new_figure((15.0, 0.42 * len(labels) + 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    index = np.arange(len(labels))
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$\\hat\\epsilon$ 호출 수 (한 장당)", None, "비용")
    ax.barh(index, calls, color=cp.MUTED, height=0.66)
    ax.set_yticks(index, labels, fontsize=8.5)
    ax.invert_yaxis()
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "지정한 숫자와 맞은 비율", None, "조건이 지켜진 정도")
    ax2.barh(index, accuracy, color=cp.BLUE, height=0.66)
    ax2.set_yticks(index, ["" for _ in labels])
    ax2.set_xlim(0, 1.05)
    ax2.invert_yaxis()
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "샘플 좌표의 평균 길이", None, f"데이터와 같은 규모인가 (금색 = 데이터 {data_norm:.2f})")
    ax3.barh(index, norms, color=cp.RED, height=0.66)
    ax3.axvline(data_norm, color=cp.GOLD, lw=1.5, ls="--")
    ax3.set_yticks(index, ["" for _ in labels])
    ax3.invert_yaxis()
    cp.label(ax2, "클래스 평균 최근접으로 센 것 (분류기가 아님)", y=-0.16)
    cp.show(fig)
