"""24 Jensen 부등식·ELBO 노트북의 개념 그림.

Jensen 부등식의 두 값, 이산 잠재의 ELBO·KL 곡선, 재구성/KL 두 항, φ 경사 상승 경로,
선형 Gaussian의 정확한 사후확률과 학습된 q는 모두 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 원칙은 docs/DESIGN.md.

색: 하한(ELBO)·q = 파랑, 차이(KL)·현(chord) = 빨강, 증거 log p(x)·정답 사후확률 = 금색, 나머지 회색.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes

CHORD_COLOR = cp.RED
BOUND_COLOR = cp.BLUE
EVIDENCE_COLOR = cp.GOLD


def _value_bars(ax, names, values, colors, fmt="{:.4f}"):
    """세로 막대와 그 위(또는 아래)의 값. 0 기준선을 둔다."""
    values = np.asarray(values, dtype=float)
    ax.bar(names, values, color=colors, width=0.6)
    ax.axhline(0, color=cp.MUTED, lw=0.8)
    span = float(np.abs(values).max() or 1.0)
    for i, value in enumerate(values):
        offset = 0.04 * span if value >= 0 else -0.04 * span
        ax.text(i, value + offset, fmt.format(value), ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8.5, color=cp.INK)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.margins(y=0.22)
    return ax


def plot_jensen_chord(t_grid, log_curve, values, mean_value, log_of_mean, mean_of_log,
                      w_grid, curve_log_of_mean, curve_mean_of_log, w0):
    """오목한 log 곡선 위의 두 점과 현: 평균의 log(곡선 위)가 log의 평균(현 위)보다 크다.

    t_grid/log_curve: 곡선, values: 두 값 (2,), mean_value: 가중 평균,
    w_grid/curve_*: 가중치 w_1을 0에서 1까지 바꾼 두 값, w0: 손계산에 쓴 가중치.
    """
    values = np.asarray(values, dtype=float)
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1, 1.2], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$t$", "$\\log t$", "오목한 곡선은 현보다 위에 있습니다")
    ax.plot(t_grid, log_curve, color=cp.INK, lw=2.0, label="$\\log t$")
    ax.plot(values, np.log(values), color=CHORD_COLOR, lw=1.6, ls="--", marker="o",
            ms=5, label="두 점을 이은 현")
    ax.scatter([mean_value], [log_of_mean], color=BOUND_COLOR, s=55, zorder=5)
    ax.scatter([mean_value], [mean_of_log], color=CHORD_COLOR, s=55, zorder=5)
    ax.plot([mean_value, mean_value], [mean_of_log, log_of_mean], color=EVIDENCE_COLOR, lw=2.6, zorder=4)
    ax.plot([mean_value, mean_value], [mean_of_log - 0.30, mean_of_log - 0.06],
            color=EVIDENCE_COLOR, lw=0.9, ls=":")                       # 금색 막대와 아래 라벨을 잇는 짧은 지시선
    ax.text(mean_value, mean_of_log - 0.36, f"차이 {log_of_mean - mean_of_log:.4f}",
            ha="center", va="top", fontsize=9, color=EVIDENCE_COLOR)    # 현 아래 빈 자리라 log 곡선에 가리지 않는다
    ax.text(mean_value + 0.05, log_of_mean + 0.24, "$\\log \\mathbb{E}[f]$", color=BOUND_COLOR, fontsize=9.5)
    ax.text(mean_value + 0.12, mean_of_log - 0.06, "$\\mathbb{E}[\\log f]$", color=CHORD_COLOR, fontsize=9.5,
            va="top")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "nat", f"$f = ({values[0]:g}, {values[1]:g})$, 가중치 ${w0:g}$")
    _value_bars(ax_b, ["$\\log \\mathbb{E}[f]$", "$\\mathbb{E}[\\log f]$", "차이"],
                [log_of_mean, mean_of_log, log_of_mean - mean_of_log],
                [BOUND_COLOR, CHORD_COLOR, EVIDENCE_COLOR])

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "가중치 $w_1$", "nat", "가중치를 바꾸어도 위아래가 바뀌지 않습니다")
    ax_c.plot(w_grid, curve_log_of_mean, color=BOUND_COLOR, lw=1.8, label="$\\log \\mathbb{E}[f]$")
    ax_c.plot(w_grid, curve_mean_of_log, color=CHORD_COLOR, lw=1.8, ls="--", label="$\\mathbb{E}[\\log f]$")
    ax_c.fill_between(w_grid, curve_mean_of_log, curve_log_of_mean, color=EVIDENCE_COLOR, alpha=0.22)
    ax_c.axvline(w0, color=cp.MUTED, lw=0.9, ls=":")
    ax_c.legend(frameon=False, fontsize=8.5, loc="lower center")
    cp.label(ax_c, "금색 넓이가 차이. 양 끝(한 값만 쓰는 경우)에서만 0", y=-0.26)
    cp.show(fig)


def plot_elbo_gap(q1_grid, elbo_curve, kl_curve, evidence_value, q1_star, q1_hand,
                  hand_elbo, hand_kl):
    """q를 바꾸며 본 ELBO와 KL: 하한은 증거를 넘지 않고, q가 사후확률일 때 정확히 닿습니다."""
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 1], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$q_1 = q(z=1)$", "nat", "$\\mathrm{ELBO}(q) \\leq \\log p(x)$")
    ax.plot(q1_grid, elbo_curve, color=BOUND_COLOR, lw=2.0, label="$\\mathrm{ELBO}(q)$")
    ax.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.8, ls="--", label="$\\log p(x)$ (증거)")
    ax.scatter([q1_star], [evidence_value], color=EVIDENCE_COLOR, s=70, zorder=5, edgecolor=cp.INK, lw=0.6)
    ax.scatter([q1_hand], [hand_elbo], color=CHORD_COLOR, s=55, zorder=5)
    ax.plot([q1_hand, q1_hand], [hand_elbo, evidence_value], color=CHORD_COLOR, lw=2.2, zorder=4)
    ax.text(q1_hand + 0.03, 0.5 * (hand_elbo + evidence_value), f"KL {hand_kl:.4f}",
            color=CHORD_COLOR, fontsize=8.5)
    ax.set_ylim(min(elbo_curve.min(), hand_elbo) - 0.6, evidence_value + 0.9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$q_1 = q(z=1)$", "nat", "$D_{KL}(q \\| p(z|x))$ = 남은 차이")
    ax_b.plot(q1_grid, kl_curve, color=CHORD_COLOR, lw=2.0)
    ax_b.axvline(q1_star, color=EVIDENCE_COLOR, lw=1.4, ls="--")
    ax_b.scatter([q1_star], [0.0], color=EVIDENCE_COLOR, s=60, zorder=5, edgecolor=cp.INK, lw=0.6)
    ax_b.set_ylim(-0.3, min(float(np.max(kl_curve)), 8.0))
    cp.label(ax_b, f"금색 세로선: 사후확률 $r_1 = {q1_star:.4f}$", y=-0.26)

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, None, "nat", f"$q = ({1 - q1_hand:g}, {q1_hand:g})$ 에서 세 숫자")
    _value_bars(ax_c, ["ELBO", "KL", "합"], [hand_elbo, hand_kl, hand_elbo + hand_kl],
                [BOUND_COLOR, CHORD_COLOR, EVIDENCE_COLOR])
    ax_c.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.2, ls="--")
    cp.show(fig)


def plot_two_terms(labels, recon_values, prior_kl_values, elbo_values, evidence_value):
    """같은 ELBO를 재구성 항과 prior KL 항으로 나눈 것. q 세 가지에서 두 항의 크기를 비교합니다."""
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.4)
    colors = [cp.MUTED, BOUND_COLOR, EVIDENCE_COLOR][:len(labels)]

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, None, "nat", "재구성 $\\mathbb{E}_q[\\log p(x|z)]$")
    _value_bars(ax, labels, recon_values, colors)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "nat", "$D_{KL}(q \\| p(z))$ (사전분포와의 거리)")
    _value_bars(ax_b, labels, prior_kl_values, colors)

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, None, "nat", "차이 = ELBO (점선은 $\\log p(x)$)")
    _value_bars(ax_c, labels, elbo_values, colors)
    ax_c.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.4, ls="--")
    cp.label(ax_c, "두 항의 절충: 재구성을 키우면 KL이 커집니다", y=-0.26)
    cp.show(fig)


def plot_phi_ascent(q1_grid, elbo_curve, grad_curve, path_q1, path_elbo, kl_history,
                    evidence_value, q1_star):
    """logit φ의 경사 상승: 기울기가 0이 되는 자리가 정확히 사후확률이고, KL이 0으로 내려갑니다."""
    path_q1, path_elbo = np.asarray(path_q1), np.asarray(path_elbo)
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.2, 1.1], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$q_1 = q(z=1)$", "nat", "경사 상승이 곡선의 꼭대기로 올라갑니다")
    ax.plot(q1_grid, elbo_curve, color=BOUND_COLOR, lw=2.0)
    ax.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.6, ls="--")
    ax.scatter(path_q1, path_elbo, color=EVIDENCE_COLOR, s=np.linspace(18, 60, len(path_q1)),
               zorder=5, edgecolor=cp.INK, lw=0.4)
    ax.annotate("시작", xy=(path_q1[0], path_elbo[0]), xytext=(11, -14), textcoords="offset points",
                fontsize=8.5, color=cp.MUTED)
    ax.set_ylim(float(np.min(path_elbo)) - 0.8, evidence_value + 0.6)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$q_1 = q(z=1)$", "nat", "$\\partial \\mathrm{ELBO} / \\partial \\varphi_1$")
    ax_b.plot(q1_grid, grad_curve, color=CHORD_COLOR, lw=2.0)
    ax_b.axhline(0, color=cp.MUTED, lw=0.9)
    ax_b.axvline(q1_star, color=EVIDENCE_COLOR, lw=1.4, ls="--")
    cp.label(ax_b, f"기울기가 0인 자리 = 사후확률 $r_1 = {q1_star:.4f}$", y=-0.26)

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "경사 상승 스텝", "nat", "$D_{KL}(q \\| p(z|x))$ 가 0으로")
    ax_c.plot(np.arange(len(kl_history)), np.maximum(kl_history, 1e-16), color=CHORD_COLOR, lw=1.8)
    ax_c.set_yscale("log")
    cp.show(fig)


def plot_gaussian_elbo_landscape(z_grid, prior_density, posterior_density, q_density, q_label,
                                 mu_grid, elbo_mu, logvar_grid, elbo_logvar,
                                 evidence_value, m_star, logvar_star):
    """연속 잠재: 사전분포·정확한 사후확률·현재 q의 밀도와, μ·ℓ 축을 따라 본 정확한 ELBO."""
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$z$", "밀도", "사전분포 · 정확한 사후확률 · 지금의 $q$")
    ax.plot(z_grid, prior_density, color=cp.MUTED, lw=3.4, label="$p(z) = \\mathcal{N}(0, 1)$")
    ax.fill_between(z_grid, 0, posterior_density, color=EVIDENCE_COLOR, alpha=0.28)
    ax.plot(z_grid, posterior_density, color=EVIDENCE_COLOR, lw=2.0, label="$p(z|x)$ (닫힌식)")
    ax.plot(z_grid, q_density, color=BOUND_COLOR, lw=1.8, ls="--", label=q_label)
    ax.legend(frameon=False, fontsize=8.5)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$\\mu$", "nat", "$\\sigma$ 를 정답에 두고 $\\mu$ 만 움직임")
    ax_b.plot(mu_grid, elbo_mu, color=BOUND_COLOR, lw=2.0)
    ax_b.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.5, ls="--")
    ax_b.axvline(m_star, color=EVIDENCE_COLOR, lw=1.3, ls=":")
    ax_b.set_ylim(float(np.min(elbo_mu)) - 0.2, evidence_value + 0.5)
    cp.label(ax_b, f"꼭대기 $\\mu = m = {m_star:.4f}$", y=-0.26)

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "$\\ell = \\log \\sigma^2$", "nat", "$\\mu$ 를 정답에 두고 $\\ell$ 만 움직임")
    ax_c.plot(logvar_grid, elbo_logvar, color=BOUND_COLOR, lw=2.0)
    ax_c.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.5, ls="--")
    ax_c.axvline(logvar_star, color=EVIDENCE_COLOR, lw=1.3, ls=":")
    ax_c.set_ylim(float(np.min(elbo_logvar)) - 0.2, evidence_value + 0.5)
    cp.label(ax_c, f"꼭대기 $\\ell = \\log v = {logvar_star:.4f}$", y=-0.26)
    cp.show(fig)


def plot_elbo_sample_spread(k_list, shown_estimates, means, stds, exact_value, reference):
    """샘플 수 K에 따른 ELBO 추정의 흩어짐. 왼쪽은 추정값 흩뿌리기와 K마다의 평균, 오른쪽은 표준편차와 1/√K 기준선.

    means: 본문에서 계산한 K마다의 평균 (len(k_list),), reference: σ_1/√K 기준선 (len(k_list),).
    """
    k_list = list(k_list)
    fig = cp.new_figure((11.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "샘플 수 $K$", "nat", "같은 $(\\mu, \\ell)$ 에서 되풀이한 ELBO 추정값")
    for i, values in enumerate(shown_estimates):
        values = np.asarray(values, dtype=float)
        offsets = np.linspace(-0.3, 0.3, len(values))
        ax.scatter(i + offsets, values, s=6, color=BOUND_COLOR, alpha=0.45)
    ax.axhline(exact_value, color=EVIDENCE_COLOR, lw=1.8, ls="--")
    for i, mean_value in enumerate(np.asarray(means, dtype=float)):
        ax.hlines(mean_value, i - 0.42, i + 0.42, color=cp.INK, lw=4.6, zorder=4)        # 금색 눈금의 테두리
        ax.hlines(mean_value, i - 0.42, i + 0.42, color=EVIDENCE_COLOR, lw=2.6, zorder=5)
    ax.plot([], [], color=EVIDENCE_COLOR, lw=2.6, label="$K$ 마다의 평균")
    ax.set_xticks(range(len(k_list)), [str(k) for k in k_list])
    ax.text(len(k_list) - 1.0, exact_value + 0.7, "정확한 ELBO", color=EVIDENCE_COLOR, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "샘플 수 $K$", "표준편차 (nat)", "흩어짐은 $1/\\sqrt{K}$ 로 줄어듭니다")
    stds = np.asarray(stds, dtype=float)
    ax_b.bar([str(k) for k in k_list], stds, color=BOUND_COLOR, width=0.6)
    ax_b.plot(range(len(k_list)), np.asarray(reference, dtype=float), color=EVIDENCE_COLOR, lw=1.6, marker="o", ms=5,
              label="$\\sigma_1 / \\sqrt{K}$")
    for i, value in enumerate(stds):
        ax_b.text(i, value + 0.03 * float(stds.max()), f"{value:.3f}", ha="center", fontsize=8.5, color=cp.INK)
    ax_b.legend(frameon=False, fontsize=8.5)
    ax_b.margins(y=0.2)
    cp.show(fig)


def plot_gaussian_q_learning(elbo_history, evidence_value, z_grid, posterior_density,
                             q_curves, q_labels, mu_history, sigma_history, m_star, sigma_star):
    """학습 결과: 스텝마다의 정확한 ELBO, 학습 전후의 q와 정확한 사후확률, (μ, σ) 경로."""
    fig = cp.new_figure((13.5, 3.8))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "스텝", "nat", "지금의 $(\\mu, \\ell)$ 로 잰 정확한 ELBO")
    ax.plot(np.arange(len(elbo_history)), elbo_history, color=BOUND_COLOR, lw=1.6)
    ax.axhline(evidence_value, color=EVIDENCE_COLOR, lw=1.5, ls="--")
    ax.text(0.45 * len(elbo_history), evidence_value + 0.25, "$\\log p(x)$", color=EVIDENCE_COLOR, fontsize=9)
    ax.set_ylim(float(np.min(elbo_history)) - 0.5, evidence_value + 1.1)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$z$", "밀도", "학습 전후의 $q$ 와 정확한 사후확률")
    ax_b.fill_between(z_grid, 0, posterior_density, color=EVIDENCE_COLOR, alpha=0.28)
    ax_b.plot(z_grid, posterior_density, color=EVIDENCE_COLOR, lw=2.0, label="$p(z|x)$ (닫힌식)")
    styles = [(cp.MUTED, ":"), (BOUND_COLOR, "--")]
    for curve, label, (color, style) in zip(q_curves, q_labels, styles):
        ax_b.plot(z_grid, curve, color=color, lw=1.8, ls=style, label=label)
    ax_b.legend(frameon=False, fontsize=8.5)

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "$\\mu$", "$\\sigma$", "$(\\mu, \\sigma)$ 가 닫힌식 값으로")
    ax_c.plot(mu_history, sigma_history, color=BOUND_COLOR, lw=1.3, alpha=0.85)
    ax_c.scatter([mu_history[0]], [sigma_history[0]], color=cp.MUTED, s=45, zorder=5)
    ax_c.scatter([m_star], [sigma_star], color=EVIDENCE_COLOR, s=80, marker="*", zorder=6,
                 edgecolor=cp.INK, lw=0.5)
    ax_c.annotate("시작", xy=(mu_history[0], sigma_history[0]), xytext=(8, -2), textcoords="offset points",
                  color=cp.MUTED, fontsize=8.5)
    cp.label(ax_c, f"별: 닫힌식 $(m, \\sqrt{{v}}) = ({m_star:.4f}, {sigma_star:.4f})$", y=-0.26)
    cp.show(fig)


def plot_posterior_variants(z_grid, panels):
    """완성 예제: (a, s)를 바꾼 세 모형의 정확한 사후확률과 학습된 q.

    panels: [(제목, 사후확률 밀도 (G,), 학습된 q 밀도 (G,)), …]
    """
    fig = cp.new_figure((4.5 * len(panels), 3.8))
    gs = fig.add_gridspec(1, len(panels), wspace=0.34)
    for i, (title, posterior, q_curve) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "$z$", "밀도" if i == 0 else None, title)
        ax.fill_between(z_grid, 0, posterior, color=EVIDENCE_COLOR, alpha=0.28)
        ax.plot(z_grid, posterior, color=EVIDENCE_COLOR, lw=2.0, label="$p(z|x)$ (닫힌식)")
        ax.plot(z_grid, q_curve, color=BOUND_COLOR, lw=1.8, ls="--", label="학습된 $q$")
        ax.legend(frameon=False, fontsize=8.5)
    cp.show(fig)
