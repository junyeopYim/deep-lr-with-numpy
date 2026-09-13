"""02번에서 이미 계산한 확률·밀도·손실·예측 결과를 그립니다."""

import matplotlib.pyplot as plt


def plot_discrete(values, probabilities, mean):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(values, probabilities, width=0.5)
    ax.axvline(mean, color="tab:red", ls="--", label=f"기댓값 {mean:.2f}")
    ax.set(xticks=values, xlabel="확률변수의 값", ylabel="확률", title="확률은 각 값에 배정한 질량입니다")
    ax.legend()
    plt.show()


def plot_density(x, density, interval, interval_probability):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, density, label="확률밀도")
    ax.fill_between(x, 0, density, where=interval, alpha=0.3,
                    label=f"색칠한 구간의 확률 {interval_probability:.2f}")
    ax.set(xlabel="값 x", ylabel="밀도 p(x)", title="높이가 4여도 전체 넓이는 1입니다")
    ax.legend()
    plt.show()


def plot_estimates(counts, estimates, exact):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(counts, estimates, alpha=0.8, label="같은 표본열의 누적 평균")
    ax.axhline(exact, color="tab:red", ls="--", label="확률 가중합으로 계산한 기댓값")
    ax.set(xscale="log", xlabel="사용한 표본 수", ylabel="추정값", title="Monte Carlo: 표본 평균으로 기댓값 추정하기")
    ax.legend()
    plt.show()


def plot_likelihood(theta, likelihood, nll, estimate):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, y, title in zip(axes, [likelihood, nll], ["우도: 크게", "평균 음의 로그우도: 작게"]):
        ax.plot(theta, y)
        ax.axvline(estimate, color="tab:red", ls="--", label=f"표본 평균 {estimate:.2f}")
        ax.set(xlabel="성공확률 후보 θ", title=title)
        ax.legend()
    plt.show()


def plot_loss_curves(logits, losses, gradients):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, curve in losses.items():
        axes[0].plot(logits, curve, label=name)
    for name, curve in gradients.items():
        axes[1].plot(logits, curve, label=name)
    axes[0].set(xlabel="logit z", ylabel="한 표본의 BCE", title="틀린 확신에는 큰 손실")
    axes[1].set(xlabel="logit z", ylabel="z에 대한 미분", title="기울기가 가리키는 갱신 방향")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend()
    plt.show()


def plot_classifier(x, y, grid, probabilities, losses):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].scatter(x, y, label="관측한 라벨", s=22)
    axes[0].plot(grid, probabilities, color="tab:green", label="학습한 P(Y=1 | x)")
    axes[0].set(xlabel="입력 x", ylabel="확률 / 라벨", ylim=(-0.08, 1.08), title="logit → 확률 → 예측")
    axes[1].plot(range(len(losses)), losses)
    axes[1].set(xlabel="갱신 횟수", ylabel="평균 BCE", title="같은 관측을 더 잘 설명하도록 학습")
    axes[0].legend()
    plt.show()


# ---------------------------------------------------------------------------
# 개념 그림. 확률·밀도·우도·손실 값은 본문에서 계산하고 여기서는 받은 배열만 그립니다.
# 부품은 concept_plots, 원칙은 docs/DESIGN.md.
# ---------------------------------------------------------------------------

import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_pmf(values, probabilities, mean, title="흠집 수의 확률"):
    """이산 확률: 값마다 세운 막대, 기댓값은 금색 선."""
    fig = cp.new_figure((5.6, 3.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "값 $a$", "확률 $P(A=a)$", title)
    ax.bar(values, probabilities, width=0.5, color=cp.BLUE)
    for v, p in zip(values, probabilities):
        ax.text(v, p + 0.01, f"{p:.1f}", ha="center", va="bottom", fontsize=9, color=cp.INK)
    ax.axvline(mean, color=cp.GOLD, lw=1.8)
    ax.text(mean + 0.05, max(probabilities) * 0.95, f"기댓값 {mean:.1f}", color=cp.GOLD, fontsize=9)
    ax.set_xticks(list(values)); ax.set_ylim(0, max(probabilities) * 1.25)
    cp.show(fig)


def plot_joint_table(joint, p_a, p_b, b_given_a, a_names, b_names):
    """결합확률 표 → 행 합·열 합(주변확률) → 행을 1로 맞춘 조건부확률."""
    fig, axes = cp.flow([1, 1, 1], height=3.0, unit=2.2, wspace=1.0)
    cp.draw_grid(axes[0], joint, "결합 $P(A, B)$", kind="plain", fmt="{:.1f}", row_labels=a_names, col_labels=b_names)
    marg = np.zeros((3, 3)); marg[:2, :2] = joint; marg[:2, 2] = p_a; marg[2, :2] = p_b; marg[2, 2] = 1
    cp.draw_grid(axes[1], marg, "행 합 = $P(A)$ · 열 합 = $P(B)$", kind="plain", fmt="{:.1f}",
                 row_labels=a_names + ["열 합"], col_labels=b_names + ["행 합"],
                 highlight=[(0, 2), (1, 2), (2, 0), (2, 1)])
    cp.draw_grid(axes[2], b_given_a, "조건부 $P(B \\mid A)$ · 행마다 합 1", kind="plain", fmt="{:.1f}",
                 row_labels=a_names, col_labels=b_names)
    cp.connect(fig, axes[0], axes[1], "축 하나를 더함"); cp.connect(fig, axes[1], axes[2], "행 합으로 나눔")
    cp.show(fig)


def plot_density_area(x, density, interval, interval_probability, total_area):
    """밀도의 높이와 넓이. 색칠한 구간의 넓이가 확률."""
    fig = cp.new_figure((6.4, 3.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "무게 $u$ (kg)", "밀도 $p(u)$", f"높이는 {density.max():g}이지만 전체 넓이는 {total_area:g}")
    ax.plot(x, density, color=cp.INK, lw=1.8)
    ax.fill_between(x, 0, density, where=interval, color=cp.BLUE, alpha=0.45)
    xi = x[interval]
    ax.text(xi.mean(), density.max() * 0.45, f"넓이 = 확률\n{interval_probability:.2f}", ha="center", color=cp.INK, fontsize=9.5)
    ax.set_ylim(0, density.max() * 1.25)
    cp.show(fig)


def plot_likelihood_curves(theta, likelihood, nll, best):
    """같은 관측에 대한 파라미터 후보의 우도(크게)와 평균 음의 로그우도(작게)."""
    fig, axes = cp.flow([1, 1], height=3.0, unit=2.6, wspace=0.4)
    for ax, y, title, ylabel in zip(axes, [likelihood, nll], ["우도 $L(\\theta)$ : 클수록 좋음", "$-\\frac{1}{N}\\log L(\\theta)$ : 작을수록 좋음"],
                                    ["", ""]):
        _clean_axes(ax, "달 확률 후보 $\\theta$", ylabel, title)
        ax.plot(theta, y, color=cp.INK, lw=1.8)
        ax.axvline(best, color=cp.GOLD, lw=1.6)
        ax.text(best + 0.02, np.max(y) * 0.9, f"$\\theta={best:.2f}$", color=cp.GOLD, fontsize=9)
    cp.show(fig)


def plot_gaussian_observation(x, y, line_x, line_y, sigma, bell_at, bell_mu):
    """직선 예측 주변에 관측이 종 모양으로 흩어진다는 가정. 종의 높이가 그 관측의 밀도."""
    fig = cp.new_figure((7.0, 3.6))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "붉기 $x$", "당도 $y$", "관측 $y$는 예측 $\\mu(x)$ 주변에 가우스 잡음으로 흩어진다")
    ax.plot(line_x, line_y, color=cp.BLUE, lw=1.8, label="$\\mu(x) = xw + b$")
    ax.scatter(x, y, s=12, color=cp.INK, zorder=3, label="관측")
    grid = np.linspace(-3.5 * sigma, 3.5 * sigma, 80)
    bell = np.exp(-0.5 * (grid / sigma) ** 2)
    for xb, mu in zip(bell_at, bell_mu):
        ax.plot(xb + 0.45 * bell, mu + grid, color=cp.RED, lw=1.3)
        ax.plot([xb, xb + 0.45], [mu, mu], color=cp.RED, lw=0.8, ls=":")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    cp.label(ax, f"빨간 종: 예측 주변의 밀도 $p(y \\mid x)$, 표준편차 $\\sigma={sigma:g}$ · 종 위에 있을수록 그 관측이 잘 설명됨", y=-0.18)
    cp.show(fig)


def plot_bce_examples(images, probabilities, targets, losses, labels):
    """이미지 10장 → 7일 확률 → 정답 → 한 장씩의 BCE. 틀린 확신에 큰 손실."""
    n = len(images)
    fig = cp.new_figure((11, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.3, 1.3], wspace=0.7)
    cols = (n + 1) // 2
    sub = gs[0].subgridspec(2, cols, wspace=0.08, hspace=0.35)
    for i in range(n):
        ax = fig.add_subplot(sub[i // cols, i % cols]); cp.blank(ax)
        cp.draw_image(ax, images[i], frame=False)
        ax.set_title(f"{labels[i]}", color=cp.MUTED, fontsize=8.5, pad=2)
    ax_p = fig.add_subplot(gs[1]); cp.blank(ax_p)
    cp.draw_bars(ax_p, probabilities, "$p = \\sigma(z)$ : 7일 확률", labels=[str(l) for l in labels], xlim=(0, 1),
                 color=cp.MUTED)
    for i, t in enumerate(np.asarray(targets).ravel()):
        ax_p.plot([t], [i], marker="|", color=cp.GOLD, ms=13, mew=2.2)
    cp.label(ax_p, "금색 = 정답 $y$ (7이면 1)", y=-0.06)
    ax_l = fig.add_subplot(gs[2]); cp.blank(ax_l)
    cp.draw_bars(ax_l, losses, "$-\\log p_y$ : 정답 쪽 확률의 $-\\log$", labels=[str(l) for l in labels],
                 highlight=int(np.argmax(losses)), color=cp.MUTED)
    cp.label(ax_l, f"평균 = BCE = {np.mean(losses):.2f}", y=-0.06, color=cp.INK, fontsize=9.5)
    cp.connect(fig, fig.axes[cols - 1], ax_p, "sigmoid"); cp.connect(fig, ax_p, ax_l, "$-\\log$")
    cp.show(fig)


def plot_softmax_examples(images, logits, probabilities, targets, losses):
    """행마다: 이미지 → 점수 10개 → softmax 확률 10개(정답 금색) → −log p_y."""
    n = len(images)
    fig, rows = cp.flow([0.8, 1.1, 1.1, 0.9], rows=n, height=2.9, unit=2.0, wspace=0.45, hspace=0.3)
    for r, (ax_x, ax_z, ax_p, ax_l) in enumerate(rows):
        t = int(targets[r])
        cp.draw_image(ax_x, images[r], f"$x$ : 숫자 {t}")
        cp.draw_bars(ax_z, logits[r], "$z$ : 점수 10개" if r == 0 else None, signed=True, marker=t, fmt="{:+.1f}")
        cp.draw_bars(ax_p, probabilities[r], "softmax$(z)$ : 확률, 합 1" if r == 0 else None, xlim=(0, 1),
                     highlight=t, marker=t)
        cp.draw_formula(ax_l, f"$-\\log p_{{{t}}} = {losses[r]:.2f}$", "정답 확률이 클수록 작음")
        cp.connect(fig, ax_x, ax_z); cp.connect(fig, ax_z, ax_p, "$e^{z}$/합" if r == 0 else None); cp.connect(fig, ax_p, ax_l)
    cp.label(rows[-1][2], "금색 테두리 = 정답 클래스", y=-0.08)
    cp.show(fig)
