"""27번 Monte Carlo 노트북의 개념 그림과 실험 결과.

표본·함숫값·확률·기울기·표준오차·학습 경로는 본문에서 계산합니다.
이 파일은 전달된 배열을 concept_plots 부품으로 배치하고 그립니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_mean_flow(samples, values, estimate, exact):
    """표본 네 개 → 제곱값 → 표본 평균 → 정확한 적분의 흐름."""
    fig, axes = cp.flow([2.2, 2.2, 2.0, 2.0], height=2.8, wspace=0.4)
    cp.draw_grid(axes[0], samples[None, :], "뽑은 잡음 $z_n$", fmt="{:.2f}")
    cp.draw_grid(axes[1], values[None, :], "제곱 크기 $f(z_n)$", fmt="{:.2f}")
    cp.draw_formula(axes[2], f"표본 평균 = {estimate:.3f}", "이번 네 표본의 요약")
    cp.draw_formula(axes[3], f"기댓값 = {exact:.3f}", "분포 전체를 적분한 값")
    cp.connect(fig, axes[0], axes[1], "제곱")
    cp.connect(fig, axes[1], axes[2], "평균")
    cp.show(fig)


def plot_replication_flow(values, summaries, exact):
    """독립 실험을 행으로 쌓고 각 행에서 평균·SE를 구하는 축의 의미."""
    fig, axes = cp.flow([3.6, 2.5, 2.1], height=3.4, wspace=0.45)
    cp.draw_grid(axes[0], values, "행 하나 = 독립 실험 하나", fmt="{:.2f}",
                 row_labels=[f"반복 {i + 1}" for i in range(len(values))])
    cp.draw_grid(axes[1], summaries, "행마다 표본 축을 줄입니다", fmt="{:.3f}",
                 col_labels=["평균", "추정 SE"])
    cp.draw_formula(axes[2], f"정확한 평균 {exact:.3f}", "행 사이의 흔들림과\n한 행 안의 흔들림을 구분")
    cp.connect(fig, axes[0], axes[1], "axis=1")
    cp.show(fig)


def plot_mc_precision(sizes, observed_sd, predicted_sd, mean_reported_se, estimates, exact):
    """표본 수에 따른 독립 추정치의 SD와 한 실행의 추정 SE를 비교합니다."""
    fig = cp.new_figure((11.5, 3.7))
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.35})
    _clean_axes(axes[0], "표본 수 N", "표준편차 / 표준오차", "독립 반복으로 본 1/√N")
    axes[0].loglog(sizes, predicted_sd, color=cp.INK, ls="--", label="정확한 SD")
    axes[0].loglog(sizes, observed_sd, color=cp.BLUE, marker="o", label="반복 추정치의 SD")
    axes[0].loglog(sizes, mean_reported_se, color=cp.RED, marker="s", label="각 실행의 SE 평균")
    axes[0].legend(frameon=False, fontsize=8)
    _clean_axes(axes[1], "독립 반복 번호", "평균 추정값", "같은 N으로 다시 뽑으면")
    axes[1].plot(np.arange(len(estimates)), estimates, ".", color=cp.BLUE, ms=5)
    axes[1].axhline(exact, color=cp.GOLD, ls="--", label="정확한 기댓값")
    axes[1].legend(frameon=False, fontsize=8)
    cp.show(fig)


def plot_importance_flow(probabilities, weights, contributions, exact):
    """다르게 뽑은 비율을 p/q로 되돌리는 유한 상태의 정확한 계산."""
    fig, axes = cp.flow([2.7, 2.3, 2.4, 2.0], height=3.0, wspace=0.5)
    cp.draw_grid(axes[0], probabilities, "상태 0·1·2의 확률", fmt="{:.2f}", row_labels=["p", "q"])
    cp.draw_grid(axes[1], weights[None, :], "$w=p/q$", fmt="{:.2f}")
    cp.draw_grid(axes[2], contributions[None, :], "$qwf$의 상태별 기여", fmt="{:.2f}")
    cp.draw_formula(axes[3], f"합 = {exact:.2f}", "p에서 직접 평균낸 값과 같음")
    cp.connect(fig, axes[0], axes[1], "비율")
    cp.connect(fig, axes[2], axes[3], "합")
    cp.show(fig)


def plot_importance_results(grid, target_pdf, proposal_pdfs, names, means, sds, ess_means, exact):
    """제안 분포·독립 반복 평균과 SD·관측한 weight ESS를 나란히 표시합니다."""
    fig = cp.new_figure((13.0, 3.8))
    axes = fig.subplots(1, 3, gridspec_kw={"wspace": 0.4})
    colors = [cp.MUTED, cp.BLUE, cp.RED]
    _clean_axes(axes[0], "z", "밀도", "오른쪽 꼬리를 얼마나 자주 보나")
    axes[0].plot(grid, target_pdf, color=cp.INK, lw=2, label="목표 p")
    for name, density, color in zip(names, proposal_pdfs, colors):
        axes[0].plot(grid, density, color=color, ls="--", label=name)
    axes[0].axvline(2.0, color=cp.GOLD, lw=1.4)
    axes[0].legend(frameon=False, fontsize=8)
    _clean_axes(axes[1], None, "확률 추정", "반복 평균 ± 반복 SD (SE 아님)")
    positions = np.arange(len(names))
    axes[1].errorbar(positions - 0.1, means[:, 0], yerr=sds[:, 0], fmt="o", color=cp.BLUE,
                     capsize=4, label="raw IS")
    axes[1].errorbar(positions + 0.1, means[:, 1], yerr=sds[:, 1], fmt="s", color=cp.RED,
                     capsize=4, label="self-normalized")
    axes[1].axhline(exact, color=cp.GOLD, ls="--")
    axes[1].set_xticks(positions, names)
    axes[1].legend(frameon=False, fontsize=8)
    _clean_axes(axes[2], None, "평균 ESS", "가중치의 불균형만 재는 진단값")
    axes[2].bar(names, ess_means, color=colors)
    cp.label(axes[2], "놓친 영역이나 적분의 정확성을 보장하지 않습니다", y=-0.23)
    cp.show(fig)


def plot_pathwise_flow(eps, latent, loss_and_grad, mean_grad, exact_grad):
    """고정 잡음에서 이동한 잠재와 그 경로를 따라 미분한 표본 기울기."""
    fig, axes = cp.flow([1.8, 2.0, 3.0, 2.5], height=3.4, wspace=0.5)
    cp.draw_grid(axes[0], eps[:, None], "고정 ε", fmt="{:.2f}")
    cp.draw_grid(axes[1], latent[:, None], "$z=\\theta+\\sigma\\epsilon$", fmt="{:.2f}")
    cp.draw_grid(axes[2], loss_and_grad, "표본마다 손실과 미분", fmt="{:.2f}",
                 col_labels=["f(z)", "df/dθ"])
    cp.draw_formula(axes[3], f"표본 기울기 {mean_grad:.2f}", f"분포의 정확한 기울기 {exact_grad:.2f}\n둘은 서로 다른 검산 대상")
    cp.connect(fig, axes[0], axes[1], "∂z/∂θ = 1")
    cp.show(fig)


def plot_optimization(steps, paths, exact_losses, labels, target, minimum):
    """본문의 확률적 경사하강 경로와 닫힌식 목적값을 그립니다."""
    fig = cp.new_figure((11.0, 3.5))
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.32})
    colors = [cp.RED, cp.BLUE]
    _clean_axes(axes[0], "갱신 수", "위치 모수 θ", "새 잡음으로 기울기를 재며 갱신")
    _clean_axes(axes[1], "갱신 수", "정확한 J(θ)", "학습 손실의 잡음 없이 경로 평가")
    for path, losses, label, color in zip(paths, exact_losses, labels, colors):
        axes[0].plot(steps, path, color=color, label=label)
        axes[1].plot(steps, losses, color=color, label=label)
    axes[0].axhline(target, color=cp.GOLD, ls="--", label="정확한 최적 θ")
    axes[1].axhline(minimum, color=cp.GOLD, ls="--", label="정확한 최솟값")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    cp.show(fig)


def plot_score_flow(probabilities, reward_score, contributions, exact_grad):
    """Bernoulli 두 상태에서 확률의 미분을 기여로 더하는 그림."""
    fig, axes = cp.flow([2.1, 3.0, 2.2, 2.1], height=3.0, wspace=0.5)
    cp.draw_grid(axes[0], probabilities[:, None], "상태의 확률", fmt="{:.2f}", row_labels=["b=0", "b=1"])
    cp.draw_grid(axes[1], reward_score, "보상 × 확률의 score", fmt="{:.2f}", col_labels=["보상", "b−p"])
    cp.draw_grid(axes[2], contributions[:, None], "기댓값의 두 기여", fmt="{:.2f}")
    cp.draw_formula(axes[3], f"합 = {exact_grad:.2f}", "열거한 정확한 ∂E[r]/∂θ")
    cp.connect(fig, axes[2], axes[3], "합")
    cp.show(fig)


def plot_baselines(names, exact_means, exact_variances, target_grad):
    """동일한 세 번의 Bernoulli 추첨을 모두 열거한 baseline별 정확한 평균과 분산."""
    fig = cp.new_figure((12.0, 3.7))
    axes = fig.subplots(1, 2, gridspec_kw={"wspace": 0.3})
    colors = [cp.MUTED, cp.BLUE, cp.RED, cp.BLUE]
    _clean_axes(axes[0], None, "기울기의 정확한 평균", "같은 배치 평균을 빼면 편향")
    axes[0].bar(names, exact_means, color=colors)
    axes[0].axhline(target_grad, color=cp.GOLD, ls="--", label="참 기울기")
    axes[0].legend(frameon=False, fontsize=8)
    _clean_axes(axes[1], None, "기울기의 정확한 분산", "상수 baseline도 잘 골라야 합니다")
    axes[1].bar(names, exact_variances, color=colors)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    cp.show(fig)
