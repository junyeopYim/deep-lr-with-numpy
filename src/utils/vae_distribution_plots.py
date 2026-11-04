"""26번 VAE 분포 읽기의 그림 배치. 밀도·모델·통계·샘플은 본문에서 계산합니다."""

import numpy as np
from . import concept_plots as cp


def _axes(ax, xlabel, ylabel, title):
    """실험 수치에 필요한 축만 남깁니다."""
    ax.set_axis_on()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=cp.INK)
    ax.tick_params(labelsize=8)


def plot_distribution_roles(grid, prior, posterior, aggregate, kl):
    """개별 q 두 개, 모인 q, 고정 p와 개별 KL을 나란히 그립니다."""
    fig, axes = cp.flow([3.2, 3.2, 2.4], height=3.3, wspace=.45)
    _axes(axes[0], '잠재 z', '밀도', '관측마다 다른 q(z|x,c)')
    axes[0].plot(grid, prior, color=cp.MUTED, ls='--', label='p(z)')
    for index, color in enumerate([cp.RED, cp.BLUE]):
        axes[0].plot(grid, posterior[index], color=color, label=f'q · 관측 {index}')
    axes[0].legend(frameon=False, fontsize=8)
    _axes(axes[1], '잠재 z', '밀도', '두 q를 같은 비중으로 모으면')
    axes[1].plot(grid, prior, color=cp.MUTED, ls='--', label='p(z)')
    axes[1].plot(grid, aggregate, color=cp.BLUE, label='모인 q')
    axes[1].legend(frameon=False, fontsize=8)
    cp.draw_grid(axes[2], kl[:, None], '개별 KL (nat)', row_labels=['관측 0', '관측 1'], fmt='{:.3f}')
    cp.label(axes[1], '평균 0·분산 1이어도 같은 밀도는 아닙니다', y=-.25)
    cp.show(fig)


def plot_beta_schedule(steps, schedules, names, weighted_terms):
    """고정 손실 항에 붙는 계수와 시간에 따른 계수 규칙을 그립니다."""
    fig, axes = cp.flow([4., 5.], height=3.3, wspace=.42)
    cp.draw_grid(axes[0], weighted_terms, '같은 R·K에 β만 다르게', kind='plain',
                 row_labels=names[:3], col_labels=['R', 'βK', '합'], fmt='{:.2f}')
    _axes(axes[1], '0부터 세는 갱신 번호', 'β', '학습 전에 고정한 네 규칙')
    for values, name, color in zip(schedules, names, [cp.MUTED, cp.BLUE, cp.RED, cp.GOLD]):
        axes[1].plot(steps, values, label=name, color=color, lw=1.8)
    axes[1].legend(frameon=False, fontsize=8)
    cp.show(fig)


def plot_conditional_flow(joined_x, mu_sd, joined_z, mean):
    """같은 x·ε에서 조건을 바꾸는 두 행이 encoder와 decoder를 지나는 모양입니다."""
    fig, axes = cp.flow([2.5, 2.5, 2.5, 2.], height=2.9, wspace=.6)
    arrays = [joined_x, mu_sd, joined_z, mean]
    titles = ['[x, c] · 2×4', '[μ, σ] · 2×4', '[z, c] · 2×4', '관측 평균 m · 2×2']
    for ax, values, title in zip(axes, arrays, titles):
        cp.draw_grid(ax, values, title, fmt='{:.2f}', row_labels=['c=0', 'c=1'], fontsize=8.5)
    cp.connect(fig, axes[0], axes[1], 'encoder')
    cp.connect(fig, axes[1], axes[2], '같은 ε')
    cp.connect(fig, axes[2], axes[3], 'decoder')
    cp.show(fig)


def plot_evaluation_flow(train_x, valid_x, eps_slice, recon_draws):
    """학습·독립 평가 관측과 반복별 잡음 및 재구성 항의 축을 보여 줍니다."""
    fig, axes = cp.flow([2.8, 2.8, 2.3, 2.3], height=3.1, wspace=.55)
    for ax, points, title in zip(axes[:2], [train_x, valid_x], ['학습 관측', '별도 평가 관측']):
        ax.scatter(points[:, 0], points[:, 1], s=7, alpha=.35, color=cp.MUTED)
        ax.set_title(title, fontsize=10.5)
        cp.label(ax, '붉기·무게 두 측정값')
    cp.draw_grid(axes[2], eps_slice, 'ε: 반복 K × 관측 N', fmt='{:.2f}', fontsize=8)
    cp.draw_grid(axes[3], recon_draws[:, None], 'Rₖ: 관측 평균 하나씩', fmt='{:.2f}', fontsize=9)
    cp.connect(fig, axes[1], axes[2], '자료 고정')
    cp.connect(fig, axes[2], axes[3], '모델 고정')
    cp.show(fig)


def plot_learning(steps, history_mean, names):
    """고정 valid 자료의 seed 평균 R·KL·−ELBO 곡선입니다."""
    fig = cp.new_figure((12.8, 3.5))
    axes = fig.subplots(1, 3)
    for column, title in enumerate(['재구성 R', 'KL', '−ELBO = R + KL']):
        _axes(axes[column], '갱신 수', 'nat / 관측', title)
        for values, name, color in zip(history_mean, names, [cp.MUTED, cp.BLUE, cp.RED, cp.GOLD]):
            axes[column].plot(steps, values[:, column], label=name, color=color, lw=1.8)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(wspace=.32, bottom=.2)
    cp.show(fig)


def plot_test_metrics(names, means, standard_deviations):
    """본문이 계산한 test 지표의 세 seed 평균±SD를 표시합니다. MC SE와 다릅니다."""
    fig = cp.new_figure((12.8, 3.6))
    axes = fig.subplots(1, 3)
    for column, title in enumerate(['재구성 R', 'KL', '−ELBO = R + KL']):
        _axes(axes[column], '', 'nat / 관측', title)
        axes[column].errorbar(np.arange(len(names)), means[:, column],
                              yerr=standard_deviations[:, column], fmt='o',
                              color=cp.BLUE, ecolor=cp.MUTED, capsize=4)
        axes[column].set_xticks(np.arange(len(names)), names, rotation=18, fontsize=8)
    fig.subplots_adjust(wspace=.32, bottom=.25)
    cp.show(fig)


def plot_generation_path(z, condition, mean, generated):
    """사전 잠재 → 조건을 붙인 decoder 평균 → 관측 잡음이 더해진 실제 표본입니다."""
    fig, axes = cp.flow([2., 1.6, 2., 2.], height=2.9, wspace=.55)
    for ax, values, title in zip(axes, [z, condition, mean, generated],
                                  ['prior z', '선택한 조건 c', 'decoder 평균 m', '실제 생성 x']):
        cp.draw_grid(ax, values, title, fmt='{:.2f}', fontsize=9)
    cp.connect(fig, axes[1], axes[2], 'z도 입력')
    cp.connect(fig, axes[2], axes[3], '+ sη')
    cp.show(fig)


def plot_generated_comparison(test_x, test_c, means, generated, names, bounds):
    """첫 고정 seed의 네 모델. 위는 평균, 아래는 관측 잡음까지 포함한 표본입니다."""
    fig = cp.new_figure((13.5, 6.0))
    axes = fig.subplots(2, len(names))
    for column, name in enumerate(names):
        for row, sample_list in enumerate([means, generated]):
            ax = axes[row, column]
            cp.blank(ax)
            for label, color in enumerate([cp.RED, cp.BLUE]):
                selected = test_c[:, label] == 1
                ax.scatter(test_x[selected, 0], test_x[selected, 1], s=9, color=cp.MUTED, alpha=.22)
                values = sample_list[column][label]
                ax.scatter(values[:, 0], values[:, 1], s=7, color=color, alpha=.35)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_title(name + (' · 평균 m' if row == 0 else ' · 생성 x'), fontsize=10)
    cp.label(axes[-1, 1], '회색: 독립 test · 빨강: c=0 · 파랑: c=1 · 모든 판의 범위 동일', y=-.08)
    fig.subplots_adjust(wspace=.15, hspace=.25)
    cp.show(fig)


def plot_latent_diagnostics(names, kl_dimension, within_variance, sensitivity):
    """첫 seed의 차원별 세 지표와 의도적으로 잠재 경로를 끊은 통제 모형입니다."""
    fig, axes = cp.flow([3.5, 3.5, 3.5], height=3.6, wspace=.65)
    for ax, values, title in zip(axes, [kl_dimension, within_variance, sensitivity],
                                  ['평균 KL · 차원별', '조건 안의 μ 분산', 'decoder의 z 민감도']):
        cp.draw_grid(ax, values, title, row_labels=names, col_labels=['z₀', 'z₁'],
                     fmt='{:.3f}', fontsize=9, kind='plain')
    cp.show(fig)
