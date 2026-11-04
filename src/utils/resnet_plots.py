"""17번 잔차 연결의 개념 그림. 전달받은 배열·진단값만 배치하고 계산·학습은 하지 않습니다."""

import numpy as np
from . import concept_plots as cp


def _axes(ax, xlabel, ylabel, title):
    """실험 곡선에 필요한 축만 남깁니다."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.set_xlabel(xlabel, color=cp.INK, fontsize=9)
    ax.set_ylabel(ylabel, color=cp.INK, fontsize=9)
    ax.set_title(title, color=cp.INK, fontsize=10.5)
    ax.tick_params(labelsize=8)


def plot_two_paths(x, branch, output, direct_gradient, branch_gradient, total_gradient):
    """같은 칸에서 더하는 forward와 두 경로에서 모이는 backward를 보여 줍니다."""
    fig, axes = cp.flow([2.5, 2.5, 2.5], rows=2, height=1.9, wspace=.4, hspace=.8)
    for row, values, titles in zip(axes, [(x, branch, output), (direct_gradient, branch_gradient, total_gradient)],
                                   [('입력 x', '수정 F(x)', '합 y'), ('직접 기울기 G', '수정 기울기 GWᵀ', '입력 기울기 dX')]):
        for ax, value, title in zip(row, values, titles):
            cp.draw_grid(ax, value, title, fmt='{:.2f}')
        cp.connect(fig, row[0], row[1], '+')
        cp.connect(fig, row[1], row[2], '=')
    cp.show(fig)


def plot_dense_block(image, features, hidden, branch, output):
    """고정 숫자 7의 두 특징에서 세 은닉 값과 수정량을 거쳐 다시 두 특징으로 갑니다."""
    fig, axes = cp.flow([2.0, 2.1, 2.6, 2.1, 2.1], height=2.5, wspace=.5)
    cp.draw_image(axes[0], image, '고정 숫자 7')
    cp.draw_grid(axes[1], features, '왼쪽·오른쪽 잉크', fmt='{:.2f}')
    cp.draw_grid(axes[2], hidden, 'tanh 은닉 세 값', fmt='{:.2f}')
    cp.draw_grid(axes[3], branch, '수정량 F', fmt='{:.2f}')
    cp.draw_grid(axes[4], output, '원래 특징 + F', fmt='{:.2f}')
    cp.connect(fig, axes[1], axes[2], 'W₁, b₁')
    cp.connect(fig, axes[2], axes[3], 'W₂, b₂')
    cp.connect(fig, axes[3], axes[4], '+ x')
    cp.show(fig)


def plot_projection(image, branch, skip, output, shapes):
    """해상도와 채널을 맞춘 주경로·투영을 같은 공간에 나란히 놓습니다."""
    fig, axes = cp.flow([2.4, 2.4, 2.4, 2.4], height=3.1, wspace=.4)
    cp.draw_image(axes[0], image, '입력의 한 채널')
    cp.draw_image(axes[1], branch, '3×3 주경로 · 첫 채널', kind='signed')
    cp.draw_image(axes[2], skip, '1×1 투영 · 첫 채널', kind='signed')
    cp.draw_image(axes[3], output, '두 경로의 합', kind='signed')
    for ax, shape in zip(axes, shapes):
        cp.label(ax, f'C×H×W = {shape}', y=-.14)
    cp.connect(fig, axes[1], axes[2], '+')
    cp.connect(fig, axes[2], axes[3], '=')
    cp.show(fig)


def plot_depth_flow(steps, plain_values, residual_values, plain_gain, residual_gain):
    """스칼라 수정 함수 .5x를 반복한 실제 값과 전체 미분을 표시합니다."""
    fig, axes = cp.flow([1.6, 3.4, 3.4], height=3.2, wspace=.45)
    cp.draw_formula(axes[0], '$F(x)=0.5x$', '같은 수정 규칙\n시작 x₀ = 1')
    labels = [str(step) for step in steps]
    cp.draw_grid(axes[1], plain_values[None, :], 'plain: x → 0.5x', fmt='{:.3f}', col_labels=labels)
    cp.draw_grid(axes[2], residual_values[None, :], '잔차: x → x + 0.5x', fmt='{:.3f}', col_labels=labels)
    cp.label(axes[1], f'세 블록 뒤 미분 = {plain_gain:.3f}', y=-.2)
    cp.label(axes[2], f'세 블록 뒤 미분 = {residual_gain:.3f}', y=-.2)
    cp.show(fig)


def plot_depth_statistics(depths, means, samples, names):
    """본문에서 구한 깊이별 신호의 seed별 곡선과 평균을 그립니다."""
    fig = cp.new_figure((11, 3.6))
    axes = fig.subplots(1, 2, gridspec_kw={'wspace': .35})
    titles = ['초기화 뒤 출력 크기', '고정 출력 방향의 입력 기울기']
    ylabels = ['활성 RMS · 로그 축', '입력 / 출력 기울기 노름 · 로그 축']
    for j, ax in enumerate(axes):
        _axes(ax, '블록 수', ylabels[j], titles[j])
        for i, (name, color) in enumerate(zip(names, [cp.MUTED, cp.BLUE])):
            for sample in samples[i]:
                ax.plot(depths, sample[:, j], color=color, alpha=.3, lw=.8)
            ax.plot(depths, means[i, :, j], marker='o', color=color, label=name, lw=1.7)
        ax.set_yscale('log')
        ax.set_xticks(depths)
        ax.legend(frameon=False, fontsize=8)
    cp.show(fig)


def plot_classifier_flow(original, reduced, stem, last, probabilities):
    """고정 예시의 축소·stem·잔차 특징·클래스 확률을 그립니다."""
    fig, axes = cp.flow([2., 1.9, 1.9, 1.9, 2.6], height=3.3, wspace=.5)
    cp.draw_image(axes[0], original, '원래 28×28')
    cp.draw_image(axes[1], reduced, '고정 평균 · 14×14')
    cp.draw_image(axes[2], stem, 'stem 첫 채널 · 7×7', kind='signed')
    cp.draw_image(axes[3], last, '두 블록 뒤 · 7×7', kind='signed')
    cp.draw_bars(axes[4], probabilities, '초기 분류 확률', labels=[str(i) for i in range(10)], marker=7, xlim=(0, 1))
    cp.connect(fig, axes[0], axes[1], '2×2 평균')
    cp.connect(fig, axes[2], axes[3], '수정 두 번')
    cp.connect(fig, axes[3], axes[4], '196 → 10')
    cp.label(axes[4], '금색 테두리는 정답 7', y=-.12)
    cp.show(fig)


def plot_training_comparison(means, sds, names):
    """동일 seed 예산으로 계산한 학습·검증·stem 기울기 기록을 그립니다."""
    fig = cp.new_figure((13.6, 3.6))
    axes = fig.subplots(1, 3, gridspec_kw={'wspace': .35})
    _axes(axes[0], 'epoch', 'CE', '손실 · train은 갱신 전 배치 평균')
    _axes(axes[1], 'epoch', '정확도', 'validation · epoch 끝 모델')
    _axes(axes[2], 'epoch', 'stem 가중치 기울기 노름', '배치별 노름의 표본 가중평균')
    for i, (name, color) in enumerate(zip(names, [cp.MUTED, cp.BLUE])):
        epochs = means[i, :, 0]
        axes[0].plot(epochs, means[i, :, 1], ls='--', color=color, label=f'{name} train')
        axes[0].plot(epochs, means[i, :, 2], color=color, label=f'{name} validation')
        axes[0].fill_between(epochs, means[i, :, 2]-sds[i, :, 2], means[i, :, 2]+sds[i, :, 2], color=color, alpha=.13)
        for ax, column in [(axes[1], 3), (axes[2], 4)]:
            ax.plot(epochs, means[i, :, column], marker='o', color=color, label=name)
            ax.fill_between(epochs, means[i, :, column]-sds[i, :, column], means[i, :, column]+sds[i, :, column], color=color, alpha=.13)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(means[0, :, 0])
    cp.show(fig)


def plot_test_predictions(images, labels, predictions, seed):
    """사전 고정한 첫 seed·test 앞 10장의 결과를 모두 표시합니다."""
    fig, axes = cp.flow([1.8]*5, rows=2, height=2.1, wspace=.25, hspace=.55)
    for ax, image, label, prediction in zip([a for row in axes for a in row], images, labels, predictions):
        cp.draw_image(ax, image, f'정답 {label} · 예측 {prediction}')
        if label != prediction:
            cp.label(ax, '틀린 예측', y=-.15)
    fig.suptitle(f'미리 정한 seed {seed} · test 앞 10장 (14×14 입력)', y=1.02, color=cp.INK, fontsize=11)
    cp.show(fig)
