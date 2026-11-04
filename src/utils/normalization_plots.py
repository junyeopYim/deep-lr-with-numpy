"""16 정규화 층의 배열·통계·학습 기록을 전달받아 그립니다.

평균·분산·정규화·역전파·모델 예측은 노트북에서 계산합니다.
이 파일은 concept_plots 부품을 배치하고 받은 값만 표시합니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def _grids(values, titles, labels=None, height=3.0):
    """작은 행렬들을 계산 순서대로 배열하고 선택적인 설명을 붙입니다."""
    fig, axes = cp.flow([2.4] * len(values), height=height, wspace=0.42)
    for index, (ax, array, title) in enumerate(zip(axes, values, titles)):
        cp.draw_grid(ax, array, title=title, kind="signed", fmt="{:.2f}", fontsize=9)
        if labels is not None:
            cp.label(ax, labels[index], fontsize=8)
    cp.show(fig)


def plot_axis_normalization(x, mean, centered, normalized):
    """열별 평균 제거와 크기 조절의 손계산을 네 판으로 표시합니다."""
    _grids([x, mean, centered, normalized],
           ["입력 $X$ · 행=표본", "열별 평균 · (1,D)", "$X-\\mu$", r"$\widehat X=(X-\mu)/\sqrt{v+\epsilon}$"],
           ["같은 열의 값끼리 묶습니다", "axis=0 · 표본 축을 평균", "각 열의 평균은 0", "분산은 v / (v + ε)"])


def plot_bn_modes(x, train_y, running_stats, eval_y):
    """현재 배치와 저장 통계를 사용하는 두 경로를 비교합니다."""
    _grids([x, train_y, running_stats, eval_y],
           ["같은 입력 배치", "train · 현재 배치로", "저장한 평균 / 분산", "eval · 저장 통계로"],
           ["행마다 표본 하나", "통계도 X의 함수", "첫 행 평균 · 둘째 행 분산", "통계는 미분에서 상수"])


def plot_batch_independence(batch_a, batch_b, ln_target_rows, bn_target_rows):
    """같은 첫 표본과 다른 동료들에 대한 LN·BN 결과를 표시합니다."""
    _grids([batch_a, batch_b, ln_target_rows, bn_target_rows],
           ["배치 A · 첫 행 고정", "배치 B · 동료만 변경", "LN 첫 행 · A / B", "BN 첫 행 · A / B"],
           ["첫 행이 비교 대상", "첫 행은 같은 값", "표본 안의 열끼리 계산", "다른 표본도 통계를 바꿈"], height=3.5)


def plot_norm_comparison(x, bn_y, ln_y, rms_y):
    """같은 행렬에 적용한 BN·LN·RMS의 정규화 출력을 표시합니다."""
    _grids([x, bn_y, ln_y, rms_y],
           ["입력 $X$", "BN · 열별 중심화", "LN · 행별 중심화", "RMS · 행별 크기만"],
           ["γ=1, β=0 조건", "표본 축 평균", "특징 축 평균", "평균을 빼지 않음"])


def plot_mlp_norm_flow(image, preactivation, normalized, activated, scores):
    """고정 MNIST 한 장의 은닉층·정규화·활성화·점수를 표시합니다."""
    fig, axes = cp.flow([2.0, 2.0, 2.0, 2.0, 2.2], height=3.0, wspace=0.48)
    cp.draw_image(axes[0], image, "입력 · 784화소")
    for ax, values, title in zip(axes[1:4], [preactivation, normalized, activated],
                                  ["선형 출력 $U$", "LN 출력 $V$", "tanh 출력 $H$"]):
        cp.draw_strip(ax, values, title, kind="signed", vertical=False)
    cp.draw_grid(axes[4], np.asarray(scores)[:, None], "10개 logit", fmt="{:.2f}", fontsize=8,
                 row_labels=[str(i) for i in range(10)], kind="signed")
    axes[4].set_aspect("auto")
    for left, right, title in zip(axes[:-1], axes[1:], ["선형", "정규화", "활성화", "선형"]):
        cp.connect(fig, left, right, title)
    cp.show(fig)


def plot_norm_training(histories, names):
    """각 방식의 epoch별 validation loss와 accuracy를 표시합니다."""
    fig = cp.new_figure((10.5, 3.6))
    axes = fig.subplots(1, 2)
    colors = [cp.MUTED, cp.BLUE, cp.GOLD, cp.RED]
    _clean_axes(axes[0], "epoch", "validation CE", "고정한 검증 자료의 손실")
    _clean_axes(axes[1], "epoch", "validation accuracy", "고정한 검증 자료의 정확도")
    for history, name, color in zip(histories, names, colors):
        axes[0].plot(history[:, 0], history[:, 2], label=name, color=color)
        axes[1].plot(history[:, 0], history[:, 3], label=name, color=color)
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.3, bottom=0.17)
    cp.show(fig)


def plot_norm_predictions(images, true_labels, predictions, names):
    """독립 평가 이미지와 각 방식의 예측 숫자를 함께 표시합니다."""
    fig, axes = cp.flow([1.7] * len(images), height=3.0, wspace=0.2)
    for column, ax in enumerate(axes):
        cp.draw_image(ax, images[column], f"정답 {true_labels[column]}")
        lines = [f"{name}: {predictions[row, column]}" for row, name in enumerate(names)]
        cp.label(ax, "\n".join(lines), y=-0.08, fontsize=8.5)
    cp.show(fig)
