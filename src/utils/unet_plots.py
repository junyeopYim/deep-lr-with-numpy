"""18 작은 U-Net 노트북의 그림 배치. 모델·손실·거리·지표는 본문에서 계산합니다."""

import numpy as np

from . import concept_plots as cp


def plot_pixel_task(image, target, logits, probability):
    """같은 좌표의 입력·정답·logit·확률을 나란히 보여 줍니다."""
    fig, axes = cp.flow([2.4]*4, height=3.1, wspace=.3)
    for ax, values, title, kind in zip(axes, [image, target, logits, probability],
                                      ["입력: 흐린 도형", "정답: 화소마다 0/1", "logit: 같은 H×W", "sigmoid: 전경 확률"],
                                      ["gray", "gray", "signed", "gray"]):
        cp.draw_image(ax, values, title, kind=kind)
    fig.subplots_adjust(top=.82, bottom=.14)
    cp.show(fig)


def plot_resize_coordinates(original, padded, pooled, restored, counts):
    """홀수 격자의 우하단 padding·평균·확대/crop·역방향 합산 개수를 그립니다."""
    fig, axes = cp.flow([2.1, 2.6, 2.1, 2.1, 2.2], height=3.3, wspace=.25)
    for ax, values, title in zip(axes, [original, padded, pooled, restored, counts],
                               ["원본 3×3", "우하단 0 → 4×4", "평균 → 2×2", "확대 후 3×3 crop", "확대 backward의 합"]):
        cp.draw_grid(ax, values, title, fmt="{:.3g}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.81, bottom=.13)
    cp.show(fig)


def plot_skip_channels(encoder, upsampled, joined, encoder_grad, upsampled_grad):
    """채널을 붙인 표와 역방향으로 채널 구간을 나눈 표를 보여 줍니다."""
    fig, axes = cp.flow([2.2, 1.8, 2.5, 2.2, 1.8], height=3.3, wspace=.3)
    for ax, values, title in zip(axes, [encoder, upsampled, joined, encoder_grad, upsampled_grad],
                               ["E: 자세한 위치 2채널", "U: 넓은 문맥 1채널", "[E,U]: 3채널", "dE: 앞 2채널", "dU: 뒤 1채널"]):
        cp.draw_grid(ax, values, title, fmt="{:g}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.8, bottom=.14)
    cp.show(fig)


def plot_unet_trace(encoder, bottleneck, decoder, logits, labels):
    """실제 forward에서 받은 네 단계의 첫 채널 특성 맵을 표시합니다."""
    fig, axes = cp.flow([2.4]*4, height=3.1, wspace=.35)
    for ax, values, label in zip(axes, [encoder, bottleneck, decoder, logits], labels):
        cp.draw_image(ax, values, label, kind="signed")
    fig.subplots_adjust(top=.8, bottom=.16)
    cp.show(fig)


def plot_pixel_losses(target, probability, bce_each, gradient, losses):
    """화소 BCE 비용·결합 손실 기울기와 soft Dice 값을 그립니다."""
    fig, axes = cp.flow([2.0, 2.0, 2.3, 2.3, 2.1], height=3.1, wspace=.25)
    for ax, values, title in zip(axes[:4], [target, probability, bce_each, gradient],
                               ["정답 y", "확률 p", "화소 BCE", "BCE + 0.5 Dice의 dz"]):
        cp.draw_grid(ax, values, title, fmt="{:.2f}", kind="plain", fontsize=9)
    cp.draw_grid(axes[4], np.asarray(losses)[:, None], "이미지 한 장의 손실", row_labels=["BCE", "soft Dice"], fmt="{:.3f}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.81, bottom=.13)
    cp.show(fig)


def plot_segmentation_plan(counts, budget, parameters):
    """분할 수·고정 학습 예산·전체/활성 파라미터 수를 그립니다."""
    fig, axes = cp.flow([2.8, 2.6, 3.0], height=3.2, wspace=.6)
    cp.draw_grid(axes[0], np.asarray(counts)[:, None], "독립 seed의 데이터", row_labels=["train", "validation", "test"], kind="plain")
    cp.draw_grid(axes[1], np.asarray(budget)[:, None], "양쪽에 같은 예산", row_labels=["스텝", "배치", "본 표본 수"], kind="plain")
    cp.draw_grid(axes[2], parameters, "저장 수와 활성 수 구분", row_labels=["skip 사용", "skip 0"], col_labels=["전체", "활성"], kind="plain")
    fig.subplots_adjust(top=.8, bottom=.14, left=.05, right=.96)
    cp.show(fig)


def plot_training_curves(steps, losses, dices, names):
    """본문에서 기록한 검증 손실·hard Dice 경로를 그립니다."""
    fig = cp.new_figure((10.7, 3.7))
    axes = fig.subplots(1, 2)
    for name, loss, dice, color in zip(names, losses, dices, [cp.BLUE, cp.MUTED]):
        axes[0].plot(steps, loss, label=name, color=color)
        axes[1].plot(steps, dice, label=name, color=color)
    axes[0].set(xlabel="갱신 수", ylabel="BCE + 0.5 soft Dice", title="고정 validation 손실")
    axes[1].set(xlabel="갱신 수", ylabel="이미지별 hard Dice 평균", title="임계값 0.5의 판정")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_segmentation_predictions(images, targets, skip_probability, zero_probability, labels):
    """미리 선택한 test 관측의 입력·정답·두 모델 확률을 표시합니다."""
    fig, axes = cp.flow([2.2]*4, rows=len(images), height=2.5, wspace=.25, hspace=.4)
    for row, image, target, skip, zero, label in zip(axes, images, targets, skip_probability, zero_probability, labels):
        for ax, values, title in zip(row, [image, target, skip, zero], [label, "정답 mask", "skip 사용 확률", "skip 0 확률"]):
            cp.draw_image(ax, values, title)
    fig.subplots_adjust(top=.94, bottom=.06)
    cp.show(fig)


def plot_resolution_variants(names, foregrounds, ious, gradient_norms):
    """홀수 해상도와 전경 비율 변형에서 측정한 양을 서로 다른 축에 표시합니다."""
    fig = cp.new_figure((12.5, 3.7))
    axes = fig.subplots(1, 3)
    for ax, values, title in zip(axes, [foregrounds, ious, gradient_norms], ["실제 전경 화소 비율", "hard IoU 평균", "입력 기울기 크기"]):
        ax.bar(np.arange(len(names)), values, color=cp.BLUE)
        ax.set_xticks(np.arange(len(names)), names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)
