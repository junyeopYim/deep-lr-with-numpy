"""22 Autoencoder의 개념 그림과 실험 결과를 그립니다.

입력·잠재 좌표·복원·오차·잡음·학습 기록은 모두 노트북에서 계산합니다.
이 파일은 받은 배열의 표시와 배치만 담당하며 모델을 실행하지 않습니다.
"""

import numpy as np

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_linear_flow(x, z, reconstructed, residual, loss):
    """두 표본이 2 → 1 → 2차원으로 흐르는 손계산 배열을 표시합니다."""
    fig, axes = cp.flow([2.4, 1.8, 2.4, 2.4], height=3.0, wspace=0.5)
    for ax, values, title in zip(
        axes, [x, z, reconstructed, residual],
        ["입력 $X$ · (2, 2)", "잠재 $Z$ · (2, 1)", "복원 $R$ · (2, 2)", "잔차 $R-X$"],
    ):
        cp.draw_grid(ax, values, title=title, kind="signed", fmt="{:.0f}")
    cp.connect(fig, axes[0], axes[1], "$W_e$")
    cp.connect(fig, axes[1], axes[2], "$W_d$")
    cp.label(axes[3], f"제곱의 평균 ÷ 2 = {loss:.2f}")
    cp.show(fig)


def plot_pca_projection(centered, latent, reconstructed, residual):
    """중심화된 작은 자료의 좌표·투영·잔차를 나란히 표시합니다."""
    fig, axes = cp.flow([2.4, 1.7, 2.4, 2.4], height=3.4, wspace=0.5)
    titles = ["중심화 $X_c$", "$Z=X_cV_L$", "$R=ZV_L^T$", "$X_c-R$ · 버린 방향"]
    for ax, values, title in zip(axes, [centered, latent, reconstructed, residual], titles):
        cp.draw_grid(ax, values, title=title, fmt="{:.1f}", kind="signed", fontsize=10)
    cp.connect(fig, axes[0], axes[1], "상위 방향")
    cp.connect(fig, axes[1], axes[2], "되돌리기")
    cp.show(fig)


def plot_nonlinear_flow(image, hidden, latent, reconstructed):
    """본문에서 계산한 한 장의 은닉 표현과 잠재 좌표, 복원을 표시합니다."""
    fig, axes = cp.flow([2.1, 2.1, 2.1, 2.1], height=3.0, wspace=0.5)
    cp.draw_image(axes[0], image, "입력 · 784화소")
    cp.draw_strip(axes[1], hidden, "은닉 $A$ · tanh", kind="signed", vertical=False)
    cp.draw_strip(axes[2], latent, "병목 $Z$ · 16좌표", kind="signed", vertical=False)
    cp.draw_image(axes[3], reconstructed, "초기 복원 · 784화소")
    cp.connect(fig, axes[0], axes[1], "$W_1,b_1$")
    cp.connect(fig, axes[1], axes[2], "$W_z,b_z$")
    cp.connect(fig, axes[2], axes[3], "decoder")
    cp.label(axes[3], "표시만 [0, 1]로 제한 · 손실은 원래 값")
    cp.show(fig)


def plot_comparison_setup(image, mean_image, pca_image, untrained_image, latent_dim):
    """같은 예시 한 장에 대한 평균·PCA·미학습 복원을 표시합니다."""
    fig, axes = cp.flow([2.2, 2.2, 2.2, 2.2], height=3.2, wspace=0.35)
    titles = ["고정 예시 · 목표 $x$", "train 평균만 복원", f"PCA · L={latent_dim}", f"미학습 AE · L={latent_dim}"]
    for ax, values, title in zip(axes, [image, mean_image, pca_image, untrained_image], titles):
        cp.draw_image(ax, values, title)
    cp.label(axes[0], "이 그림은 평가 표본이 아닌 고정 예시입니다")
    cp.show(fig)


def plot_training_history(history, title):
    """전달받은 epoch별 train/validation MSE 기록을 그립니다."""
    fig = cp.new_figure((6.8, 3.2))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "epoch", "화소당 MSE", title)
    ax.plot(history[:, 0], history[:, 1], color=cp.MUTED, label="train · 해당 입력 조건")
    ax.plot(history[:, 0], history[:, 2], color=cp.BLUE, label="validation · 고정 입력 조건")
    ax.legend(frameon=False, fontsize=9)
    cp.show(fig)


def plot_reconstruction_rows(rows, labels, title):
    """이미지 배열 (행, 장수, 28, 28)을 받아 비교 격자를 그립니다."""
    rows = np.asarray(rows)
    nrows, ncols = rows.shape[:2]
    fig = cp.new_figure((1.25 * ncols + 1.8, 1.45 * nrows + 0.6))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    fig.suptitle(title, color=cp.INK, fontsize=12, y=0.98)
    for i in range(nrows):
        for j in range(ncols):
            cp.blank(axes[i, j])
            cp.draw_image(axes[i, j], rows[i, j])
            if j == 0:
                axes[i, j].text(-0.13, 0.5, labels[i], transform=axes[i, j].transAxes,
                                ha="right", va="center", color=cp.INK, fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.04, hspace=0.12, wspace=0.08)
    cp.show(fig)


def plot_denoising_flow(clean, noise, noisy, target):
    """깨끗한 그림·추가 잡음·실제 입력·학습 타깃을 구분해 표시합니다."""
    fig, axes = cp.flow([2.1, 2.1, 2.1, 2.1], height=3.2, wspace=0.4)
    cp.draw_image(axes[0], clean, "깨끗한 $x$")
    cp.draw_image(axes[1], noise, r"고정한 $\sigma\epsilon$", kind="signed")
    cp.draw_image(axes[2], noisy, r"입력 $\widetilde{x}$")
    cp.draw_image(axes[3], target, "타깃은 깨끗한 $x$")
    cp.connect(fig, axes[0], axes[1], "+")
    cp.connect(fig, axes[1], axes[2], "섞기")
    cp.connect(fig, axes[2], axes[3], "학습 목표")
    cp.label(axes[2], "입력의 범위는 자르지 않음 · 그림만 [0, 1]")
    cp.show(fig)
