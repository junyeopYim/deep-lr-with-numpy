"""01b 미니배치 노트북의 개념 그림. 시간·기울기·손실 값은 본문에서 계산하고 여기서는 받은 배열만 그립니다."""

import numpy as np

from . import concept_plots as cp
from . import schematic_plots as sp
from .gradient_plots import _clean_axes


def plot_step_cost(sizes, times):
    """배치 크기별 한 스텝 시간(ms, 로그 눈금)과 1초에 가능한 갱신 횟수."""
    sizes = np.asarray(sizes); ms = np.asarray(times) * 1e3
    fig, axes = cp.flow([1, 1], height=3.2, unit=2.6, wspace=0.5)
    _clean_axes(axes[0], "한 스텝에 쓰는 이미지 수 $B$", "시간 (ms)", "한 스텝의 시간")
    axes[0].loglog(sizes, ms, "o-", color=cp.BLUE, lw=1.6, ms=5)
    for s, m in zip(sizes, ms):
        axes[0].text(s, m * 1.6, f"{m:.3g}", ha="center", fontsize=8.5, color=cp.INK)
    _clean_axes(axes[1], "한 스텝에 쓰는 이미지 수 $B$", "갱신 횟수", "1초 동안 할 수 있는 갱신")
    axes[1].loglog(sizes, 1 / np.asarray(times), "s-", color=cp.RED, lw=1.6, ms=5)
    for s, t in zip(sizes, times):
        axes[1].text(s, (1 / t) * 1.6, f"{1/t:,.0f}", ha="center", fontsize=8.5, color=cp.INK)
    cp.show(fig)


def plot_batch_gradient_images(full_image, small_images, large_images, small_B, large_B, N):
    """전체 데이터의 기울기 이미지와, 작은 배치·큰 배치에서 계산한 기울기 이미지들."""
    k = len(small_images)
    fig = cp.new_figure((11.5, 5.2))
    gs = fig.add_gridspec(2, 1 + k, width_ratios=[1.15] + [1] * k, wspace=0.12, hspace=0.35)
    vmax = float(np.abs(full_image).max()) * 1.6
    ax = fig.add_subplot(gs[:, 0]); cp.blank(ax)
    cp.draw_image(ax, full_image, f"전체 {N:,}장의 기울기", kind="signed", vmax=vmax)
    cp.label(ax, "모든 배치 기울기의 기댓값", y=-0.05)
    for r, (imgs, B) in enumerate([(small_images, small_B), (large_images, large_B)]):
        for j, img in enumerate(imgs):
            a = fig.add_subplot(gs[r, 1 + j]); cp.blank(a)
            cp.draw_image(a, img, f"배치 {B}장 · {j+1}번째", kind="signed", vmax=vmax)
    cp.show(fig)


def plot_batch_noise(batch_sizes, observed, expected):
    """배치 크기에 따른 기울기 한 좌표의 표준편차: 관측값과 σ/√B."""
    fig = cp.new_figure((6.4, 3.4))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "배치 크기 $B$", "기울기 한 좌표의 표준편차", "잡음은 $1/\\sqrt{B}$로 줄어듭니다")
    ax.loglog(batch_sizes, observed, "o", color=cp.BLUE, ms=6, label="배치 2,000개에서 관측")
    ax.loglog(batch_sizes, expected, "--", color=cp.GOLD, lw=1.6, label="$\\sigma/\\sqrt{B}$ (이론)")
    ax.legend(frameon=False, fontsize=9)
    cp.show(fig)


def plot_epoch_batches(orders, batch_size):
    """에폭마다 섞인 표본 번호를 배치 크기로 끊어 색칠합니다."""
    palette = ["#cfe0f3", "#f6d5d5", "#d9ecd0", "#fce8c3", "#e3d9f2", "#d0ecec"]
    fig = cp.new_figure((11, 0.9 * len(orders) + 0.8))
    gs = fig.add_gridspec(len(orders), 1, hspace=0.6)
    for e, order in enumerate(orders):
        ax = fig.add_subplot(gs[e])
        colors = [[palette[(i // batch_size) % len(palette)] for i in range(len(order))]]
        sp.draw_cells(ax, np.asarray(order)[None, :], f"에폭 {e+1}: 섞은 순서를 {batch_size}개씩 끊음", cell_colors=colors, fontsize=9)
        for k in range(batch_size, len(order), batch_size):
            ax.axvline(k, color=cp.INK, lw=1.4)
    cp.label(fig.axes[-1], "같은 색 = 같은 배치 · 마지막 배치는 작을 수 있음 · 에폭마다 순서가 다름", y=-0.35)
    cp.show(fig)


def plot_loss_vs_seen(curves, batch_trace=None, title="같은 표본 처리량에서의 전체 손실"):
    """curves = {이름: (처리한 표본 수, 전체 손실)}. batch_trace = (표본 수, 배치 손실)을 옅게 겹칠 수 있음."""
    fig = cp.new_figure((7.2, 3.8))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "처리한 표본 수", "전체 60,000장의 손실", title)
    if batch_trace is not None:
        ax.plot(batch_trace[0], batch_trace[1], color=cp.MUTED, lw=0.6, alpha=0.6, label="현재 배치의 손실 (잡음)")
    for (name, (seen, loss)), color in zip(curves.items(), [cp.BLUE, cp.RED, cp.GOLD, "#2a9d8f"]):
        ax.plot(seen, loss, "o-" if len(seen) < 8 else "-", color=color, lw=1.8, ms=5, label=name)
    ax.legend(frameon=False, fontsize=9)
    cp.show(fig)


def plot_batch_size_effect(curves, times, images, batch_sizes):
    """왼쪽: 배치 크기별 손실 vs 처리한 표본 수. 가운데: 한 에폭 시간. 오른쪽: 한 에폭 뒤 가중치 이미지."""
    n = len(batch_sizes)
    fig = cp.new_figure((13, 3.6))
    gs = fig.add_gridspec(1, 2 + n, width_ratios=[2.2, 1.3] + [0.8] * n, wspace=0.45)
    colors = [cp.BLUE, cp.RED, cp.GOLD, "#2a9d8f", "#7b5ea7"]
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "처리한 표본 수", "전체 손실", "한 에폭 동안의 손실")
    for (B, (seen, loss)), color in zip(curves.items(), colors):
        ax.plot(seen, loss, "-", color=color, lw=1.6, label=f"B={B}")
    ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    _clean_axes(ax, "배치 크기 $B$", "초", "한 에폭에 걸린 시간")
    ax.bar([str(B) for B in batch_sizes], times, color=colors[:n], width=0.6)
    for i, t in enumerate(times):
        ax.text(i, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    vmax = float(max(np.abs(im).max() for im in images))
    for j, (B, img) in enumerate(zip(batch_sizes, images)):
        a = fig.add_subplot(gs[2 + j]); cp.blank(a)
        cp.draw_image(a, img, f"B={B}", kind="signed", vmax=vmax)
    cp.label(fig.axes[2], "한 에폭 뒤의 $w$", y=-0.08)
    cp.show(fig)
