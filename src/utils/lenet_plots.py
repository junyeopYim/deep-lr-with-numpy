"""05b LeNet 노트북의 개념 그림과 구조 도식.

padding 격자·특성 맵·수용 영역·데이터 분할·학습 결과·이동 반응·혼동 행렬은 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import math

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .architecture_plots import plot_curves  # noqa: F401  (노트북이 한 파일에서 가져가도록)
from .gradient_plots import _clean_axes

PAD_FILL = "#eeeeee"      # 0을 두른 칸
PARAM_FILL = "#fce8c3"    # 파라미터가 있는 블록
UNUSED_FILL = "#e5e7eb"   # 쓰지 않는 데이터


def _frame(ax, w, h, size):
    """draw_cells 격자를 size×size 틀 가운데에 두어 옆 판과 칸 크기를 맞춘다."""
    cx, cy = w / 2, h / 2
    ax.set_xlim(cx - size / 2, cx + size / 2)
    ax.set_ylim(cy - size / 2, cy + size / 2)
    return ax


def _mosaic(maps, cols, gap=1):
    """(F, h, w) 특성 맵 F장을 cols열의 격자로 이어 붙인 2차원 배열. 칸 사이는 NaN(흰색)."""
    maps = np.asarray(maps, float)
    F, h, w = maps.shape
    rows = math.ceil(F / cols)
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), np.nan)
    for f in range(F):
        r, c = divmod(f, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = maps[f]
    return out


# ---------------------------------------------------------------- 1절
def plot_padding_scene(X, Xp, K, Y, window):
    """입력 → 0을 두른 격자(회색 칸) 위의 금색 창 → 커널 → 입력과 같은 크기의 출력. padding이 크기를 지키는 장면."""
    X, Xp, K, Y = (np.asarray(a, float) for a in (X, Xp, K, Y))
    h, w = X.shape
    ph, pw = Xp.shape
    kh, kw = K.shape
    i, j = window
    pad = (ph - h) // 2
    size = ph + 0.3
    fig = cp.new_figure((12.5, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.45)
    ax_x = fig.add_subplot(gs[0])
    sp.draw_cells(ax_x, X, f"입력 $X$ ({h}×{w})")
    _frame(ax_x, w, h, size)
    colors = [[PAD_FILL if (r < pad or r >= pad + h or c < pad or c >= pad + w) else "white" for c in range(pw)]
              for r in range(ph)]
    ax_p = fig.add_subplot(gs[1])
    sp.draw_cells(ax_p, Xp, f"0을 두른 $X^{{pad}}$ ({ph}×{pw}), $p={pad}$", window=(i, j, kh, kw), fill=cp.GOLD,
                  cell_colors=colors)
    cp.label(ax_p, f"회색 = 두른 0 · 창의 가운데가 원래 ({i}, {j})", y=-0.04)
    ax_k = fig.add_subplot(gs[2])
    sp.draw_cells(ax_k, K, f"커널 $K$ ({kh}×{kw})")
    _frame(ax_k, kw, kh, size)
    ax_y = fig.add_subplot(gs[3])
    sp.draw_cells(ax_y, Y, f"출력 $Y$ ({Y.shape[0]}×{Y.shape[1]}) = 입력과 같은 크기", highlight_cells=[(i, j)])
    _frame(ax_y, Y.shape[1], Y.shape[0], size)
    cp.connect(fig, ax_x, ax_p, f"사방에 0을 {pad}겹")
    cp.connect(fig, ax_p, ax_k, "창 ⊙ 커널")
    cp.connect(fig, ax_k, ax_y, f"합 = {Y[i, j]:g}")
    cp.show(fig)


def plot_feature_flow(maps, titles, arrows, rf_box, cell):
    """예시 7이 conv → pool → conv → pool을 지나며 작아지는 특성 맵. 마지막 금색 칸과 그 칸이 보는 입력 창(금색 틀)."""
    maps = [np.asarray(m, float) for m in maps]
    sizes = [m.shape[0] for m in maps]
    widths = [max(0.5, s / sizes[0]) for s in sizes]
    fig, axes = cp.flow(widths, height=3.4, unit=2.4, wspace=0.4)
    for k, (ax, m, title) in enumerate(zip(axes, maps, titles)):
        if k == 0:
            cp.draw_image(ax, m, f"{title} ({m.shape[0]}×{m.shape[1]})")
        else:
            cp.draw_image(ax, m, f"{title} ({m.shape[0]}×{m.shape[1]})", kind="signed")
    top, left, hh, ww = rf_box
    axes[0].add_patch(Rectangle((left - 0.5, top - 0.5), ww, hh, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
    ci, cj = cell
    axes[-1].add_patch(Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False, ec=cp.GOLD, lw=2.2, zorder=3))
    for a, b, text in zip(axes, axes[1:], arrows):
        cp.connect(fig, a, b, text)
    cp.label(axes[0], f"금색 틀 {hh}×{ww} = 맨 오른쪽 금색 칸 {cell}이 보는 입력", y=-0.05)
    cp.label(axes[-1], "파랑 양수 · 빨강 음수", y=-0.05)
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_lenet_blocks(top_blocks, bottom_blocks, param_top=(), param_bottom=()):
    """두 줄의 블록 흐름: 윗줄 conv 두 단, 아랫줄 펼침과 완전 연결 세 층. 파라미터가 있는 블록은 옅은 금색."""
    fig = cp.new_figure((13.5, 4.4))
    gs = fig.add_gridspec(2, 1, hspace=0.5)
    for row, (blocks, params) in enumerate(((top_blocks, param_top), (bottom_blocks, param_bottom))):
        ax = fig.add_subplot(gs[row])
        colors = [PARAM_FILL if k in params else sp.FILL for k in range(len(blocks))]
        sp.draw_blocks(ax, blocks, colors=colors, width=1.75, gap=0.5, fontsize=9.5)
    fig.text(0.5, 0.5, "윗줄의 끝 (N, 16, 5, 5)이 아랫줄의 시작입니다", ha="center", va="center", fontsize=8.5, color=cp.MUTED)
    cp.label(fig.axes[-1], "금색 상자 = 학습하는 파라미터가 있는 층(다섯) · 상자 아래 = 그 단계의 텐서 모양", y=-0.02)
    cp.show(fig)


def plot_lenet_trace(image, maps1, maps2, hidden3, hidden4, logits, label):
    """예시 한 장이 다섯 층을 지나는 값: 입력 → conv1 특성 맵 6장 → pool2 특성 맵 16장 → 은닉 120·84 띠 → logit 막대."""
    maps1, maps2 = np.asarray(maps1, float), np.asarray(maps2, float)
    fig, axes = cp.flow([1, 1.5, 0.95, 0.2, 0.2, 1.05], height=3.4, unit=2.2, wspace=0.7)
    cp.draw_image(axes[0], image, f"$x_{{{label}}}$ (28×28)")
    m1 = _mosaic(maps1, 3)
    cp.draw_image(axes[1], m1, f"conv1 특성 맵 {len(maps1)}장 (28×28)", kind="signed", vmax=float(np.nanmax(np.abs(m1))))
    m2 = _mosaic(maps2, 4)
    cp.draw_image(axes[2], m2, f"pool2 특성 맵 {len(maps2)}장 (5×5)", kind="signed", vmax=float(np.nanmax(np.abs(m2))))
    cp.draw_strip(axes[3], hidden3, f"$A_3$ ({len(hidden3)},)", kind="signed")
    cp.draw_strip(axes[4], hidden4, f"$A_4$ ({len(hidden4)},)", kind="signed")
    cp.match_height(axes[3], axes[0])
    cp.match_height(axes[4], axes[0])
    cp.draw_bars(axes[5], logits, "logit 10개", labels=[str(k) for k in range(10)], highlight=int(np.argmax(logits)),
                 marker=label, signed=True)
    cp.match_height(axes[5], axes[0])
    cp.connect(fig, axes[0], axes[1], "conv1")
    cp.connect(fig, axes[1], axes[2], "pool → conv2\n→ pool")
    cp.connect(fig, axes[2], axes[3], "펼침 → FC")
    cp.connect(fig, axes[4], axes[5], "FC")
    cp.label(axes[5], "파랑 = 가장 큰 logit · 금색 테두리 = 정답", y=-0.06)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_data_plan(counts, n_test, n_train, n_steps, batch_size, last_batch, batch_images):
    """왼쪽: 학습 6만 장을 학습·검증·안 씀으로 나눈 띠와 따로 둔 test. 가운데: 한 에폭을 B장씩 끊은 스텝. 오른쪽: 첫 배치."""
    fig = cp.new_figure((13.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    cp.blank(ax)
    total = sum(counts.values())
    x = 0.0
    band_color = {"학습": cp.BLUE, "검증": cp.GOLD, "안 씀": UNUSED_FILL}   # 색은 자리가 아니라 이름으로 정한다
    for name, n in counts.items():
        color = next(c for key, c in band_color.items() if name.startswith(key))
        ax.add_patch(Rectangle((x / total, 0.55), n / total, 0.28, facecolor=color, edgecolor="white"))
        if n / total < 0.08:                                  # 좁은 구간의 이름은 띠 위에 적고 선으로 잇는다
            center = (x + n / 2) / total
            ax.plot([center] * 2, [0.83, 0.9], color=color, lw=1.0)
            ax.text(center, 0.91, name, ha="right" if center > 0.9 else "center",   # 오른쪽 끝이면 글자를 안쪽으로
                    va="bottom", fontsize=8.5, color=cp.INK)
        else:
            ax.text((x + n / 2) / total, 0.69, name, ha="center", va="center", fontsize=8.5,
                    color=cp.INK if color == UNUSED_FILL else "white")
        x += n
    ax.add_patch(Rectangle((0, 0.12), n_test / total, 0.28, facecolor=cp.RED, edgecolor="white"))
    ax.text(n_test / total + 0.02, 0.26, f"test {n_test:,}장 · 5절에서 한 번만", va="center", fontsize=8.5, color=cp.INK)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"학습 집합 {total:,}장: 앞에서 학습, 뒤에서 검증", color=cp.INK, fontsize=10.5, pad=6)

    ax = fig.add_subplot(gs[1])
    cp.blank(ax)
    shown = [(1, batch_size), (2, batch_size), (3, batch_size), (4, batch_size), None, (n_steps - 1, batch_size), (n_steps, last_batch)]
    x = 0.0
    for item in shown:
        if item is None:
            ax.text(x + 0.3, 0.5, "⋯", ha="center", va="center", fontsize=14, color=cp.INK)
            x += 0.7
            continue
        step, b = item
        w = b / batch_size
        ax.add_patch(Rectangle((x, 0.3), w, 0.4, facecolor=cp.GOLD if b < batch_size else cp.BLUE, edgecolor="white"))
        ax.text(x + w / 2, 0.5, f"{b}장", ha="center", va="center", fontsize=8.5, color="white")
        ax.text(x + w / 2, 0.24, str(step), ha="center", va="top", fontsize=8, color=cp.MUTED)
        x += w + 0.1
    ax.set_xlim(-0.1, x)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"한 에폭 = ⌈{n_train:,}/{batch_size}⌉ = {n_steps}스텝 (아래 숫자 = 스텝 번호)", color=cp.INK, fontsize=10.5, pad=6)
    cp.label(ax, f"에폭마다 순서를 섞고 {batch_size}장씩 끊음 · 마지막 배치 {last_batch}장(금색)", y=-0.02)

    ax = fig.add_subplot(gs[2])
    cp.blank(ax)
    cp.draw_image(ax, _mosaic(batch_images, 8), f"첫 배치 {len(batch_images)}장 (64, 1, 28, 28)")
    cp.show(fig)


def plot_training_result(losses, window, epoch_steps, valid_acc):
    """왼쪽: 스텝별 배치 CE(회색)와 이동평균(파랑). 오른쪽: 에폭 끝마다 잰 검증 정확도."""
    losses = np.asarray(losses, float)
    fig = cp.new_figure((11, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "스텝", "배치 CE", "학습 손실")
    ax.plot(np.arange(1, len(losses) + 1), losses, color=cp.MUTED, lw=0.6, alpha=0.6, label="배치 64장의 CE")
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(losses) + 1), smooth, color=cp.BLUE, lw=1.8, label=f"{window}스텝 이동평균")
    for s in epoch_steps[:-1]:
        ax.axvline(s, color=cp.GOLD, lw=1.0, ls=":")
    ax.set_ylim(0, min(2.5, float(losses.max()) * 1.05))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "에폭", "검증 정확도", "검증 2,000장 (갱신에 쓰지 않음)")
    epochs = np.arange(1, len(valid_acc) + 1)
    ax2.plot(epochs, valid_acc, "o-", color=cp.BLUE, lw=1.6, ms=6)
    for e, a in zip(epochs, valid_acc):
        ax2.text(e, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    ax2.set_xticks(epochs)
    ax2.set_ylim(min(valid_acc) - 0.03, 1.0)
    cp.show(fig)


def plot_kernels_before_after(K_before, K_after, K_delta):
    """conv1 커널 6개를 학습 전(윗줄)·학습 뒤(가운데)·학습이 더한 변화(아랫줄, 뒤 − 전)로. 파랑 양수 · 빨강 음수, 줄마다 같은 색 눈금."""
    rows = [(np.asarray(K_before, float), "학습 전"), (np.asarray(K_after, float), "학습 뒤"), (np.asarray(K_delta, float), "변화 (뒤 − 전)")]
    n = len(rows[0][0])
    fig = cp.new_figure((1.55 * n + 1.4, 5.2))
    gs = fig.add_gridspec(3, n, wspace=0.15, hspace=0.3, left=0.12)
    for row, (K, name) in enumerate(rows):
        vmax = float(np.abs(K).max())
        for k in range(n):
            ax = fig.add_subplot(gs[row, k])
            cp.blank(ax)
            cp.draw_image(ax, K[k], f"커널 {k + 1}" if row == 0 else None, kind="signed", vmax=vmax)
        fig.text(0.02, 0.81 - 0.3 * row, f"{name}\n|값| 최대 {vmax:.2f}", ha="left", va="center", fontsize=9, color=cp.INK)
    cp.label(fig.axes[-1], "줄마다 색 눈금이 다름: 셋째 줄은 첫째 줄보다 작은 값을 같은 진하기로", y=-0.12)
    cp.show(fig)


def plot_feature_maps(image, maps, label):
    """예시 한 장과 학습된 conv1 커널 6개의 특성 맵. 같은 색 눈금."""
    maps = np.asarray(maps, float)
    fig, axes = cp.flow([1] * (len(maps) + 1), height=2.6, unit=1.55, wspace=0.5)
    cp.draw_image(axes[0], image, f"$x_{{{label}}}$")
    vmax = float(np.abs(maps).max())
    for k, ax in enumerate(axes[1:]):
        cp.draw_image(ax, maps[k], f"커널 {k + 1}의 특성 맵", kind="signed", vmax=vmax)
    cp.connect(fig, axes[0], axes[1], "conv1")
    cp.label(axes[-1], "파랑 = 커널과 잘 맞는 자리", y=-0.06)
    cp.show(fig)


# ---------------------------------------------------------------- 4절
def plot_shift_response(images, maps, probs, shifts, peak, label):
    """행마다 입력을 오른쪽으로 d칸 옮긴 것: 입력 · 학습된 conv1 특성 맵(금색 칸 = 원본 최댓값 자리를 d칸 옮긴 곳) · 예측 확률."""
    maps = [np.asarray(m, float) for m in maps]
    probs = np.asarray(probs, float)
    rows = len(shifts)
    vmax = float(max(np.abs(m).max() for m in maps))
    fig = cp.new_figure((9.5, 2.35 * rows))
    gs = fig.add_gridspec(rows, 3, width_ratios=[1, 1, 1.3], wspace=0.55, hspace=0.3)
    pi, pj = int(peak[0]), int(peak[1])
    for r, (img, m, p, d) in enumerate(zip(images, maps, probs, shifts)):
        ax_x = fig.add_subplot(gs[r, 0])
        cp.blank(ax_x)
        cp.draw_image(ax_x, img, "원본" if d == 0 else f"오른쪽으로 {d}칸")
        ax_m = fig.add_subplot(gs[r, 1])
        cp.blank(ax_m)
        cp.draw_image(ax_m, m, "학습된 conv1 특성 맵" if r == 0 else None, kind="signed", vmax=vmax)
        ax_m.add_patch(Rectangle((pj + d - 0.5, pi - 0.5), 1, 1, fill=False, ec=cp.GOLD, lw=2.0, zorder=3))
        ax_b = fig.add_subplot(gs[r, 2])
        cp.blank(ax_b)
        cp.draw_bars(ax_b, p, "softmax 확률" if r == 0 else None, labels=[str(k) for k in range(10)],
                     highlight=int(np.argmax(p)), marker=label, xlim=(0, 1), fmt="{:.3f}")
        cp.match_height(ax_b, ax_x)
        if r == 0:
            cp.connect(fig, ax_x, ax_m, "conv1")
            cp.connect(fig, ax_m, ax_b, "pool·conv2\n·pool·FC")
    cp.label(fig.axes[-2], f"금색 칸 = 원본의 최댓값 자리 ({pi}, {pj})를 함께 옮긴 것", y=-0.08)
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def plot_confusion_matrix(confusion, accuracy):
    """10×10 혼동 행렬을 축 없이. 색은 틀린 칸의 장수 기준(대각선은 포화)."""
    confusion = np.asarray(confusion)
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    fig = cp.new_figure((6.4, 5.8))
    ax = fig.add_subplot(111)
    cp.blank(ax)
    cp.draw_grid(ax, confusion, f"test {int(confusion.sum()):,}장의 혼동 행렬 · 정확도 {accuracy:.4f}", kind="count",
                 fmt="{:g}", vmax=max(int(off.max()), 1), fontsize=8.5,
                 row_labels=[f"정답 {k}" for k in range(10)], col_labels=[f"예측 {k}" for k in range(10)])
    cp.label(ax, "행 = 정답, 열 = 예측 · 색 = 틀린 칸의 장수 (대각선은 맞힌 것, 색 포화)", y=-0.03)
    cp.show(fig)


def plot_wrong_examples(images, titles):
    """가장 자신 있게 틀린 예 8장을 두 줄로. 제목은 정답·예측·확률."""
    cols = math.ceil(len(images) / 2)
    fig, grid = cp.flow([1] * cols, rows=2, height=2.5, unit=1.5, wspace=0.2, hspace=0.45)
    axes = [ax for row in grid for ax in row]
    for ax, img, title in zip(axes, images, titles):
        cp.draw_image(ax, img, title)
    for ax in axes[len(images):]:
        ax.set_visible(False)
    cp.show(fig)
