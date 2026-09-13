"""개념 설명용 최소 그림 문법.

"왜 이런 수식을 쓰는가", "이 모델이 어떻게 학습하나"를 MNIST 배열로 보여 주는
작은 부품들입니다. 축·격자·눈금·colorbar를 끄고, 숫자는 화소·띠·막대처럼
물건으로 그립니다. 계산은 노트북 본문에서 하고, 여기서는 받은 배열만 그립니다.

부품
----
- ``flow(widths)``            : 왼쪽에서 오른쪽으로 흐르는 패널 묶음을 만든다.
- ``draw_image(ax, img)``     : 28×28 같은 2차원 배열을 화소로 그린다. kind="gray" | "signed".
- ``draw_strip(ax, v)``       : 1차원 벡터를 가로·세로 띠로 그린다.
- ``draw_grid(ax, M)``        : 작은 행렬을 칸으로 그리고 값을 적는다.
- ``draw_bars(ax, values)``   : 값 여러 개를 가로 막대로 그린다. 강조 인덱스를 받는다.
- ``draw_rows(ax, X)``        : (N, D) 배치를 얇은 행 N개로 그린다.
- ``draw_formula(ax, tex)``   : 패널 사이에 수식 한 줄과 짧은 설명을 둔다.
- ``match_height(ax, ref)``   : 띠 패널의 높이를 옆 이미지 패널에 맞춘다.
- ``connect(fig, a, b)``      : 두 패널 사이에 화살표를 긋는다.
- ``label(ax, text)``         : 패널 아래에 회색 주석을 적는다.

색은 의미가 고정되어 있습니다. 양수는 파랑, 음수는 빨강, 강조는 금색, 나머지는 회색입니다.
설계 원칙과 노트북별 그림 목록은 docs/DESIGN.md에 있습니다.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = "#2b2d42"      # 제목·수식
MUTED = "#9aa0a6"    # 화살표·보조 설명·강조되지 않은 막대
BLUE = "#3b82c4"     # 양수·정답·선택된 것
RED = "#d1495b"      # 음수·오차
GOLD = "#d9a63a"     # 한 번에 하나만 강조
SIGNED_CMAP = "RdBu"   # 음수 빨강 · 양수 파랑
GRAY_CMAP = "gray_r"   # 0 흰색 · 1 검정 (잉크처럼 보이게)


def flow(widths, *, rows=1, height=3.2, wspace=0.12, hspace=0.35, unit=1.0):
    """폭 비율 목록으로 가로 흐름 패널을 만든다. (fig, axes) 반환. 모든 축은 비어 있다.

    rows=1이면 axes는 패널 목록, rows>1이면 행별 목록의 목록입니다.
    """
    widths = list(widths)
    fig = new_figure((unit * sum(widths) * 1.15 + 0.6, height * rows))
    gs = fig.add_gridspec(rows, len(widths), width_ratios=widths, wspace=wspace, hspace=hspace)
    grid = [[fig.add_subplot(gs[r, i]) for i in range(len(widths))] for r in range(rows)]
    for row in grid:
        for ax in row:
            blank(ax)
    return fig, (grid[0] if rows == 1 else grid)


def new_figure(figsize):
    """흰 배경의 빈 그림. 전역 autolayout을 끄고 gridspec 위치를 그대로 써서 화살표 좌표가 어긋나지 않게 한다."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.set_layout_engine("none")
    return fig


def blank(ax):
    """축·격자·눈금을 모두 끈다. 배열을 물건처럼 보이게 하는 첫 단계."""
    ax.set_axis_off()
    ax.grid(False)
    return ax


def _title(ax, text):
    if text:
        ax.set_title(text, color=INK, fontsize=10.5, pad=6)


def draw_image(ax, img, title=None, *, kind="gray", vmax=None, frame=True):
    """2차원 배열을 화소로 그린다.

    kind="gray"   : 0~1 잉크 (0 흰색, 1 검정)
    kind="signed" : 0을 중심으로 음수 빨강, 양수 파랑. vmax는 |값|의 상한.
    """
    a = np.asarray(img, dtype=float)
    if kind == "gray":
        ax.imshow(a, cmap=GRAY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    else:
        v = float(np.abs(a).max()) if vmax is None else vmax
        v = v if v > 0 else 1.0
        ax.imshow(a, cmap=SIGNED_CMAP, vmin=-v, vmax=v, interpolation="nearest")
    if frame:
        h, w = a.shape
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, ec=MUTED, lw=0.8))
    _title(ax, title)
    return ax


def draw_strip(ax, v, title=None, *, kind="gray", vertical=True, vmax=None, every=1):
    """1차원 벡터를 띠로 그린다. 긴 벡터는 every개마다 하나만 보여 길이를 암시한다."""
    a = np.asarray(v, dtype=float)[::every]
    a = a[:, None] if vertical else a[None, :]
    if kind == "gray":
        ax.imshow(a, cmap=GRAY_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    else:
        m = float(np.abs(a).max()) if vmax is None else vmax
        m = m if m > 0 else 1.0
        ax.imshow(a, cmap=SIGNED_CMAP, vmin=-m, vmax=m, aspect="auto", interpolation="nearest")
    _title(ax, title)
    return ax


def draw_grid(ax, M, title=None, *, kind="signed", fmt="{:g}", vmax=None, highlight=None,
              row_labels=None, col_labels=None, fontsize=10, frame_to=None):
    """작은 행렬을 칸으로 그리고 값을 적는다. highlight는 (행, 열) 목록.

    frame_to=(열 수, 행 수)를 주면 그 크기의 틀 가운데에 놓아, 나란한 여러 판의 칸 크기를 같게 만든다.
    """
    a = np.atleast_2d(np.asarray(M, dtype=float))
    if kind == "signed":
        m = float(np.abs(a).max()) if vmax is None else vmax
        ax.imshow(a, cmap=SIGNED_CMAP, vmin=-(m or 1), vmax=(m or 1), alpha=0.55, interpolation="nearest")
    elif kind == "plain":
        ax.imshow(np.zeros_like(a), cmap="Greys", vmin=0, vmax=1, alpha=0.0, interpolation="nearest")
        for (i, j) in np.ndindex(*a.shape):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=MUTED, lw=0.6))
    else:
        ax.imshow(a, cmap="Blues", vmin=0, vmax=(vmax or a.max() or 1), alpha=0.55, interpolation="nearest")
    for (i, j), val in np.ndenumerate(a):
        ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=fontsize, color=INK)
    for (i, j) in (highlight or []):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=GOLD, lw=2.2))
    h, w = a.shape
    ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, ec=MUTED, lw=0.8))
    ax.set_axis_on()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.set_xticks(range(w), col_labels if col_labels is not None else [""] * w)
    ax.set_yticks(range(h), row_labels if row_labels is not None else [""] * h)
    ax.xaxis.tick_top()
    if frame_to is not None:
        fw, fh = frame_to
        cx, cy = (w - 1) / 2, (h - 1) / 2
        ax.set_xlim(cx - fw / 2, cx + fw / 2)
        ax.set_ylim(cy + fh / 2, cy - fh / 2)
    _title(ax, title)
    return ax


def draw_bars(ax, values, title=None, *, labels=None, highlight=None, marker=None,
              xlim=None, fmt="{:.2f}", color=MUTED, signed=False):
    """값들을 가로 막대로 그린다. highlight는 파랑, marker는 금색 테두리(정답 표시용)."""
    v = np.asarray(values, dtype=float)
    k = np.arange(len(v))
    colors = [BLUE if i == highlight else (RED if signed and x < 0 else color) for i, x in zip(k, v)]
    ax.set_axis_on()
    ax.grid(False)
    ax.barh(k, v, color=colors, height=0.72)
    if marker is not None:
        ax.add_patch(Rectangle((min(0, v[marker]), marker - 0.36), abs(v[marker]) or 1e-9, 0.72,
                               fill=False, ec=GOLD, lw=2.0))
    ax.set_yticks(k, labels if labels is not None else [str(i) for i in k])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.tick_params(length=0, labelsize=9, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    if signed:
        ax.axvline(0, color=MUTED, lw=0.8)
    lo, hi = (xlim if xlim is not None else (min(0, v.min()), max(0, v.max())))
    pad = 0.18 * (hi - lo or 1)
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    if highlight is not None:
        ax.text(v[highlight] + 0.02 * (hi - lo or 1), highlight, fmt.format(v[highlight]),
                va="center", color=BLUE, fontsize=9)
    _title(ax, title)
    return ax


def draw_rows(ax, X, title=None, *, kind="gray", vmax=None):
    """(N, D) 배치를 얇은 가로 행 N개로 그린다. 행 하나가 샘플 하나."""
    a = np.asarray(X, dtype=float)
    if kind == "gray":
        ax.imshow(a, cmap=GRAY_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    else:
        m = float(np.abs(a).max()) if vmax is None else vmax
        ax.imshow(a, cmap=SIGNED_CMAP, vmin=-(m or 1), vmax=(m or 1), aspect="auto", interpolation="nearest")
    n = a.shape[0]
    for i in range(1, n):
        ax.axhline(i - 0.5, color="white", lw=1.0)
    _title(ax, title)
    return ax


def match_height(ax, ref):
    """ax의 세로 범위를 ref(정사각형 이미지 축)의 줄어든 높이에 맞춘다. 띠를 이미지 옆에 나란히 둘 때 쓴다."""
    ref.apply_aspect()
    r, p = ref.get_position(), ax.get_position()
    ax.set_position([p.x0, r.y0, p.width, r.height])
    return ax


def draw_formula(ax, tex, sub=None, *, fontsize=12):
    """빈 패널에 수식 한 줄과 회색 설명을 둔다."""
    ax.text(0.5, 0.62, tex, ha="center", va="center", fontsize=fontsize, color=INK, transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.38, sub, ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    return ax


def label(ax, text, *, y=-0.06, color=MUTED, fontsize=8.5):
    """패널 아래(또는 y 위치)에 짧은 주석을 적는다."""
    ax.text(0.5, y, text, ha="center", va="top", fontsize=fontsize, color=color, transform=ax.transAxes)
    return ax


def _extent(fig, ax):
    """패널의 경계(그림 좌표). 정사각형 축은 줄어든 위치를 쓰고, 축 밖의 눈금 글자(막대의 라벨)는 포함한다.

    제목·주석은 패널보다 넓어도 화살표 길이에 영향을 주지 않도록 일부러 포함하지 않는다.
    """
    ax.apply_aspect()
    box = ax.get_position()
    if not ax.axison:                 # 축이 꺼진 패널(이미지·띠)은 그림 영역만 경계로 쓴다
        return box
    try:
        renderer = fig.canvas.get_renderer()
        for axis in (ax.xaxis, ax.yaxis):
            tb = axis.get_tightbbox(renderer)
            if tb is not None and tb.width > 0:
                box = box.union([box, tb.transformed(fig.transFigure.inverted())])
    except Exception:
        pass
    return box


def connect(fig, ax_from, ax_to, text=None, *, gap=0.01):
    """두 패널 사이에 가로 화살표를 긋는다. new_figure()로 만든 그림에서만 위치가 정확하다."""
    a, b = _extent(fig, ax_from), _extent(fig, ax_to)
    y = 0.5 * (a.y0 + a.y1)
    fig.patches.append(FancyArrowPatch((a.x1 + gap, y), (b.x0 - gap, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.2))
    if text:
        fig.text(0.5 * (a.x1 + b.x0), y + 0.025, text, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return fig


def show(fig):
    """노트북에서 그림을 내보내고 닫는다."""
    plt.show()
    plt.close(fig)
