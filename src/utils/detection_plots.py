"""18b 물체 검출(YOLO) 노트북의 개념 그림.

상자·IoU·격자 타깃·손실 항·NMS·precision–recall 값은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.

상자는 (cx, cy, w, h) 순서입니다. 화소 단위면 scale=1, 캔버스 비율이면 scale=캔버스 한 변을 줍니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .architecture_plots import plot_curves  # noqa: F401  (노트북이 한 파일에서 가져가도록)
from .gradient_plots import _clean_axes

PARAM_FILL = "#fce8c3"      # 파라미터가 있는 블록
OBJ_FILL = "#cfe0f3"        # 책임 셀 배경
TARGET_LABELS = ["x", "y", "w", "h", "C"] + [str(k) for k in range(10)]


def _box_patch(box, color, *, scale=1.0, lw=1.6, alpha=1.0, fill=False, ls="-"):
    """(cx, cy, w, h) 상자를 imshow 화소 좌표의 사각형으로. 화소 경계는 i − 0.5에 있으므로 0.5를 뺀다."""
    cx, cy, w, h = np.asarray(box, dtype=float) * scale
    return Rectangle((cx - w / 2 - 0.5, cy - h / 2 - 0.5), w, h, fill=fill, ec=color,
                     fc=color if fill else "none", lw=lw, alpha=alpha, ls=ls, zorder=3)


def _box_text(ax, box, text, color, *, scale=1.0, fontsize=9, above=True):
    """상자의 왼쪽 위(또는 왼쪽 아래)에 짧은 글자를 흰 바탕으로 적는다."""
    cx, cy, w, h = np.asarray(box, dtype=float) * scale
    y = cy - h / 2 - 0.8 if above else cy + h / 2 + 0.2
    ax.text(cx - w / 2 - 0.3, y, text, color=color, fontsize=fontsize, ha="left", va="bottom" if above else "top",
            zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85})


def _grid_lines(ax, size, S):
    """캔버스 위에 S×S 격자 선을 긋는다."""
    for k in range(1, S):
        ax.axhline(k * size / S - 0.5, color=cp.MUTED, lw=0.7, alpha=0.8, zorder=2)
        ax.axvline(k * size / S - 0.5, color=cp.MUTED, lw=0.7, alpha=0.8, zorder=2)


# ---------------------------------------------------------------- 📎 데이터: 캔버스와 정답 상자
def plot_canvases(canvases, boxes_list, classes_list, titles=None):
    """캔버스 여러 장과 정답 상자(파랑)·클래스 번호. boxes_list[k]는 (개수, 4) 캔버스 비율, classes_list[k]는 (개수,)."""
    n = len(canvases)
    size = canvases[0].shape[0]
    fig, axes = cp.flow([1] * n, height=3.4, wspace=0.15, unit=1.7)
    for k, ax in enumerate(axes):
        cp.draw_image(ax, canvases[k], titles[k] if titles else f"캔버스 {k}")
        for box, cls in zip(boxes_list[k], classes_list[k]):
            ax.add_patch(_box_patch(box, cp.BLUE, scale=size, lw=1.6))
            _box_text(ax, box, str(int(cls)), cp.BLUE, scale=size)
    cp.label(axes[n // 2], "파랑 = 정답 상자 (클래스, 중심 x, 중심 y, 너비, 높이) · 글자 = 클래스", y=-0.05)
    cp.show(fig)


# ---------------------------------------------------------------- 1절: IoU
def plot_iou_boxes(canvas, box_a, box_b, inter_corners, areas):
    """캔버스 위 상자 A(파랑)·B(빨강) → 교집합 사각형(금색) → IoU 식. areas = (A 넓이, B 넓이, 교집합 넓이). 화소 단위."""
    area_a, area_b, area_i = areas
    union = area_a + area_b - area_i
    fig, (ax0, ax1, ax2) = cp.flow([1, 1, 1.15], height=3.4, unit=2.3)
    cp.draw_image(ax0, canvas, "상자 A (파랑)와 B (빨강)")
    ax0.add_patch(_box_patch(box_a, cp.BLUE))
    ax0.add_patch(_box_patch(box_b, cp.RED))
    _box_text(ax0, box_a, "A", cp.BLUE)
    _box_text(ax0, box_b, "B", cp.RED, above=False)
    cp.label(ax0, f"|A| = {area_a:g} · |B| = {area_b:g}", y=-0.05)
    cp.draw_image(ax1, canvas, "겹친 부분 = 교집합 (금색)")
    ax1.add_patch(_box_patch(box_a, cp.BLUE, alpha=0.6))
    ax1.add_patch(_box_patch(box_b, cp.RED, alpha=0.6))
    x1, y1, x2, y2 = inter_corners
    ax1.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fc=cp.GOLD, ec=cp.GOLD, alpha=0.55, lw=1.5, zorder=4))
    cp.label(ax1, f"|A ∩ B| = {area_i:g} · |A ∪ B| = {area_a:g} + {area_b:g} − {area_i:g} = {union:g}", y=-0.05)
    cp.draw_formula(ax2, f"$\\mathrm{{IoU}} = \\frac{{{area_i:g}}}{{{union:g}}} = {area_i / union:g}$", "교집합 넓이 ÷ 합집합 넓이", fontsize=13)
    cp.connect(fig, ax0, ax1, "같은 상자")
    cp.connect(fig, ax1, ax2, "넓이의 비")
    cp.show(fig)


# ---------------------------------------------------------------- 2절: 격자와 타깃
def plot_grid_targets(canvas, S, boxes, classes, cells, highlight_cell, obj_grid, target_vec):
    """캔버스 + S×S 격자 + 정답 상자(파랑)·중심점 → 책임 셀 mask 격자 → 금색 셀의 타깃 벡터 막대. boxes는 캔버스 비율."""
    size = canvas.shape[0]
    fig, (ax0, ax1, ax2) = cp.flow([1.5, 0.9, 1.1], height=3.8, wspace=0.25, unit=2.4)
    cp.draw_image(ax0, canvas, f"캔버스 {size}×{size} · 격자 {S}×{S} · 파랑 = 정답 상자")
    _grid_lines(ax0, size, S)
    for box, cls, (row, col) in zip(boxes, classes, cells):
        ax0.add_patch(_box_patch(box, cp.BLUE, scale=size))
        cx, cy = box[0] * size - 0.5, box[1] * size - 0.5
        ax0.plot([cx], [cy], marker="o", ms=4, color=cp.BLUE, zorder=4)
        ax0.text(cx + 0.6, cy - 0.6, f"{int(cls)} → 셀 ({row}, {col})", color=cp.BLUE, fontsize=8.5, ha="left", va="bottom", zorder=5,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85})
    r, c = highlight_cell
    cell = size / S
    ax0.add_patch(Rectangle((c * cell - 0.5, r * cell - 0.5), cell, cell, fill=False, ec=cp.GOLD, lw=2.4, zorder=4))
    cp.label(ax0, "점 = 상자 중심 · 금색 = 그 중심이 속한 책임 셀 (행, 열)", y=-0.05)
    cp.draw_grid(ax1, obj_grid, "책임 셀 mask $1^{obj}$", kind="count", fmt="{:g}", highlight=[highlight_cell],
                 row_labels=[f"행 {i}" for i in range(S)], col_labels=[f"열 {j}" for j in range(S)], fontsize=9.5)
    cp.draw_bars(ax2, target_vec, "금색 셀의 타깃 15개", labels=TARGET_LABELS, marker=5 + int(np.argmax(target_vec[5:])), fmt="{:.2f}", color=cp.BLUE)
    for k, v in enumerate(target_vec[:5]):
        ax2.text(v + 0.02, k, f"{v:.2f}", va="center", fontsize=8.5, color=cp.INK)
    cp.label(ax2, "x·y = 셀 안 상대 위치, w·h = 캔버스 비율, C = 1, 뒤 10칸 = 클래스 one-hot", y=-0.06)
    cp.connect(fig, ax0, ax1, "중심 → 셀")
    cp.connect(fig, ax1, ax2, "셀의 15칸")
    cp.show(fig)


# ---------------------------------------------------------------- 3절: 손실 항
def plot_loss_terms(conf_grid, obj_grid, term_names, term_values):
    """셀마다의 신뢰도 예측 격자(책임 셀은 파란 배경·금색 테두리) → 네 손실 항 막대."""
    conf_grid, obj_grid = np.asarray(conf_grid, float), np.asarray(obj_grid, bool)
    S = conf_grid.shape[0]
    colors = [[OBJ_FILL if obj_grid[i, j] else "white" for j in range(S)] for i in range(S)]
    cells = [(i, j) for i in range(S) for j in range(S) if obj_grid[i, j]]
    fig, (ax0, ax1) = cp.flow([1.1, 1.3], height=3.4, wspace=0.55, unit=2.6)
    sp.draw_cells(ax0, conf_grid, "셀마다의 신뢰도 예측 $\\hat{C}$", cell_colors=colors, highlight_cells=cells, fmt="{:.2f}", fontsize=9)
    cp.label(ax0, "파란 셀(책임 셀): 좌표·obj·클래스 항 · 흰 셀: noobj 항만", y=-0.04)
    cp.draw_bars(ax1, term_values, "네 항의 값 (합 = 손실)", labels=term_names, fmt="{:.3f}", color=cp.BLUE)
    for k, v in enumerate(term_values):
        ax1.text(v + 0.01 * max(term_values), k, f"{v:.4f}", va="center", fontsize=9, color=cp.INK)
    cp.label(ax1, "어느 셀이 어느 항을 받는지는 mask가 정합니다", y=-0.04)
    cp.show(fig)


# ---------------------------------------------------------------- 4절: 블록 흐름과 검출 결과
def plot_detector_blocks(blocks, param_blocks=(1, 4, 7, 10)):
    """입력에서 출력 텐서까지의 블록 흐름. 아래에 텐서 모양, 파라미터가 있는 블록은 옅은 금색."""
    fig = cp.new_figure((15.5, 2.2))
    ax = fig.add_subplot(111)
    colors = [PARAM_FILL if i in param_blocks else sp.FILL for i in range(len(blocks))]
    sp.draw_blocks(ax, blocks, colors=colors, width=1.5, gap=0.4, fontsize=8.6)
    cp.label(ax, "금색 상자 = 학습하는 파라미터가 있는 층 · 상자 아래 = 그 단계의 텐서 모양", y=0.0)
    cp.show(fig)


def plot_detections(canvases, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls, threshold, titles=None):
    """검증 캔버스 여러 장. 파랑 = 정답 상자, 빨강 = 모든 셀의 예측 상자(진하기 = 신뢰도). 임계값 이상이면 클래스·신뢰도를 적는다."""
    n = len(canvases)
    size = canvases[0].shape[0]
    fig, axes = cp.flow([1] * n, height=3.9, wspace=0.15, unit=2.9)
    for k, ax in enumerate(axes):
        cp.draw_image(ax, canvases[k], titles[k] if titles else f"검증 캔버스 {k}")
        for box, cls in zip(gt_boxes[k], gt_cls[k]):
            ax.add_patch(_box_patch(box, cp.BLUE, scale=size, lw=1.8))
        S = pred_conf[k].shape[0]
        for i in range(S):
            for j in range(S):
                conf = float(pred_conf[k][i, j])
                ax.add_patch(_box_patch(pred_boxes[k][i, j], cp.RED, scale=size, lw=1.2, alpha=max(0.06, min(1.0, conf))))
                if conf >= threshold:
                    _box_text(ax, pred_boxes[k][i, j], f"{int(pred_cls[k][i, j])} · {conf:.2f}", cp.RED, scale=size, fontsize=7.5, above=(i + j) % 2 == 0)
    cp.label(axes[n // 2], f"파랑 = 정답 · 빨강 = 셀 16개의 예측 상자(진할수록 신뢰도 큼) · 글자 = 신뢰도 {threshold:g} 이상인 상자의 클래스", y=-0.09)
    cp.show(fig)


# ---------------------------------------------------------------- 5절: NMS와 precision–recall
def plot_nms_hand(canvas, boxes, scores, names, iou_matrix, keep, iou_threshold):
    """손계산 NMS: 상자 셋과 점수 → 상자끼리의 IoU 표(임계값 초과는 금색) → 남긴 상자(파랑)와 지운 상자(빨강 점선). 화소 단위."""
    keep = set(int(k) for k in keep)
    n = len(boxes)
    fig, (ax0, ax1, ax2) = cp.flow([1, 0.95, 1], height=3.5, wspace=0.3, unit=2.5)
    cp.draw_image(ax0, canvas, "후보 상자와 점수")
    for k, (box, score, name) in enumerate(zip(boxes, scores, names)):
        ax0.add_patch(_box_patch(box, cp.INK, lw=1.4))
        _box_text(ax0, box, f"{name} {score:g}", cp.INK, above=k % 2 == 0)
    hits = [(i, j) for i in range(n) for j in range(n) if i != j and iou_matrix[i, j] > iou_threshold]
    cp.draw_grid(ax1, iou_matrix, f"상자끼리의 IoU (금색 > {iou_threshold:g})", kind="count", fmt="{:.2f}", highlight=hits,
                 row_labels=names, col_labels=names, fontsize=9.5)
    cp.draw_image(ax2, canvas, "점수 순서로 남기고 겹치면 지움")
    for k, (box, score, name) in enumerate(zip(boxes, scores, names)):
        kept = k in keep
        ax2.add_patch(_box_patch(box, cp.BLUE if kept else cp.RED, lw=1.8 if kept else 1.3, ls="-" if kept else "--"))
        _box_text(ax2, box, f"{name} {'남김' if kept else '지움'}", cp.BLUE if kept else cp.RED, above=k % 2 == 0)
    cp.connect(fig, ax0, ax1, "모든 쌍")
    cp.connect(fig, ax1, ax2, "greedy")
    cp.show(fig)


def plot_nms_result(canvas, boxes_before, scores_before, boxes_after, scores_after, cls_after, gt_boxes, threshold):
    """학습한 모델의 검증 캔버스 한 장: 임계값을 넘은 상자 전부(빨강) → NMS 뒤 남은 상자(빨강, 클래스·점수)와 정답(파랑). 상자는 캔버스 비율."""
    size = canvas.shape[0]
    fig, (ax0, ax1) = cp.flow([1, 1], height=3.6, wspace=0.34, unit=2.8)
    cp.draw_image(ax0, canvas, f"신뢰도 ≥ {threshold:g}인 상자 {len(boxes_before)}개")
    for box, score in zip(boxes_before, scores_before):
        ax0.add_patch(_box_patch(box, cp.RED, scale=size, lw=1.2, alpha=max(0.35, float(score))))
    cp.draw_image(ax1, canvas, f"NMS 뒤 {len(boxes_after)}개 · 파랑 = 정답")
    for box in gt_boxes:
        ax1.add_patch(_box_patch(box, cp.BLUE, scale=size, lw=1.8))
    for k, (box, score, cls) in enumerate(zip(boxes_after, scores_after, cls_after)):
        ax1.add_patch(_box_patch(box, cp.RED, scale=size, lw=1.4))
        _box_text(ax1, box, f"{int(cls)} · {score:.2f}", cp.RED, scale=size, fontsize=8, above=k % 2 == 0)
    cp.connect(fig, ax0, ax1, "겹치면 지움")
    cp.show(fig)


def plot_pr_curve(recall, precision, envelope, ap, *, title="검증 캔버스의 precision–recall 곡선"):
    """precision–recall 계단 곡선(파랑)과 all-point 보간 상한(금색). ap는 금색 아래 넓이."""
    recall = np.concatenate([[0.0], np.asarray(recall, dtype=float)])          # 0에서 시작해야 칠한 넓이가 AP와 같다
    precision = np.concatenate([np.asarray(precision, dtype=float)[:1], precision])
    envelope = np.concatenate([np.asarray(envelope, dtype=float)[:1], envelope])
    fig = cp.new_figure((6.2, 3.8))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "recall (정답 중 찾은 비율)", "precision (예측 중 맞은 비율)", title)
    ax.step(recall, precision, where="post", color=cp.BLUE, lw=1.6, label="precision–recall")
    ax.step(recall, envelope, where="post", color=cp.GOLD, lw=2.0, label="오른쪽 최댓값 상한")
    ax.fill_between(recall, 0, envelope, step="post", color=cp.GOLD, alpha=0.15)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.text(0.03, 0.06, f"AP = {ap:.3f}", color=cp.INK, fontsize=11, transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.16)
    cp.show(fig)
