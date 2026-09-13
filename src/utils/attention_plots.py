"""08 Attention 노트북의 개념 그림.

점수·가중치·읽은 값·mask·분산은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from .gradient_plots import _clean_axes
from .architecture_plots import plot_curves  # noqa: F401


def _heat(ax, M, title=None, *, vmax=None, frame=True):
    """0 이상의 값을 파랑 진하기로. 축·눈금 없음."""
    a = np.asarray(M, float)
    ax.imshow(a, cmap="Blues", vmin=0, vmax=(vmax if vmax is not None else (a.max() or 1)), interpolation="nearest")
    if frame:
        h, w = a.shape
        ax.add_patch(Rectangle((-0.5, -0.5), w, h, fill=False, ec=cp.MUTED, lw=0.8))
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)
    return ax


def plot_attention_lookup(Q, K, S, P, V, O):
    """질문 하나: Q·K^T/√d → softmax → P·V. 손계산 예제의 작은 행렬 흐름."""
    fig = cp.new_figure((13, 3.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.0, 1.0, 1.0, 1.4, 1.0, 1.0], wspace=0.55)
    ax_q = fig.add_subplot(gs[0]); cp.blank(ax_q)
    cp.draw_grid(ax_q, Q, "$Q$ 찾을 조건 (1, 2)", kind="signed", fmt="{:.2f}", fontsize=9, frame_to=(2, 2))
    ax_k = fig.add_subplot(gs[1]); cp.blank(ax_k)
    cp.draw_grid(ax_k, K, "$K$ 항목의 특징 (2, 2)", kind="signed", fmt="{:g}", fontsize=9, row_labels=["key 0", "key 1"])
    ax_s = fig.add_subplot(gs[2]); cp.blank(ax_s)
    cp.draw_grid(ax_s, S, "$S = QK^{\\top}/\\sqrt{d_k}$", kind="signed", fmt="{:.2f}", fontsize=9, frame_to=(2, 2))
    ax_p = fig.add_subplot(gs[3]); cp.blank(ax_p)
    cp.draw_bars(ax_p, np.asarray(P).ravel(), "$P$ = softmax (합 1)", labels=["key 0", "key 1"], xlim=(0, 1), color=cp.BLUE)
    for i, v in enumerate(np.asarray(P).ravel()):
        ax_p.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9, color=cp.INK)
    cp.match_height(ax_p, ax_k)                                                  # 막대 판의 높이를 옆 2×2 격자에 맞춤
    ax_v = fig.add_subplot(gs[4]); cp.blank(ax_v)
    cp.draw_grid(ax_v, V, "$V$ 읽어 올 내용 (2, 2)", kind="count", fmt="{:g}", fontsize=9, row_labels=["key 0", "key 1"])
    ax_o = fig.add_subplot(gs[5]); cp.blank(ax_o)
    cp.draw_grid(ax_o, O, "$O = PV$ 읽은 값 (1, 2)", kind="count", fmt="{:.1f}", fontsize=9, frame_to=(2, 2))
    cp.connect(fig, ax_q, ax_k, "비교"); cp.connect(fig, ax_k, ax_s, "내적 / $\\sqrt{2}$")
    cp.connect(fig, ax_s, ax_p, "softmax"); cp.connect(fig, ax_p, ax_v, "가중치로"); cp.connect(fig, ax_v, ax_o, "섞어 읽음")
    cp.label(ax_o, "$0.75\\cdot[10, 0] + 0.25\\cdot[0, 20]$", y=-0.1, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_patch_attention(image, patch_size, query_index, P_map, O_patch, query_patch):
    """7을 패치 토큰으로 자르고, 한 패치(금색)가 다른 패치를 얼마나 읽는지(P)를 이미지 격자 위에 겹칩니다."""
    n = image.shape[0] // patch_size
    fig = cp.new_figure((12.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.55, 0.55], wspace=0.45)
    qi, qj = divmod(query_index, n)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_image(ax_x, image, f"$x_7$ 을 {patch_size}×{patch_size} 패치 {n * n}개로")
    for k in range(1, n):
        ax_x.axhline(k * patch_size - 0.5, color=cp.MUTED, lw=0.5)
        ax_x.axvline(k * patch_size - 0.5, color=cp.MUTED, lw=0.5)
    ax_x.add_patch(Rectangle((qj * patch_size - 0.5, qi * patch_size - 0.5), patch_size, patch_size, fill=False, ec=cp.GOLD, lw=2.2, zorder=3))
    cp.label(ax_x, f"금색 = query 패치 {query_index} (Q = K = V = 화소 16개)", y=-0.05)
    ax_p = fig.add_subplot(gs[1]); cp.blank(ax_p)
    _heat(ax_p, P_map, f"$P$ : 패치 {query_index}이 각 패치를 읽는 가중치 ({n}×{n})")
    ax_p.add_patch(Rectangle((qj - 0.5, qi - 0.5), 1, 1, fill=False, ec=cp.GOLD, lw=2.2, zorder=3))
    cp.label(ax_p, f"합 1 · 가장 큰 값 {P_map.max():.2f}", y=-0.05)
    ax_q = fig.add_subplot(gs[2]); cp.blank(ax_q)
    cp.draw_image(ax_q, query_patch, "query 패치")
    ax_o = fig.add_subplot(gs[3]); cp.blank(ax_o)
    cp.draw_image(ax_o, O_patch, "읽은 값 $O = PV$")
    cp.connect(fig, ax_x, ax_p, "점수 → softmax"); cp.connect(fig, ax_q, ax_o, "$P$ 로 섞음")
    cp.show(fig)


def plot_score_variance(dimensions, raw_variance, scaled_variance, p_raw, p_scaled):
    """왼쪽: 비교 차원에 따른 점수의 분산(로그). 오른쪽: 같은 점수를 나눈 것과 안 나눈 것의 softmax."""
    fig = cp.new_figure((11.5, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "비교 차원 $d_k$", "점수의 분산 (로그 눈금)", "독립·평균 0·분산 1 입력의 점수 분산")
    ax.loglog(dimensions, raw_variance, "o-", color=cp.RED, lw=1.5, ms=4, label="$q\\cdot k$ 그대로 ($\\approx d_k$)")
    ax.loglog(dimensions, scaled_variance, "s-", color=cp.BLUE, lw=1.5, ms=4, label="$q\\cdot k/\\sqrt{d_k}$ ($\\approx 1$)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a, p, title, color in [(fig.add_subplot(gs[1]), p_raw, "나누지 않은 점수의 softmax", cp.RED),
                               (fig.add_subplot(gs[2]), p_scaled, "$\\sqrt{d_k}$ 로 나눈 점수의 softmax", cp.BLUE)]:
        cp.blank(a)
        cp.draw_bars(a, p, title, labels=[f"key {i}" for i in range(len(p))], xlim=(0, 1), color=color)
        cp.label(a, f"가장 큰 가중치 {max(p):.2f}", y=-0.06)
    cp.show(fig)


def plot_softmax_backward(p, g, jacobian, ds, ds_wrong):
    """한 행의 softmax backward: 상류 g → Jacobian p_i(δ_ij − p_j) → ds. 원소별 오답과 비교."""
    fig = cp.new_figure((12.5, 3.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.1, 1, 1], wspace=0.6)
    labels = [f"$s_{i}$" for i in range(len(p))]
    ax_p = fig.add_subplot(gs[0]); cp.blank(ax_p)
    cp.draw_bars(ax_p, p, "$p$ = softmax($s$)", labels=labels, xlim=(0, 1), color=cp.BLUE)
    ax_g = fig.add_subplot(gs[1]); cp.blank(ax_g)
    cp.draw_bars(ax_g, g, "상류 $g = \\partial L/\\partial p$", labels=labels, signed=True)
    ax_j = fig.add_subplot(gs[2]); cp.blank(ax_j)
    cp.draw_grid(ax_j, jacobian, "$\\partial p_i/\\partial s_j = p_i(\\delta_{ij} - p_j)$", kind="signed", fmt="{:.3f}", fontsize=9,
                 row_labels=[f"$p_{i}$" for i in range(len(p))], col_labels=labels)
    ax_d = fig.add_subplot(gs[3]); cp.blank(ax_d)
    cp.draw_bars(ax_d, ds, "$ds = p\\odot(g - \\sum g p)$", labels=labels, signed=True, color=cp.BLUE)
    ax_w = fig.add_subplot(gs[4]); cp.blank(ax_w)
    cp.draw_bars(ax_w, ds_wrong, "원소별 오답 $g\\odot p(1-p)$", labels=labels, signed=True, color=cp.RED)
    lo, hi = min(min(ds), min(ds_wrong), 0) - 0.15, max(max(ds), max(ds_wrong), 0) + 0.15
    for a in (ax_d, ax_w):
        a.set_xlim(lo, hi)
    for a, vals in [(ax_d, ds), (ax_w, ds_wrong)]:
        for i, v in enumerate(vals):
            a.text(max(v, 0) + 0.03, i, f"{v:+.3f}", va="center", ha="left", fontsize=8.5, color=cp.INK)   # 라벨은 항상 0의 오른쪽
    cp.connect(fig, ax_g, ax_j, "$g^{\\top}J$"); cp.connect(fig, ax_j, ax_d, "=")
    cp.label(ax_w, "분모를 공유한 다른 위치의 기여가 빠짐", y=-0.08, color=cp.RED)
    cp.show(fig)


def plot_masks(causal, valid_keys, combined, P):
    """causal mask AND padding mask = 허용 표, 그 위에서만 합이 1인 P."""
    T = causal.shape[0]
    fig = cp.new_figure((12.5, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.5)
    labels_q = [f"q{i}" for i in range(T)]
    labels_k = [f"k{j}" for j in range(T)]
    ax_c = fig.add_subplot(gs[0]); cp.blank(ax_c)
    cp.draw_grid(ax_c, causal.astype(float), "causal: 미래 key 금지", kind="count", fmt="{:g}", fontsize=9, row_labels=labels_q, col_labels=labels_k, vmax=1)
    ax_v = fig.add_subplot(gs[1]); cp.blank(ax_v)
    cp.draw_grid(ax_v, np.broadcast_to(valid_keys.astype(float), (T, T)), "padding: 가짜 key 금지", kind="count", fmt="{:g}", fontsize=9,
                 row_labels=labels_q, col_labels=labels_k, vmax=1)
    ax_a = fig.add_subplot(gs[2]); cp.blank(ax_a)
    cp.draw_grid(ax_a, combined.astype(float), "허용 = causal AND padding", kind="count", fmt="{:g}", fontsize=9, row_labels=labels_q, col_labels=labels_k, vmax=1)
    ax_p = fig.add_subplot(gs[3]); cp.blank(ax_p)
    cp.draw_grid(ax_p, P, "$P$ : 허용 위치에서만 합 1", kind="count", fmt="{:.2f}", fontsize=8.5, row_labels=labels_q, col_labels=labels_k, vmax=1)
    cp.connect(fig, ax_c, ax_v, "AND"); cp.connect(fig, ax_v, ax_a, "="); cp.connect(fig, ax_a, ax_p, "softmax")
    cp.label(ax_p, "금지 점수는 $-\\infty$ · 행마다 허용된 key 사이에서만 나눔", y=-0.05)
    cp.show(fig)


def plot_retrieval_example(query, keys, values, key_ids, bits, label, target_position):
    """검색 문제 한 예: query 정체 → 같은 정체의 key 행(금색) → 그 행의 값이 정답."""
    fig = cp.new_figure((11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 0.8], wspace=0.55)
    ax_q = fig.add_subplot(gs[0]); cp.blank(ax_q)
    cp.draw_grid(ax_q, query, f"query $X_q$ : 정체 {int(np.argmax(query))}", kind="count", fmt="{:g}", fontsize=9,
                 col_labels=[f"정체 {i}" for i in range(query.shape[1])], frame_to=(4, 4), vmax=1)
    ax_k = fig.add_subplot(gs[1]); cp.blank(ax_k)
    cp.draw_grid(ax_k, keys, "key $X_k$ : 행마다 다른 정체 (순서는 섞임)", kind="count", fmt="{:g}", fontsize=9,
                 row_labels=[f"위치 {i}" for i in range(keys.shape[0])], col_labels=[f"정체 {i}" for i in range(keys.shape[1])],
                 highlight=[(target_position, j) for j in range(keys.shape[1])], vmax=1)
    ax_v = fig.add_subplot(gs[2]); cp.blank(ax_v)
    cp.draw_grid(ax_v, values, "값 $V$ : 위치마다 0/1 (one-hot)", kind="count", fmt="{:g}", fontsize=9,
                 row_labels=[f"값 {b}" for b in bits], col_labels=["0", "1"], highlight=[(target_position, 0), (target_position, 1)], vmax=1)
    cp.connect(fig, ax_q, ax_k, "같은 정체를 찾음"); cp.connect(fig, ax_k, ax_v, "그 위치의 값")
    cp.label(ax_v, f"정답 $y = {label}$ · 값은 예제마다 무작위", y=-0.05, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_attention_weights(before, after, target_positions, query_ids):
    """학습 전후 검증 예제 4개의 가중치 (행 = 예제, 열 = key 위치). 금색 = query 와 같은 정체의 key 위치."""
    fig = cp.new_figure((9, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.5)
    rows = [f"예제 {i} (query {q})" for i, q in enumerate(query_ids)]
    cols = [f"위치 {j}" for j in range(before.shape[1])]
    for ax, M, title in [(fig.add_subplot(gs[0]), before, "학습 전 $P$"), (fig.add_subplot(gs[1]), after, "학습 후 $P$")]:
        cp.blank(ax)
        cp.draw_grid(ax, M, title, kind="count", fmt="{:.2f}", fontsize=9, row_labels=rows, col_labels=cols,
                     highlight=[(i, int(t)) for i, t in enumerate(target_positions)], vmax=1)
    cp.label(fig.axes[-1], "금색 = query 와 같은 정체의 key 위치 · 학습 뒤 그 칸에 가중치가 몰림", y=-0.05)
    cp.show(fig)
