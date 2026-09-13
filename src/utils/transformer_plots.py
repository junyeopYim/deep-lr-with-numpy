"""09 Transformer 노트북의 개념 그림과 구조 도식.

정규화·head 분리·가중치·embedding·위치 값은 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import concept_plots as cp
from . import schematic_plots as sp
from .architecture_plots import plot_curves  # noqa: F401

HEAD_COLORS = ["#cfe0f3", "#fce8c3"]


def plot_layernorm(X, mu, std, Xhat, Y):
    """토큰 두 개의 특성 4개: 행마다 평균을 빼고 표준편차로 나눔. 전체에 같은 수를 더한 차이는 사라집니다."""
    T, D = X.shape
    rows = [f"토큰 {t}" for t in range(T)]
    fig = cp.new_figure((12.5, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[D, 2.2, D, D], wspace=0.7)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_grid(ax_x, X, "$x$ : 토큰마다 특성 $D=4$", kind="count", fmt="{:g}", fontsize=9.5, row_labels=rows)
    ax_s = fig.add_subplot(gs[1]); cp.blank(ax_s)
    cp.draw_grid(ax_s, np.column_stack([mu, std]), "행의 $\\mu$, $\\sqrt{v+\\epsilon}$", kind="plain", fmt="{:.2f}", fontsize=9,
                 row_labels=rows, col_labels=["$\\mu$", "$\\sqrt{v+\\epsilon}$"], frame_to=(2, T))
    ax_h = fig.add_subplot(gs[2]); cp.blank(ax_h)
    cp.draw_grid(ax_h, Xhat, "$\\hat x = (x-\\mu)/\\sqrt{v+\\epsilon}$", kind="signed", fmt="{:.2f}", fontsize=9, row_labels=rows)
    ax_y = fig.add_subplot(gs[3]); cp.blank(ax_y)
    cp.draw_grid(ax_y, Y, "$y = \\gamma\\odot\\hat x + \\beta$ ($\\gamma=1, \\beta=0$)", kind="signed", fmt="{:.2f}", fontsize=9, row_labels=rows)
    cp.connect(fig, ax_x, ax_s, "통계"); cp.connect(fig, ax_s, ax_h, "정규화"); cp.connect(fig, ax_h, ax_y, "affine")
    cp.label(ax_y, "두 행이 같아짐: 전체 +3의 차이는 사라짐", y=-0.06, color=cp.RED)
    cp.show(fig)


def plot_head_split(X, P_heads, O_merged):
    """토큰 T×폭 D 의 열을 head 별로 나누고(색), head 마다 T×T 가중치를 따로 만든 뒤 열을 다시 합칩니다."""
    T, D = X.shape
    heads = len(P_heads)
    dh = D // heads
    rows = [f"토큰 {t}" for t in range(T)]
    fig = cp.new_figure((12.5, 3.2))
    gs = fig.add_gridspec(1, 2 + heads, width_ratios=[D + 0.5] + [T + 0.5] * heads + [D + 0.5], wspace=0.55)
    ax_x = fig.add_subplot(gs[0]); cp.blank(ax_x)
    cp.draw_grid(ax_x, X, f"$X$ : ({T}, {D}) → head {heads}개 × 폭 {dh}", kind="signed", fmt="{:.1f}", fontsize=8.5, row_labels=rows)
    for h in range(heads):
        ax_x.add_patch(Rectangle((h * dh - 0.5, -0.5), dh, T, facecolor=HEAD_COLORS[h % 2], alpha=0.35, edgecolor="none", zorder=0))
        ax_x.text(h * dh + dh / 2 - 0.5, T - 0.3, f"head {h}", ha="center", va="top", fontsize=8.5, color=cp.INK)
    axes_p = []
    for h in range(heads):
        ax = fig.add_subplot(gs[1 + h]); cp.blank(ax)
        cp.draw_grid(ax, P_heads[h], f"head {h}의 $P$ : ({T}, {T})", kind="count", fmt="{:.2f}", fontsize=8.5,
                     row_labels=[f"q{t}" for t in range(T)], col_labels=[f"k{t}" for t in range(T)], vmax=1)
        ax.add_patch(Rectangle((-0.5, -0.5), T, T, fill=False, ec=HEAD_COLORS[h % 2], lw=4, zorder=0))
        axes_p.append(ax)
    ax_o = fig.add_subplot(gs[-1]); cp.blank(ax_o)
    cp.draw_grid(ax_o, O_merged, f"head 출력을 열로 합침 : ({T}, {D})", kind="signed", fmt="{:.1f}", fontsize=8.5, row_labels=rows)
    for h in range(heads):
        ax_o.add_patch(Rectangle((h * dh - 0.5, -0.5), dh, T, facecolor=HEAD_COLORS[h % 2], alpha=0.35, edgecolor="none", zorder=0))
    cp.connect(fig, ax_x, axes_p[0], "softmax")
    cp.connect(fig, axes_p[-1], ax_o, "merge")
    cp.label(ax_o, "$P_hV_h$ 를 합친 뒤 $W_o$ 로 섞음", y=-0.06)
    cp.show(fig)


def plot_transformer_block(shape_text="(N, T, D)"):
    """Pre-LN 블록: 두 하위 층(MHA, FFN)과 두 잔차 연결(위로 도는 화살표)."""
    fig = cp.new_figure((13, 3.0))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_aspect("equal")
    names = ["$X$", "LN$_1$", "MHA", "⊕", "LN$_2$", "FFN", "⊕", "$Y$"]
    fills = [sp.FILL, sp.FILL, "#fce8c3", "white", sp.FILL, "#fce8c3", "white", sp.FILL]
    w, h, gap = 1.3, 0.8, 0.45
    xs = []
    for i, (name, fill) in enumerate(zip(names, fills)):
        x = i * (w + gap)
        xs.append(x)
        if name == "⊕":
            ax.add_patch(FancyBboxPatch((x + w / 2 - 0.3, h / 2 - 0.3), 0.6, 0.6, boxstyle="circle,pad=0.0",
                                        facecolor="white", edgecolor=cp.MUTED, lw=1.0))
            ax.text(x + w / 2, h / 2, "+", ha="center", va="center", fontsize=13, color=cp.INK)
        else:
            ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fill, edgecolor=cp.MUTED, lw=1.0))
            ax.text(x + w / 2, h / 2, name, ha="center", va="center", fontsize=10, color=cp.INK)
        ax.text(x + w / 2, -0.12, shape_text, ha="center", va="top", fontsize=8, color=cp.MUTED)
        if i < len(names) - 1:
            x0 = x + (w / 2 + 0.32 if name == "⊕" else w + 0.04)
            x1 = x + w + gap + (w / 2 - 0.32 if names[i + 1] == "⊕" else -0.04)
            ax.add_patch(FancyArrowPatch((x0, h / 2), (x1, h / 2), arrowstyle="-|>", mutation_scale=11, color=cp.MUTED, lw=1.0))
    for src, dst, text in [(0, 3, "$X$ 그대로 (잔차)"), (3, 6, "$U$ 그대로 (잔차)")]:
        x0, x1 = xs[src] + w / 2, xs[dst] + w / 2
        ax.add_patch(FancyArrowPatch((x0, h + 0.05), (x1, h / 2 + 0.34), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                     mutation_scale=11, color=cp.GOLD, lw=1.6))
        ax.text((x0 + x1) / 2, h + 0.95, text, ha="center", va="bottom", fontsize=9, color=cp.GOLD)
    ax.text(xs[2] + w / 2, -0.55, "위치 사이의 정보 교환", ha="center", va="top", fontsize=8.5, color=cp.INK)
    ax.text(xs[5] + w / 2, -0.55, "위치마다 같은 MLP", ha="center", va="top", fontsize=8.5, color=cp.INK)
    ax.set_xlim(-0.3, xs[-1] + w + 0.3)
    ax.set_ylim(-1.0, h + 1.6)
    ax.set_title("$U = X + \\mathrm{MHA}(\\mathrm{LN}_1(X))$,  $Y = U + \\mathrm{FFN}(\\mathrm{LN}_2(U))$ · 금색 = 항등 경로", color=cp.INK, fontsize=10.5, pad=4)
    cp.show(fig)


def plot_embedding_positions(tokens, E, E_rows, P_rows, X, sinusoid):
    """위: 정수 토큰 → E 의 행 고르기 → 위치 표를 더함. 아래: 원 논문의 고정 sinusoidal 위치 표."""
    T = len(tokens)
    V, D = E.shape
    fig = cp.new_figure((13, 6.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.2, V + 0.5, D * 0.42, D * 0.42, D * 0.42], height_ratios=[1, 1.15], wspace=0.5, hspace=0.55)
    ax_t = fig.add_subplot(gs[0, 0]); cp.blank(ax_t)
    cp.draw_grid(ax_t, np.asarray(tokens, float)[:, None], "tokens (T,)", kind="plain", fmt="{:g}", fontsize=10,
                 row_labels=[f"$t={t}$" for t in range(T)], frame_to=(1, T))
    ax_e = fig.add_subplot(gs[0, 1]); cp.blank(ax_e)
    cp.draw_rows(ax_e, E, f"$E$ : ({V}, {D}) 표", kind="signed")
    for k, tok in enumerate(tokens):
        ax_e.add_patch(Rectangle((-0.5, tok - 0.5), D, 1, fill=False, ec=cp.GOLD, lw=1.6, zorder=3))
    ax_e.set_yticks(range(V), [f"행 {v}" for v in range(V)]); ax_e.set_axis_on(); ax_e.set_xticks([])
    for s in ax_e.spines.values():
        s.set_visible(False)
    ax_e.tick_params(length=0, labelsize=8.5, colors=cp.MUTED)
    vmax = float(max(np.abs(E_rows).max(), np.abs(P_rows).max(), np.abs(X).max()))
    ax_r = fig.add_subplot(gs[0, 2]); cp.blank(ax_r)
    cp.draw_rows(ax_r, E_rows, "$E[\\mathrm{tokens}]$ : (T, D)", kind="signed", vmax=vmax)
    ax_p = fig.add_subplot(gs[0, 3]); cp.blank(ax_p)
    cp.draw_rows(ax_p, P_rows, "$P[:T]$ 위치 표 : (T, D)", kind="signed", vmax=vmax)
    ax_x = fig.add_subplot(gs[0, 4]); cp.blank(ax_x)
    cp.draw_rows(ax_x, X, "$X = E[\\mathrm{tokens}] + P[:T]$", kind="signed", vmax=vmax)
    cp.connect(fig, ax_t, ax_e, "행 번호로"); cp.connect(fig, ax_e, ax_r, "고른 행"); cp.connect(fig, ax_r, ax_p, "+"); cp.connect(fig, ax_p, ax_x, "=")
    cp.label(ax_x, "같은 토큰이라도 위치가 다르면 다른 벡터", y=-0.08)
    ax_s = fig.add_subplot(gs[1, 1:4]); cp.blank(ax_s)
    ax_s.imshow(sinusoid, cmap=cp.SIGNED_CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax_s.add_patch(Rectangle((-0.5, -0.5), sinusoid.shape[1], sinusoid.shape[0], fill=False, ec=cp.MUTED, lw=0.8))
    ax_s.set_title(f"고정 sinusoidal 위치 표 : ({sinusoid.shape[0]} 위치, {sinusoid.shape[1]} 좌표) · 행 = 위치, 열 = 좌표", color=cp.INK, fontsize=10.5, pad=6)
    cp.label(ax_s, "왼쪽 열은 빠르게, 오른쪽 열은 느리게 진동 · 파랑 +1, 빨강 −1", y=-0.08)
    cp.show(fig)


def plot_reverse_examples(inputs, targets):
    """역순 변환 예: 입력 토큰열 → 뒤집은 정답."""
    n = len(inputs)
    T = inputs.shape[1]
    fig = cp.new_figure((7.5, 0.9 * n + 1.0))
    gs = fig.add_gridspec(n, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.6)
    for r in range(n):
        ax_i = fig.add_subplot(gs[r, 0])
        sp.draw_cells(ax_i, inputs[r][None, :], "입력 tokens" if r == 0 else None, fontsize=10)
        ax_o = fig.add_subplot(gs[r, 1])
        sp.draw_cells(ax_o, targets[r][None, :], "정답 = 역순" if r == 0 else None, fontsize=10)
        cp.connect(fig, ax_i, ax_o, "뒤집기" if r == 0 else None)
    cp.label(fig.axes[-2], f"위치 $t$의 출력은 위치 ${T - 1}-t$의 입력을 읽어야 함", y=-0.5, color=cp.INK, fontsize=9.5)
    cp.show(fig)


def plot_trained_heads(P_heads, tokens):
    """학습된 두 head 의 가중치. 금색 = 역순 변환에 필요한 위치 (query t → key T−1−t)."""
    T = len(tokens)
    heads = len(P_heads)
    fig = cp.new_figure((4.2 * heads + 0.5, 3.6))
    gs = fig.add_gridspec(1, heads, wspace=0.45)
    for h in range(heads):
        ax = fig.add_subplot(gs[h]); cp.blank(ax)
        cp.draw_grid(ax, P_heads[h], f"학습된 head {h} : query → key", kind="count", fmt="{:.2f}", fontsize=9,
                     row_labels=[f"출력 {t}" for t in range(T)], col_labels=[f"입력 {t}: {tokens[t]}" for t in range(T)],
                     highlight=[(t, T - 1 - t) for t in range(T)], vmax=1)
    cp.label(fig.axes[-1], "금색 = 역대각선 (출력 $t$ ← 입력 $T-1-t$) · head 마다 같을 필요는 없음", y=-0.05)
    cp.show(fig)
