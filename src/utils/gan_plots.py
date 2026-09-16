"""28 GAN 노트북의 개념 그림과 구조 도식.

손실·기울기·최적 판별기·생성 샘플·책임확률·PCA 좌표는 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp)·schematic_plots(sp), 원칙은 docs/DESIGN.md.
학습 곡선(plot_curves)과 이미지 묶음(plot_images)은 architecture_plots의 것을 그대로 내보냅니다.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import concept_plots as cp
from . import schematic_plots as sp
from .architecture_plots import plot_curves, plot_images  # noqa: F401  (노트북이 한 파일에서 가져가도록)

GOLD_FILL = "#fbf0d4"      # 금색 강조 영역의 옅은 배경
BLUE_FILL = "#dbe7f5"      # 파랑 강조 영역의 옅은 배경


def _clean_axes(ax, xlabel=None, ylabel=None, title=None):
    """곡선·산점 패널: 위·오른쪽 테두리를 없애고 회색 눈금만 남깁니다."""
    ax.set_axis_on()
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(cp.MUTED)
    ax.tick_params(colors=cp.MUTED, labelsize=8.5, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=cp.INK, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=cp.INK, fontsize=9.5)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)


# ---------------------------------------------------------------- 1절: 두 손실
def plot_generator_losses(p, real_term, fake_term, loss_mm, loss_ns, grad_mm, grad_ns, p_mark):
    """가로축 D(G(z)) = p 위에 (왼쪽) 판별기의 두 항, (가운데) 생성기의 두 손실, (오른쪽) logit 기울기. 금색 띠 = 학습 초기(p 작음)."""
    fig, axes = cp.flow([1, 1, 1], height=3.3, unit=3.6, wspace=0.42)
    early = p < 0.1
    for ax in axes:
        ax.axvspan(0, 0.1, color=GOLD_FILL, lw=0)
    ax = axes[0]
    _clean_axes(ax, "$p = D(x)$", "손실", "판별기의 두 항")
    ax.plot(p, real_term, color=cp.BLUE, lw=1.8, label="진짜: $-\\log p$")
    ax.plot(p, fake_term, color=cp.RED, lw=1.8, label="가짜: $-\\log(1-p)$")
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    ax = axes[1]
    _clean_axes(ax, "$p = D(G(z))$", "손실", "생성기의 두 손실")
    ax.plot(p, loss_mm, color=cp.MUTED, lw=1.8, label="minimax: $\\log(1-p)$")
    ax.plot(p, loss_ns, color=cp.BLUE, lw=1.8, label="non-saturating: $-\\log p$")
    ax.axhline(0, color=cp.MUTED, lw=0.6)
    ax.set_ylim(-5, 5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[2]
    _clean_axes(ax, "$p = D(G(z))$", "$\\partial L_G / \\partial a$", "logit 기울기 (하나의 가짜)")
    ax.plot(p, grad_mm, color=cp.MUTED, lw=1.8, label="minimax: $-p$")
    ax.plot(p, grad_ns, color=cp.BLUE, lw=1.8, label="non-saturating: $-(1-p)$")
    ax.axhline(0, color=cp.MUTED, lw=0.6)
    k = int(np.argmin(np.abs(p - p_mark)))
    ax.scatter([p[k], p[k]], [grad_mm[k], grad_ns[k]], color=cp.GOLD, s=36, zorder=4)
    ax.annotate(f"{grad_mm[k]:.2f}", (p[k], grad_mm[k]), xytext=(14, 6), textcoords="offset points", fontsize=8.5, color=cp.INK)
    ax.annotate(f"{grad_ns[k]:.2f}", (p[k], grad_ns[k]), xytext=(14, -4), textcoords="offset points", fontsize=8.5, color=cp.INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    cp.label(axes[1], f"금색 띠 = 학습 초기 ($p < 0.1$) · 금색 점 = $p = {p_mark}$ 손계산", y=-0.2, fontsize=9)
    assert early.any(), "p 격자에 학습 초기 영역이 있어야 합니다"
    cp.show(fig)


# ---------------------------------------------------------------- 2절: 기울기 경로 도식
def _box(ax, x, y, w, h, text, sub=None, fill=sp.FILL, edge=cp.MUTED, lw=1.0, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=fill, edgecolor=edge, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=cp.INK)
    if sub:
        ax.text(x + w / 2, y - 0.1, sub, ha="center", va="top", fontsize=8.2, color=cp.MUTED)


def _arrow(ax, p0, p1, color=cp.MUTED, lw=1.0, text=None, dy=0.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=8.5, color=color)


def plot_gan_paths(blocks, real_shape):
    """위: 생성기 갱신 — 손실에서 D를 거슬러 x̂까지 온 dX_fake(금색)가 G로 이어지고 D의 기울기는 버립니다.
    아래: 판별기 갱신 — 진짜·가짜 두 갈래가 D로 들어가고 기울기(파랑)는 D에서 멈춥니다(x̂는 상수).
    blocks: [(이름, 모양)] 여섯 개 = z, G, x̂, D, a, 손실 순서."""
    (z_t, z_s), (g_t, g_s), (x_t, x_s), (d_t, d_s), (a_t, a_s), (l_t, l_s) = blocks
    fig = cp.new_figure((12.5, 5.6))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_aspect("equal")
    w, h, gap = 1.6, 0.8, 0.75
    xs = [i * (w + gap) for i in range(6)]
    # 1. 위 줄: 생성기 갱신
    y1 = 2.6
    ax.text(-0.3, y1 + h + 0.55, "생성기 갱신 · 금색 = 흐르는 기울기", ha="left", va="bottom", fontsize=10.5, color=cp.INK)
    fills = [sp.FILL, cp.GOLD, sp.FILL, sp.FILL, sp.FILL, sp.FILL]
    for x, (t, s), f in zip(xs, blocks, fills):
        _box(ax, x, y1, w, h, t, s, fill=GOLD_FILL if f == cp.GOLD else f, edge=cp.GOLD if f == cp.GOLD else cp.MUTED, lw=1.8 if f == cp.GOLD else 1.0)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, (x0 + w + 0.04, y1 + h * 0.65), (x1 - 0.04, y1 + h * 0.65))
    back = [("$d\\theta_G$ 얻음", cp.GOLD), ("G의 backward", cp.GOLD), ("$dX_{fake}$", cp.GOLD), ("D의 backward", cp.GOLD), ("$dL_G/da$", cp.GOLD)]
    for (x0, x1), (txt, col) in zip(zip(xs[:-1], xs[1:]), back):
        _arrow(ax, (x1 - 0.04, y1 + h * 0.25), (x0 + w + 0.04, y1 + h * 0.25), color=col, lw=1.8, text=txt, dy=-0.42)
    ax.text(xs[3] + w / 2, y1 + h + 0.12, "$d\\theta_D$는 버림 (갱신 안 함)", ha="center", va="bottom", fontsize=8.5, color=cp.RED)
    # 2. 아래 줄: 판별기 갱신
    y2 = 0.0
    ax.text(-0.3, y2 + h + 0.95, "판별기 갱신 · 파랑 = 흐르는 기울기", ha="left", va="bottom", fontsize=10.5, color=cp.INK)
    _box(ax, xs[0], y2, w, h, z_t, z_s)
    _box(ax, xs[1], y2, w, h, g_t, "고정")
    _box(ax, xs[2], y2, w, h, x_t + " (상수)", x_s)
    _box(ax, xs[2], y2 + 1.15, w, h, "진짜 $x$", real_shape)
    _box(ax, xs[3], y2, w, h, d_t, d_s, fill=BLUE_FILL, edge=cp.BLUE, lw=1.8)
    _box(ax, xs[4], y2, w, h, "$a_{real}, a_{fake}$", a_s)
    _box(ax, xs[5], y2, w, h, "$L_D$", l_s)
    _arrow(ax, (xs[0] + w + 0.04, y2 + h * 0.65), (xs[1] - 0.04, y2 + h * 0.65))
    _arrow(ax, (xs[1] + w + 0.04, y2 + h * 0.65), (xs[2] - 0.04, y2 + h * 0.65))
    _arrow(ax, (xs[2] + w + 0.04, y2 + h * 0.65), (xs[3] - 0.04, y2 + h * 0.65))
    _arrow(ax, (xs[2] + w + 0.04, y2 + 1.15 + h * 0.5), (xs[3] - 0.04, y2 + h * 0.85), rad=0.0)
    _arrow(ax, (xs[3] + w + 0.04, y2 + h * 0.65), (xs[4] - 0.04, y2 + h * 0.65))
    _arrow(ax, (xs[4] + w + 0.04, y2 + h * 0.65), (xs[5] - 0.04, y2 + h * 0.65))
    _arrow(ax, (xs[5] - 0.04, y2 + h * 0.25), (xs[4] + w + 0.04, y2 + h * 0.25), color=cp.BLUE, lw=1.8, text="$dL_D/da$", dy=-0.42)
    _arrow(ax, (xs[4] - 0.04, y2 + h * 0.25), (xs[3] + w + 0.04, y2 + h * 0.25), color=cp.BLUE, lw=1.8, text="$d\\theta_D$", dy=-0.42)
    ax.text(xs[2] + w / 2, y2 - 0.42, "가짜 표본으로 기울기를 보내지 않음 (stop-gradient)", ha="center", va="top", fontsize=8.5, color=cp.RED)
    ax.set_xlim(-0.4, xs[-1] + w + 0.3)
    ax.set_ylim(-0.8, y1 + h + 1.0)
    cp.show(fig)


# ---------------------------------------------------------------- 3절: 최적 판별기와 1차원 학습
def plot_optimal_discriminator(x, p_data, p_g, d_star, logit_star, marks):
    """왼쪽: 진짜 밀도와 생성 밀도, 가운데: D* = p_data/(p_data + p_g), 오른쪽: D*의 logit = log p_data − log p_g. marks = 손계산한 x 값들."""
    fig, axes = cp.flow([1, 1, 1], height=3.3, unit=3.6, wspace=0.42)
    ax = axes[0]
    _clean_axes(ax, "$x$", "밀도", "진짜 $p_{data}$ 와 생성 $p_g$")
    ax.plot(x, p_data, color=cp.BLUE, lw=1.8, label="$p_{data}$ (두 봉우리)")
    ax.plot(x, p_g, color=cp.RED, lw=1.8, label="$p_g$ (고정한 G)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    _clean_axes(ax, "$x$", "$D^*(x)$", "$D^* = p_{data} / (p_{data} + p_g)$")
    ax.plot(x, d_star, color=cp.INK, lw=1.8)
    ax.axhline(0.5, color=cp.MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    for m in marks:
        k = int(np.argmin(np.abs(x - m)))
        ax.scatter([x[k]], [d_star[k]], color=cp.GOLD, s=36, zorder=4)
        ax.annotate(f"{d_star[k]:.2f}", (x[k], d_star[k]), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=cp.INK)
    ax = axes[2]
    _clean_axes(ax, "$x$", "logit", "$\\log p_{data} - \\log p_g$")
    ax.plot(x, logit_star, color=cp.INK, lw=1.8)
    ax.axhline(0, color=cp.MUTED, lw=0.8, ls="--")
    cp.label(axes[1], "점선 = 1/2: 두 밀도가 같은 자리 · 금색 = 손계산 값", y=-0.2, fontsize=9)
    cp.show(fig)


def plot_discriminator_fit(x, d_learned, d_star, region, d_loss, d_loss_star, steps):
    """학습한 D(x)와 닫힌식 D*(x). region = (왼쪽, 오른쪽): 두 밀도가 모두 있는 구간(금색 띠), steps = 판별기만 학습한 스텝 수. 오른쪽은 차이."""
    fig, axes = cp.flow([1.4, 1], height=3.3, unit=3.6, wspace=0.4)
    ax = axes[0]
    _clean_axes(ax, "$x$", "$D(x)$", f"판별기만 {steps:,}스텝 학습한 뒤 (생성기 고정)")
    ax.axvspan(region[0], region[1], color=GOLD_FILL, lw=0)
    ax.plot(x, d_star, color=cp.INK, lw=2.4, alpha=0.35, label="닫힌식 $D^*$")
    ax.plot(x, d_learned, color=cp.BLUE, lw=1.6, label="학습한 $D$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    _clean_axes(ax, "$x$", "$|D - D^*|$", "차이")
    ax.axvspan(region[0], region[1], color=GOLD_FILL, lw=0)
    gap = np.abs(d_learned - d_star)
    ax.plot(x, gap, color=cp.RED, lw=1.6)
    ax.set_ylim(0, max(0.1, 1.15 * float(gap.max())))
    cp.label(axes[0], f"금색 띠 = 데이터가 있는 구간 · 손실 {d_loss:.4f} vs 최적 {d_loss_star:.4f}", y=-0.2, fontsize=9)
    cp.show(fig)


def plot_gan_1d(x, p_data, snapshots, d_curve, bins):
    """학습 전·중·후 생성 샘플의 히스토그램(파랑) vs 진짜 밀도(검정), 마지막 패널은 학습이 끝난 D(x). snapshots = [(제목, (N,) 샘플)]."""
    n = len(snapshots)
    fig, axes = cp.flow([1] * (n + 1), height=3.2, unit=3.0, wspace=0.35)
    last = np.histogram(snapshots[-1][1], bins=bins, density=True)[0]
    top = max(float(np.max(p_data)), float(last.max()))
    for ax, (title, samples) in zip(axes, snapshots):
        _clean_axes(ax, "$x$", None, title)
        ax.hist(samples, bins=bins, density=True, color=cp.BLUE, alpha=0.55, label="생성 샘플")
        ax.plot(x, p_data, color=cp.INK, lw=1.6, label="진짜 밀도")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1.15 * top)
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[-1]
    _clean_axes(ax, "$x$", "$D(x)$", "학습 뒤의 판별기")
    ax.plot(x, d_curve, color=cp.RED, lw=1.6)
    ax.axhline(0.5, color=cp.MUTED, lw=0.8, ls="--")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    cp.label(axes[1], "세로축은 마지막 패널에 맞췄습니다(학습 전의 좁고 높은 막대는 위로 잘림) · 생성기가 두 봉우리를 덮으면 판별기는 1/2 근처에서 구분을 포기합니다", y=-0.2, fontsize=9)
    cp.show(fig)


# ---------------------------------------------------------------- 4절: 봉우리별로 세기
def plot_mode_counting(clouds, titles, assignments, counts, bar_labels, mu, boundary, expected):
    """앞 세 패널: 점 구름을 배정된 봉우리 색(파랑/빨강)으로 칠하고 금색 점선은 책임확률이 1/2인 경계.
    마지막 패널: 봉우리별 개수 막대와 기대 개수(금색 점선). boundary = (xx, yy, r1) 격자, mu = (2, 2) 봉우리 중심."""
    n = len(clouds)
    fig, axes = cp.flow([1] * (n + 1), height=3.4, unit=3.2, wspace=0.32)
    xx, yy, r1 = boundary
    lim = (-5, 5)
    for ax, X, title, assign in zip(axes, clouds, titles, assignments):
        _clean_axes(ax, None, None, title)
        ax.contour(xx, yy, r1, levels=[0.5], colors=[cp.GOLD], linewidths=1.3, linestyles="--")
        assign = np.asarray(assign)
        ax.scatter(X[assign == 0, 0], X[assign == 0, 1], s=6, color=cp.BLUE, alpha=0.45, lw=0)
        ax.scatter(X[assign == 1, 0], X[assign == 1, 1], s=6, color=cp.RED, alpha=0.45, lw=0)
        ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=60, color=cp.INK, lw=2, zorder=4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    ax = axes[-1]
    _clean_axes(ax, None, "개수", "봉우리별 개수")
    counts = np.asarray(counts, float)
    k = np.arange(len(bar_labels))
    ax.bar(k - 0.19, counts[:, 0], width=0.38, color=cp.BLUE, label="봉우리 1")
    ax.bar(k + 0.19, counts[:, 1], width=0.38, color=cp.RED, label="봉우리 2")
    ax.axhline(expected[0], color=cp.GOLD, lw=1.2, ls="--")
    ax.axhline(expected[1], color=cp.GOLD, lw=1.2, ls="--")
    ax.set_xticks(k, bar_labels, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    cp.label(axes[1], "금색 점선 = 책임확률 1/2 경계와 기대 개수 $N\\pi_k$ · ✕ = 봉우리 중심", y=-0.16, fontsize=9)
    cp.show(fig)


# ---------------------------------------------------------------- 4절: mode coverage
def plot_mode_coverage(real, balanced, weak, counts, labels, mu, expected=None):
    """위 줄: 진짜 샘플과 균형 잡힌 설정의 seed 3개. 아래 줄: 개수 막대와 D가 약한 설정의 seed 3개. mu = 봉우리 중심 (2, 2), expected = 기대 개수 N·π (2,)."""
    fig = cp.new_figure((13, 6.2))
    gs = fig.add_gridspec(2, 4, wspace=0.3, hspace=0.45)
    lim = (-5, 5)

    def cloud(ax, X, title, color):
        _clean_axes(ax, None, None, title)
        ax.scatter(X[:, 0], X[:, 1], s=4, color=color, alpha=0.35, lw=0)
        ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=60, color=cp.GOLD, lw=2, zorder=4)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    cloud(fig.add_subplot(gs[0, 0]), real, "진짜 샘플", cp.INK)
    for j, X in enumerate(balanced):
        cloud(fig.add_subplot(gs[0, j + 1]), X, f"균형 설정 · seed {j + 1}", cp.BLUE)
    for j, X in enumerate(weak):
        cloud(fig.add_subplot(gs[1, j + 1]), X, f"D가 약한 설정 · seed {j + 1}", cp.RED)
    ax = fig.add_subplot(gs[1, 0])
    _clean_axes(ax, None, "봉우리별 개수", "책임확률 argmax로 센 개수")
    counts = np.asarray(counts, float)
    k = np.arange(len(labels))
    ax.bar(k - 0.19, counts[:, 0], width=0.38, color=cp.BLUE, label="봉우리 1")
    ax.bar(k + 0.19, counts[:, 1], width=0.38, color=cp.RED, label="봉우리 2")
    if expected is not None:
        ax.axhline(expected[0], color=cp.GOLD, lw=1.2, ls="--")
        ax.axhline(expected[1], color=cp.GOLD, lw=1.2, ls="--")
    ax.set_ylim(0, 1.3 * float(counts.max()))
    ax.set_xticks(k, labels, fontsize=8, rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    cp.label(ax, "금색 × = 진짜 봉우리 중심 · 금색 점선 = 기대 개수 $N\\pi_k$", y=-0.42, fontsize=9)
    cp.show(fig)


# ---------------------------------------------------------------- 5절: PCA 좌표
def plot_pca_roundtrip(image, coords, recon, basis, kept):
    """고정 예시 7 → 32개 좌표(띠) → 복원 이미지, 아래 줄은 기저 이미지 6장. kept = 32개 좌표가 남긴 분산 비율."""
    fig = cp.new_figure((12, 5.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], wspace=0.25, hspace=0.45)
    ax_x = fig.add_subplot(gs[0, 0:2])
    cp.blank(ax_x)
    cp.draw_image(ax_x, image, "$x_7$ (28×28)")
    ax_z = fig.add_subplot(gs[0, 2:4])
    cp.blank(ax_z)
    cp.draw_grid(ax_z, np.asarray(coords).reshape(4, 8), "좌표 $z = (x - \\mu) Q / s$ (32개)", kind="signed", fmt="{:.1f}", fontsize=8)
    ax_r = fig.add_subplot(gs[0, 4:6])
    cp.blank(ax_r)
    cp.draw_image(ax_r, np.clip(recon, 0, 1), "복원 $z\\,s\\,Q^{\\top} + \\mu$")
    cp.connect(fig, ax_x, ax_z, "$(x - \\mu) Q / s$")
    cp.connect(fig, ax_z, ax_r, "$z s Q^{\\top} + \\mu$")
    for j, b in enumerate(basis):
        ax = fig.add_subplot(gs[1, j])
        cp.blank(ax)
        cp.draw_image(ax, b, f"$q_{{{j}}}$", kind="signed")
    cp.label(fig.axes[-3], f"아래 줄 = 기저 이미지(고유벡터) · 32개 좌표가 남긴 분산 {kept:.0%}", y=-0.12, fontsize=9)
    cp.show(fig)


def plot_generated_images(images, labels, title, columns=8):
    """생성한 이미지들을 잉크(0 흰색, 1 검정)로 늘어놓습니다. images: (N, 28, 28) 0-1 배열, labels: 이미지마다 붙일 짧은 글."""
    rows = int(np.ceil(len(images) / columns))
    fig, grid = cp.flow([1] * columns, rows=rows, height=1.9, unit=1.5, wspace=0.12, hspace=0.45)
    panels = grid if rows > 1 else [grid]
    for index, ax in enumerate(ax for row in panels for ax in row):
        if index < len(images):
            cp.draw_image(ax, np.asarray(images[index]), str(labels[index]))
    fig.suptitle(title, color=cp.INK, fontsize=11)
    cp.show(fig)


def plot_digit_counts(gen_counts, real_counts):
    """생성 샘플과 진짜 샘플의 숫자별 비율(클래스 평균 최근접으로 배정)."""
    fig, axes = cp.flow([1], height=3.2, unit=7.5)
    ax = axes[0]
    _clean_axes(ax, "가장 가까운 클래스 평균", "비율", "생성 샘플의 숫자별 비율 (클래스 평균 최근접)")
    k = np.arange(10)
    gen, real = np.asarray(gen_counts, float), np.asarray(real_counts, float)
    ax.bar(k - 0.2, gen / gen.sum(), width=0.4, color=cp.BLUE, label="생성")
    ax.bar(k + 0.2, real / real.sum(), width=0.4, color=cp.MUTED, label="진짜 (앞 5,000장)")
    ax.set_xticks(k, [str(i) for i in k])
    ax.legend(frameon=False, fontsize=8.5)
    cp.show(fig)


def plot_variant_bars(names, metrics):
    """설정별 막대 여러 판. metrics = {제목: (값 목록, 기준선 또는 None)}. 차이가 작아 세로축을 값 범위에 맞춰 자릅니다."""
    fig, axes = cp.flow([1] * len(metrics), height=3.1, unit=3.6, wspace=0.45)
    k = np.arange(len(names))
    for ax, (title, (values, baseline)) in zip(axes, metrics.items()):
        _clean_axes(ax, None, None, title)
        values = np.asarray(values, float)
        ax.bar(k, values, color=cp.BLUE, width=0.6)
        marks = np.append(values, baseline) if baseline is not None else values
        low, high = float(marks.min()), float(marks.max())
        pad = max(0.4 * (high - low), 0.02)
        ax.set_ylim(low - pad, high + pad)
        if baseline is not None:
            ax.axhline(baseline, color=cp.GOLD, lw=1.4, ls="--")
        ax.set_xticks(k, names, fontsize=8.5)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    cp.label(axes[0], "금색 점선 = 균형점의 값 · 세로축은 값 범위에 맞춰 잘랐습니다", y=-0.16, fontsize=9)
    cp.show(fig)
