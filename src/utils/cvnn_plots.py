"""10b CVNN 노트북의 개념 그림.

복소수 곱·마이크 신호·펼친 실수 행렬·크기와 위상 표·푸리에 교환 복원·복소 경사하강·|tanh|·활성화의 화살표·
방향 맞히기 학습 결과는 노트북 본문에서 계산하고 여기서는 받은 배열만 그립니다.
부품은 concept_plots(cp), 곡선 축 정리는 gradient_plots._clean_axes, 원칙은 docs/DESIGN.md.
"""

import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from . import concept_plots as cp
from .gradient_plots import _clean_axes

PLANE_LINE = "#c9ced6"    # 복소평면의 실수축·허수축·단위원


# ---------------------------------------------------------------- 공용 부품
def _plane(ax, lim, *, unit=True, labels=True, title=None):
    """축을 끈 복소평면: 옅은 실수축·허수축, 점선 단위원, 가로세로 같은 축척."""
    cp.blank(ax)
    ax.set_aspect("equal")
    ax.axhline(0, color=PLANE_LINE, lw=0.8, zorder=0)
    ax.axvline(0, color=PLANE_LINE, lw=0.8, zorder=0)
    if unit:
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=PLANE_LINE, lw=0.8, ls="--", zorder=0))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if labels:
        ax.text(lim * 0.97, lim * 0.03, "Re", ha="right", va="bottom", fontsize=8.5, color=cp.MUTED)
        ax.text(lim * 0.03, lim * 0.97, "Im", ha="left", va="top", fontsize=8.5, color=cp.MUTED)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=10.5, pad=6)
    return ax


def _arrow(ax, z, color, *, start=0j, lw=1.8, alpha=1.0, zorder=3, scale=12):
    """start 에서 z 로 가는 화살표(복소수 좌표)."""
    ax.annotate("", xy=(float(np.real(z)), float(np.imag(z))), xytext=(float(np.real(start)), float(np.imag(start))),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "alpha": alpha, "mutation_scale": scale,
                            "shrinkA": 0, "shrinkB": 0}, zorder=zorder)


def _dial(ax, z, title=None, note=None, *, color=cp.BLUE):
    """반지름 1 원 안의 화살표 하나(페이저 한 개). 마이크 하나의 신호."""
    cp.blank(ax)
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=PLANE_LINE, lw=0.9))
    ax.plot([-1, 1], [0, 0], color=PLANE_LINE, lw=0.6)
    _arrow(ax, z, color, lw=1.6, scale=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    if title:
        ax.set_title(title, color=cp.INK, fontsize=9, pad=2)
    if note:
        ax.text(0, -1.3, note, ha="center", va="top", fontsize=8, color=cp.MUTED)


# ---------------------------------------------------------------- 1절
def plot_complex_multiply(z, wz, w, mic_z):
    """왼쪽: 화살표 z(회색)에 w를 곱한 wz(파랑). 첫 화살표(금색)는 손계산. 오른쪽: 마이크 8개의 신호를 시계 8개로."""
    z, wz, mic_z = np.asarray(z), np.asarray(wz), np.asarray(mic_z)
    fig = cp.new_figure((13.5, 4.9))
    gs = fig.add_gridspec(2, 6, width_ratios=[2.7, 0.35, 1, 1, 1, 1], wspace=0.12, hspace=0.55, top=0.84, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    lim = 1.12 * float(np.abs(wz).max())
    _plane(ax, lim, title=f"길이 {np.abs(w):.0f}·각도 {np.rad2deg(np.angle(w)):.0f}°인 $w$ 를 곱하면: 모두 {np.rad2deg(np.angle(w)):.0f}° 돌고 {np.abs(w):.0f}배")
    for k in range(len(z)):
        hand = k == 0
        _arrow(ax, z[k], cp.GOLD if hand else cp.MUTED, lw=2.0 if hand else 1.4)
        _arrow(ax, wz[k], cp.GOLD if hand else cp.BLUE, lw=2.2 if hand else 1.6)
    t0, t1 = np.rad2deg(np.angle(z[0])), np.rad2deg(np.angle(wz[0]))
    ax.add_patch(Arc((0, 0), 1.3, 1.3, theta1=t0, theta2=t1, color=cp.GOLD, lw=1.4))
    ax.text(z[0].real + 0.08, z[0].imag - 0.05, f"$z$ : 길이 {np.abs(z[0]):.3f}, {t0:.0f}°", fontsize=8.5, color=cp.INK, va="top")
    ax.text(wz[0].real + 0.1, wz[0].imag, f"$wz$ : 길이 {np.abs(wz[0]):.3f}, {t1:.0f}°", fontsize=8.5, color=cp.INK, va="center")
    cp.label(ax, "회색 = 곱하기 전, 파랑 = 곱한 뒤, 금색 = 손계산", y=-0.02)
    for m in range(len(mic_z)):
        axd = fig.add_subplot(gs[m // 4, 2 + m % 4])
        _dial(axd, mic_z[m], f"마이크 {m}", f"{np.rad2deg(np.angle(mic_z[m])):.1f}°")
    fig.text(0.69, 0.955, f"마이크 신호 $z_m$ : 길이는 모두 {np.abs(mic_z[0]):.1f}, 각도만 이웃마다 같은 폭으로 돎",
             ha="center", va="top", fontsize=10.5, color=cp.INK)
    cp.show(fig)


def plot_real_block(W, R):
    """복소 W:(D, K)의 실수부·허수부 → 번갈아 펼친 실수 행렬 R:(2D, 2K). 2×2 블록 경계와 첫 블록(금색)."""
    W, R = np.asarray(W), np.asarray(R, float)
    D, K = W.shape
    vmax = float(np.abs(R).max())
    fig = cp.new_figure((10.5, 4.6))
    gs = fig.add_gridspec(3, 3, width_ratios=[K, K, 2 * K], height_ratios=[1, 2, 1], wspace=0.4, hspace=0.0,
                          left=0.05, right=0.97, top=0.8, bottom=0.14)
    ax_re = fig.add_subplot(gs[1, 0])
    cp.blank(ax_re)
    cp.draw_grid(ax_re, W.real, r"$\mathrm{Re}\,W$ $(D, K)$", fmt="{:.1f}", vmax=vmax, fontsize=9,
                 row_labels=[f"d={d}" for d in range(D)], col_labels=[f"k={k}" for k in range(K)])
    ax_im = fig.add_subplot(gs[1, 1])
    cp.blank(ax_im)
    cp.draw_grid(ax_im, W.imag, r"$\mathrm{Im}\,W$ $(D, K)$", fmt="{:.1f}", vmax=vmax, fontsize=9,
                 row_labels=[f"d={d}" for d in range(D)], col_labels=[f"k={k}" for k in range(K)])
    ax_r = fig.add_subplot(gs[:, 2])
    cp.blank(ax_r)
    rows = [f"Re $x_{d}$" if i % 2 == 0 else f"Im $x_{d}$" for d in range(D) for i in range(2)]
    cols = [f"Re $z_{k}$" if i % 2 == 0 else f"Im $z_{k}$" for k in range(K) for i in range(2)]
    cp.draw_grid(ax_r, R, f"펼친 실수 행렬 $R$ $(2D, 2K)$ = ({2 * D}, {2 * K})", fmt="{:.1f}", vmax=vmax, fontsize=8.5,
                 row_labels=rows, col_labels=cols)
    for d in range(1, D):
        ax_r.axhline(2 * d - 0.5, color=cp.INK, lw=1.0)
    for k in range(1, K):
        ax_r.axvline(2 * k - 0.5, color=cp.INK, lw=1.0)
    ax_r.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, ec=cp.GOLD, lw=2.6, zorder=4))
    ax_r.tick_params(axis="x", labelsize=7.5)
    cp.label(ax_r, f"금색 블록 = $W_{{00}}$ 한 칸의 [[a, b], [−b, a]]\n{R.size}칸 중 자유로운 수는 {2 * W.size}개", y=-0.03, fontsize=9)
    cp.connect(fig, ax_im, ax_r, "번갈아 펼치기")
    cp.show(fig)


# ---------------------------------------------------------------- 2절
def plot_magnitude_vs_phase(alpha_deg, magnitude, phase_deg, step_deg, recovered_deg):
    """방향 4개 × 마이크 8개의 크기 표(모두 같음)와 위상 표(행마다 일정하게 늘어남), 이웃 위상 차이와 되찾은 각도."""
    magnitude, phase_deg = np.asarray(magnitude, float), np.asarray(phase_deg, float)
    n_dir, M = magnitude.shape
    rows = [f"α = {a:g}°" for a in alpha_deg]
    cols = [f"m={m}" for m in range(M)]
    fig = cp.new_figure((15, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[M, M, 2.6], wspace=0.28, left=0.05, right=0.98, top=0.78, bottom=0.08)
    ax_m = fig.add_subplot(gs[0])
    cp.blank(ax_m)
    cp.draw_grid(ax_m, magnitude, "크기 $|z_m|$ : 방향과 무관하게 모두 같음", kind="count", fmt="{:.2f}", fontsize=9,
                 vmax=2 * float(magnitude.max()), row_labels=rows, col_labels=cols)
    ax_p = fig.add_subplot(gs[1])
    cp.blank(ax_p)
    cp.draw_grid(ax_p, phase_deg, r"위상 $\arg z_m$ (도) : 이웃마다 같은 폭으로 늘어남", fmt="{:.0f}", fontsize=9, vmax=180,
                 row_labels=rows, col_labels=cols)
    ax_s = fig.add_subplot(gs[2])
    cp.blank(ax_s)
    table = np.stack([np.asarray(step_deg, float), np.asarray(recovered_deg, float)], axis=1)
    cp.draw_grid(ax_s, table, "이웃 곱의 각도 → α", kind="plain", fmt="{:.1f}", fontsize=9, col_labels=["πsinα (도)", "α (도)"])
    cp.connect(fig, ax_m, ax_p, "크기 말고 위상")
    cp.connect(fig, ax_p, ax_s, "arg($z_{m+1}\\bar z_m$)")
    cp.show(fig)


def plot_phase_swap(images, log_magnitudes, phases, swaps, names, correlations):
    """행마다 원본 · 푸리에 크기(로그, 가운데가 낮은 주파수) · 푸리에 위상 · 위상을 이 숫자에서 가져온 교환 복원."""
    fig = cp.new_figure((11.5, 6.2))
    gs = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.42, left=0.03, right=0.97, top=0.9, bottom=0.08)
    for r in range(2):
        other = names[1 - r]
        ax = fig.add_subplot(gs[r, 0])
        cp.blank(ax)
        cp.draw_image(ax, images[r], f"숫자 {names[r]}")
        ax = fig.add_subplot(gs[r, 1])
        cp.blank(ax)
        mag = np.asarray(log_magnitudes[r], float)
        ax.imshow(mag, cmap="Blues", vmin=0, vmax=float(mag.max()), interpolation="nearest")
        ax.set_title(f"크기 $\\log(1+|F_{names[r]}|)$", color=cp.INK, fontsize=10.5, pad=6)
        ax = fig.add_subplot(gs[r, 2])
        cp.blank(ax)
        cp.draw_image(ax, phases[r], f"위상 $\\angle F_{names[r]}$ (−π–π)", kind="signed", vmax=np.pi)
        ax = fig.add_subplot(gs[r, 3])
        cp.blank(ax)
        swap = np.asarray(swaps[r], float)
        ax.imshow(swap, cmap=cp.GRAY_CMAP, vmin=0, vmax=float(swap.max()), interpolation="nearest")
        ax.add_patch(Rectangle((-0.5, -0.5), swap.shape[1], swap.shape[0], fill=False, ec=cp.GOLD, lw=1.6))
        ax.set_title(f"{names[r]}의 위상 + {other}의 크기", color=cp.INK, fontsize=10.5, pad=6)
        c = correlations[r]
        cp.label(ax, f"{names[0]}과 상관 {c[0]:.2f} · {names[1]}과 상관 {c[1]:.2f}", y=-0.04, fontsize=9, color=cp.INK)
    cp.show(fig)


# ---------------------------------------------------------------- 3절
def plot_complex_descent(grid_re, grid_im, grid_loss, arrow_w, arrow_g, path_good, path_wrong, w_star, z, c):
    """왼쪽: 복소평면 위 손실 등고선, −g 화살표, g로 내려간 경로(파랑)와 켤레를 잘못 둔 경로(빨강). 오른쪽: 경로를 따라 w_t z 화살표."""
    path_good, path_wrong = np.asarray(path_good), np.asarray(path_wrong)
    fig = cp.new_figure((13, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.15, left=0.03, right=0.97, top=0.88, bottom=0.08)
    ax = fig.add_subplot(gs[0])
    cp.blank(ax)
    ax.set_aspect("equal")
    lo_x, hi_x = float(np.min(grid_re)), float(np.max(grid_re))
    lo_y, hi_y = float(np.min(grid_im)), float(np.max(grid_im))
    ax.contour(grid_re, grid_im, grid_loss, levels=12, cmap="Blues", linewidths=0.9, alpha=0.8)
    ax.axhline(0, color=PLANE_LINE, lw=0.8, zorder=0)
    ax.axvline(0, color=PLANE_LINE, lw=0.8, zorder=0)
    step = -np.asarray(arrow_g)
    length = np.abs(step)
    unit = step / np.maximum(length, 1e-12) * 0.28
    ax.quiver(np.real(arrow_w), np.imag(arrow_w), unit.real, unit.imag, color=cp.MUTED, angles="xy", scale_units="xy",
              scale=1, width=0.0035, headwidth=4)
    inside = (path_wrong.real > lo_x) & (path_wrong.real < hi_x) & (path_wrong.imag > lo_y) & (path_wrong.imag < hi_y)
    ax.plot(path_wrong[inside].real, path_wrong[inside].imag, "o--", color=cp.RED, lw=1.4, ms=4,
            label="켤레를 잘못 둔 $2\\partial L/\\partial w$ 로 갱신")
    if not inside.all():
        cp.label(ax, f"빨간 경로의 점 {int((~inside).sum())}개는 그림 밖", y=-0.02, color=cp.RED)
    ax.plot(path_good.real, path_good.imag, "o-", color=cp.BLUE, lw=1.6, ms=4, label="$g=\\partial L/\\partial x+i\\,\\partial L/\\partial y$ 로 갱신")
    ax.scatter([path_good[0].real], [path_good[0].imag], marker="s", s=60, color=cp.INK, zorder=5)
    ax.text(path_good[0].real + 0.12, path_good[0].imag + 0.1, "시작 $w_0$", fontsize=9, color=cp.INK)
    ax.scatter([np.real(w_star)], [np.imag(w_star)], marker="*", s=220, color=cp.GOLD, zorder=6)
    ax.text(np.real(w_star) - 0.2, np.imag(w_star) - 0.12, "$w^* = c/z_1$", fontsize=9.5, color=cp.INK, ha="right", va="top")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.text(hi_x * 0.97, 0.05, "Re $w$", ha="right", va="bottom", fontsize=8.5, color=cp.MUTED)
    ax.text(0.05, hi_y * 0.97, "Im $w$", ha="left", va="top", fontsize=8.5, color=cp.MUTED)
    ax.set_title("손실 $L(w)=|wz_1-c|^2$ 의 등고선과 $-g$ 방향(회색 화살표)", color=cp.INK, fontsize=10.5, pad=6)
    ax.legend(frameon=True, facecolor="white", edgecolor="white", fontsize=8.5, loc="lower left")
    ax2 = fig.add_subplot(gs[1])
    lim = 1.35
    _plane(ax2, lim, title="경로를 따라 $w_t z_1$ : 돌고 늘어나 $c = 1$ 에 겹침")
    _arrow(ax2, z, cp.MUTED, lw=1.4)
    ax2.text(np.real(z) - 0.05, np.imag(z) + 0.06, "$z_1$", fontsize=9.5, color=cp.MUTED, ha="right")
    shades = np.linspace(0.25, 1.0, len(path_good))
    for t, w in enumerate(path_good):
        _arrow(ax2, w * z, cp.BLUE, lw=1.2 + 0.6 * shades[t], alpha=float(shades[t]))
    _arrow(ax2, c, cp.GOLD, lw=2.4, zorder=5)
    ax2.text(np.real(c), -0.12, "$c = 1$", fontsize=9.5, color=cp.INK, ha="center", va="top")
    cp.label(ax2, "옅은 파랑 = 처음, 진한 파랑 = 나중 스텝", y=-0.02)
    cp.show(fig)


# ---------------------------------------------------------------- 4절
def plot_tanh_blowup(grid_re, grid_im, abs_tanh, line_t, real_axis, imag_axis, cap=5.0):
    """왼쪽: 복소평면의 |tanh z| (cap 에서 자름)와 극 ±iπ/2. 오른쪽: 실수축의 |tanh x| 와 허수축의 |tanh iy|."""
    fig = cp.new_figure((12.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25, left=0.04, right=0.97, top=0.86, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    cp.blank(ax)
    extent = [float(np.min(grid_re)), float(np.max(grid_re)), float(np.min(grid_im)), float(np.max(grid_im))]
    ax.imshow(np.minimum(np.asarray(abs_tanh, float), cap), cmap="Blues", vmin=0, vmax=cap, origin="lower", extent=extent,
              interpolation="nearest")
    ax.axhline(0, color=cp.INK, lw=1.2)
    ax.axvline(0, color=cp.GOLD, lw=1.6, ls="--")
    ax.scatter([0, 0], [np.pi / 2, -np.pi / 2], marker="x", s=70, color=cp.INK, zorder=5)
    bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    ax.text(0.55, np.pi / 2, "극 $i\\pi/2$", fontsize=9.5, color=cp.INK, va="center", bbox=bbox)
    ax.text(0.55, -np.pi / 2, "극 $-i\\pi/2$", fontsize=9.5, color=cp.INK, va="center", bbox=bbox)
    ax.text(extent[1] * 0.97, 0.1, "Re", ha="right", va="bottom", fontsize=8.5, color=cp.INK)
    ax.text(0.1, extent[3] * 0.97, "Im", ha="left", va="top", fontsize=8.5, color=cp.INK)
    ax.set_title(f"$|\\tanh z|$ ({cap:g}에서 자름, 진할수록 큼)", color=cp.INK, fontsize=10.5, pad=6)
    cp.label(ax, "검은 실선 = 실수축, 금색 점선 = 허수축", y=-0.03)
    ax2 = fig.add_subplot(gs[1])
    _clean_axes(ax2, "$x$ 또는 $y$", "크기", "실수축에서는 1 이하, 허수축에서는 $\\pi/2$ 에서 발산")
    imag_axis = np.asarray(imag_axis, float)
    ax2.plot(line_t, real_axis, color=cp.INK, lw=1.8, label="$|\\tanh x|$ (실수축)")
    ax2.plot(line_t, np.where(imag_axis > cap * 1.2, np.nan, imag_axis), color=cp.GOLD, lw=2.0, ls="--", label="$|\\tanh iy|=|\\tan y|$ (허수축)")
    for pole in (-np.pi / 2, np.pi / 2):
        ax2.axvline(pole, color=cp.MUTED, lw=0.8, ls=":")
    ax2.set_ylim(0, cap)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    cp.show(fig)


def plot_activation_fields(z, mod_out, crelu_out, b, turn_deg, gap_mod, gap_crelu):
    """격자점 z에서 f(z)로 가는 화살표: modReLU(방향 유지, 짧은 것은 0)와 CReLU(사분면에 따라 꺾임), 그리고 입력을 φ만큼 돌렸을 때 등변에서 벗어난 정도."""
    z, mod_out, crelu_out = np.asarray(z), np.asarray(mod_out), np.asarray(crelu_out)
    fig = cp.new_figure((15, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.22, left=0.02, right=0.98, top=0.86, bottom=0.14)
    lim = 1.25 * float(np.abs(z).max())
    panels = [(mod_out, f"modReLU ($b = {b:g}$) : 방향은 그대로, 길이만 줄임"), (crelu_out, "CReLU : 실수부·허수부의 음수를 따로 자름")]
    for i, (out, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        _plane(ax, lim, unit=False, title=title)
        if i == 0:
            ax.add_patch(Circle((0, 0), -b, fill=False, ec=cp.GOLD, lw=1.6, ls="--", zorder=1))
            ax.text(-lim * 0.97, -lim * 0.97, f"금색 점선 원 안 $|z| < {-b:g}$ 는 0", ha="left", va="bottom", fontsize=8.5,
                    color=cp.INK, zorder=6)
        ax.scatter(z.real, z.imag, s=10, color=cp.MUTED, zorder=2)
        for zi, oi in zip(z, out):
            if abs(oi - zi) > 1e-9:
                _arrow(ax, oi, cp.BLUE, start=zi, lw=1.0, scale=8)
        cp.label(ax, "회색 점 = 입력 $z$, 파란 화살표 = $z \\to f(z)$", y=-0.02)
    ax3 = fig.add_subplot(gs[2])
    _clean_axes(ax3, "돌린 각도 φ (도)", r"$\max|f(e^{i\varphi}z)-e^{i\varphi}f(z)|$", "입력을 φ만큼 돌렸을 때 등변에서 벗어난 정도")
    ax3.plot(turn_deg, gap_crelu, color=cp.INK, lw=1.8, label="CReLU")
    ax3.plot(turn_deg, gap_mod, color=cp.BLUE, lw=2.2, label="modReLU (모든 φ에서 0)")
    ax3.set_xticks([0, 90, 180, 270, 360])
    ax3.set_ylim(-0.05 * float(np.max(gap_crelu)), 1.15 * float(np.max(gap_crelu)))
    ax3.legend(frameon=False, fontsize=9, loc="upper right")
    cp.show(fig)


# ---------------------------------------------------------------- 5절
def plot_doa_samples(sample_z, sample_real, sweep_deg, response, class_deg, true_deg):
    """두 표본(둘째는 첫째의 전체 위상만 돌린 것)의 마이크별 화살표, 실수 MLP가 보는 16개 숫자, 각도를 훑은 빔포머 응답."""
    sample_z = np.asarray(sample_z)
    n, M = sample_z.shape
    fig = cp.new_figure((15, 5.4))
    gs = fig.add_gridspec(n + 1, M + 2, width_ratios=[1] * M + [1.1, 6.5], height_ratios=[1] * n + [0.75],
                          wspace=0.12, hspace=0.5, left=0.07, right=0.98, top=0.86, bottom=0.14)
    scale = float(np.abs(sample_z).max())
    for r in range(n):
        for m in range(M):
            ax = fig.add_subplot(gs[r, m])
            _dial(ax, sample_z[r, m] / scale, f"$m={m}$" if r == 0 else None, color=cp.BLUE if r == 0 else cp.INK)
        fig.text(0.015, 1 - (0.86 - 0.12) * (r + 0.5) / (n + 0.75) - 0.14, f"표본 {'A' if r == 0 else 'B'}",
                 ha="left", va="center", fontsize=10, color=cp.INK)
    ax_rows = fig.add_subplot(gs[n, :M])
    cp.blank(ax_rows)
    cp.draw_rows(ax_rows, sample_real, "실수 MLP가 보는 16개 숫자: 실수부·허수부 번갈아 (윗줄 A, 아랫줄 B)", kind="signed")
    ax = fig.add_subplot(gs[:, M + 1])
    _clean_axes(ax, "훑는 각도 α (도)", "$|a(\\alpha)^H z|^2/8$", f"빔포머 응답: {true_deg:g}°에서 온 소리")
    for k in class_deg:
        ax.axvline(k, color=cp.MUTED, lw=0.7, ls=":")
    ax.axvline(true_deg, color=cp.GOLD, lw=1.6)
    ax.plot(sweep_deg, response[0], color=cp.BLUE, lw=3.2, label="표본 A")
    ax.plot(sweep_deg, response[1], color=cp.INK, lw=1.2, ls="--", label="표본 B (A의 전체 위상만 돌림)")
    ax.set_xticks(list(class_deg))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    cp.label(ax, "점선 = 방향 6개, 금색 = 참 방향", y=-0.13)
    fig.text(0.36, 0.955, "같은 소리의 전체 위상만 돌리면: 화살표와 16개 숫자는 달라지고, $|a^H z|^2$ 응답은 그대로",
             ha="center", va="top", fontsize=10.5, color=cp.INK)
    cp.show(fig)


def plot_sample_curves(sizes, acc_complex, acc_real, acc_magnitude, beam_acc, chance, n_params):
    """학습 표본 수(로그 축)별 test 정확도: seed마다 점, 평균은 선. 빔포머(금색)와 우연 수준(회색 점선).
    n_params: 세 모델 (a)·(b)·(c)의 실수 파라미터 수 세 개(범례에 적음)."""
    sizes = np.asarray(sizes)
    fig = cp.new_figure((9.5, 4.4))
    ax = fig.add_subplot(111)
    _clean_axes(ax, "학습 표본 수 (로그 축)", "test 정확도 (3,000개)", "학습 표본 수에 따른 방향 맞히기 정확도")
    series = [(acc_complex, cp.BLUE, f"(a) 복소 모델 · 파라미터 {n_params[0]}개"), (acc_real, cp.INK, f"(b) 실수 MLP · 파라미터 {n_params[1]}개"),
              (acc_magnitude, cp.MUTED, f"(c) 크기만 MLP · 파라미터 {n_params[2]}개")]
    for acc, color, name in series:
        acc = np.asarray(acc, float)
        for seed_acc in acc.T:
            ax.scatter(sizes, seed_acc, s=12, color=color, alpha=0.35, zorder=2)
        ax.plot(sizes, acc.mean(axis=1), "o-", color=color, lw=1.9, ms=5, label=name, zorder=3)
    ax.axhline(beam_acc, color=cp.GOLD, lw=1.6, ls="--", label=f"빔포머 (학습 없음) {beam_acc:.3f}")
    ax.axhline(chance, color=cp.MUTED, lw=1.0, ls=":", label=f"우연 수준 1/6 = {chance:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, 1)
    for acc, color, dy in ((acc_complex, cp.BLUE, -0.055), (acc_real, cp.INK, -0.03)):
        mean = np.asarray(acc, float).mean(axis=1)
        ax.text(sizes[1] * 1.1, mean[1] + dy, f"{mean[1]:.3f}", color=color, fontsize=9.5, ha="left", va="center")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    cp.show(fig)


def plot_phase_invariance(turn_deg, prob_complex, prob_real, changed, names):
    """왼쪽: test 표본 하나의 전체 위상을 φ만큼 돌리며 잰 P(정답). 오른쪽: 무작위 위상으로 돌린 test 전체에서 예측이 바뀐 비율."""
    fig = cp.new_figure((13, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.3, left=0.07, right=0.97, top=0.86, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "돌린 각도 φ (도)", "P(정답)", "test 표본 0번에 $e^{i\\varphi}$ 를 곱하며 잰 정답 확률")
    ax.plot(turn_deg, prob_real, color=cp.INK, lw=1.6, label=names[1])
    ax.plot(turn_deg, prob_complex, color=cp.BLUE, lw=2.4, label=names[0])
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax2 = fig.add_subplot(gs[1])
    cp.blank(ax2)
    changed = np.asarray(changed, float)
    right = max(0.05, 1.35 * float(changed.max()))
    ax2.barh([0, 1], changed, height=0.42, color=[cp.BLUE, cp.INK])
    for k, value in enumerate(changed):
        ax2.text(value + 0.02 * right, k, f"{value:.1%}", va="center", fontsize=10, color=cp.INK)
        ax2.text(-0.02 * right, k, names[k], va="center", ha="right", fontsize=10, color=cp.INK)
    ax2.axvline(0, color=cp.MUTED, lw=0.8)
    ax2.set_xlim(-0.45 * right, right)
    ax2.set_ylim(1.8, -0.8)
    ax2.set_title("test 3,000개를 표본마다 무작위 위상으로 돌렸을 때\n예측이 바뀐 비율", color=cp.INK, fontsize=10.5, pad=6)
    cp.show(fig)
