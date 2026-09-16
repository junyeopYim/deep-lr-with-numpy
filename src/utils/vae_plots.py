"""25 VAE 노트북의 개념 그림과 구조 도식.

encoder가 낸 (μ, logσ²), 재매개화 샘플 z, decoder의 복원 확률, 두 손실 항, 기울기 두 갈래,
학습 기록, 잠재 평면 격자·산점, 혼합 데이터의 ELBO는 모두 노트북 본문에서 계산하고
여기서는 받은 배열만 그립니다. 부품은 concept_plots(cp), 계산 그래프는 13번의
autodiff_plots.draw_compute_graph, 원칙은 docs/DESIGN.md.

색: 재구성 항·q·복원 = 파랑, KL 항·오차 = 빨강, 지금 보는 표본 하나·기준선 = 금색, 나머지 회색.
"""

import numpy as np
from matplotlib.patches import Circle, Ellipse

from . import concept_plots as cp
from .autodiff_plots import draw_compute_graph
from .gradient_plots import _clean_axes

RECON_COLOR = cp.BLUE      # 재구성 항
KL_COLOR = cp.RED          # KL 항
MARK_COLOR = cp.GOLD       # 지금 보는 표본 하나 · 기준선
DIGIT_CMAP = "tab10"       # 숫자 0–9 산점 (이 파일에서 유일하게 범주형 색을 쓰는 자리)


def _prior_rings(ax, radii=(1.0, 2.0)):
    """잠재 평면에 사전분포 N(0, I)의 반지름 1σ·2σ 원을 회색 점선으로 두른다."""
    for radius in radii:
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, ec=cp.MUTED, lw=1.0, ls=":"))
    ax.scatter([0.0], [0.0], color=cp.MUTED, s=16, marker="+")


def _value_bars(ax, names, values, colors, fmt="{:.2f}", log=False):
    """세로 막대와 값. 음수는 막대 아래에, 로그 눈금이면 막대 위 조금 떨어진 자리에 적는다."""
    values = np.asarray(values, dtype=float)
    if log:
        ax.set_yscale("log")
        ax.bar(names, values, color=colors, width=0.6)
        for i, value in enumerate(values):
            ax.text(i, value * 1.35, fmt.format(value), ha="center", va="bottom", fontsize=8.5, color=cp.INK)
        ax.set_ylim(float(values.min()) * 0.35, float(values.max()) * 6.0)
    else:
        ax.bar(names, values, color=colors, width=0.6)
        ax.axhline(0, color=cp.MUTED, lw=0.8)
        span = float(np.abs(values).max() or 1.0)
        for i, value in enumerate(values):
            offset = 0.05 * span if value >= 0 else -0.05 * span
            ax.text(i, value + offset, fmt.format(value), ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8.5, color=cp.INK)
        ax.margins(y=0.26)
    ax.tick_params(axis="x", labelsize=8.5)
    return ax


# ---------------------------------------------------------------- 1절: 두 신경망과 그 사이의 샘플
def plot_vae_flow(image, mu, sigma, z, recon_prob):
    """고정 예시 한 장이 x → (μ, σ) → z → 복원 확률로 흐르는 그림.

    image: (28, 28), mu·sigma·z: (2,) 잠재 좌표, recon_prob: (28, 28) 화소별 확률.
    """
    mu, sigma, z = np.asarray(mu, float), np.asarray(sigma, float), np.asarray(z, float)
    fig = cp.new_figure((14, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1], wspace=0.62)

    ax_x = fig.add_subplot(gs[0])
    cp.blank(ax_x)
    cp.draw_image(ax_x, image, "$x$ · 784개 화소")

    ax_q = fig.add_subplot(gs[1])
    _clean_axes(ax_q, "값", None, "$q(z|x) = \\mathcal{N}(\\mu,\\ \\sigma^2)$")
    ax_q.axvspan(-1.0, 1.0, color=cp.MUTED, alpha=0.18, lw=0)
    for k in range(len(mu)):
        ax_q.plot([mu[k] - sigma[k], mu[k] + sigma[k]], [k, k], color=RECON_COLOR, lw=5, solid_capstyle="round")
        ax_q.scatter([mu[k]], [k], color=cp.INK, s=40, zorder=5)                # μ는 짙은 점 (금색은 셋째 판의 z 하나에만)
        ax_q.text(mu[k], k + 0.2, f"$\\mu_{k} = {mu[k]:.2f}$, $\\sigma_{k} = {sigma[k]:.2f}$",
                  ha="center", va="bottom", fontsize=8.5, color=cp.INK)
    ax_q.set_yticks(range(len(mu)), [f"$z_{k}$" for k in range(len(mu))])
    ax_q.set_ylim(-0.6, len(mu) - 0.15)
    cp.label(ax_q, "회색 띠 = 사전분포 $\\pm 1\\sigma$ · 파랑 = $\\mu \\pm \\sigma$ · 짙은 점 = $\\mu$", y=-0.22)

    ax_z = fig.add_subplot(gs[2])
    _clean_axes(ax_z, "$z_0$", "$z_1$", "$z = \\mu + \\sigma \\odot \\epsilon$ 한 번 뽑기")
    _prior_rings(ax_z)
    ax_z.add_patch(Ellipse(tuple(mu), 2 * sigma[0], 2 * sigma[1], fill=True,
                           facecolor=RECON_COLOR, alpha=0.25, ec=RECON_COLOR, lw=1.2))
    ax_z.scatter([mu[0]], [mu[1]], color=cp.INK, s=30, zorder=4)              # 둘째 판과 같은 짙은 점 = μ
    ax_z.scatter([z[0]], [z[1]], color=MARK_COLOR, s=90, marker="*", zorder=5, edgecolor=cp.INK, lw=0.5)
    ax_z.annotate("$z$", xy=(z[0], z[1]), xytext=(9, 4), textcoords="offset points",
                  fontsize=10, color=cp.INK)
    ax_z.set_aspect("equal", adjustable="datalim")
    cp.label(ax_z, "점선 원 = 사전분포 $1\\sigma$·$2\\sigma$ · 짙은 점 = $\\mu$ · 금색 별 = 뽑은 $z$", y=-0.22)

    ax_r = fig.add_subplot(gs[3])
    cp.blank(ax_r)
    cp.draw_image(ax_r, recon_prob, "복원 확률 $p = \\sigma(s)$")
    cp.connect(fig, ax_x, ax_q, "encoder")
    cp.connect(fig, ax_q, ax_z, "$+\\ \\sigma \\odot \\epsilon$")
    cp.connect(fig, ax_z, ax_r, "decoder")
    cp.show(fig)


# ---------------------------------------------------------------- 2절: 두 손실 항
def plot_loss_terms(image, prob_map, bce_map, prob_title, recon_names, recon_values, kl_value):
    """정답 잉크·후보 복원 확률·화소별 BCE와, 후보별 재구성 항과 KL 항의 크기(로그 눈금)."""
    fig = cp.new_figure((13.5, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.42)
    contrast = float(np.quantile(bce_map, 0.985))          # 한두 화소의 큰 값에 가려지지 않게 상한을 자른다
    for i, (img, title, kind) in enumerate([(image, "$x$ (정답 잉크)", "gray"),
                                            (prob_map, prob_title, "gray"),
                                            (bce_map, "화소별 BCE", "signed")]):
        ax = fig.add_subplot(gs[i])
        cp.blank(ax)
        cp.draw_image(ax, img, title, kind=kind, vmax=contrast if kind == "signed" else None)
    cp.label(fig.axes[2], f"784개를 더하면 재구성 항 {float(np.sum(bce_map)):.1f} nat", y=-0.08)

    ax_b = fig.add_subplot(gs[3])
    _clean_axes(ax_b, None, "nat (이미지 한 장당, 로그 눈금)", "재구성 항 세 후보와 KL 항")
    names = list(recon_names) + ["KL"]
    values = list(recon_values) + [kl_value]
    colors = [cp.MUTED] * (len(names) - 2) + [RECON_COLOR, KL_COLOR]
    _value_bars(ax_b, names, values, colors, fmt="{:.2f}", log=True)
    ax_b.tick_params(axis="x", labelsize=8)
    cp.label(ax_b, "학습 전에는 재구성 항이 KL 항보다 세 자릿수 큽니다", y=-0.22)
    cp.show(fig)


# ---------------------------------------------------------------- 3절: 기울기 두 갈래
def plot_reparam_graph(dz_value, sigma_value, eps_value, dmu_parts, dlogvar_parts):
    """왼쪽은 z 노드에서 μ·ℓ·ε 세 갈래로 갈라지는 계산 그래프, 오른쪽은 두 갈래 기여의 크기.

    dmu_parts·dlogvar_parts: (z 경로 기여, KL 경로 기여) 숫자 두 개씩.
    """
    fig = cp.new_figure((14, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1], wspace=0.2)

    ax = fig.add_subplot(gs[0])
    nodes = {
        "mu": (0.0, 1.8, "$\\mu$", "(N, 2)"),
        "eps": (0.0, 0.0, "$\\epsilon$", "상수"),
        "lv": (0.0, -1.8, "$\\ell$", "(N, 2)"),
        "op:re": (3.4, 0.0, "$\\mu + e^{\\ell/2}\\epsilon$", None),
        "z": (6.0, 0.0, "$z$", "(N, 2)"),
        "op:dec": (7.9, 0.0, "decoder", None),
        "s": (9.8, 0.0, "$s$", "(N, 784)"),
        "op:bce": (11.7, 0.0, "BCE", None),
        "L": (13.8, 0.0, "$L$", "재구성 항"),
    }
    edges = [("mu", "op:re", "$\\partial z / \\partial \\mu = 1$"),
             ("eps", "op:re", None),
             ("lv", "op:re", "$\\partial z / \\partial \\ell = \\sigma\\epsilon/2$"),
             ("op:re", "z", None), ("z", "op:dec", None), ("op:dec", "s", None),
             ("s", "op:bce", None), ("op:bce", "L", None)]
    draw_compute_graph(ax, nodes, edges, title="$z$ 에서 되돌아온 $dz$ 가 $\\mu$ 와 $\\ell$ 두 갈래로 갈립니다")
    cp.label(ax, f"금색 = 그 화살표에서 곱하는 국소 미분 · $\\epsilon$ 은 상수라 화살표가 되돌아가지 않습니다 "
                 f"($dz = {dz_value:.4f}$, $\\sigma = {sigma_value:.3f}$, $\\epsilon = {eps_value:.3f}$)", y=-0.01)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "기울기", "한 성분의 두 갈래 기여")
    width = 0.36
    index = np.arange(2)
    ax_b.bar(index - width / 2, [dmu_parts[0], dlogvar_parts[0]], width, color=RECON_COLOR, label="$z$ 경로")
    ax_b.bar(index + width / 2, [dmu_parts[1], dlogvar_parts[1]], width, color=KL_COLOR, label="KL 경로")
    ax_b.axhline(0, color=cp.MUTED, lw=0.8)
    ax_b.set_xticks(index, ["$d\\mu$", "$d\\ell$"])
    ax_b.legend(frameon=False, fontsize=8.5)
    ax_b.margins(y=0.3)
    cp.show(fig)


# ---------------------------------------------------------------- 4절: ε을 다시 뽑을 때마다
def plot_eps_draws(mu, sigma, z_draws, loss_values, mean_loss):
    """같은 x에 ε을 여러 번 뽑으면 z가 q 안에서 흩어지고 −ELBO 추정값도 흔들립니다."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    z_draws, loss_values = np.asarray(z_draws, float), np.asarray(loss_values, float)
    fig = cp.new_figure((11.5, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$z_0$", "$z_1$", "$\\epsilon$ 을 다시 뽑을 때마다 다른 $z$")
    _prior_rings(ax)
    ax.add_patch(Ellipse(tuple(mu), 2 * sigma[0], 2 * sigma[1], fill=True,
                         facecolor=RECON_COLOR, alpha=0.22, ec=RECON_COLOR, lw=1.2))
    ax.scatter(z_draws[:, 0], z_draws[:, 1], s=12, color=RECON_COLOR, alpha=0.5)
    ax.scatter(z_draws[:3, 0], z_draws[:3, 1], s=80, marker="*", color=MARK_COLOR,
               zorder=5, edgecolor=cp.INK, lw=0.5)
    ax.scatter([mu[0]], [mu[1]], color=cp.INK, s=26, zorder=6)
    ax.set_aspect("equal", adjustable="datalim")
    cp.label(ax, "파란 타원 = $\\mu \\pm \\sigma$ · 금색 별 = 처음 세 번 뽑은 $z$", y=-0.22)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "뽑은 순서", "nat", "같은 배치의 $-\\mathrm{ELBO}$ 추정값")
    colors = [MARK_COLOR if i < 3 else cp.MUTED for i in range(len(loss_values))]
    ax_b.bar(np.arange(len(loss_values)), loss_values, color=colors, width=0.7)
    ax_b.axhline(mean_loss, color=RECON_COLOR, lw=1.6, ls="--")
    low, high = float(loss_values.min()), float(loss_values.max())
    pad = max(0.35 * (high - low), 0.5)
    ax_b.set_ylim(low - pad, high + pad)
    ax_b.text(len(loss_values) - 0.5, mean_loss, f"평균 {mean_loss:.2f}", va="center", ha="right",
              fontsize=9, color=RECON_COLOR, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    cp.label(ax_b, "금색 = 이 배치에서 처음 세 번 잰 값(왼쪽 별과는 다른 추출) · 한 번의 값은 흔들려도 평균은 같은 자리를 향합니다", y=-0.22)
    cp.show(fig)


def plot_loss_history(steps, recon_history, kl_history):
    """학습 기록: 재구성 항과 KL 항을 따로 그립니다(크기가 20배 넘게 달라 한 축에 겹치지 않습니다)."""
    fig = cp.new_figure((11.5, 3.5))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    for i, (values, color, title) in enumerate([
            (recon_history, RECON_COLOR, "재구성 항 (화소 784개의 합)"),
            (kl_history, KL_COLOR, "KL 항 (잠재 2개의 합)")]):
        ax = fig.add_subplot(gs[i])
        _clean_axes(ax, "스텝", "nat", title)
        ax.plot(steps, values, color=color, lw=1.8)
        ax.scatter([steps[-1]], [values[-1]], color=MARK_COLOR, s=45, zorder=5, edgecolor=cp.INK, lw=0.5)
        ax.annotate(f"{values[-1]:.2f}", xy=(steps[-1], values[-1]), xytext=(-48, 12),
                    textcoords="offset points", fontsize=9, color=cp.INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    cp.show(fig)


def plot_recon_pairs(originals, recons, labels, title="위: 원본 · 아래: 복원 확률"):
    """고정 예시 이미지와 그 복원 확률을 위아래로 나란히. originals·recons: (n, 28, 28)."""
    n = len(originals)
    fig, grid = cp.flow([1] * n, rows=2, height=1.8, unit=1.35, wspace=0.14, hspace=0.3)
    for j in range(n):
        cp.draw_image(grid[0][j], originals[j], str(labels[j]))
        cp.draw_image(grid[1][j], recons[j])
    fig.suptitle(title, color=cp.INK, fontsize=11)
    cp.show(fig)


def plot_latent_map(tiled_grid, x_values, y_values, points, digits):
    """왼쪽은 잠재 평면 격자를 decoder에 통과시킨 이미지 판, 오른쪽은 학습 데이터의 μ 산점(숫자별 색)."""
    fig = cp.new_figure((12, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    side = len(x_values)
    tile = tiled_grid.shape[0] // side

    ax = fig.add_subplot(gs[0])
    ax.imshow(tiled_grid, cmap=cp.GRAY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([(j + 0.5) * tile for j in range(side)], [f"{v:.1f}" for v in x_values])
    ax.set_yticks([(i + 0.5) * tile for i in range(side)], [f"{v:.1f}" for v in y_values[::-1]])
    ax.set_xlabel("$z_0$", color=cp.INK, fontsize=9.5)
    ax.set_ylabel("$z_1$", color=cp.INK, fontsize=9.5)
    ax.set_title("잠재 평면 격자를 decoder에 넣은 그림", color=cp.INK, fontsize=10.5, pad=6)
    ax.tick_params(colors=cp.MUTED, labelsize=8.5, length=0)
    ax.grid(False)
    for name in ax.spines:
        ax.spines[name].set_color(cp.MUTED)

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, "$z_0$", "$z_1$", "학습 데이터 5,000장의 $\\mu$ (숫자별 색)")
    scatter = ax_b.scatter(points[:, 0], points[:, 1], c=digits, cmap=DIGIT_CMAP, s=5,
                           alpha=0.7, vmin=-0.5, vmax=9.5)
    _prior_rings(ax_b)
    bar = fig.colorbar(scatter, ax=ax_b, ticks=range(10), fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=8.5, colors=cp.MUTED, length=0)
    bar.outline.set_edgecolor(cp.MUTED)
    ax_b.set_aspect("equal", adjustable="datalim")
    cp.label(ax_b, "점선 원 = 사전분포 $1\\sigma$·$2\\sigma$", y=-0.13)
    cp.show(fig)


# ---------------------------------------------------------------- 4절 🔍 소절: 혼합 데이터
def plot_mixture_vae(data, model_samples, curve, elbo_value, true_value):
    """왼쪽은 데이터와 모델 샘플·decoder 곡선, 오른쪽은 하한과 진짜 평균 log 밀도 막대."""
    fig = cp.new_figure((11.5, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.34)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, "$x_0$", "$x_1$", "두 봉우리 혼합과 잠재 1차원 VAE")
    ax.scatter(data[:, 0], data[:, 1], s=8, color=cp.MUTED, alpha=0.45, label="데이터")
    ax.scatter(model_samples[:, 0], model_samples[:, 1], s=8, color=RECON_COLOR, alpha=0.5, label="모델 샘플")
    ax.plot(curve[:, 0], curve[:, 1], color=MARK_COLOR, lw=2.4, label="decoder 평균 $m(z)$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "nat", "하한은 진짜 평균 log 밀도를 넘지 못합니다")
    _value_bars(ax_b, ["ELBO 평균", "진짜 $\\log p(x)$ 평균"], [elbo_value, true_value],
                [RECON_COLOR, MARK_COLOR], fmt="{:.3f}")
    cp.label(ax_b, f"모자란 양 {true_value - elbo_value:.3f} nat", y=-0.2)
    cp.show(fig)


# ---------------------------------------------------------------- 5절: 세 가지 사용
def plot_vae_uses(originals, recons, samples, interpolation):
    """한 줄에 한 가지 사용: 복원(encoder→decoder), 생성(prior→decoder), 보간(두 잠재점 사이)."""
    columns = max(2 * len(originals), len(samples), len(interpolation))
    fig = cp.new_figure((1.28 * columns + 2.6, 5.4))
    gs = fig.add_gridspec(3, columns + 1, width_ratios=[2.0] + [1] * columns,
                          wspace=0.14, hspace=0.32)
    titles = ["1. 복원\n$x \\to \\mu \\to$ decoder",
              "2. 생성\n$z \\sim \\mathcal{N}(0, I) \\to$ decoder",
              "3. 보간\n두 잠재점 사이를 직선으로"]
    for row, text in enumerate(titles):
        ax = fig.add_subplot(gs[row, 0])
        cp.blank(ax)
        ax.text(1.0, 0.5, text, ha="right", va="center", fontsize=9.5, color=cp.INK)
    panels = [(gs[0, 1 + 2 * j], originals[j], "원본") for j in range(len(originals))]
    panels += [(gs[0, 2 + 2 * j], recons[j], "복원") for j in range(len(originals))]
    panels += [(gs[1, 1 + j], image, None) for j, image in enumerate(samples)]
    panels += [(gs[2, 1 + j], image, None) for j, image in enumerate(interpolation)]
    for slot, image, title in panels:
        ax = fig.add_subplot(slot)
        cp.blank(ax)
        cp.draw_image(ax, image, title)
    cp.show(fig)


# ---------------------------------------------------------------- 완성 예제: 잠재 차원 변형
def plot_latent_dim_variants(names, recon_values, kl_values, mu_stds):
    """잠재 차원을 바꾼 두 모델의 재구성 항·KL 항과 차원별 μ 표준편차."""
    fig = cp.new_figure((13, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.4)

    ax = fig.add_subplot(gs[0])
    _clean_axes(ax, None, "nat", "재구성 항 (작을수록 잘 복원)")
    _value_bars(ax, names, recon_values, [cp.MUTED, RECON_COLOR], fmt="{:.1f}")

    ax_b = fig.add_subplot(gs[1])
    _clean_axes(ax_b, None, "nat", "KL 항 (클수록 사전분포에서 멂)")
    _value_bars(ax_b, names, kl_values, [cp.MUTED, KL_COLOR], fmt="{:.2f}")

    ax_c = fig.add_subplot(gs[2])
    _clean_axes(ax_c, "잠재 차원 번호", "$\\mu$ 의 표준편차", "차원마다 실제로 쓰이고 있나")
    for (name, values), color in zip(zip(names, mu_stds), [cp.MUTED, RECON_COLOR]):
        values = np.asarray(values, float)
        ax_c.plot(np.arange(len(values)), values, marker="o", ms=5, lw=1.6, color=color, label=name)
    ax_c.axhline(0.3, color=MARK_COLOR, lw=1.3, ls="--")
    ax_c.set_ylim(0.0, None)
    ax_c.text(0.05, 0.36, "0.3 (쓰이는 것으로 보는 기준)", fontsize=8.5, color=MARK_COLOR)
    ax_c.legend(frameon=False, fontsize=8.5, loc="lower right")
    cp.show(fig)
