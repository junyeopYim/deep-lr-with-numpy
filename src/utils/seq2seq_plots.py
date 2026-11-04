"""20 Encoder–Decoder의 토큰·마스크·기울기·생성 기록을 표시합니다.

모든 확률, attention, 손실, 학습과 생성은 노트북에서 계산합니다.
이 모듈은 전달받은 배열과 문자열의 배치만 담당합니다.
"""

import numpy as np
from matplotlib.patches import Rectangle

from . import concept_plots as cp
from .gradient_plots import _clean_axes


def _tokens(ax, names, title):
    """토큰 이름을 동일한 칸 안에 표시합니다."""
    cp.blank(ax)
    for i, name in enumerate(names):
        color = "#f1f2f4" if name == "PAD" else "#e5eef6"
        ax.add_patch(Rectangle((i, 0), 1, 1, fc=color, ec="white", lw=2))
        ax.text(i + 0.5, 0.5, name, ha="center", va="center", fontsize=9, color=cp.INK)
    ax.set_xlim(0, len(names))
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, color=cp.INK)


def plot_teacher_shift(source, decoder, target, valid):
    """source·decoder 입력·목표·손실 포함 여부를 나란히 표시합니다."""
    fig, axes = cp.flow([3.0, 3.0, 3.0, 2.3], height=2.5, wspace=0.25)
    _tokens(axes[0], source, "source · EOS까지 읽기")
    _tokens(axes[1], decoder, "decoder 입력 · 한 칸 오른쪽")
    _tokens(axes[2], target, "예측할 target")
    cp.draw_grid(axes[3], np.asarray(valid)[None, :], "CE에 포함하는 칸", kind="plain", fmt="{:g}")
    cp.show(fig)


def plot_attention_masks(scores, allowed, probability, output):
    """점수·허용 mask·확률·가중합을 직접 비교합니다."""
    fig, axes = cp.flow([2.5, 2.5, 2.5, 2.2], height=3.1, wspace=0.35)
    for ax, values, title, kind in zip(axes, [scores, allowed, probability, output],
                                      ["점수 $S$", "허용 mask · 1만 읽기", "확률 $A$", "가중합 $AV$"],
                                      ["signed", "plain", "plain", "signed"]):
        cp.draw_grid(ax, values, title, kind=kind, fmt="{:.2f}", fontsize=9)
    cp.label(axes[2], "막힌 칸은 0 · 전부 막힌 행도 0")
    cp.show(fig)


def plot_encoder_decoder_flow(source, memory, decoder, cross_weights, output):
    """source 표현과 decoder 상태가 cross-attention에서 만나는 배열을 표시합니다."""
    fig, axes = cp.flow([2.2, 2.2, 2.2, 2.7, 2.5], height=3.2, wspace=0.5)
    values = [source, memory, decoder, cross_weights, output]
    titles = ["source 특징 $X_s$", "encoder 기억 $M$", "causal decoder $D$", "cross 확률 · T×S", "다음 토큰 logit"]
    for ax, value, title in zip(axes, values, titles):
        cp.draw_grid(ax, value, title, fmt="{:.2f}", kind="signed", fontsize=9)
    cp.connect(fig, axes[0], axes[1], "self + residual")
    cp.connect(fig, axes[2], axes[3], "Q")
    cp.connect(fig, axes[3], axes[4], "가중합 + D\n출력층")
    cp.label(axes[1], "cross의 K와 V로 보냅니다")
    cp.show(fig)


def plot_masked_loss(target, valid, nll, gradient):
    """타깃·valid mask·위치별 비용·logit 기울기를 표시합니다."""
    fig, axes = cp.flow([2.1, 2.1, 2.5, 5.2], height=3.0, wspace=0.45)
    for ax, array, title in zip(axes, [target, valid, nll, gradient],
                               ["target ID", "valid · 1인 칸만", "위치별 음의 log 확률", "한 표본의 dlogit"]):
        cp.draw_grid(ax, array, title, fmt="{:.2f}", kind="signed", fontsize=9)
    cp.label(axes[2], "분모는 배치의 valid target 수")
    cp.show(fig)


def plot_generation_protocol(source_text, teacher_lines, free_lines):
    """이미 계산한 teacher prefix와 자유 생성 prefix 기록을 표시합니다."""
    fig, axes = cp.flow([2.8, 4.2, 4.2], height=3.1, wspace=0.3)
    cp.draw_formula(axes[0], source_text, sub="source는 두 방식에서 같습니다", fontsize=12)
    for ax, lines, title in zip(axes[1:], [teacher_lines, free_lines],
                                ["teacher forcing · 정답 앞부분", "greedy · 자신이 낸 앞부분"]):
        ax.set_title(title, color=cp.INK, fontsize=10.5)
        ax.text(0.03, 0.85, "\n\n".join(lines), transform=ax.transAxes,
                color=cp.INK, fontsize=10, va="top")
    cp.show(fig)


def plot_sequence_history(history):
    """고정 validation의 teacher CE와 토큰 정확도 기록을 그립니다."""
    fig = cp.new_figure((10, 3.3))
    axes = fig.subplots(1, 2)
    _clean_axes(axes[0], "갱신 수", "valid target 평균 CE", "teacher-forcing 손실")
    _clean_axes(axes[1], "갱신 수", "valid target 정확도", "정답 prefix를 준 토큰 정확도")
    axes[0].plot(history[:, 0], history[:, 1], color=cp.BLUE)
    axes[1].plot(history[:, 0], history[:, 2], color=cp.BLUE)
    fig.subplots_adjust(wspace=0.3, bottom=0.18)
    cp.show(fig)


def plot_generated_examples(source_texts, reference_texts, generated_texts):
    """미리 정한 held-out 표본의 입력·정답·실제 생성 문자열을 표시합니다."""
    fig = cp.new_figure((10.2, 0.65 * len(source_texts) + 1.1))
    ax = fig.add_subplot(111)
    cp.blank(ax)
    rows = list(zip(source_texts, reference_texts, generated_texts))
    table = ax.table(cellText=rows, colLabels=["source", "정답 · EOS 포함", "greedy 생성"],
                     loc="center", cellLoc="left", colColours=["#edf1f5"] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for cell in table.get_celld().values():
        cell.set_edgecolor("white")
        cell.set_text_props(color=cp.INK)
    cp.show(fig)
