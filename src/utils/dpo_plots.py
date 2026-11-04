"""49 선호 확률·sequence 합·기준 정책·평가를 그립니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_preference_probability(differences,probabilities):
    """점수 차이가 선호 확률로 바뀌는 모습을 표시합니다."""
    fig,axes=cp.flow([3,3],height=2.8,wspace=.4)
    cp.draw_grid(axes[0],differences[None,:],'두 답변 점수 차이',kind='signed',fmt='{:.1f}')
    cp.draw_grid(axes[1],probabilities[None,:],'첫 답변 선호 확률',kind='prob',fmt='{:.2f}')
    cp.show(fig)


def plot_sequence_sum(selected,weights,sums):
    """답변 위치의 로그확률과 문서별 합을 표시합니다."""
    fig,axes=cp.flow([4,4,2],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[selected,weights,sums[:,None]],['target별 log p','답변+EOS 선택','sequence 합']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_reference_margin(policy,reference,margin):
    """actor와 reference의 두 답변 차이를 표시합니다."""
    fig,axes=cp.flow([3,3,2],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[policy,reference,margin[:,None]],['actor [chosen,rejected]','고정 reference','β × 차이의 차이']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_objective_signals(sft,dpo):
    """선호하는 답변만 쓰는 SFT와 두 경로의 DPO 미분을 표시합니다."""
    fig,axes=cp.flow([3.5,3.5],height=2.7,wspace=.4)
    cp.draw_grid(axes[0],sft,'SFT: [chosen,rejected] 상류',kind='signed',fmt='{:.2f}')
    cp.draw_grid(axes[1],dpo,'DPO: [chosen,rejected] 상류',kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_dpo_learning(histories,labels):
    """같은 valid 후보쌍의 DPO 목적을 표시합니다."""
    fig=cp.new_figure((7,3.4))
    ax=fig.subplots()
    _clean_axes(ax,'갱신 횟수','valid DPO loss','같은 선호 쌍에서의 진단')
    for history,label,color in zip(histories,labels,[cp.BLUE,cp.GOLD]):
        ax.plot(history[:,0],history[:,1],label=label,color=color)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=.15,bottom=.2)
    cp.show(fig)


def plot_absolute_probabilities(log_probabilities,labels):
    """주어진 두 답변의 절대 로그확률을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,label in zip(axes,log_probabilities,labels):
        cp.draw_grid(ax,value[None,:],label+' [chosen,rejected]',kind='signed',fmt='{:.2f}')
    cp.show(fig)
