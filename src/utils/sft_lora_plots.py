"""48 답변 선택·저랭크 갱신·고정 경로·학습 비교를 표시합니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_response_mask(inputs,targets,weights):
    """prompt 조건과 답변 target의 다른 역할을 표시합니다."""
    fig,axes=cp.flow([3.5,3.5,3.5],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[inputs,targets,weights],['입력 prompt+앞 답변','한 칸 이동한 target','답변+EOS 손실 mask']):
        cp.draw_grid(ax,value,title,kind='plain')
    cp.show(fig)


def plot_lora_weights(base,delta,effective):
    """고정 행렬과 저랭크 변화량의 합을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    for ax,value,title in zip(axes,[base,delta,effective],['고정 W₀','학습 변화 sAB','실제 W₀+sAB']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_frozen_gradient(upstream,base_path,adapter_path):
    """고정 W의 입력 미분도 남아야 함을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[upstream,base_path,adapter_path],['출력 상류 G','고정 W₀를 지나는 dX','adapter를 지나는 dX']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_trainable_budget(counts,labels):
    """실제 셀에서 센 학습 배열 수를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.7,wspace=.4)
    for ax,value,label in zip(axes,counts,labels):
        cp.draw_grid(ax,np.array([[value]]),label+' 학습 원소',kind='plain')
    cp.show(fig)


def plot_merge_logits(unmerged,merged,difference):
    """합치기 전후 실제 logits와 차이를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    for ax,value,title in zip(axes,[unmerged,merged,difference],['adapter 경로 logits','merge한 logits','두 결과 차이']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.3f}')
    cp.show(fig)


def plot_adaptation_learning(histories,labels):
    """세 방법의 고정 valid 답변 NLL을 표시합니다."""
    fig=cp.new_figure((7.5,3.5))
    ax=fig.subplots()
    _clean_axes(ax,'예정 갱신 번호','답변+EOS 평균 NLL','같은 과제의 세 갱신 범위')
    for history,label,color in zip(histories,labels,[cp.MUTED,cp.BLUE,cp.GOLD]):
        ax.plot(history[:,0],history[:,2],label=label,color=color)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=.14,bottom=.2)
    cp.show(fig)
