"""47 MLM의 허용 문맥·corruption·선택 손실·복원을 표시합니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_context_masks(tokens,bidirectional,causal):
    """같은 token열의 양방향/causal 허용 영역을 비교합니다."""
    fig,axes=cp.flow([2.5,3.5,3.5],height=3.4,wspace=.4)
    cp.draw_grid(axes[0],tokens[:,None],'입력 token IDs',kind='plain')
    cp.draw_grid(axes[1],bidirectional,'양방향 허용',kind='plain')
    cp.draw_grid(axes[2],causal,'causal 허용',kind='plain')
    cp.show(fig)


def plot_corruption(clean,selected,corrupted):
    """선택한 위치와 실제 바뀐 위치가 다를 수 있음을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[clean,selected,corrupted],['원래 target','손실 선택 mask','corruption 입력']):
        cp.draw_grid(ax,value,title,kind='plain')
    cp.show(fig)


def plot_selected_loss(probabilities,selected,delta):
    """선택 mask에 따른 logits 미분의 위치를 보여 줍니다."""
    fig,axes=cp.flow([3,2.5,3.5],height=3,wspace=.4)
    cp.draw_grid(axes[0],probabilities,'위치별 확률',kind='plain',fmt='{:.2f}')
    cp.draw_grid(axes[1],selected[:,None],'선택 위치',kind='plain')
    cp.draw_grid(axes[2],delta,'logits 미분',kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_information_access(mlm_input,ar_input,answer):
    """동일 첫 색 target을 예측할 때 실제 주는 입력의 차이를 표시합니다."""
    fig,axes=cp.flow([4,3,2],height=2.7,wspace=.5)
    cp.draw_grid(axes[0],mlm_input[None,:],'MLM: 오른쪽 문맥도 입력',kind='plain')
    cp.draw_grid(axes[1],ar_input[None,:],'첫 token 자기회귀 입력',kind='plain')
    cp.draw_grid(axes[2],np.array([[answer]]),'정답 색',kind='plain')
    cp.show(fig)


def plot_fill(original,masked,filled):
    """정답·빈칸 입력·실제 모델이 채운 token열을 표시합니다."""
    fig,axes=cp.flow([3.5,3.5,3.5],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[original,masked,filled],['보관한 정답','모델에 준 입력','한 칸 복원 결과']):
        cp.draw_grid(ax,np.asarray(value)[None,:],title,kind='plain')
    cp.show(fig)


def plot_mlm_learning(history):
    """항상 첫 색을 MASK로 만든 고정 평가 손실을 그립니다."""
    fig=cp.new_figure((7,3.5))
    ax=fig.subplots()
    _clean_axes(ax,'Adam 갱신 수','선택 위치 NLL','고정 첫 색 MASK 평가')
    ax.plot(history[:,0],history[:,1],color=cp.BLUE,label='train 문장')
    ax.plot(history[:,0],history[:,2],color=cp.MUTED,label='valid 문장 · 선택 없음')
    ax.legend(frameon=False)
    fig.subplots_adjust(left=.14,bottom=.2)
    cp.show(fig)
