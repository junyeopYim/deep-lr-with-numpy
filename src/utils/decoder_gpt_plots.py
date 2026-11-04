"""45 작은 decoder의 본문 계산 배열·확률·기록을 표시합니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_decoder_contract(inputs, targets, mask):
    """입력·다음 정답·유효 위치라는 언어모델 계약을 보여 줍니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[inputs,targets,mask],['BOS와 앞부분','다음 token 정답','유효 target']):
        cp.draw_grid(ax,value,title,kind='plain')
    cp.show(fig)


def plot_causal_attention(scores, allowed, probabilities):
    """점수에서 미래·padding을 막고 남은 확률로 가는 과정을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3.2,wspace=.4)
    for ax,value,title in zip(axes,[scores,allowed,probabilities],['QKᵀ / √D','허용 query–key','attention 확률']):
        cp.draw_grid(ax,value,title,kind='signed' if ax is axes[0] else 'plain',fmt='{:.2f}')
    cp.show(fig)


def plot_residual_block(inputs, normalized, output):
    """정규화된 가지와 원래 입력을 더한 잔차 결과를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    for ax,value,title in zip(axes,[inputs,normalized,output],['입력 X','token별 LN(X)','X + FFN(LN(X))']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_loss_path(probabilities, losses, gradients):
    """정답 확률·위치 손실·상류 점수 미분을 나란히 보여 줍니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    cp.draw_grid(axes[0],probabilities,'token별 어휘 확률',kind='plain',fmt='{:.2f}')
    cp.draw_grid(axes[1],losses[:,None],'유효 위치 NLL',kind='plain',fmt='{:.2f}')
    cp.draw_grid(axes[2],gradients,'(p−one_hot)/M',kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_next_token(prefix_ids, probabilities, labels, selected):
    """이미 주어진 prefix와 마지막 위치의 다음 token 분포를 표시합니다."""
    fig,axes=cp.flow([3,4,2.5],height=3,wspace=.5)
    cp.draw_grid(axes[0],np.asarray(prefix_ids)[None,:],'주어진 prefix IDs',kind='plain',fontsize=8)
    cp.draw_grid(axes[1],probabilities[None,:],'표시한 다음 token 확률',kind='plain',fmt='{:.2f}',col_labels=labels,fontsize=8)
    axes[2].text(.5,.5,f'다음 ID\n{selected}',ha='center',va='center',fontsize=16,color=cp.INK)
    cp.show(fig)


def plot_decoder_learning(history):
    """고정 train·valid 손실 기록을 그립니다."""
    fig=cp.new_figure((7,3.5))
    ax=fig.subplots()
    _clean_axes(ax,'Adam 갱신 수','유효 target당 NLL','전체 decoder 학습')
    ax.plot(history[:,0],history[:,1],color=cp.BLUE,label='train')
    ax.plot(history[:,0],history[:,2],color=cp.MUTED,label='valid · 선택 없음')
    ax.legend(frameon=False)
    fig.subplots_adjust(left=.13,bottom=.2)
    cp.show(fig)
