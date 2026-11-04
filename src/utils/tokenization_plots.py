"""44 토큰화: 본문에서 계산한 문자열·번호·확률·학습 기록만 그립니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_token_units(text, characters, byte_ids):
    """같은 문자열의 문자 단위와 byte 번호를 구분합니다."""
    fig, axes = cp.flow([2.2, 2.5, 4.0], height=2.8, wspace=.45)
    axes[0].text(.5, .5, repr(text), ha='center', va='center', fontsize=18, color=cp.INK)
    axes[0].set_title('원래 문자열', fontsize=11)
    axes[1].text(.5, .5, ' | '.join(characters), ha='center', va='center', fontsize=15, color=cp.INK)
    axes[1].set_title(f'문자 {len(characters)}개', fontsize=11)
    cp.draw_grid(axes[2], np.asarray(byte_ids)[None,:], 'UTF-8 byte 번호', kind='plain', fmt='{:g}', fontsize=9)
    cp.show(fig)


def plot_pair_merge(before, counts, pair_labels, after):
    """입력 번호·인접 쌍 빈도·한 번 합친 결과를 표시합니다."""
    fig, axes = cp.flow([2.8, 3.5, 2.8], height=3, wspace=.4)
    cp.draw_grid(axes[0], np.asarray(before)[None,:], '병합 전', kind='plain')
    cp.draw_grid(axes[1], np.asarray(counts)[None,:], '인접 쌍의 횟수', kind='plain', col_labels=pair_labels)
    cp.draw_grid(axes[2], np.asarray(after)[None,:], '병합 후 · 왼쪽부터', kind='plain')
    cp.show(fig)


def plot_shift(inputs, targets, mask):
    """문서별 한 칸 이동과 손실에 포함할 위치를 보여 줍니다."""
    fig, axes = cp.flow([3,3,3], height=3, wspace=.35)
    for ax,value,title in zip(axes,[inputs,targets,mask],['입력 IDs','다음 token 정답','유효 target mask']):
        cp.draw_grid(ax,value,title,kind='plain')
    cp.show(fig)


def plot_embedding_path(ids, embedded, probabilities):
    """번호로 고른 행이 다음 token 분포로 바뀌는 계산을 표시합니다."""
    fig, axes = cp.flow([2.3,3,3.5], height=3.3, wspace=.5)
    cp.draw_grid(axes[0],np.asarray(ids)[:,None],'현재 token ID',kind='plain')
    cp.draw_grid(axes[1],embedded,'고른 embedding 행',kind='signed',fmt='{:.2f}')
    cp.draw_grid(axes[2],probabilities,'다음 token 확률',kind='plain',fmt='{:.2f}')
    cp.show(fig)


def plot_token_budget(counts, labels, costs):
    """동일 문자열의 단위 수와 token 분모의 변화를 보여 줍니다."""
    fig, axes = cp.flow([3.3,3.3,3.3],height=3,wspace=.5)
    cp.draw_grid(axes[0],np.asarray(counts)[None,:],'문자열 한 개의 단위 수',col_labels=labels,kind='plain')
    cp.draw_grid(axes[1],costs,'동일 총 NLL의 다른 분모',row_labels=['token 수','NLL/token','PPL'],kind='plain',fmt='{:.2f}')
    axes[2].text(.5,.5,'PPL 비교는\n같은 tokenizer·같은\n평가 규약 안에서',ha='center',va='center',fontsize=14,color=cp.INK)
    cp.show(fig)


def plot_learning(history):
    """전달된 train·test NLL 기록만 표시합니다."""
    fig=cp.new_figure((7,3.5));ax=fig.subplots()
    _clean_axes(ax,'SGD 갱신 수','유효 target당 NLL','고정 tokenizer의 bigram 학습')
    ax.plot(history[:,0],history[:,1],color=cp.BLUE,label='train')
    ax.plot(history[:,0],history[:,2],color=cp.MUTED,label='test · 선택에 사용하지 않음')
    ax.legend(frameon=False);fig.subplots_adjust(left=.13,bottom=.2);cp.show(fig)
