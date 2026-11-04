"""46 생성 정책·K/V 재사용의 본문 계산 결과를 표시합니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_temperature(logits, probabilities, temperatures):
    """같은 logits에 다른 temperature를 적용한 분포를 표시합니다."""
    fig,axes=cp.flow([3,4,3],height=3,wspace=.4)
    cp.draw_grid(axes[0],logits[None,:],'고정 logits',kind='signed',fmt='{:.1f}')
    cp.draw_grid(axes[1],probabilities,'temperature별 확률',row_labels=[str(t) for t in temperatures],kind='plain',fmt='{:.2f}')
    axes[2].text(.5,.5,'모델 가중치는 고정\n분포의 뾰족함만 변경',ha='center',va='center',fontsize=12,color=cp.INK)
    cp.show(fig)


def plot_truncation(original,keep,filtered):
    """원래 확률·유지할 token·재정규화한 분포를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.7,wspace=.4)
    for ax,value,title in zip(axes,[original,keep,filtered],['원래 확률','남길 token','재정규화 확률']):
        cp.draw_grid(ax,np.asarray(value)[None,:],title,kind='plain',fmt='{:.2f}')
    cp.show(fig)


def plot_prefill(features,keys,values):
    """모든 prefix 위치에서 만든 K/V를 그립니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    for ax,value,title in zip(axes,[features,keys,values],['prefix 정규화 특성','저장 K','저장 V']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_cached_step(keys,query,weights,mixed):
    """과거+새 key에 대해 새 query 하나가 읽는 가중합을 그립니다."""
    fig,axes=cp.flow([2.5,2.5,2.5,2.5],height=3,wspace=.45)
    for ax,value,title in zip(axes,[keys,query,weights,mixed],['과거+새 K','새 Q 한 행','현재 attention','현재 가중합']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_cache_budget(lengths,counts,memory):
    """본문에서 계산한 누적 score 원소 수와 K/V byte 수를 보여 줍니다."""
    fig,axes=cp.flow([3.5,3.5,3],height=3,wspace=.5)
    cp.draw_grid(axes[0],counts,'한 token씩 처리한 score 원소',row_labels=['full','cache'],col_labels=[str(t) for t in lengths],kind='plain')
    cp.draw_grid(axes[1],memory[None,:],'최종 K/V 저장 byte',col_labels=[str(t) for t in lengths],kind='plain')
    axes[2].text(.5,.5,'연산 수와 시간은 다릅니다\n작은 배열·할당 비용도 측정',ha='center',va='center',fontsize=12,color=cp.INK)
    cp.show(fig)


def plot_cache_timing(lengths,times):
    """같은 예정 token열의 실제 full/cache 시간만 그립니다."""
    fig=cp.new_figure((7,3.5))
    ax=fig.subplots()
    _clean_axes(ax,'처리한 token 수','반복당 초','같은 입력열·같은 가중치의 실제 측정')
    ax.plot(lengths,times[:,0],color=cp.MUTED,label='full recomputation')
    ax.plot(lengths,times[:,1],color=cp.BLUE,label='KV cache')
    ax.legend(frameon=False)
    fig.subplots_adjust(left=.15,bottom=.2)
    cp.show(fig)
