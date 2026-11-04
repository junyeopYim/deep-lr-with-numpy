"""62 선호 보상·token 경로·KL shaping·PPO·정책 평가를 그립니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_reward_preference(scores,probabilities):
    """두 답변의 학습 보상 점수와 선호 확률을 표시합니다."""
    fig,axes=cp.flow([4,3],height=2.7,wspace=.4)
    cp.draw_grid(axes[0],scores,'[chosen,rejected] 보상 점수',kind='signed',fmt='{:.2f}')
    cp.draw_grid(axes[1],probabilities[:,None],'chosen 선호 확률',kind='prob',fmt='{:.2f}')
    cp.show(fig)


def plot_token_outcomes(token_probabilities,sequence_probabilities,labels):
    """token 조건부 확률과 전체 일곱 답변의 확률을 표시합니다."""
    fig,axes=cp.flow([4,6],height=3,wspace=.4)
    cp.draw_grid(axes[0],token_probabilities,'다음 [R,G,EOS] 확률',kind='prob',fmt='{:.2f}',row_labels=['BOS','R 뒤','G 뒤'])
    cp.draw_grid(axes[1],sequence_probabilities[None,:],'모든 종료 답변 확률',kind='prob',fmt='{:.2f}',col_labels=labels)
    cp.show(fig)


def plot_kl_rewards(log_ratios,rewards,returns):
    """KL token 보상과 마지막 reward가 return에 더해지는 모습을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.7,wspace=.4)
    for ax,value,title in zip(axes,[log_ratios,rewards,returns],['선택 token log-ratio','KL shaping + 마지막 reward','뒤에서 더한 return']):
        cp.draw_grid(ax,value[None,:],title,kind='signed',fmt='{:.3f}')
    cp.show(fig)


def plot_ppo_clip(ratios,unclipped,surrogate):
    """양·음 advantage에서 다른 방향으로 막히는 surrogate를 표시합니다."""
    fig,axes=cp.flow([3,3.5,3.5],height=2.8,wspace=.4)
    cp.draw_grid(axes[0],ratios[None,:],'새/old 확률비',kind='plain',fmt='{:.1f}')
    cp.draw_grid(axes[1],unclipped,'ratio × A',kind='signed',fmt='{:.1f}',row_labels=['A=+1','A=−1'])
    cp.draw_grid(axes[2],surrogate,'min(ratio A, clip A)',kind='signed',fmt='{:.1f}',row_labels=['A=+1','A=−1'])
    cp.show(fig)


def plot_rlhf_budgets(budgets,labels):
    """방법별 피드백 단위 수와 갱신 횟수를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.4)
    for ax,value,label in zip(axes,budgets,labels):
        cp.draw_grid(ax,value[:,None],label,kind='plain',row_labels=['기본 데이터 단위','actor 갱신'])
    cp.show(fig)


def plot_rlhf_learning(histories,labels):
    """각 seed의 정확한 valid 성공확률과 sequence KL을 별도 축에 그립니다."""
    fig,axes=cp.flow([4,4],height=3.4,wspace=.45)
    for ax,column,ylabel in zip(axes,[4,3],['정확한 답변 확률','sequence KL']):
        _clean_axes(ax,'rollout 라운드',ylabel,'고정 valid prompt')
        for history,label,color in zip(histories,labels,[cp.BLUE,cp.GOLD,cp.MUTED]):
            ax.plot(history[:,0],history[:,column],label=label,color=color)
        ax.legend(frameon=False,fontsize=8)
    fig.subplots_adjust(left=.1,bottom=.2)
    cp.show(fig)


def plot_policy_metrics(values,labels):
    """learned reward와 별도 oracle 품질·KL·성공확률을 표로 표시합니다."""
    fig,axes=cp.flow([10],height=3.7)
    cp.draw_grid(axes[0],values,'전체7답변 열거: 방법별 test 지표',kind='plain',fmt='{:.3f}',row_labels=labels,
                 col_labels=['learned reward','oracle 품질','KL to ref','정확 답변','EOS'])
    axes[0].set_aspect('auto')
    fig.subplots_adjust(left=.14,right=.98,top=.78,bottom=.05)
    cp.show(fig)
