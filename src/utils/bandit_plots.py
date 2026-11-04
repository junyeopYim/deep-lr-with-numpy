"""51번 bandit의 그림. 확률·가치·행동·학습 통계는 노트북에서 계산합니다."""
import numpy as np
from . import concept_plots as cp


def plot_reward_expectation(means, policy, contributions):
    """행동별 참 평균과 선택 확률의 곱이 기대 보상을 만드는 모습을 그립니다."""
    fig, axes = cp.flow([2.6,2.6,2.6],height=2.8,wspace=.5)
    for ax, values, title in zip(axes,[means,policy,contributions],['판매대별 μ','선택 확률 π','기여 π × μ']):
        cp.draw_grid(ax,values[:,None],title,row_labels=['A','B','C'],fmt='{:.3f}',kind='plain')
    cp.connect(fig,axes[0],axes[1],'×')
    cp.connect(fig,axes[1],axes[2],'행동별 곱')
    cp.show(fig)


def plot_average_update(rewards, averages, selected):
    """선택한 행동의 보상만 표본평균에 반영되는 작은 기록을 그립니다."""
    fig, axes=cp.flow([3.,3.,2.5],height=2.8,wspace=.6)
    cp.draw_grid(axes[0],rewards[:,None],'A를 고른 순서의 보상',fmt='{:.0f}',kind='plain')
    cp.draw_grid(axes[1],averages[:,None],'관측 직후 Q(A)',fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[2],selected,'마지막 Q와 N',col_labels=['Q','N'],row_labels=['A','B','C'],fmt='{:.3g}',kind='plain')
    cp.connect(fig,axes[0],axes[1],'누적 평균')
    cp.show(fig)


def plot_epsilon_policy(values, policies, names):
    """동률 가치와 epsilon별 실제 행동 확률을 표로 보여 줍니다."""
    fig,axes=cp.flow([2.5,4.5,2.5],height=2.8,wspace=.6)
    cp.draw_grid(axes[0],values[:,None],'현재 Q',row_labels=['A','B','C'],fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[1],policies,'행동별 선택 확률',row_labels=names,col_labels=['A','B','C'],fmt='{:.3f}',kind='plain')
    cp.draw_formula(axes[2],r'$(1-\epsilon)g+\epsilon/A$','최댓값 동률도 균등 분할',fontsize=12)
    cp.show(fig)


def plot_regret_paths(steps, cumulative_reward, pseudo, realized):
    """작은 고정 경로의 보상과 두 regret 정의를 분리합니다."""
    fig,axes=cp.flow([3.,3.,3.],height=2.8,wspace=.5)
    for ax,values,title in zip(axes,[cumulative_reward,pseudo,realized],['실제 누적 보상','누적 pseudo-regret','tμ* − 실제 보상']):
        cp.draw_grid(ax,np.column_stack([steps,values]),title,col_labels=['t','값'],fmt='{:.2f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_bandit_protocol(configuration, initial_table, learned_policy):
    """고정 학습 예산과 초기 기록, 평가할 정책을 그립니다."""
    fig,axes=cp.flow([3.2,3.,3.],height=3.,wspace=.55)
    cp.draw_grid(axes[0],configuration,'실험 계획',row_labels=['행동','학습 step','독립 seed','평가 선택'],fmt='{:.0f}',kind='plain')
    cp.draw_grid(axes[1],initial_table,'첫 관측 뒤 Q·N',col_labels=['Q','N'],fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[2],learned_policy[:,None],'평가에서는 고정 π',fmt='{:.3f}',kind='plain')
    cp.connect(fig,axes[1],axes[2],'학습 중단')
    cp.show(fig)


def plot_bandit_learning(steps, reward_means, regret_means, regret_se, names):
    """실제 학습에서 계산한 누적 평균 보상과 pseudo-regret 평균±seed SE를 그립니다."""
    fig=cp.new_figure((11.7,3.8))
    axes=fig.subplots(1,2)
    colors=[cp.MUTED,cp.BLUE,cp.RED,cp.GOLD]
    for row,name,color in zip(range(len(names)),names,colors):
        axes[0].plot(steps,reward_means[row],label=name,color=color)
        axes[1].plot(steps,regret_means[row],label=name,color=color)
        axes[1].fill_between(steps,regret_means[row]-regret_se[row],regret_means[row]+regret_se[row],color=color,alpha=.12)
    for ax,title,ylabel in zip(axes,['학습 중 얻은 보상','기회 손실의 기대값 성분'],['누적 보상 / step','누적 pseudo-regret']):
        ax.set(xlabel='학습 step',ylabel=ylabel,title=title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    axes[1].legend(frameon=False,fontsize=9)
    fig.subplots_adjust(wspace=.32,bottom=.18)
    cp.show(fig)


def plot_bandit_evaluation(names, exact_scores, sampled_scores, final_regret):
    """모든 seed의 고정 정책 정확한 가치와 새 보상 표본의 평균, 최종 regret를 보존합니다."""
    fig=cp.new_figure((12.5,3.8))
    axes=fig.subplots(1,3)
    for index,name in enumerate(names):
        offset=(np.arange(exact_scores.shape[1])%8-3.5)*.018
        axes[0].scatter(index+offset,exact_scores[index],s=10,color=cp.BLUE,alpha=.4)
        axes[1].scatter(index+offset,sampled_scores[index],s=10,color=cp.BLUE,alpha=.4)
        axes[2].scatter(index+offset,final_regret[index],s=10,color=cp.BLUE,alpha=.4)
    for ax,title in zip(axes,['고정 greedy π의 정확한 가치','새 1,000회 평가 보상 평균','학습 누적 pseudo-regret']):
        ax.set_xticks(range(len(names)),names,rotation=15,fontsize=8)
        ax.set_title(title,fontsize=10)
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(wspace=.33,bottom=.22)
    fig.canvas.draw()
    cp.show(fig)
