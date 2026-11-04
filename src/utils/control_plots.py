"""54번의 순수 그림: Q·정책·타깃·경로 확률·실제 학습 결과를 인자로 받습니다."""
import numpy as np
from . import concept_plots as cp


def plot_control_environment(next_state,reward,policy):
    """경로의 다음 상태·보상·행동 확률을 같은 행 순서로 보여 줍니다."""
    fig,axes=cp.flow([3.2,3.2,3.2],height=3.,wspace=.65)
    for ax,table,title in zip(axes,[next_state,reward,policy],['다음 상태','즉시 보상','행동 확률 μ']):
        cp.draw_grid(ax,table,title,row_labels=['시작0','위험1','안전2','종료3'],col_labels=['a0','a1'],fmt='{:.2g}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_sarsa_targets(next_q,probabilities,targets,expected):
    """같은 다음 상태에서 표본 행동 타깃과 기대 타깃을 보여 줍니다."""
    fig,axes=cp.flow([2.7,2.7,2.7,2.7],height=2.8,wspace=.5)
    for ax,table,title in zip(axes,[next_q,probabilities,targets,expected],['다음 Q','다음 행동 μ','SARSA 타깃들','Expected SARSA']):
        cp.draw_grid(ax,np.asarray(table).reshape(-1,1),title,fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_q_target_comparison(q,targets,updated):
    """같은 Q에서 다른 타깃과 시작 상태의 한 번 갱신을 보여 줍니다."""
    fig,axes=cp.flow([2.5,3.2,3.2],height=2.8,wspace=.55)
    cp.draw_grid(axes[0],q[None,:],'위험 상태의 Q',col_labels=['좋은 행동','실수 행동'],fmt='{:.1f}',kind='plain')
    for ax,table,title in zip(axes[1:],[targets,updated],['고정된 다음 타깃','현재 .2에서 α=.1 갱신']):
        cp.draw_grid(ax,table[:,None],title,row_labels=['SARSA: a1','Expected SARSA','Q-learning'],fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_importance_paths(probabilities,ratios,returns,contributions):
    """네 경로의 행동·타깃 확률과 IS 기여를 보여 줍니다."""
    fig,axes=cp.flow([3.,2.,2.,2.],height=3.2,wspace=.6)
    for ax,table,title in zip(axes,[probabilities,ratios[:,None],returns[:,None],contributions[:,None]],['경로 확률 μ · π','비율 π/μ','경로 return','μ × 비율 × return']):
        cp.draw_grid(ax,table,title,fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_control_protocol(settings,policies,values):
    """고정 예산과 같은 Q를 두 평가 규약으로 읽는 장면을 그립니다."""
    fig,axes=cp.flow([2.6,3.2,3.2],height=3.1,wspace=.7)
    cp.draw_grid(axes[0],settings,'사전 고정 계획',row_labels=['seed','episode','전이/경로','ε','α'],fmt='{:.2g}',kind='plain')
    cp.draw_grid(axes[1],policies,'시작 상태의 행동 확률',row_labels=['ε 유지','ε 제거'],col_labels=['안전','위험'],fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[2],values[:,None],'같은 Q: 평가 정책별 가치',row_labels=['ε 유지','ε 제거'],fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_control_learning(episodes,means,standard_errors,names):
    """독립 seed들의 정확한 정책 가치 평균과 SE를 두 평가 규약별로 그립니다."""
    fig=cp.new_figure((10.5,3.8))
    axes=fig.subplots(1,2)
    for mode,ax,title in zip(range(2),axes,['탐색을 유지한 고정 행동정책','탐색을 끈 고정 배포정책']):
        for method,name,color in zip(range(len(names)),names,[cp.BLUE,cp.RED,cp.GOLD]):
            ax.plot(episodes,means[method,:,mode],label=name,color=color)
            ax.fill_between(episodes,means[method,:,mode]-standard_errors[method,:,mode],means[method,:,mode]+standard_errors[method,:,mode],alpha=.12,color=color)
        ax.set(xlabel='완료 episode',ylabel='정확 시작 가치',title=title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_control_final(names,values,training_returns):
    """모든 seed의 최종 가치 점과 실제 학습 중 받은 return을 보여 줍니다."""
    fig=cp.new_figure((10.5,3.7))
    axes=fig.subplots(1,2)
    for mode,color,offset,label in [(0,cp.BLUE,-.1,'ε 유지'),(1,cp.GOLD,.1,'ε 제거')]:
        for method in range(len(names)):
            jitter=(np.arange(values.shape[1])%8-3.5)*.009
            axes[0].scatter(method+offset+jitter,values[method,:,mode],color=color,s=15,alpha=.5,label=label if method==0 else None)
    axes[0].set_xticks(range(len(names)),names,fontsize=8)
    axes[0].set(ylabel='정확 시작 가치',title='최종 정책: 모든 학습 seed')
    axes[0].legend(frameon=False,fontsize=8)
    for method,name in enumerate(names):
        jitter=(np.arange(training_returns.shape[1])%8-3.5)*.02
        axes[1].scatter(method+jitter,training_returns[method],s=15,alpha=.6,color=cp.BLUE)
    axes[1].set_xticks(range(len(names)),names,fontsize=8)
    axes[1].set(ylabel='학습 중 경로 return 평균',title='초기 탐색 비용까지 포함한 관측값')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)
