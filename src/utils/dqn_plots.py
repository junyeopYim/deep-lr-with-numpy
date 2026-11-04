"""55번 DQN의 그림: 신경망·타깃·학습 지표는 모두 본문에서 계산해 전달합니다."""
import numpy as np
from . import concept_plots as cp


def plot_q_network(x,hidden,q,selected):
    """one-hot 입력에서 은닉값·Q·선택한 행동 값으로 이어지는 배열 흐름을 그립니다."""
    fig,axes=cp.flow([2.5,3.,2.5,1.8],height=2.9,wspace=.55)
    for ax,table,title in zip(axes,[x,hidden,q,selected[:,None]],['상태 one-hot X','tanh 은닉 H','행동별 Q','선택한 Q']):
        cp.draw_grid(ax,table,title,fmt='{:.2f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_td_batch(batch,targets,residual,dq):
    """한 replay 배치의 고정 타깃·오차·선택한 출력의 미분을 그립니다."""
    fig,axes=cp.flow([3.8,2.,2.,2.8],height=2.8,wspace=.55)
    cp.draw_grid(axes[0],batch,'저장한 전이 배치',col_labels=['s','a','r','s′','종료'],fmt='{:.2g}',kind='plain',fontsize=8)
    for ax,table,title in zip(axes[1:],[targets[:,None],residual[:,None],dq],['고정 타깃 y','Q선택 − y','dQ: 선택한 칸만']):
        cp.draw_grid(ax,table,title,fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_target_freeze(steps,online,target,copied):
    """온라인 숫자와 주기적으로 복사된 타깃의 시간 순서를 보여 줍니다."""
    fig,axes=cp.flow([5.,5.],height=3.,wspace=.55)
    cp.draw_grid(axes[0],np.stack([online,target,copied]),'갱신 후 online / target / 복사 여부',row_labels=['online','target','copy'],col_labels=[str(x) for x in steps],fmt='{:.0f}',kind='plain',fontsize=9)
    cp.draw_formula(axes[1],r'$y=r+\gamma(1-d)\max Q_{\bar{\theta}}(s^\prime)$',sub='현재 θ만 미분 · θ̄는 복사 시점에만 변경')
    cp.show(fig)


def plot_double_selection(online,target,values):
    """선택과 평가에 서로 다른 Q를 쓰는 Double 타깃을 보여 줍니다."""
    fig,axes=cp.flow([3.,3.,3.],height=2.7,wspace=.55)
    cp.draw_grid(axes[0],online,'online: 행동 선택',col_labels=['a0','a1'],fmt='{:.1f}',kind='plain')
    cp.draw_grid(axes[1],target,'target: 선택한 행동 평가',col_labels=['a0','a1'],fmt='{:.1f}',kind='plain')
    cp.draw_grid(axes[2],values,'최종 TD 타깃',col_labels=['DQN','Double'],fmt='{:.2f}',kind='plain')
    cp.show(fig)


def plot_dqn_protocol(settings,transitions):
    """같은 환경 상호작용 예산과 replay 수집·갱신 주기를 그립니다."""
    fig,axes=cp.flow([3.3,5.5],height=3.7,wspace=.7)
    cp.draw_grid(axes[0],settings,'사전 고정 학습 계획',row_labels=['독립 seed','episode','replay 용량','batch','Adam lr','파라미터'],fmt='{:.3g}',kind='plain',fontsize=9)
    cp.draw_grid(axes[1],transitions,'전이 수집 → replay 저장 → 배치 갱신',col_labels=['환경 전이','저장 수','갱신 수'],fmt='{:.0f}',kind='plain',fontsize=10)
    cp.show(fig)


def plot_dqn_learning(episodes,rmse,values,gaps,mean_rmse,mean_values,mean_gaps,names):
    """각 학습 seed의 Q RMSE·정확 가치·online-target 차이를 모두 그립니다."""
    fig=cp.new_figure((12.,3.8))
    axes=fig.subplots(1,3)
    for data,means,ax,title in zip([rmse,values,gaps],[mean_rmse,mean_values,mean_gaps],axes,['비종료 6칸 Q RMSE','동결 greedy 정책의 정확 가치','online-target Q의 RMS 차이']):
        for index,name,color in zip(range(len(names)),names,[cp.BLUE,cp.RED,cp.GOLD]):
            for run in data[index]:
                ax.plot(episodes,run,color=color,alpha=.3,lw=.9)
            ax.plot(episodes,means[index],color=color,label=name,lw=1.8)
        ax.set(xlabel='완료 episode',title=title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    axes[0].legend(frameon=False,fontsize=7)
    fig.subplots_adjust(wspace=.25,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_dqn_final(names,training_returns,q_values,truth):
    """모든 seed의 실제 수집 return과 최종 평균 Q를 그립니다."""
    fig,axes=cp.flow([4.7,5.5],height=3.7,wspace=.6)
    ax=axes[0]
    ax.set_axis_on()
    for index in range(len(names)):
        offsets=np.linspace(-.08,.08,training_returns.shape[1])
        ax.scatter(index+offsets,training_returns[index],s=30,color=cp.BLUE)
    ax.set_xticks(range(len(names)),names,fontsize=8)
    ax.set(ylabel='학습 중 실제 경로 return 평균',title='초기 탐색 비용까지 포함한 관측값')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    cp.draw_grid(axes[1],np.concatenate([q_values,truth[None,:]],axis=0),'최종 seed 평균 Q · 비종료6칸',row_labels=names+['정확해'],col_labels=['0a0','0a1','1a0','1a1','2a0','2a1'],fmt='{:.2f}',kind='plain',fontsize=8)
    fig.canvas.draw()
    cp.show(fig)
