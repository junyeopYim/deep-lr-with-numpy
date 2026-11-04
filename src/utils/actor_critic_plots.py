"""57번 actor·critic·GAE 그림. 타깃·기울기·학습 지표는 본문에서 계산합니다."""
import numpy as np
from . import concept_plots as cp


def plot_actor_critic_values(policy,q,values,advantages):
    """정책·행동 가치·상태 가치·advantage를 시작 상태의 숫자로 비교합니다."""
    fig,axes=cp.flow([2.6,2.6,2.,2.6],height=2.8,wspace=.55)
    for ax,table,title in zip(axes,[policy,q,values,advantages],['actor 행동 확률','Q: 행동별 기대','critic의 정확 기준 V','advantage Q − V']):
        cp.draw_grid(ax,np.atleast_2d(table),title,fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_gae_masks(rewards,values,next_values,terminated,boundary,deltas):
    """수집 경계와 실제 종료가 서로 다른 두 마스크임을 표로 보여 줍니다."""
    fig,axes=cp.flow([4.6,4.6],height=3.1,wspace=.6)
    cp.draw_grid(axes[0],np.column_stack([rewards,values,next_values]),'두 전이: 보상·현재V·다음V',col_labels=['r','V','V다음'],fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[1],np.column_stack([terminated,boundary,deltas]),'서로 다른 마스크와 TD잔차',col_labels=['terminated','boundary','δ'],fmt='{:.2f}',kind='plain')
    cp.show(fig)


def plot_gae_mixture(lambdas,coefficients,advantages,returns):
    """같은 TD잔차에 다른 λ를 적용한 계수·advantage·critic타깃을 그립니다."""
    fig,axes=cp.flow([3.1,3.1,3.1],height=3.1,wspace=.6)
    labels=[f'λ={value:g}' for value in lambdas]
    for ax,table,title in zip(axes,[coefficients,advantages,returns],['δ0·δ1에 붙는 계수','GAE advantage','Vold + advantage']):
        cp.draw_grid(ax,table,title,row_labels=labels,fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_gae_bias_variance(values,labels):
    """전수열거한 경로기울기의 편향norm과 분산trace를 그립니다."""
    fig,axes=cp.flow([6.],height=3.5)
    cp.draw_grid(axes[0],values,'정확 열거: 기울기 편향 norm · 분산 trace',row_labels=labels,col_labels=['편향','분산'],fmt='{:.4f}',kind='plain',fontsize=10)
    cp.show(fig)


def plot_ac_gradients(actor_gradient,critic_gradient,targets):
    """서로 다른 목적과 분모에서 나온 두 모델 미분을 보여 줍니다."""
    fig,axes=cp.flow([3.4,2.4,3.2],height=3.1,wspace=.6)
    cp.draw_grid(axes[0],actor_gradient,'actor θ 미분',col_labels=['a0','a1'],fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[1],critic_gradient[:,None],'critic w 미분',fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[2],targets,'미리 고정한 A · return',col_labels=['A','return'],fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_ac_protocol(settings,sequence):
    """고정 예산과 oldcritic→타깃고정→각모델갱신 순서를 보여 줍니다."""
    fig,axes=cp.flow([3.,5.7],height=3.3,wspace=.6)
    cp.draw_grid(axes[0],settings,'사전 고정 실험',row_labels=['seed','갱신','episode/배치','actor lr','critic lr'],fmt='{:.3g}',kind='plain')
    cp.draw_grid(axes[1],sequence,'이전 값 → 고정 타깃 → 새 값',col_labels=['이전V','return','V 갱신예'],fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_ac_learning(updates,values,errors,mean_values,mean_errors,names):
    """모든 학습 seed와 평균의 정확정책가치·critic오차를 그립니다."""
    fig=cp.new_figure((10.3,3.7))
    axes=fig.subplots(1,2)
    for ax,data,means,title in zip(axes,[values,errors],[mean_values,mean_errors],['학습 확률정책의 정확 J','현재 정책의 정확 V와 critic RMSE']):
        for index,name,color in zip(range(len(names)),names,[cp.BLUE,cp.RED,cp.GOLD]):
            for run in data[index]:
                ax.plot(updates,run,color=color,alpha=.25,lw=.8)
            ax.plot(updates,means[index],color=color,label=name)
        ax.set(xlabel='새 rollout 배치 갱신 수',title=title)
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_ac_final(names,values,evaluation):
    """최종 모든 seed와 첫 seed 독립경로평가 평균·SE를 보여 줍니다."""
    fig,axes=cp.flow([4.4,5.3],height=3.3,wspace=.65)
    ax=axes[0]
    ax.set_axis_on()
    for index in range(len(names)):
        ax.scatter(index+np.linspace(-.08,.08,values.shape[1]),values[index],color=cp.BLUE,s=28)
    ax.set_xticks(range(len(names)),names)
    ax.set(ylabel='정확 J',title='점 하나 = 독립 학습 seed')
    ax.grid(False)
    ax.spines[['top','right']].set_visible(False)
    cp.draw_grid(axes[1],evaluation,'첫seed: 새2048경로 평가',row_labels=names,col_labels=['평균','MC SE','정확 J'],fmt='{:.3f}',kind='plain')
    fig.canvas.draw()
    cp.show(fig)
