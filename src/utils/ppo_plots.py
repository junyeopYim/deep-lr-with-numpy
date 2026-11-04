"""58번 PPO의 순수 그림: 확률비·목적·기울기·학습 진단은 본문 계산을 받습니다."""
import numpy as np
from . import concept_plots as cp


def plot_policy_ratios(old,current,ratios):
    """같은 선택 행동의 old확률·현재확률·그 비율을 비교합니다."""
    fig,axes=cp.flow([3.,3.,3.],height=2.8,wspace=.6)
    for ax,table,title in zip(axes,[old,current,ratios],['수집 당시 old π(a|s)','현재 π(a|s)','현재 / old 확률비']):
        cp.draw_grid(ax,table[:,None],title,fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_signed_clip(ratios,raw,clipped,example):
    """advantage 부호에 따라 달라지는 surrogate의 평평한 가지를 보여 줍니다."""
    fig=cp.new_figure((10.8,3.6))
    axes=fig.subplots(1,3,gridspec_kw={'width_ratios':[1,1,1.35]})
    for index,ax,title in zip(range(2),axes[:2],['A=+1','A=−1']):
        ax.plot(ratios,raw[index],color=cp.MUTED,ls='--',label='rA')
        ax.plot(ratios,clipped[index],color=cp.BLUE,label='min surrogate')
        ax.axvline(.8,color=cp.GOLD,lw=.9)
        ax.axvline(1.2,color=cp.GOLD,lw=.9)
        ax.set(xlabel='확률비 r',title=title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    axes[0].legend(frameon=False,fontsize=8)
    cp.draw_grid(axes[2],example,'네 가지 바깥 비율',col_labels=['r','A','미분활성'],fmt='{:.1f}',kind='plain',fontsize=9)
    fig.subplots_adjust(wspace=.35,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_ppo_rollout(states,old_values,advantages,returns):
    """수집한 두 경로의 old V·고정 A·critic return을 그립니다."""
    fig,axes=cp.flow([2.4,2.6,2.6,2.6],height=2.8,wspace=.55)
    for ax,table,title in zip(axes,[states,old_values,advantages,returns],['수집한 상태','old critic V','고정 GAE A','고정 λ return']):
        cp.draw_grid(ax,table,title,fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_ppo_objective(components,coefficients,contributions):
    """actor·value·entropy 항의 부호와 계수를 보여 줍니다."""
    fig,axes=cp.flow([3.,3.,3.],height=3.,wspace=.6)
    for ax,table,title in zip(axes,[components,coefficients,contributions],['각 목적의 값','붙이는 계수','총손실의 기여']):
        cp.draw_grid(ax,table[:,None],title,row_labels=['actor','value','entropy'],fmt='{:.4f}',kind='plain')
    cp.show(fig)


def plot_ppo_protocol(budgets,labels,stored):
    """같은 상호작용 예산과 다른 optimizer 예산, 고정해 둘 old 배열을 그립니다."""
    fig,axes=cp.flow([5.2,4.2],height=3.2,wspace=.6)
    cp.draw_grid(axes[0],budgets,'실행 하나의 고정 예산',row_labels=labels,col_labels=['경로','전이','갱신'],fmt='{:.0f}',kind='plain',fontsize=9)
    cp.draw_grid(axes[1],stored,'한 rollout에서 복사한 상수',col_labels=['old logπ','old V','A','return'],fmt='{:.3f}',kind='plain',fontsize=8)
    cp.show(fig)


def plot_ppo_learning(updates,values,means,names):
    """모든seed의정확J·oldKL·비율범위초과및미분포화진단을그립니다."""
    fig=cp.new_figure((12.,3.8))
    axes=fig.subplots(1,3)
    for index,name,color in zip(range(len(names)),names,[cp.BLUE,cp.RED,cp.GOLD]):
        for run in values[index]:
            axes[0].plot(updates,run,color=color,alpha=.25,lw=.8)
        axes[0].plot(updates,means[0][index],color=color,label=name)
        axes[1].plot(updates,means[1][index],color=color)
        axes[2].plot(updates,means[2][index],color=color,label=name+' 범위밖')
        axes[2].plot(updates,means[3][index],color=color,ls=':',label=name+' 미분포화')
    for ax,title in zip(axes,['학습 확률정책의 정확 J','관측상태의 categorical KL','범위 초과율과 clip 미분 포화율']):
        ax.set(xlabel='새 rollout 배치 수',title=title)
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=7)
    axes[2].legend(frameon=False,fontsize=6)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_ppo_epochs(audits,names):
    """첫 수집 배치를 여러 epoch 쓰는 동안의 실제 KL·비율 범위를 그립니다."""
    fig=cp.new_figure((10.6,3.7))
    axes=fig.subplots(1,2)
    for audit,name,color in zip(audits,names,[cp.BLUE,cp.RED,cp.GOLD]):
        axes[0].plot(audit[:,0],audit[:,2],label=name,color=color)
        axes[1].plot(audit[:,0],audit[:,5],color=color,ls=':')
        axes[1].plot(audit[:,0],audit[:,6],color=color,label=name)
    axes[1].axhline(.8,color=cp.MUTED,lw=.8)
    axes[1].axhline(1.2,color=cp.MUTED,lw=.8)
    axes[0].set(xlabel='같은 첫 배치의 epoch',ylabel='KL(old || current)',title='old 분포는 epoch 중 고정')
    axes[1].set(xlabel='같은 첫 배치의 epoch',ylabel='선택행동 확률비',title='점선 min · 실선 max: clip은 강제 범위가 아님')
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_ppo_final(names,values,evaluation):
    """최종 모든 학습seed와 첫seed 독립평가를 그립니다."""
    fig,axes=cp.flow([4.5,5.2],height=3.2,wspace=.6)
    ax=axes[0]
    ax.set_axis_on()
    for index in range(len(names)):
        ax.scatter(index+np.linspace(-.08,.08,values.shape[1]),values[index],color=cp.BLUE,s=28)
    ax.set_xticks(range(len(names)),names,fontsize=8)
    ax.set(ylabel='정확 J',title='점 하나 = 독립 학습 seed')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    cp.draw_grid(axes[1],evaluation,'첫seed: 새로운2048경로',row_labels=names,col_labels=['평균','MC SE','정확 J'],fmt='{:.3f}',kind='plain',fontsize=9)
    fig.canvas.draw()
    cp.show(fig)
