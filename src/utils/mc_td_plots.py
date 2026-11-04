"""53번 MC·TD의 그림. 경로·타깃·편향·분산·학습 결과는 본문에서 계산합니다."""
import numpy as np
from . import concept_plots as cp


def plot_mc_path(states,rewards,returns):
    """완료한 한 경로의 상태·보상·시점별 return을 그립니다."""
    fig,axes=cp.flow([3.4,2.7,2.7],height=2.9,wspace=.55)
    cp.draw_grid(axes[0],states[None,:],'완료한 상태 경로',fmt='{:.0f}',kind='plain')
    cp.draw_grid(axes[1],rewards[:,None],'다음에 받은 보상',fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[2],returns[:,None],'각 시점의 MC return',fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_td_update(current,next_values,rewards,targets,errors):
    """한 전이의 현재 값·보상+다음 값·타깃·TD 오차를 그립니다."""
    fig,axes=cp.flow([2.4,2.4,2.4,2.4],height=2.8,wspace=.6)
    for ax,values,title in zip(axes,[current,np.column_stack([rewards,next_values]),targets,errors],['현재 V(s)','r와 종료마스크 후 V다음','TD 타깃','타깃 − 현재 V']):
        cp.draw_grid(ax,np.atleast_2d(values).reshape(2,-1),title,row_labels=['비종료 전이','종료 전이'],fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_lambda_weights(targets,weights,contributions):
    """n-step 타깃들을 유한 λ 가중치로 섞는 모습을 그립니다."""
    fig,axes=cp.flow([3.,3.,3.],height=3.,wspace=.55)
    for ax,values,title in zip(axes,[targets,weights,contributions],['1·2·3-step 타깃','유한 λ 가중치','가중 기여']):
        cp.draw_grid(ax,values[:,None],title,row_labels=['n=1','n=2','n=3'],fmt='{:.4f}',kind='plain')
    cp.connect(fig,axes[0],axes[1],'×')
    cp.connect(fig,axes[1],axes[2],'마지막 가중치 보존')
    cp.show(fig)


def plot_target_distribution(weights,targets,moments,names):
    """완전 경로의 정확한 확률과 타깃·분포 통계를 그립니다."""
    fig,axes=cp.flow([1.8,4.8,4.8],height=3.6,wspace=.6)
    cp.draw_grid(axes[0],weights[:,None],'완전 경로 확률',fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[1],targets,'경로마다 다른 타깃',col_labels=names,fmt='{:.3f}',kind='plain',fontsize=8)
    cp.draw_grid(axes[2],moments,'정확한 타깃 통계',row_labels=names,col_labels=['평균','분산','편향','MSE'],fmt='{:.3f}',kind='plain',fontsize=8)
    cp.show(fig)


def plot_prediction_protocol(settings,values,targets,updated):
    """고정 계획과 V 복사본에서 타깃을 만든 뒤 갱신하는 순서를 보여 줍니다."""
    fig,axes=cp.flow([3.1,2.4,2.4,2.4],height=2.9,wspace=.55)
    cp.draw_grid(axes[0],settings,'고정 학습 계획',row_labels=['seed','episode','α','γ','λ'],fmt='{:.2g}',kind='plain')
    for ax,table,title in zip(axes[1:],[values,targets,updated],['갱신 전 V 고정','타깃 계산','V ← V+α(타깃−V)']):
        cp.draw_grid(ax,table[:,None],title,fmt='{:.3f}',kind='plain',fontsize=9)
    cp.show(fig)


def plot_prediction_learning(episodes,mean_rmse,se_rmse,names):
    """정확한 V에 대한 독립 seed 평균 RMSE와 평균의 SE를 그립니다."""
    fig=cp.new_figure((9.2,3.6))
    ax=fig.subplots()
    for index,name,color in zip(range(len(names)),names,[cp.MUTED,cp.BLUE,cp.RED,cp.GOLD]):
        ax.plot(episodes,mean_rmse[index],label=name,color=color)
        ax.fill_between(episodes,mean_rmse[index]-se_rmse[index],mean_rmse[index]+se_rmse[index],color=color,alpha=.12)
    ax.set(xlabel='같은 수의 완료 episode',ylabel='비종료 3상태 RMSE',title='같은 경로·초기값·α에서 타깃만 바꾼 실제 학습')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.subplots_adjust(bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_prediction_final(names,final_rmse,mean_values,truth):
    """실패 실행도 포함한 최종 오차 점과 상태별 seed 평균 예측을 그립니다."""
    fig,axes=cp.flow([4.8,4.8],height=3.4,wspace=.55)
    ax=axes[0]
    ax.set_axis_on()
    for index,name in enumerate(names):
        offsets=(np.arange(final_rmse.shape[1])%8-3.5)*.02
        ax.scatter(index+offsets,final_rmse[index],s=16,color=cp.BLUE,alpha=.55)
    ax.set_xticks(range(len(names)),names)
    ax.set_ylabel('최종 RMSE')
    ax.set_title('점 하나 = 독립 학습 seed',fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    table=np.concatenate([mean_values,truth[None,:]],axis=0)
    cp.draw_grid(axes[1],table,'학습 seed 평균 V와 정확해',row_labels=names+['정확해'],col_labels=['s0','s1','s2'],fmt='{:.3f}',kind='plain',fontsize=9)
    fig.canvas.draw()
    cp.show(fig)
