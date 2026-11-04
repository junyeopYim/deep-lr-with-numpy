"""36 Flow Matching의 이미 계산한 경로·속도·실험 통계를 그립니다."""
from . import concept_plots as cp


def plot_straight_paths(paths,positions,velocities):
    """직선 경로와 중간점·속도 배열을 보여 줍니다."""
    fig,axes=cp.flow([4,3,3],height=3.5,wspace=.45)
    for path in paths:
        axes[0].plot(path[:,0],path[:,1],'-o',ms=3)
    axes[0].set_axis_on()
    axes[0].set(title='독립 endpoint를 이은 직선',xlabel='첫 좌표',ylabel='둘째 좌표')
    cp.draw_grid(axes[1],positions,'t=.5의 위치',kind='plain',fmt='{:.2f}')
    cp.draw_grid(axes[2],velocities,'각 쌍의 조건부 속도',kind='plain',fmt='{:.2f}')
    cp.show(fig)


def plot_conditional_average(targets,mean,terms):
    """같은 위치를 지나는 서로 다른 속도와 손실 분해를 표시합니다."""
    fig,axes=cp.flow([3,3,4],height=3,wspace=.5)
    cp.draw_grid(axes[0],targets,'같은 위치의 두 타깃 u',kind='plain')
    cp.draw_grid(axes[1],mean[None,:],'조건부 평균 속도',kind='plain')
    cp.draw_grid(axes[2],terms[None,:],'조건부 손실 = 평균 손실 + 잔여',col_labels=['CFM','FM','상수'],kind='plain',fmt='{:.2f}',fontsize=9)
    cp.show(fig)


def plot_velocity_network(inputs,hidden,prediction,gradient):
    """좌표와 시간부터 속도·손실 미분까지 계산한 배열을 나열합니다."""
    fig,axes=cp.flow([3,4,3,3],height=3.2,wspace=.4)
    for ax,value,title in zip(axes,[inputs,hidden,prediction,gradient],['x₁,x₂,t','tanh 은닉값','예측 속도','손실의 속도 미분']):
        cp.draw_grid(ax,value,title,kind='plain',fmt='{:.2f}',fontsize=9)
    cp.show(fig)


def plot_flow_data(source,target,plan):
    """학습용 데이터 상황과 고정 계획을 보여 줍니다."""
    fig,axes=cp.flow([4,4,3],height=3.4,wspace=.5)
    for ax,points,title in zip(axes[:2],[source,target],['출발: 표준 Gaussian','도착: 두 봉우리 관측']):
        ax.set_axis_on()
        ax.scatter(points[:,0],points[:,1],s=5,alpha=.35)
        ax.set(title=title,xlim=(-4,4),ylim=(-3,3))
    cp.draw_grid(axes[2],plan[:,None],'고정 계획',row_labels=['train','valid','test','갱신'],kind='plain',fontsize=9)
    cp.show(fig)


def plot_flow_learning(history):
    """고정 validation 경로 쌍의 실제 학습 기록을 그립니다."""
    fig=cp.new_figure((7,3.2));ax=fig.subplots()
    ax.plot(history[:,0],history[:,1],color=cp.BLUE)
    ax.set(xlabel='Adam 갱신 수',ylabel='고정 valid 조건부 속도 손실')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_ode_steps(states,increments):
    """Euler·Heun의 실제 한 스텝과 기울기 평가를 표시합니다."""
    fig,axes=cp.flow([4,4],height=3,wspace=.5)
    cp.draw_grid(axes[0],states,'x′=x, x(0)=1, h=.5',col_labels=['Euler','Heun','정확해'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.draw_grid(axes[1],increments,'Heun의 두 기울기와 증분',col_labels=['k₁','k₂','증분'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.show(fig)


def plot_solver_error(nfes,errors,labels):
    """정확한 Gaussian 벡터장을 적분한 끝점 오차를 NFE별로 표시합니다."""
    fig=cp.new_figure((7,3.4));ax=fig.subplots()
    for nfe,error,label in zip(nfes,errors,labels):
        ax.loglog(nfe,error,'-o',label=label)
    ax.set(xlabel='속도장 평가 횟수 NFE',ylabel='정확한 경로 끝점과 최대 절대 오차')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_flow_samples(groups,titles,paths):
    """동일 초기 샘플을 보낸 생성 결과와 일부 이동 경로를 표시합니다."""
    fig,axes=cp.flow([4]*len(groups),height=3.7,wspace=.35)
    for ax,points,title,path in zip(axes,groups,titles,paths):
        ax.set_axis_on()
        ax.scatter(points[:,0],points[:,1],s=5,alpha=.3)
        if path is not None:
            for index in range(min(path.shape[1],8)):
                ax.plot(path[:,index,0],path[:,index,1],lw=.7,alpha=.5)
        ax.set(title=title,xlim=(-4,4),ylim=(-3,3))
    cp.show(fig)


def plot_flow_fields(points,fields,titles):
    """본문에서 계산한 같은 격자의 기준·학습 속도장을 그립니다."""
    fig,axes=cp.flow([4]*len(fields),height=3.6,wspace=.35)
    for ax,field,title in zip(axes,fields,titles):
        ax.set_axis_on()
        ax.quiver(points[:,0],points[:,1],field[:,0],field[:,1],color=cp.BLUE,
                  angles='xy',scale_units='xy',scale=5)
        ax.set(title=title,xlim=(-3,3),ylim=(-2,2))
    cp.show(fig)
