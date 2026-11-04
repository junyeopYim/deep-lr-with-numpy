"""63 논문 재현의 본문 계산·실험 기록을 표시합니다."""
from . import concept_plots as cp


def plot_scope_table(values):
    """원자료와 선택한 경로 끝점의 통계를 구분합니다."""
    fig,axes=cp.flow([7],height=2.8)
    cp.draw_grid(axes[0],values,'원자료와 이번 재현의 생성 목표',row_labels=['원자료 y','경로의 t=1'],
                 col_labels=['평균','분산','표준편차'],kind='plain',fmt='{:.4f}',fontsize=10)
    cp.show(fig)


def plot_specification(inputs,design,outputs):
    """입력·설계행렬·타깃의 shape를 수치와 함께 보여 줍니다."""
    fig,axes=cp.flow([3,5,3],height=3,wspace=.5)
    cp.draw_grid(axes[0],inputs,'위치·시간',col_labels=['x','t'],kind='plain',fmt='{:.2f}')
    cp.draw_grid(axes[1],design,'시간2차 설계행렬 Φ',kind='plain',fmt='{:.2f}',fontsize=9)
    cp.draw_grid(axes[2],outputs,'예측·타깃',col_labels=['v','u'],kind='plain',fmt='{:.2f}')
    cp.show(fig)


def plot_independent_checks(design,residual,gradient):
    """손계산할 수 있는 작은 회귀 문제의 기울기 경로를 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3,wspace=.5)
    for ax,value,title in zip(axes,[design,residual[:,None],gradient[:,None]],['독립 검산 입력 Φ','예측−타깃','Φᵀr / N']):
        cp.draw_grid(ax,value,title,kind='plain',fmt='{:.3f}')
    cp.show(fig)


def plot_reproduction_plan(settings):
    """사전에 고정한 분할·갱신·모형 크기를 그립니다."""
    fig,axes=cp.flow([7],height=3)
    cp.draw_grid(axes[0],settings,'모든 seed에 적용한 계획',row_labels=['시간 없음','시간2차'],
                 col_labels=['train','valid','test','GD','파라미터'],kind='plain',fmt='{:.0f}',fontsize=10)
    cp.show(fig)


def plot_reproduction_learning(histories,labels):
    """같은 실험의 실제 train/valid 기록을 표시합니다."""
    fig=cp.new_figure((10,3.4));axes=fig.subplots(1,2)
    for history,label in zip(histories,labels):
        for ax,column in zip(axes,[1,2]):
            ax.plot(history[:,0],history[:,column],label=label)
    for ax,name in zip(axes,['train CFM','고정 valid CFM']):
        ax.set(xlabel='GD 갱신',ylabel=name);ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_exact_paths(times,paths):
    """알려진 Gaussian ODE의 정확한 경로를 표시합니다."""
    fig=cp.new_figure((7,3));ax=fig.subplots()
    ax.plot(times,paths)
    ax.set(xlabel='시간 t',ylabel='정확한 위치 x(t)')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_paired_evidence(metrics,labels):
    """seed별 두 후보의 지표를 연결해 모든 반복을 보입니다."""
    fig=cp.new_figure((11,3.3));axes=fig.subplots(1,3)
    for ax,column,title in zip(axes,[0,1,2],['test 주변 속도 MSE','정확 끝점과 평균 절대 거리','Heun32·64 끝점 차이']):
        for row in metrics:
            ax.plot([0,1],row[:,column],'-o',alpha=.7)
        ax.set(xticks=[0,1],xticklabels=labels,title=title)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_paired_samples(initial,generated,reference,labels):
    """같은 초기 샘플의 끝점을 정답 경로와 비교합니다."""
    fig=cp.new_figure((10,3.5));axes=fig.subplots(1,2)
    for ax,points,label in zip(axes,generated,labels):
        ax.scatter(initial,points,s=10,alpha=.4,label=label)
        ax.plot(initial,reference,color=cp.RED,alpha=.8,label='같은 초기점의 정확 끝점')
        ax.set(xlabel='고정 출발점 x0',ylabel='생성 끝점');ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)
