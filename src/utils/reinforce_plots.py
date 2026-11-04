"""56번의 본문에서 계산한 경로·gradient·분산·학습 결과만 그립니다."""
from . import concept_plots as cp


def plot_pg_panels(tables,titles,row_labels=None):
    """작은 손계산 배열들을 동일한 순서의 표로 표시합니다."""
    fig,axes=cp.flow([3.4]*len(tables),height=3.3,wspace=.65)
    for index,(ax,table,title) in enumerate(zip(axes,tables,titles)):
        rows=None if row_labels is None else row_labels[index]
        cp.draw_grid(ax,table,title,kind='plain',fmt='{:.3f}',row_labels=rows,fontsize=9)
    cp.show(fig)


def plot_pg_variance(names,variances,biases):
    """본문에서 전수열거한 분산 trace와 정확gradient와의 차이를 그립니다."""
    fig=cp.new_figure((10.5,3.5));axes=fig.subplots(1,2)
    for ax,values,title in zip(axes,[variances,biases],['경로1개 gradient의 분산 trace','정확gradient와 기대 추정치의 거리']):
        ax.bar(range(len(names)),values,color=[cp.BLUE,cp.GOLD,cp.INK,cp.RED])
        ax.set_xticks(range(len(names)),names,fontsize=8)
        ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_pg_learning(histories,names,upper):
    """두 방법의 모든3seed 정책가치를 그리며 각 실행을 평균 뒤 숨기지 않습니다."""
    fig=cp.new_figure((10,3.6));axes=fig.subplots(1,2)
    for ax,group,name in zip(axes,histories,names):
        for index,history in enumerate(group):
            ax.plot(history[:,0],history[:,1],label=f'seed {5601+index}')
        ax.axhline(upper,color=cp.RED,ls='--',label='정확 최적값')
        ax.set(xlabel='새 배치로 갱신한 횟수',ylabel='동결 확률 정책의 정확 J',title=name,ylim=(-3,1.85))
        ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)


def plot_pg_evaluation(labels,exact,means,standard_errors):
    """정확가치와 독립 평가 경로의 평균±1SE를 동일 행으로 표시합니다."""
    fig=cp.new_figure((9,3.8));ax=fig.subplots()
    positions=list(range(len(labels)))
    ax.errorbar(means,positions,xerr=standard_errors,fmt='o',color=cp.BLUE,label='2048 새 경로 평균 ±1SE')
    ax.scatter(exact,positions,color=cp.RED,marker='x',label='4경로 정확 J')
    ax.set_yticks(positions,labels,fontsize=9);ax.invert_yaxis()
    ax.set(xlabel='시작 상태 할인 보상',title='동결 확률 정책의 독립 평가')
    ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();cp.show(fig)
