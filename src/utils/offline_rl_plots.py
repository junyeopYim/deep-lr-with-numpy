"""61 오프라인 RL 그림. 모든 수학·학습·통계는 노트북에서 계산합니다."""
from . import concept_plots as cp


def plot_panels(arrays,titles,columns=None,rows=None):
    """전달된 작은 표들을 같은 순서로 보여 줍니다."""
    fig,axes=cp.flow([3.3]*len(arrays),height=3.5,wspace=.6)
    for i,(ax,values,title) in enumerate(zip(axes,arrays,titles)):
        labels=None if columns is None else columns[i]
        row_labels=None if rows is None else rows[i]
        cp.draw_grid(ax,values,title,col_labels=labels,row_labels=row_labels,kind='plain',fmt='{:.3g}',fontsize=9)
        if len(arrays)==1:
            ax.set_aspect('auto')
            fig.set_size_inches(11 if rows is not None else 8,max(3.5,len(values)*.4+1.5))
    fig.subplots_adjust(top=.82,bottom=.1,left=.35 if rows is not None else .04,right=.97)
    cp.show(fig)


def plot_curves(histories,labels,ylabel):
    """전달된 시간·측정값만 선으로 표시합니다."""
    fig=cp.new_figure((8,3.4))
    ax=fig.subplots()
    for history,label in zip(histories,labels):
        ax.plot(history[:,0],history[:,1],label=label)
    ax.set(xlabel='고정 데이터에서 Q 갱신 수',ylabel=ylabel)
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    cp.show(fig)

