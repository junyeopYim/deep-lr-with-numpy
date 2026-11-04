"""41 분산·공분산 그림. 모든 수학·학습·통계는 노트북에서 계산합니다."""
from . import concept_plots as cp


def plot_panels(arrays,titles,columns=None):
    """전달된 작은 표들을 같은 순서로 보여 줍니다."""
    fig,axes=cp.flow([3.3]*len(arrays),height=3.5,wspace=.6)
    for i,(ax,values,title) in enumerate(zip(axes,arrays,titles)):
        labels=None if columns is None else columns[i]
        cp.draw_grid(ax,values,title,col_labels=labels,kind='plain',fmt='{:.3g}',fontsize=9)
    fig.subplots_adjust(top=.78,bottom=.12,left=.04,right=.97)
    cp.show(fig)


def plot_curves(histories,labels,ylabel):
    """전달된 시간·측정값만 선으로 표시합니다."""
    fig=cp.new_figure((8,3.4))
    ax=fig.subplots()
    for history,label in zip(histories,labels):
        ax.plot(history[:,0],history[:,1],label=label)
    ax.set(xlabel='갱신 수',ylabel=ylabel)
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_points(groups,labels,titles):
    """이미 계산된 2차원 점을 라벨 색상으로 표시합니다."""
    fig,axes=cp.flow([3.4]*len(groups),height=3.5,wspace=.35)
    for ax,x,y,title in zip(axes,groups,labels,titles):
        ax.scatter(x[:,0],x[:,1],c=y,cmap='coolwarm',s=12,alpha=.65,vmin=0,vmax=2)
        ax.set_title(title,fontsize=11)
    fig.subplots_adjust(top=.83,bottom=.1)
    cp.show(fig)
