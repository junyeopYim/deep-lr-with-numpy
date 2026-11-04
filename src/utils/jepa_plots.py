"""43 JEPA 그림. 모든 수학·학습·통계는 노트북에서 계산합니다."""
from . import concept_plots as cp


def plot_panels(arrays,titles,columns=None):
    """전달된 작은 표들을 같은 순서로 보여 줍니다."""
    fig,axes=cp.flow([3.3]*len(arrays),height=3.5,wspace=.6)
    for i,(ax,values,title) in enumerate(zip(axes,arrays,titles)):
        labels=None if columns is None else columns[i]
        cp.draw_grid(ax,values,title,col_labels=labels,kind='plain',fmt='{:.3g}',fontsize=9)
        if len(arrays)==1:
            ax.set_aspect('auto')
            fig.set_size_inches(8,3.5)
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


def plot_images(images,titles):
    """전달된 원본·가림·예측 이미지를 같은 0–1 척도로 표시합니다."""
    fig,axes=cp.flow([2.6]*len(images),height=3.1,wspace=.25)
    for ax,img,title in zip(axes,images,titles):
        cp.draw_image(ax,img,title,kind='gray')
    fig.subplots_adjust(top=.8,bottom=.1)
    cp.show(fig)


def plot_objectives(histories,titles):
    """서로 단위가 다른 두 목적을 독립 y축 패널로 표시합니다."""
    fig=cp.new_figure((9,3.5))
    axes=fig.subplots(1,len(histories))
    for ax,history,title in zip(axes,histories,titles):
        ax.plot(history[:,0],history[:,1],color=cp.BLUE)
        ax.set(xlabel='갱신 수',ylabel='자기 목적의 validation 반 MSE',title=title)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    cp.show(fig)
