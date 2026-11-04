"""31 에너지 모형의 본문 계산 결과를 표시합니다. 학습·샘플링·밀도 계산은 하지 않습니다."""
from . import concept_plots as cp


def plot_energy_table(energies,weights,masses):
    """상태별 에너지에서 가중치·확률로 가는 세 배열을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=3.1,wspace=.4)
    for ax,value,title in zip(axes,[energies,weights,masses],['상태별 에너지 E','정규화 전 exp(−E)','합이 1인 확률']):
        cp.draw_grid(ax,value[None,:],title,kind='plain',fmt='{:.3f}')
    cp.show(fig)


def plot_phase_terms(grid,density,positive,negative,gradient):
    """현재 밀도와 두 phase의 특징 평균·차이를 표시합니다."""
    fig,axes=cp.flow([3.2,3,3,3],height=3.2,wspace=.5)
    axes[0].plot(grid,density,color=cp.BLUE)
    axes[0].set_title('현재 모형 밀도',fontsize=10)
    for ax,value,title in zip(axes[1:],[positive,negative,gradient],['자료 특징 평균','모형 특징 평균','양의 phase − 음의 phase']):
        cp.draw_grid(ax,value[None,:],title,col_labels=['x','x²'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.show(fig)


def plot_training_plan(grid,density,data,settings):
    """합성 분포·관측·고정 학습 계획을 표시합니다."""
    fig,axes=cp.flow([4,3,3],height=3.2,wspace=.5)
    axes[0].plot(grid,density,color=cp.BLUE)
    axes[0].set_title('자료를 뽑는 분포',fontsize=10)
    cp.draw_grid(axes[1],data[:,None],'train의 첫 관측',kind='plain',fmt='{:.2f}',fontsize=9)
    cp.draw_grid(axes[2],settings[:,None],'미리 정한 계획',row_labels=['train','valid','test','갱신'],kind='plain',fontsize=9)
    cp.show(fig)


def plot_energy_fit(history,grid,densities,labels):
    """학습 기록과 별도로 계산한 최종 밀도 곡선을 그립니다."""
    fig=cp.new_figure((10,3.4))
    axes=fig.subplots(1,2)
    for column,label in [(1,'train'),(2,'validation')]:
        axes[0].plot(history[:,0],history[:,column],label=label)
    axes[0].set(xlabel='갱신 수',ylabel='관측별 density NLL')
    for density,label in zip(densities,labels):
        axes[1].plot(grid,density,label=label)
    axes[1].set(xlabel='과일 측정 좌표 x',ylabel='밀도')
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_langevin_proposal(states,proposal,terms):
    """현재점·drift·제안과 수락 판정에 필요한 값을 표시합니다."""
    fig,axes=cp.flow([3.5,3.5,3.5],height=3.2,wspace=.5)
    cp.draw_grid(axes[0],states,'현재점과 에너지 기울기',col_labels=['x','E′(x)'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.draw_grid(axes[1],proposal,'이동과 잡음',col_labels=['−hE′','잡음','제안 y'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.draw_grid(axes[2],terms,'MALA 판정',col_labels=['log 비율','수락 확률'],kind='plain',fmt='{:.3f}',fontsize=9)
    cp.show(fig)


def plot_dependent_blocks(values,means):
    """연속 표본을 block으로 묶는 방식과 미리 계산한 평균을 보여 줍니다."""
    fig,axes=cp.flow([5,3],height=2.8,wspace=.5)
    cp.draw_grid(axes[0],values,'행마다 연속한 네 시각',kind='plain',fmt='{:.2f}',fontsize=9)
    cp.draw_grid(axes[1],means[:,None],'block 평균',kind='plain',fmt='{:.2f}')
    cp.show(fig)


def plot_chains(paths,labels):
    """같은 네 초기값의 보관 전후 실제 경로를 그립니다."""
    fig=cp.new_figure((11,6))
    axes=fig.subplots(len(paths),1)
    for ax,path,label in zip(axes,paths,labels):
        ax.plot(path,alpha=.65,lw=.7)
        ax.axvline(400,color=cp.INK,ls='--',lw=.8)
        ax.set(title=label,ylabel='x')
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('chain 시각 · 점선 뒤 2,400개 보관')
    fig.tight_layout()
    cp.show(fig)


def plot_chain_density(grid,density,centers,histograms,labels,summary):
    """받은 histogram 밀도·격자 기준·chain별 진단 요약을 표시합니다."""
    fig=cp.new_figure((11,3.6))
    axes=fig.subplots(1,2,gridspec_kw={'width_ratios':[1,1.4]})
    axes[0].plot(grid,density,color=cp.INK,label='격자 기준')
    for histogram,label in zip(histograms,labels):
        axes[0].step(centers,histogram,where='mid',label=label,alpha=.75)
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].set(xlabel='x',ylabel='밀도')
    cp.draw_grid(axes[1],summary,'네 chain을 함께 읽기',row_labels=labels,
                 col_labels=['수락률','평균','chain 평균\nSD','block SE\n평균'],kind='plain',fmt='{:.3f}',fontsize=8)
    fig.subplots_adjust(wspace=.45,bottom=.2,top=.8,left=.06,right=.98)
    cp.show(fig)
