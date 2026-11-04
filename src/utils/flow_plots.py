"""30 Normalizing Flow의 시각화. 모델·밀도·Jacobian·적분·학습은 노트북에서 계산합니다."""

import numpy as np
from . import concept_plots as cp


def plot_affine_volume(base_table,data_table,base_vertices,data_vertices):
    """1D의 좌표·밀도와 2D의 면적 변화를 받은 값 그대로 표시합니다."""
    fig,axes=cp.flow([2.8,2.8,2.5,2.5],height=3.4,wspace=.5)
    cp.draw_grid(axes[0],base_table,"잠재 z · 표준정규",col_labels=["z","p(z)"],fmt="{:.3g}",kind="plain")
    cp.draw_grid(axes[1],data_table,"생성 x=2z+1",col_labels=["x","p(x)=p(z)/2"],fmt="{:.3g}",kind="plain")
    for ax,vertices,title in zip(axes[2:],[base_vertices,data_vertices],["2D 단위 정사각형","A 변환 뒤 면적 2"]):
        ax.fill(vertices[:,0],vertices[:,1],facecolor=cp.BLUE,alpha=.2,edgecolor=cp.BLUE)
        ax.plot(vertices[:,0],vertices[:,1],color=cp.BLUE)
        ax.set_title(title,fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-.4,3.4)
        ax.set_ylim(-.3,1.5)
    fig.subplots_adjust(top=.8,bottom=.12)
    cp.show(fig)


def plot_coupling_map(inputs,scale_shift,outputs,jacobian):
    """coupling의 유지 좌표·조건부 scale/shift·출력·삼각 Jacobian을 표시합니다."""
    fig,axes=cp.flow([2.5,2.5,2.5,2.8],height=3.3,wspace=.5)
    for ax,values,title,labels in zip(axes,[inputs,scale_shift,outputs,jacobian],
                                      ["입력 x=(a,b)","a가 정하는 s,t","밀도 방향 y=f(x)","한 점의 Jacobian"],
                                      [["a","b"],["s","t"],["a","(b−t)e⁻ˢ"],["da","db"]]):
        cp.draw_grid(ax,values,title,col_labels=labels,fmt="{:.3f}",kind="plain",fontsize=9)
    fig.subplots_adjust(top=.8,bottom=.12)
    cp.show(fig)


def plot_flow_stages(stages,labels):
    """데이터→잠재 방향의 단계별 실제 좌표 배열을 그립니다."""
    fig,axes=cp.flow([2.3]*len(stages),height=2.9,wspace=.2)
    for ax,points,label in zip(axes,stages,labels):
        ax.scatter(points[:,0],points[:,1],s=9,alpha=.55,color=cp.BLUE)
        ax.set_title(label,fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
    fig.subplots_adjust(top=.81,bottom=.12)
    cp.show(fig)


def plot_training_plan(noise,data,settings):
    """잡음으로부터 만든 곡선 자료와 고정 학습 계획을 표시합니다."""
    fig,axes=cp.flow([3.1,3.1,3.6],height=3.6,wspace=.5)
    for ax,points,title in zip(axes[:2],[noise,data],["자료 생성용 독립 잡음","비Gaussian 곡선 자료"]):
        ax.scatter(points[:,0],points[:,1],s=10,alpha=.55,color=cp.BLUE)
        ax.set_title(title,fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-3.5,3.5)
        ax.set_ylim(-2,5)
    cp.draw_grid(axes[2],settings,"미리 정한 계획",row_labels=["train","validation","test","스텝","배치"],kind="plain",fontsize=9)
    fig.subplots_adjust(top=.81,bottom=.12,right=.98)
    cp.show(fig)


def plot_nll_history(history,baseline_values):
    """관측별 validation NLL과 같은 좌표의 두 고정 기준선을 그립니다."""
    fig=cp.new_figure((8.4,3.6))
    ax=fig.subplots()
    ax.plot(history[:,0],history[:,1],color=cp.BLUE,label='flow')
    for value,label,color in zip(baseline_values,['표준정규','train-fit Gaussian'],[cp.MUTED,cp.RED]):
        ax.axhline(value,linestyle='--',linewidth=1,color=color,label=label)
    ax.set(xlabel='갱신 수',ylabel='관측별 validation NLL',title='같은 원래 좌표에서의 로그 우도')
    ax.legend(frameon=False,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_fitted_density(xx,yy,gaussian_density,flow_density,points,vmax):
    """같은 좌표·같은 색상 척도로 Gaussian과 flow의 계산된 밀도를 비교합니다."""
    fig,axes=cp.flow([3.3,3.3,3.3],height=3.4,wspace=.3)
    axes[0].scatter(points[:,0],points[:,1],s=8,alpha=.5,color=cp.BLUE)
    axes[0].set_title('고정 test 관측',fontsize=10)
    for ax,density,title in zip(axes[1:],[gaussian_density,flow_density],['train-fit Gaussian 밀도','학습 flow 밀도']):
        ax.pcolormesh(xx,yy,density,shading='auto',cmap='Blues',vmin=0,vmax=vmax)
        ax.set_title(title,fontsize=10)
    for ax in axes:
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
        ax.set_aspect('equal')
    fig.subplots_adjust(top=.81,bottom=.12)
    cp.show(fig)


def plot_quadrature_cells(density,contributions,summary):
    """셀 중앙의 밀도와 면적을 곱한 질량 근사를 구분합니다."""
    fig,axes=cp.flow([3.0,3.0,3.0],height=3.4,wspace=.5)
    cp.draw_grid(axes[0],density,'셀 중앙의 밀도',fmt='{:.3f}',kind='plain',fontsize=9)
    cp.draw_grid(axes[1],contributions,'밀도 × 셀 면적',fmt='{:.3f}',kind='plain',fontsize=9)
    cp.draw_grid(axes[2],summary,'이 작은 상자에서만 더함',row_labels=['셀 면적','질량 근사'],fmt='{:.4f}',kind='plain',fontsize=9)
    fig.subplots_adjust(top=.8,bottom=.13,right=.98)
    cp.show(fig)


def plot_generation_diagnostics(test,latent,new_latent,generated):
    """heldout 데이터→잠재와 독립 잠재→생성의 두 방향을 표시합니다."""
    fig,axes=cp.flow([2.7]*4,height=3.1,wspace=.3)
    for ax,points,title in zip(axes,[test,latent,new_latent,generated],
                               ['heldout 데이터 x','z=f(x) 진단','새 표준정규 z','x=f⁻¹(z) 생성']):
        ax.scatter(points[:,0],points[:,1],s=8,alpha=.5,color=cp.BLUE)
        ax.set_title(title,fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-4,4)
        ax.set_ylim(-4,4)
    fig.subplots_adjust(top=.81,bottom=.12)
    cp.show(fig)


def plot_mass_checks(resolution_table,box_table):
    """고정 상자의 해상도 변화와 같은 간격의 상자 확장을 별도 표로 표시합니다."""
    fig,axes=cp.flow([4.4,6.3],height=3.5,wspace=.55)
    cp.draw_grid(axes[0],resolution_table,'R=4 고정 · 해상도만 변경',col_labels=['축 칸 수','격자 질량'],fmt='{:.6g}',kind='plain',fontsize=9)
    cp.draw_grid(axes[1],box_table,'간격 .05 고정 · 상자 확장',col_labels=['반경 R','격자 질량','MC 질량','MC SE'],fmt='{:.6g}',kind='plain',fontsize=9)
    for text in axes[1].texts:
        if text.get_position()[0] == 3:
            text.set_text(f'{box_table[int(text.get_position()[1]),3]:.2e}')
    fig.subplots_adjust(top=.78,bottom=.15,left=.03,right=.98)
    cp.show(fig)
