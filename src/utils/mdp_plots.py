"""52번 MDP·Bellman 그림. 전이·보상·가치·반복·rollout 통계는 본문에서 계산합니다."""
import numpy as np
from . import concept_plots as cp


def plot_mdp_transition(transition, rewards, expected):
    """시작 상태의 행동별 전이·보상·기댓값을 그립니다."""
    fig,axes=cp.flow([4.,4.,2.3],height=2.7,wspace=.5)
    for ax,values,title in zip(axes[:2],[transition,rewards],['P(다음 상태 | 시작, 행동)','R(시작, 행동, 다음 상태)']):
        cp.draw_grid(ax,values,title,row_labels=['a=0','a=1'],col_labels=['시작0','다리1','우회2','종료3'],fmt='{:.2f}',kind='plain',fontsize=8.5)
    cp.draw_grid(axes[2],expected[:,None],'즉시 보상 평균',row_labels=['a=0','a=1'],fmt='{:.2f}',kind='plain')
    cp.show(fig)


def plot_return_path(rewards, powers, contributions, totals):
    """시간별 보상에 할인율을 곱하고 terminal/truncation을 구분합니다."""
    fig,axes=cp.flow([2.3,2.3,2.3,3.],height=2.8,wspace=.55)
    for ax,values,title in zip(axes[:3],[rewards,powers,contributions],['보상 r','시간별 γ 거듭제곱','할인 보상']):
        cp.draw_grid(ax,values[:,None],title,fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[3],totals[:,None],'마지막 상태 처리',row_labels=['진짜 종료','수집 제한+V'],fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_bellman_backup(contributions, action_values, policy, updated):
    """다음 상태 합과 행동 평균의 서로 다른 축을 그립니다."""
    fig,axes=cp.flow([4.2,2.2,2.2,2.2],height=2.8,wspace=.5)
    cp.draw_grid(axes[0],contributions,'P × (R + γV다음)',row_labels=['a=0','a=1'],col_labels=['s0','s1','s2','종료'],fmt='{:.3f}',kind='plain',fontsize=8)
    cp.draw_grid(axes[1],action_values[:,None],'다음 상태 합 Q',fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[2],policy[:,None],'행동 확률 π',fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[3],np.array([[updated]]),'행동 평균 V새값',fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_policy_solution(transition, reward, value):
    """선형 시스템의 정책 전이·즉시 보상·해를 받은 그대로 보여 줍니다."""
    fig,axes=cp.flow([4.,2.5,2.5],height=3.2,wspace=.6)
    cp.draw_grid(axes[0],transition,'종료 뒤를 제외한 Pπ',fmt='{:.2f}',kind='plain',fontsize=8.5)
    cp.draw_grid(axes[1],reward[:,None],'rπ',fmt='{:.3f}',kind='plain')
    cp.draw_grid(axes[2],value[:,None],'(I−γPπ)⁻¹rπ',fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_policy_improvement(values, q, old_policy, new_policy):
    """평가한 가치에서 모든 행동을 비교하여 정책을 개선하는 한 단계를 그립니다."""
    fig,axes=cp.flow([2.1,3.1,2.6,2.6],height=3.1,wspace=.6)
    for ax,array,title in zip(axes,[values[:,None],q,old_policy,new_policy],['현재 Vπ','한 행동 뒤 π의 Q','기존 π','greedy 개선 π']):
        cp.draw_grid(ax,array,title,fmt='{:.3f}',kind='plain',fontsize=8.5)
    cp.show(fig)


def plot_value_iterations(history, exact, policy_history):
    """실제 value iteration과 policy iteration의 수치 갱신을 보여 줍니다."""
    fig=cp.new_figure((10.8,3.7))
    axes=fig.subplots(1,2)
    for state,color in zip(range(3),[cp.BLUE,cp.RED,cp.GOLD]):
        axes[0].plot(np.arange(len(history)),history[:,state],color=color,label=f's={state}')
        axes[0].axhline(exact[state],color=color,ls=':',lw=1)
        axes[1].plot(np.arange(len(policy_history)),policy_history[:,state],marker='o',color=color)
    for ax,title in zip(axes,['value iteration · 점선=열거한 정확해','policy iteration · 매번 정확한 정책 평가']):
        ax.set(xlabel='갱신 번호',ylabel='가치',title=title)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(False)
    axes[0].legend(frameon=False)
    fig.subplots_adjust(wspace=.3,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)


def plot_rollout_boundary(path_states, rewards, terminal, bootstrap):
    """관측 경로가 종료인지 단순 수집 제한인지 구분하는 숫자 예입니다."""
    fig,axes=cp.flow([3.1,2.5,2.5,2.5],height=2.8,wspace=.6)
    cp.draw_grid(axes[0],path_states[None,:],'수집한 상태 경로',fmt='{:.0f}',kind='plain')
    cp.draw_grid(axes[1],rewards[:,None],'관측 보상',fmt='{:.2f}',kind='plain')
    cp.draw_grid(axes[2],np.array([[terminal]]),'마지막 상태 terminal?',fmt='{:.0f}',kind='plain')
    cp.draw_grid(axes[3],np.array([[bootstrap]]),'남은 V의 할인 기여',fmt='{:.3f}',kind='plain')
    cp.show(fig)


def plot_rollout_comparison(horizons, finite_means, finite_se, finite_exact, boot_means, boot_se, infinite_exact):
    """고정 시작 상태의 유한·bootstrap 추정과 각각의 정확한 목적을 대조합니다."""
    fig=cp.new_figure((11.,3.6))
    axes=fig.subplots(1,2)
    index=np.arange(len(horizons))
    axes[0].errorbar(index,finite_means,yerr=2*np.asarray(finite_se),fmt='o',color=cp.BLUE,capsize=4,label='rollout 평균 ±2SE')
    axes[0].scatter(index,finite_exact,marker='x',color=cp.RED,label='유한 H 정확해')
    axes[1].errorbar(index,boot_means,yerr=2*np.asarray(boot_se),fmt='o',color=cp.BLUE,capsize=4,label='정확 V bootstrap 평균 ±2SE')
    axes[1].axhline(infinite_exact,color=cp.RED,ls='--',label='무한 목적 정확해')
    for ax,title in zip(axes,['수집한 보상만 더한 값','수집 제한에서 남은 가치 연결']):
        ax.set_xticks(index,[str(h) for h in horizons])
        ax.set(xlabel='수집 horizon H',ylabel='시작 상태 가치',title=title)
        ax.legend(frameon=False,fontsize=8)
        ax.grid(False)
        ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(wspace=.35,bottom=.2)
    fig.canvas.draw()
    cp.show(fig)
