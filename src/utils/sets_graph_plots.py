"""21 집합·그래프 attention의 그림 배치. 수학·데이터·모델 계산은 노트북에서 수행합니다."""

import numpy as np

from . import concept_plots as cp


def plot_permutation(original, reordered, transformed, totals):
    """행 순서·함께 움직인 행 출력·변하지 않은 합을 보여 줍니다."""
    fig, axes = cp.flow([2.5, 2.5, 2.6, 2.4], height=3.3, wspace=.45)
    cp.draw_grid(axes[0], original, "과일 세 개 · X", col_labels=["붉기", "무게"], kind="plain")
    cp.draw_grid(axes[1], reordered, "순서만 바꾼 PX", col_labels=["붉기", "무게"], kind="plain")
    cp.draw_grid(axes[2], transformed, "공유 함수 출력도 이동", col_labels=["원래", "순열 뒤"], kind="plain")
    cp.draw_grid(axes[3], totals, "합은 그대로", row_labels=["원래", "순열 뒤"], kind="plain")
    fig.subplots_adjust(top=.8, bottom=.13)
    cp.show(fig)


def plot_pool_cardinality(features, duplicated, sums, means):
    """꾸러미 전체를 복제했을 때 합과 평균의 차이를 표시합니다."""
    fig, axes = cp.flow([2.4, 2.4, 2.7, 2.7], height=3.6, wspace=.5)
    for ax,values,title in zip(axes[:2],[features,duplicated],["공유 φ의 출력", "같은 원소들을 두 번"]):
        cp.draw_grid(ax, values, title, fmt="{:.2f}", kind="plain", fontsize=9)
    for ax,values,title in zip(axes[2:],[sums,means],["sum: 두 배", "mean: 그대로"]):
        cp.draw_grid(ax, values, title, row_labels=["원래", "전체 복제"], fmt="{:.2f}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.81, bottom=.12)
    cp.show(fig)


def plot_neighbor_mask(features, adjacency, allowed, weights):
    """실제 노드·padding과 adjacency·허용 이웃·정규화 비중을 보여 줍니다."""
    fig, axes = cp.flow([2.2, 2.7, 2.7, 2.9], height=3.7, wspace=.65)
    labels=["노드0", "노드1", "고립2", "padding"]
    cp.draw_grid(axes[0], features, "노드의 값", row_labels=labels, kind="plain")
    for ax,values,title in zip(axes[1:],[adjacency,allowed,weights],["A: i가 j를 읽음", "valid + self-loop", "이웃 softmax α"]):
        cp.draw_grid(ax, values, title, row_labels=labels, col_labels=["0", "1", "2", "pad"], fmt="{:.2g}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.78, bottom=.13, left=.07, right=.98)
    cp.show(fig)


def plot_gat_aggregation(features, projected, raw, weights, aggregated):
    """공유 투영·additive 점수·이웃 비중·가중합 결과를 나란히 표시합니다."""
    fig, axes = cp.flow([2.0, 2.0, 2.6, 2.6, 2.0], height=3.4, wspace=.4)
    for ax,values,title in zip(axes,[features,projected,raw,weights,aggregated],
                               ["X: 과일의 특징", "Z = XW", "LeakyReLU 전 점수", "이웃 비중 α", "tanh(αZ)"]):
        cp.draw_grid(ax, values, title, fmt="{:.2f}", kind="plain", fontsize=9)
    fig.subplots_adjust(top=.8, bottom=.12)
    cp.show(fig)


def plot_task_plan(set_target, node_targets, counts, budget):
    """두 학습 과제의 정답 단위와 미리 정한 분할·예산을 구분합니다."""
    fig, axes = cp.flow([2.7, 2.7, 2.8, 2.8], height=3.5, wspace=.6)
    cp.draw_grid(axes[0], set_target, "Deep Sets: 꾸러미당 1개", fmt="{:.3f}", kind="plain")
    cp.draw_grid(axes[1], node_targets, "GAT: 실제 노드마다 1개", fmt="{:.3f}", kind="plain")
    cp.draw_grid(axes[2], np.asarray(counts)[:,None], "각 과제의 독립 분할", row_labels=["train", "validation", "test"], kind="plain")
    cp.draw_grid(axes[3], np.asarray(budget)[:,None], "각 모델의 고정 예산", row_labels=["갱신 수", "배치 수", "누적 표본"], kind="plain")
    fig.subplots_adjust(top=.8, bottom=.12, left=.04, right=.98)
    cp.show(fig)


def plot_learning_curves(steps, set_mse, graph_mse):
    """서로 다른 과제의 validation MSE를 별도 패널에 그립니다."""
    fig=cp.new_figure((10.4,3.7))
    axes=fig.subplots(1,2)
    for ax,values,title in zip(axes,[set_mse,graph_mse],["Deep Sets · 꾸러미 평균 MSE", "GAT · graph별 평균 MSE"]):
        ax.plot(steps,values,color=cp.BLUE)
        ax.set(xlabel="갱신 수",ylabel="validation MSE",title=title)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_heldout_predictions(set_target, set_prediction, node_target, node_prediction, reference):
    """미리 고정한 test 전체의 정답과 예측을 보여 줍니다. 기준선 좌표도 본문에서 받습니다."""
    fig=cp.new_figure((10.4,4.0))
    axes=fig.subplots(1,2)
    for ax,target,prediction,title in zip(axes,[set_target,node_target],[set_prediction,node_prediction],
                                         ["꾸러미 test", "실제 노드 test"]):
        ax.scatter(target,prediction,s=13,alpha=.55,color=cp.BLUE)
        ax.plot(reference,reference,color=cp.MUTED,linestyle="--",linewidth=1)
        ax.set(xlabel="정답",ylabel="예측",title=title)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    cp.show(fig)


def plot_size_degree_metrics(set_values, graph_values, set_names, graph_names):
    """크기·이웃 수와 실제 MSE를 별도 과제의 표로 표시합니다."""
    fig,axes=cp.flow([4.6,6.1],height=3.8,wspace=.6)
    cp.draw_grid(axes[0],set_values,"Deep Sets · 같은 고정 모델",row_labels=set_names,
                 col_labels=["평균 원소 수","MSE"],fmt="{:.4g}",kind="plain",fontsize=9)
    cp.draw_grid(axes[1],graph_values,"GAT · 같은 고정 모델",row_labels=graph_names,
                 col_labels=["평균 노드 수","평균 이웃 수","MSE"],fmt="{:.4g}",kind="plain",fontsize=9)
    for ax in axes:
        ax.set_aspect("auto")
    fig.subplots_adjust(top=.8,bottom=.13,left=.1,right=.98)
    cp.show(fig)
