"""01번 노트북에서 계산한 손실·도함수·학습 경로를 그립니다."""

import numpy as np
import matplotlib.pyplot as plt

from .plotting import COLOR_ACCENT, COLOR_NEG, COLOR_POS


def plot_derivative(grid, values, tangent, point, point_value, secant_points, secant_values):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, values, label="함수 f(w)")
    ax.plot(grid, tangent, "--", label="선택한 점의 접선")
    ax.plot(secant_points, secant_values, "o-", color=COLOR_NEG, label="두 점 사이 변화율")
    ax.scatter([point], [point_value], color=COLOR_ACCENT, s=60, zorder=5)
    ax.set(xlabel="w", ylabel="함수값", title="미분: 작은 입력 변화에 출력이 얼마나 변하는가")
    ax.legend()
    plt.show()


def plot_difference_errors(epsilons, forward_errors, central_errors):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(epsilons, np.maximum(forward_errors, 1e-17), "o-", label="전진차분")
    ax.loglog(epsilons, np.maximum(central_errors, 1e-17), "o-", label="중심차분")
    ax.set(xlabel="차분 폭 ε", ylabel="도함수와의 절대 차이", title="ε를 줄이는 효과와 부동소수점 오차")
    ax.legend()
    plt.show()


def plot_loss_surface(ww, bb, losses, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    contours = ax.contour(ww, bb, losses, levels=16, linewidths=0.9, cmap="Blues")
    ax.clabel(contours, inline=True, fontsize=8)
    ax.plot(path[:, 0], path[:, 1], "o-", ms=3, color=COLOR_NEG, label="경사하강 경로")
    ax.scatter(*path[0], marker="s", s=65, color=COLOR_ACCENT, label="시작")
    ax.scatter(*path[-1], marker="*", s=130, color=COLOR_POS, label="마지막")
    ax.set(xlabel="가중치 w", ylabel="편향 b", title="손실이라는 지형 위에서 파라미터를 움직입니다")
    ax.legend()
    plt.show()


def plot_fit_snapshots(x, y, line_x, predictions, steps):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, pred, step in zip(axes.ravel(), predictions, steps):
        ax.scatter(x.ravel(), y.ravel(), s=16, alpha=0.7, color=COLOR_ACCENT, label="학습 데이터")
        ax.plot(line_x.ravel(), pred.ravel(), color=COLOR_POS, label="현재 예측")
        ax.set_title(f"{step}회 갱신 뒤")
        ax.set(xlabel="입력 x", ylabel="출력 y")
    axes.ravel()[0].legend()
    plt.show()


def plot_learning_curves(histories):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, losses in histories.items():
        ax.semilogy(np.arange(len(losses)), np.maximum(losses, 1e-17), label=label)
    ax.set(xlabel="파라미터 갱신 횟수", ylabel="평균 손실", title="같은 기울기도 학습률에 따라 다른 경로를 만듭니다")
    ax.legend()
    plt.show()
