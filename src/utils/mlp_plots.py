"""03번 본문에서 계산한 활성화·은닉 표현·예측·손실을 그립니다."""

import matplotlib.pyplot as plt


def plot_activations(x, values, derivatives):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, y in values.items():
        axes[0].plot(x, y, label=name)
    for name, y in derivatives.items():
        axes[1].plot(x, y, label=name)
    axes[0].set(xlabel="입력", ylabel="출력", title="비선형 활성화")
    axes[1].set(xlabel="입력", ylabel="도함수", title="역전파에서 곱하는 값")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_representation(inputs, hidden, labels):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, points, title in zip(axes, [inputs, hidden], ["입력 공간의 XOR", "두 은닉값으로 바꾼 XOR"]):
        ax.scatter(points[:, 0], points[:, 1], c=labels.ravel(), cmap="coolwarm",
                   vmin=0, vmax=1, s=100, edgecolors="black")
        for i, point in enumerate(points):
            ax.annotate(str(i), point, xytext=(6, 6), textcoords="offset points")
        ax.set(title=title, xlabel="첫 좌표", ylabel="둘째 좌표")
        ax.margins(0.2)
    plt.show()


def plot_regions(xx, yy, probabilities, inputs, labels, title):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    region = ax.contourf(xx, yy, probabilities, levels=[0, .1, .25, .5, .75, .9, 1],
                        cmap="RdBu_r", vmin=0, vmax=1, alpha=0.75)
    ax.contour(xx, yy, probabilities, levels=[0.5], colors="black", linewidths=1)
    ax.scatter(inputs[:, 0], inputs[:, 1], c=labels.ravel(), cmap="RdBu_r",
               vmin=0, vmax=1, edgecolors="black", s=34)
    fig.colorbar(region, ax=ax, label="P(Y=1 | x)")
    ax.set(xlabel="첫 번째 입력", ylabel="두 번째 입력", title=title)
    plt.show()


def plot_training(histories, ylabel="평균 BCE", title="같은 목적, 다른 계산 구조"):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for name, history in histories.items():
        ax.plot(range(len(history)), history, label=name)
    ax.set(xlabel="갱신 횟수", ylabel=ylabel, title=title)
    ax.legend()
    plt.show()


def plot_hidden_maps(xx, yy, activations):
    count = activations.shape[-1]
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.1), squeeze=False)
    for k, ax in enumerate(axes[0]):
        ax.contourf(xx, yy, activations[:, :, k], levels=12, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set(xlabel="첫 입력", ylabel="둘째 입력", title=f"은닉 유닛 {k}")
    plt.show()
