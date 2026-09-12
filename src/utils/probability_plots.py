"""02번에서 이미 계산한 확률·밀도·손실·예측 결과를 그립니다."""

import matplotlib.pyplot as plt


def plot_discrete(values, probabilities, mean):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(values, probabilities, width=0.5)
    ax.axvline(mean, color="tab:red", ls="--", label=f"기댓값 {mean:.2f}")
    ax.set(xticks=values, xlabel="확률변수의 값", ylabel="확률", title="확률은 각 값에 배정한 질량입니다")
    ax.legend()
    plt.show()


def plot_density(x, density, interval, interval_probability):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, density, label="확률밀도")
    ax.fill_between(x, 0, density, where=interval, alpha=0.3,
                    label=f"색칠한 구간의 확률 {interval_probability:.2f}")
    ax.set(xlabel="값 x", ylabel="밀도 p(x)", title="높이가 4여도 전체 넓이는 1입니다")
    ax.legend()
    plt.show()


def plot_estimates(counts, estimates, exact):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(counts, estimates, alpha=0.8, label="같은 표본열의 누적 평균")
    ax.axhline(exact, color="tab:red", ls="--", label="확률 가중합으로 계산한 기댓값")
    ax.set(xscale="log", xlabel="사용한 표본 수", ylabel="추정값", title="Monte Carlo: 표본 평균으로 기댓값 추정하기")
    ax.legend()
    plt.show()


def plot_likelihood(theta, likelihood, nll, estimate):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, y, title in zip(axes, [likelihood, nll], ["우도: 크게", "평균 음의 로그우도: 작게"]):
        ax.plot(theta, y)
        ax.axvline(estimate, color="tab:red", ls="--", label=f"표본 평균 {estimate:.2f}")
        ax.set(xlabel="성공확률 후보 θ", title=title)
        ax.legend()
    plt.show()


def plot_loss_curves(logits, losses, gradients):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, curve in losses.items():
        axes[0].plot(logits, curve, label=name)
    for name, curve in gradients.items():
        axes[1].plot(logits, curve, label=name)
    axes[0].set(xlabel="logit z", ylabel="한 표본의 BCE", title="틀린 확신에는 큰 손실")
    axes[1].set(xlabel="logit z", ylabel="z에 대한 미분", title="기울기가 가리키는 갱신 방향")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend()
    plt.show()


def plot_classifier(x, y, grid, probabilities, losses):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].scatter(x, y, label="관측한 라벨", s=22)
    axes[0].plot(grid, probabilities, color="tab:green", label="학습한 P(Y=1 | x)")
    axes[0].set(xlabel="입력 x", ylabel="확률 / 라벨", ylim=(-0.08, 1.08), title="logit → 확률 → 예측")
    axes[1].plot(range(len(losses)), losses)
    axes[1].set(xlabel="갱신 횟수", ylabel="평균 BCE", title="같은 관측을 더 잘 설명하도록 학습")
    axes[0].legend()
    plt.show()
