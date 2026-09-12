"""04번 본문에서 계산한 기울기·상태·궤적·초기화 통계를 그립니다."""

import matplotlib.pyplot as plt


def plot_batch_gradients(batch_sizes, empirical_std, theoretical_std):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(batch_sizes, empirical_std, "o-", label="반복 추출에서 관측한 표준편차")
    ax.plot(batch_sizes, theoretical_std, "s--", label="표본 평균의 이론 표준편차")
    ax.set(xlabel="미니배치 크기", ylabel="w 기울기의 표준편차", title="같은 파라미터에서 배치만 바꾼 실험")
    ax.legend()
    plt.show()


def plot_moments(steps, raw_mean, corrected_mean, raw_square, corrected_square):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, raw, corrected, title in zip(axes, [raw_mean, raw_square],
                                        [corrected_mean, corrected_square],
                                        ["1차 모멘트", "2차 원시 모멘트"]):
        ax.plot(steps, raw, "o-", label="0에서 시작한 이동 평균")
        ax.plot(steps, corrected, "--", label="초기값 편향 보정 후")
        ax.set(xlabel="업데이트 횟수 t", ylabel="값", title=title)
        ax.legend()
    plt.show()


def plot_optimizer_paths(xx, yy, objective, histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].contour(xx, yy, objective, levels=18, cmap="Greys", alpha=0.65)
    for name, history in histories.items():
        points = history["theta"]
        axes[0].plot(points[:, 0], points[:, 1], "-", label=name, alpha=0.85)
        axes[1].semilogy(range(len(history["loss"])), history["loss"], label=name)
    axes[0].scatter([0], [0], marker="*", s=100, color="black")
    axes[0].set(xlabel="첫 파라미터", ylabel="둘째 파라미터", title="같은 이차 목적에서의 경로")
    axes[1].set(xlabel="업데이트 횟수", ylabel="목적값", title="선택한 학습률에서의 손실 변화")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_regression_runs(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, history in histories.items():
        axes[0].plot(history["seen"], history["loss"], label=name)
        axes[1].plot(history["seen"], history["gradient_norm"], label=name)
    axes[0].set(xlabel="처리한 학습 표본 수", ylabel="전체 학습셋 손실", title="같은 데이터 순서로 학습")
    axes[1].set(xlabel="처리한 학습 표본 수", ylabel="전체 손실의 기울기 norm", title="같은 시점의 전체 기울기")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_initialization(depths, second_moments, gradient_rms):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name in second_moments:
        axes[0].semilogy(depths, second_moments[name], "o-", label=name)
        axes[1].semilogy(depths, gradient_rms[name], "o-", label=name)
    axes[0].set(xlabel="층 (0은 입력)", ylabel="활성값 제곱의 평균", title="forward 값의 크기")
    axes[1].set(xlabel="층 (0은 입력)", ylabel="전달된 기울기의 RMS", title="backward 값의 크기")
    for ax in axes:
        ax.legend()
    plt.show()


def plot_schedule(steps, learning_rates, norms_before, norms_after):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(steps, learning_rates)
    axes[0].set(xlabel="업데이트 횟수", ylabel="학습률", title="미리 정한 학습률 스케줄")
    axes[1].bar(["원래 기울기", "clipping 후"], [norms_before, norms_after])
    axes[1].set(ylabel="전체 gradient norm", title="크기를 제한하는 별도 연산")
    plt.show()
