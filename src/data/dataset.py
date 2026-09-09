"""미니배치 공급기.

셔플과 배치 분할만 한다. 전처리는 데이터셋을 만들 때 미리 끝내 두는 쪽이
디버깅하기 편하다.
"""

from __future__ import annotations

import numpy as np


class DataLoader:
    """(X, y)를 미니배치로 잘라 내놓는다.

    Args:
        X: (N, ...) 입력.
        y: (N, ...) 타깃.
        batch_size: 배치 크기.
        shuffle: 에폭마다 순서를 섞을지. 학습이면 True, 평가면 False.
        drop_last: 마지막 자투리 배치를 버릴지. 배치 크기가 일정해야 하는
            실험(예: BatchNorm 통계 비교)에서 켠다.
        seed: 셔플 재현용.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        if len(X) != len(y):
            raise ValueError(f"X({len(X)})와 y({len(y)})의 길이가 다릅니다")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.X)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.X)
        idx = self.rng.permutation(n) if self.shuffle else np.arange(n)
        limit = n - (n % self.batch_size) if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            batch = idx[start:start + self.batch_size]
            yield self.X[batch], self.y[batch]


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0):
    """데이터를 학습/검증으로 나눈다."""
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_ratio)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]
