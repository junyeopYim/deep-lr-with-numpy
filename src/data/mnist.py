"""MNIST 손글씨 숫자 로더.

데이터는 외부 미러에서 내려받는다(기본값: PyTorch가 쓰는 S3 미러).
자동으로 받지 않고 download=True를 명시할 때만 네트워크를 쓴다.
받은 파일은 IDX 매직 넘버와 개수를 검사해 온전한지 확인한다.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", 2051, 60000),
    "train_labels": ("train-labels-idx1-ubyte.gz", 2049, 60000),
    "test_images": ("t10k-images-idx3-ubyte.gz", 2051, 10000),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", 2049, 10000),
}

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "mnist")


def _download(filename: str, dest: str) -> None:
    last_err: Exception | None = None
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"내려받는 중: {url}")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:  # 미러가 죽어 있으면 다음 것을 시도
            last_err = e
            print(f"  실패: {e}")
    raise RuntimeError(f"{filename} 을(를) 모든 미러에서 받지 못했습니다") from last_err


def _read_idx(path: str, expected_magic: int, expected_count: int) -> np.ndarray:
    """IDX 형식을 읽는다. 헤더는 빅엔디언 32비트 정수들이다."""
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != expected_magic:
            raise ValueError(f"{path}: 매직 넘버가 {magic} 입니다 (기대값 {expected_magic}). "
                             "파일이 손상되었을 수 있습니다 — 지우고 다시 받으세요.")
        if count != expected_count:
            raise ValueError(f"{path}: 항목 수가 {count} 입니다 (기대값 {expected_count})")
        if expected_magic == 2051:  # 이미지: 행/열 크기가 더 붙는다
            rows, cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(count, rows, cols)
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(root: str = DEFAULT_ROOT, *, download: bool = False, flatten: bool = True,
               normalize: bool = True):
    """MNIST를 읽어 (X_train, y_train, X_test, y_test)를 반환한다.

    Args:
        root: 캐시 디렉터리.
        download: 파일이 없을 때 내려받을지. False면 없다는 안내와 함께 실패한다.
        flatten: True면 (N, 784), False면 (N, 1, 28, 28) — 합성곱 단계에서 쓴다.
        normalize: True면 [0, 1] 실수로 변환.
    """
    os.makedirs(root, exist_ok=True)
    arrays = {}
    for key, (filename, magic, count) in FILES.items():
        path = os.path.join(root, filename)
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(
                    f"{path} 가 없습니다. load_mnist(download=True) 로 내려받으세요.")
            _download(filename, path)
        arrays[key] = _read_idx(path, magic, count)

    X_train, X_test = arrays["train_images"], arrays["test_images"]
    if normalize:
        X_train = X_train.astype(np.float64) / 255.0
        X_test = X_test.astype(np.float64) / 255.0
    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    else:
        X_train = X_train.reshape(len(X_train), 1, 28, 28)
        X_test = X_test.reshape(len(X_test), 1, 28, 28)

    return X_train, arrays["train_labels"].astype(np.int64), X_test, arrays["test_labels"].astype(np.int64)
