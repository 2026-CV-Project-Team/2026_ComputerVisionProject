"""3D-Shapes 데이터 읽기와 고정 분할.

데이터 출처: DeepMind 3D Shapes (Burgess & Kim, 2018), 3dshapes.h5
- images: (480000, 64, 64, 3) uint8
- labels: (480000, 6) float64. 속성 순서는 FACTORS와 같음

이 파일의 원칙
- 분할은 seed 0으로 한 번만 정하고 모든 모델 seed에서 똑같이 사용한다.
  모델 seed는 초기값과 학습 순서만 바꾸므로, 모델 간 차이가 데이터 차이에서 오지 않는다.
- 학습 / 보정(calibration) / val / rank / test 묶음은 서로 겹치지 않는다.
  rank는 층 순위를 정하는 데만, test는 최종 평가에만 써서 선택 편향을 막는다.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# 환경변수 SHAPES3D_H5로 다른 위치를 지정할 수 있음. 기본은 레포의 data/3dshapes.h5
H5 = Path(os.environ.get("SHAPES3D_H5", ROOT / "data" / "3dshapes.h5"))
# 공식 파일의 SHA-256. 팀원이 같은 파일을 쓰는지 확인하는 용도
H5_SHA256 = "0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785"

FACTORS = ["floor", "wall", "object", "scale", "shape", "orientation"]
# 예측 과제: 이름 -> (labels의 열 번호, 클래스 수)
TASKS = {"shape": (4, 4), "color": (2, 10), "scale": (3, 8)}
# HDF5 chunk의 첫 축 크기. 한 장씩 읽으면 매번 chunk 전체를 풀어야 해서 매우 느리다
BLOCK = 15000
# 분할 크기. calib 2,048장은 이하형 문서 기준, 나머지는 잠정값
SPLIT_SIZES = {"train": 60000, "calib": 2048, "val": 5000, "rank": 10000, "test": 10000}


def check_file() -> None:
    """데이터 파일이 공식 파일과 같은지 SHA-256으로 확인. 약 1초 소요."""
    digest = hashlib.sha256()
    with open(H5, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != H5_SHA256:
        raise SystemExit(f"{H5} 파일이 공식 3dshapes.h5와 다름 (SHA-256 불일치)")


def load_classes() -> np.ndarray:
    """실수 라벨을 속성별 클래스 번호(0부터)로 변환. 반환 shape (480000, 6)."""
    with h5py.File(H5, "r") as f:
        labels = f["labels"][:]
    classes = np.empty(labels.shape, dtype=np.int64)
    for i in range(labels.shape[1]):
        # 속성값은 균등 간격의 실수이므로, 크기순 번호를 클래스 번호로 사용
        _, classes[:, i] = np.unique(labels[:, i], return_inverse=True)
    return classes


def load_images(indices: np.ndarray) -> np.ndarray:
    """주어진 번호의 이미지를 (N, 64, 64, 3) uint8로 반환. 입력 순서 유지."""
    order = np.argsort(indices)
    sorted_idx = indices[order]
    result = np.empty((len(indices), 64, 64, 3), dtype=np.uint8)
    with h5py.File(H5, "r") as f:
        images = f["images"]
        for start in range(0, images.shape[0], BLOCK):
            # chunk 한 덩어리를 통째로 읽고 그 안에서 필요한 장만 고른다
            lo = np.searchsorted(sorted_idx, start)
            hi = np.searchsorted(sorted_idx, start + BLOCK)
            if lo == hi:
                continue
            block = images[start : start + BLOCK]
            result[order[lo:hi]] = block[sorted_idx[lo:hi] - start]
    return result


def fixed_split(n_total: int = 480000, sizes: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """모든 모델 seed가 공유하는 분할. 전체를 한 번 섞고 앞에서부터 잘라 겹침이 없다."""
    sizes = sizes or SPLIT_SIZES
    pool = np.random.default_rng(0).permutation(n_total)
    out, at = {}, 0
    for name, size in sizes.items():
        out[name] = pool[at : at + size]
        at += size
    return out


def to_nchw(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) uint8 텐서.

    메모리를 아끼려고 uint8로 들고 있다가 배치 단위로만 실수로 바꾼다.
    contiguous()가 없으면 MPS 역전파에서 view 오류가 난다.
    """
    return torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> 실수 [-0.5, 0.5]. 입력 이미지는 양자화 대상이 아니다."""
    return images.float() / 255.0 - 0.5


def batches(n: int, size: int):
    """(시작, 끝) 번호를 size 단위로 생성."""
    for i in range(0, n, size):
        yield i, min(i + size, n)


def load_split(names: list[str] | None = None, sizes: dict[str, int] | None = None):
    """분할 이름별 (이미지 텐서, 과제별 정답 텐서, 원본 번호)를 반환."""
    classes = load_classes()
    split = fixed_split(len(classes), sizes)
    names = names or list(split)
    data = {k: to_nchw(load_images(split[k])) for k in names}
    labels = {k: {t: torch.from_numpy(classes[split[k], f]) for t, (f, _) in TASKS.items()} for k in names}
    return data, labels, {k: split[k] for k in names}, classes
