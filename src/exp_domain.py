"""6-Conv 도메인 실험 1차: 학습해둔 모델 그대로, 평가 입력만 바꿔서 양자화 영향을 다시 잰다.

조건 (이하형 문서 11-10 2차)
- orig:       원본 (본 실험과 같은 입력. 본 실험 기록과 일치하는지 확인하는 용도도 겸함)
- bg:         물체의 Shape·Color·Scale·Orientation은 그대로 두고 벽·바닥 색만 다른 이미지로 교체
- noise_0.01, noise_0.03: Gaussian Noise. σ는 이미지 값을 0~1로 놓았을 때 기준
- blur_0.5, blur_1.0:     Gaussian Blur. 5x5 커널, σ는 픽셀 단위

읽을 때 주의
- 지금 모델은 무작위 분할로 학습해서 벽·바닥의 모든 색을 이미 봤다. 그래서 bg는 "처음 보는
  분포"가 아니라 "같은 물체, 다른 배경"에서 결과가 달라지는지 보는 실험이다. 학습에서 색 조합을
  빼는 진짜 분포 변화 실험은 재학습이 필요한 2차로 따로 한다.
- 보정 범위는 원본 이미지로 구한 것을 그대로 쓴다(재보정 없음). 배포 뒤 입력이 바뀌는 상황을
  가정한 것이고, 잠정 설정이다.
- C→W, W→C는 같은 입력의 FP32 예측이 기준이다. 입력 변화 자체로 틀린 것은 FP32 정답률에 나타난다.

구현상 결정 (잠정)
- bg: 평가 이미지마다 벽·바닥 색을 둘 다 원래와 다른 값으로 무작위 선택(seed 0).
      학습·보정에 쓰인 이미지가 뽑히면 다시 뽑는다.
- noise: 더한 뒤 0~1 밖으로 나간 값은 잘라낸다. 두 σ는 같은 난수(seed 0)에 배율만 다르다.
- blur: 가장자리는 reflect padding (torchvision gaussian_blur 기본 동작).

출력
- <exp>/domain_index.npz:        bg에서 바꿔 넣은 이미지의 원본 번호
- <exp>/seed{n}/domain_metrics.csv: metrics.csv 형식 + domain 열
- <exp>/seed{n}/domain_nmse.csv:    nmse.csv 형식 + domain 열 (W4A4만)
상태는 FP32, 조건별 전 층 양자화, 그리고 W4A4의 restore_i / only_i.

사용 예:
    python -m src.exp_domain --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import gaussian_blur

from .data import FACTORS, TASKS, check_file, fixed_split, load_classes, load_images, to_nchw
from .exp_6conv import condition_states, evaluate, layer_nmse, load_seed
from .models import N_LAYERS
from .train import pick_device

# 속성별 값의 개수. 3D-Shapes는 이 순서의 모든 조합을 사전식으로 나열한 데이터다
FACTOR_SIZES = (10, 10, 10, 8, 4, 15)
# 전 층 양자화만 보는 조건. W4A4는 층별(restore/only) 상태까지 본다
ALL_ONLY = {"W8A8": (8, 8), "W6A6": (6, 6), "W5A5": (5, 5), "W4A8": (4, 8), "W32A4": (None, 4), "W4A32": (4, None)}
NOISE_SIGMAS = (0.01, 0.03)
BLUR_SIGMAS = (0.5, 1.0)
BLUR_KERNEL = 5
DOMAINS = ["orig", "bg"] + [f"noise_{s}" for s in NOISE_SIGMAS] + [f"blur_{s}" for s in BLUR_SIGMAS]


def factor_index(factors: np.ndarray) -> np.ndarray:
    """속성 클래스 번호 (N, 6) -> 데이터셋의 이미지 번호."""
    return np.ravel_multi_index(factors.T, FACTOR_SIZES)


def swap_background(factors: np.ndarray, forbidden: np.ndarray, seed: int = 0) -> np.ndarray:
    """벽·바닥 색만 다른 이미지의 번호. forbidden(학습·보정 이미지)에 속하면 다시 뽑는다."""
    rng = np.random.default_rng(seed)
    floor, wall = FACTORS.index("floor"), FACTORS.index("wall")
    new = factors.copy()
    todo = np.arange(len(factors))
    while len(todo):
        for col in (floor, wall):
            # 1~9를 더하고 10으로 나눈 나머지 -> 원래 값과 반드시 다른 값
            new[todo, col] = (factors[todo, col] + rng.integers(1, FACTOR_SIZES[col], len(todo))) % FACTOR_SIZES[col]
        todo = todo[np.isin(factor_index(new[todo]), forbidden)]
    return factor_index(new)


def make_domain(name: str, x01: torch.Tensor, bg01: torch.Tensor) -> torch.Tensor:
    """0~1 실수 이미지 (N, 3, 64, 64)를 도메인에 맞게 바꾼 뒤 모델 입력 범위 [-0.5, 0.5]로 반환."""
    kind, _, value = name.partition("_")
    if kind == "orig":
        out = x01
    elif kind == "bg":
        out = bg01
    elif kind == "noise":
        noise = torch.randn(x01.shape, generator=torch.Generator().manual_seed(0))
        out = (x01 + float(value) * noise).clamp_(0.0, 1.0)
    elif kind == "blur":
        out = torch.cat([gaussian_blur(x01[i : i + 2000], [BLUR_KERNEL] * 2, [float(value)] * 2)
                         for i in range(0, len(x01), 2000)])
    else:
        raise ValueError(name)
    return out - 0.5


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-nmse", type=int, default=5000)
    args = parser.parse_args()

    check_file()
    device = pick_device()
    classes = load_classes()
    split = fixed_split(len(classes))
    eval_idx = np.concatenate([split["rank"], split["test"]])
    n_rank = len(split["rank"])
    at = np.arange(len(eval_idx))
    masks = {"rank": at < n_rank, "test": at >= n_rank}
    y_eval = {t: torch.from_numpy(classes[eval_idx, f]) for t, (f, _) in TASKS.items()}

    bg_idx = swap_background(classes[eval_idx], np.concatenate([split["train"], split["calib"]]))
    # 바꾼 이미지는 벽·바닥만 다르고 나머지 네 속성은 같아야 한다(번호 계산 검증)
    keep = [FACTORS.index(f) for f in ("object", "scale", "shape", "orientation")]
    assert np.array_equal(classes[bg_idx][:, keep], classes[eval_idx][:, keep])
    assert (classes[bg_idx][:, :2] != classes[eval_idx][:, :2]).all()
    np.savez_compressed(args.exp / "domain_index.npz", eval_index=eval_idx, bg_index=bg_idx)

    x01 = to_nchw(load_images(eval_idx)).float() / 255.0
    bg01 = to_nchw(load_images(bg_idx)).float() / 255.0

    states = [("FP32", "all", [None] * N_LAYERS, [None] * N_LAYERS)]
    states += [condition_states(c, w, a)[0] for c, (w, a) in ALL_ONLY.items()]
    states += condition_states("W4A4", 4, 4)
    nets = {seed: load_seed(args.exp / f"seed{seed}", device) for seed in args.seeds}
    metric_rows = {seed: [] for seed in args.seeds}
    nmse_rows = {seed: [] for seed in args.seeds}

    for domain in DOMAINS:
        x = make_domain(domain, x01, bg01)  # 도메인마다 한 번 만들어 모든 seed가 같은 입력을 본다
        for seed, (net, table) in nets.items():
            rows, _ = evaluate(net, table, states, x, y_eval, masks, seed, device, prep=identity)
            metric_rows[seed] += [{"domain": domain, **r} for r in rows]
            for cond, state, w_bits, a_bits in states:
                if cond != "W4A4" or state.startswith("restore"):
                    continue
                values = layer_nmse(net, table, w_bits, a_bits, x[: args.n_nmse], device, prep=identity)
                nmse_rows[seed] += [{"domain": domain, "seed": seed, "condition": cond, "state": state,
                                     "layer": k, "nmse": v} for k, v in enumerate(values, start=1)]
            done = pd.DataFrame(rows)
            view = done[(done.state == "all") & (done.set == "test")].groupby("condition").acc.mean()
            print(f"{domain} seed {seed}: " + ", ".join(f"{c} {v:.2f}" for c, v in view.items()), flush=True)

    for seed in args.seeds:
        seed_dir = args.exp / f"seed{seed}"
        frame = pd.DataFrame(metric_rows[seed])
        # 재현 확인: orig 도메인의 W4A4 전 층 양자화가 본 실험 기록과 같아야 한다
        saved = pd.read_csv(seed_dir / "metrics.csv")
        saved = saved[(saved.condition == "W4A4") & (saved.state == "all")].set_index(["set", "task"]).acc
        now = frame[(frame.domain == "orig") & (frame.condition == "W4A4") & (frame.state == "all")]
        for row in now.itertuples():
            assert abs(row.acc - saved[(row.set, row.task)]) < 1e-6, (seed, row)
        frame.to_csv(seed_dir / "domain_metrics.csv", index=False)
        pd.DataFrame(nmse_rows[seed]).to_csv(seed_dir / "domain_nmse.csv", index=False)
    print("orig 도메인이 본 실험 기록과 일치")


if __name__ == "__main__":
    main()
