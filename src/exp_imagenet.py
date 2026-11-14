"""ImageNet 사전학습 모델(ResNet-18, DeiT-Tiny)의 기준 정확도와 조건별 정확도 측정.

현재 단계: FP32 기준 정확도 + 전 단위 동일 비트 조건(W8A8 등).
단위별 복원·교체, Mixed-Precision 정책 비교는 팀 확인 후 이 파일에 이어서 추가한다.

실행 예
    python -m src.exp_imagenet --models resnet18 deit_tiny --conditions W8A8 W6A6 W5A5 W4A4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .imagenet_data import load_split, to_input
from .imagenet_models import MODELS, load_model
from .quant.units import UnitQuantizer

ROOT = Path(__file__).resolve().parent.parent


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_condition(text: str) -> tuple[int | None, int | None]:
    """'W4A8' -> (4, 8). 32는 FP32(None)."""
    w, a = text.upper().lstrip("W").split("A")
    return (None if w == "32" else int(w)), (None if a == "32" else int(a))


@torch.no_grad()
def predict(model, x: torch.Tensor, device, batch_size: int = 250) -> torch.Tensor:
    """top-1 예측 번호. x는 uint8 텐서(파일 매핑)라 배치 단위로만 메모리에 올린다."""
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model(to_input(x[i : i + batch_size], device)).argmax(1).cpu())
    return torch.cat(out)


def run_model(name: str, conditions: list[str], out_dir: Path, device) -> list[dict]:
    model, units = load_model(name)
    model.to(device)
    q = UnitQuantizer(model, units)  # 모델을 device로 옮긴 뒤 만들어야 가중치 표가 같은 device에 생긴다
    n = len(units)
    calib_x, _ = load_split(name, "calib")
    # 보정: train 2,048장의 FP32 순전파에서 단위별 출력 [min, max] 수집
    ranges = q.calibrate(to_input(calib_x[i : i + 256], device) for i in range(0, len(calib_x), 256))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranges.json").write_text(
        json.dumps({u.name: [float(r[0]), float(r[1])] for u, r in zip(units, ranges)}, indent=1)
    )
    rows = []
    for split in ("rank", "test"):
        x, y = load_split(name, split)
        q.apply([None] * n, [None] * n)
        t0 = time.time()
        base = predict(model, x, device)
        rows.append({"model": name, "set": split, "condition": "FP32", "acc": float((base == y).float().mean() * 100),
                     "c2w": 0, "w2c": 0})
        print(name, split, "FP32", round(rows[-1]["acc"], 2), f"{time.time() - t0:.0f}s", flush=True)
        for cond in conditions:
            w, a = parse_condition(cond)
            q.apply([w] * n, [a] * n)
            pred = predict(model, x, device)
            ok, base_ok = pred == y, base == y
            rows.append({"model": name, "set": split, "condition": cond, "acc": float(ok.float().mean() * 100),
                         "c2w": int((base_ok & ~ok).sum()),  # FP32 정답 -> 양자화 후 오답
                         "w2c": int((~base_ok & ok).sum())})  # FP32 오답 -> 양자화 후 정답
            print(name, split, cond, round(rows[-1]["acc"], 2), flush=True)
    q.remove()
    return rows


def main() -> None:
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--conditions", nargs="*", default=["W8A8", "W6A6", "W5A5", "W4A4", "W4A8", "W32A4", "W4A32"])
    p.add_argument("--out", default="results/imagenet")
    args = p.parse_args()
    device = pick_device()
    rows = []
    for name in args.models:
        rows += run_model(name, args.conditions, ROOT / args.out / name, device)
    summary = ROOT / args.out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary / "conditions.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["model", "condition"], columns="set", values="acc").round(2))


if __name__ == "__main__":
    main()
