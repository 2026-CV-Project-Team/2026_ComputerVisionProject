"""ImageNet 모델의 최종 출력(logit, 1000 클래스) 양자화 오차 측정.

6-Conv의 exp_output_error와 같은 지표를 ResNet-18, DeiT-Tiny에 적용한다.
- out_nmse = ‖z − ẑ‖² / ‖z‖²,  out_kl = KL(softmax(z) ‖ softmax(ẑ))  (z = 같은 입력의 FP32 logit)
- 분류기(fc/head)의 logit은 양자화하지 않으므로(imagenet_models 잠정 규칙) 최종 출력 오차는
  분류기 가중치 양자화 + 앞 단위들에서 넘어온 오차다.

두 가지를 잰다
1. 조건별(all 상태): FP32, W8A8 ... W4A32 전체에 대해 rank·test 묶음의 최종 출력 오차
2. 단위별: exp_imagenet_units를 돌린 조건(results/imagenet/{모델}/{조건}/restore.csv가 있는 것)에 대해
   all, restore_i, only_i 상태의 최종 출력 오차 (rank 5,000장, 순위 산출용이라 test는 쓰지 않음)

확인: restore_i, only_i, all 상태의 top-1 예측이 단위별 실험의 per_image.npz와 같은지 본다.
보정은 같은 2,048장으로 다시 하므로 범위가 같아야 하고, 다르면 결과를 믿을 수 없다.

출력
- results/imagenet/{모델}/output_conditions.csv: model, set, condition, acc, out_nmse, out_kl
- results/imagenet/{모델}/{조건}/output_error.csv: 단위별 out_nmse_*, out_kl_*, out_gain
- 집계는 `python -m src.output_summary`

실행 예
    python -m src.exp_imagenet_output --models resnet18 deit_tiny
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .exp_imagenet import ROOT, parse_condition, pick_device
from .imagenet_data import load_split, to_input
from .imagenet_models import MODELS, load_model
from .quant.units import UnitQuantizer

CONDITIONS = ["W8A8", "W6A6", "W5A5", "W4A4", "W4A8", "W32A4", "W4A32"]


@torch.no_grad()
def logits_all(model, x: torch.Tensor, device, batch_size: int = 250) -> torch.Tensor:
    """(N, 1000) logit. 5,000장 기준 20MB라 CPU에 모아 둔다."""
    return torch.cat([model(to_input(x[i : i + batch_size], device)).float().cpu() for i in range(0, len(x), batch_size)])


def output_error(ref: torch.Tensor, out: torch.Tensor) -> tuple[float, float]:
    """이미지별로 계산한 뒤 평균한 (out_nmse, out_kl)."""
    nmse = (ref - out).pow(2).sum(1) / ref.pow(2).sum(1).clamp(min=1e-12)
    log_p, log_q = F.log_softmax(ref, 1), F.log_softmax(out, 1)
    kl = (log_p.exp() * (log_p - log_q)).sum(1)
    return float(nmse.mean()), float(kl.mean())


def run(name: str, out_root, device) -> None:
    model, units = load_model(name)
    model.to(device)
    q = UnitQuantizer(model, units)
    n = len(units)
    off = [None] * n
    calib_x, _ = load_split(name, "calib")
    q.calibrate(to_input(calib_x[i : i + 256], device) for i in range(0, len(calib_x), 256))

    # 1. 조건별 (all 상태)
    rows = []
    for split in ("rank", "test"):
        x, y = load_split(name, split)
        q.apply(off, off)
        ref = logits_all(model, x, device)
        rows.append({"model": name, "set": split, "condition": "FP32",
                     "acc": float((ref.argmax(1) == y).float().mean() * 100), "out_nmse": 0.0, "out_kl": 0.0})
        for cond in CONDITIONS:
            wb, ab = parse_condition(cond)
            q.apply([wb] * n, [ab] * n)
            out = logits_all(model, x, device)
            e_nmse, e_kl = output_error(ref, out)
            rows.append({"model": name, "set": split, "condition": cond,
                         "acc": float((out.argmax(1) == y).float().mean() * 100), "out_nmse": e_nmse, "out_kl": e_kl})
            print(name, split, cond, {k: round(v, 4) for k, v in rows[-1].items() if k in ("acc", "out_nmse", "out_kl")},
                  flush=True)
    pd.DataFrame(rows).to_csv(out_root / name / "output_conditions.csv", index=False)

    # 2. 단위별 (rank)
    x, _ = load_split(name, "rank")
    q.apply(off, off)
    ref = logits_all(model, x, device)
    for cond_dir in sorted((out_root / name).glob("W*A*")):
        if not (cond_dir / "restore.csv").exists():
            continue
        cond = cond_dir.name
        wb, ab = parse_condition(cond)
        saved = np.load(cond_dir / "per_image.npz")
        t0 = time.time()

        def measure(w, a, key):
            q.apply(w, a)
            out = logits_all(model, x, device)
            same = float((out.argmax(1).numpy() == saved[key]).mean())
            assert same > 0.999, (name, cond, key, same)  # 보정·모델이 단위별 실험과 같은지
            return output_error(ref, out)

        all_nmse, all_kl = measure([wb] * n, [ab] * n, "all")
        rows = []
        for i, u in enumerate(units):
            w, a = [wb] * n, [ab] * n
            w[i], a[i] = None, None
            r_nmse, r_kl = measure(w, a, f"restore_{i}")
            w, a = list(off), list(off)
            w[i], a[i] = wb, ab
            o_nmse, o_kl = measure(w, a, f"only_{i}")
            rows.append({"model": name, "condition": cond, "unit": u.name, "unit_idx": i,
                         "out_nmse_all": all_nmse, "out_nmse_restore": r_nmse, "out_nmse_only": o_nmse,
                         "out_kl_all": all_kl, "out_kl_restore": r_kl, "out_kl_only": o_kl,
                         "out_gain": all_nmse - r_nmse, "out_kl_gain": all_kl - r_kl})
        pd.DataFrame(rows).to_csv(cond_dir / "output_error.csv", index=False)
        print(name, cond, f"단위별 완료 {time.time() - t0:.0f}s, all out_nmse {all_nmse:.4f}", flush=True)
    q.remove()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--out", default="results/imagenet")
    args = p.parse_args()
    device = pick_device()
    for name in args.models:
        run(name, ROOT / args.out, device)


if __name__ == "__main__":
    main()
