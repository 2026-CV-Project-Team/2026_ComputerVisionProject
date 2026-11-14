"""ImageNet 모델의 단위별 복원·교체 실험 + 오차 순위와 복원 순위 비교.

6-Conv 실험 ①②와 같은 방식을 블록 단위로 적용한다.
- all:        모든 단위를 양자화
- restore_i:  전부 양자화한 상태에서 단위 i만 FP32로 복원 -> 정확도 회복량(gain_pp)
- only_i:     전부 FP32인 상태에서 단위 i만 양자화       -> 정확도 하락량(drop_pp)
- 오차:       단위 출력의 ‖y − ŷ‖² / ‖y‖² (이미지별 계산 후 평균)
              nmse_all = all 상태(앞 단위의 오차까지 쌓인 누적 오차), nmse_only = only_i 상태(단독 오차)

순위는 rank 5,000장으로만 계산한다(test는 고른 정책의 최종 평가용으로 남겨 둔다).

잠정 가정
- 오차 계산에는 rank 앞쪽 1,000장만 쓴다(단위 출력이 커서 전부 쓰면 오래 걸림).
- FC/Head는 logit을 양자화하지 않으므로, 그 단위의 오차는 가중치 양자화와 앞 단위 오차만 반영한다.

실행 예
    python -m src.exp_imagenet_units --models resnet18 --conditions W4A4 W5A5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .exp_imagenet import ROOT, parse_condition, pick_device, predict
from .imagenet_data import load_split, to_input
from .imagenet_models import MODELS, load_model, unit_param_counts
from .quant.units import UnitQuantizer

N_NMSE = 1000


@torch.no_grad()
def unit_nmse(model, q: UnitQuantizer, states: list[tuple[list, list]], x: torch.Tensor, device) -> np.ndarray:
    """(상태 수, 단위 수) 오차 표. 배치마다 FP32 출력을 기준으로 삼아 바로 비교한다(출력을 쌓아 두지 않음)."""
    n = len(q.units)
    totals = torch.zeros(len(states), n)
    off = [None] * n
    for i in range(0, len(x), 100):
        batch = to_input(x[i : i + 100], device)
        q.apply(off, off)
        q.capture = []
        model(batch)
        ref = [r.flatten(1) for r in q.capture]
        for s, (w, a) in enumerate(states):
            q.apply(w, a)
            q.capture = []
            model(batch)
            for k, (r, out) in enumerate(zip(ref, q.capture)):
                value = (r - out.flatten(1)).pow(2).sum(1) / r.pow(2).sum(1).clamp(min=1e-12)
                totals[s, k] += value.sum().item()
    q.capture = None
    q.apply(off, off)
    return (totals / len(x)).numpy()


def run(name: str, conditions: list[str], out_root: Path, device) -> list[dict]:
    model, units = load_model(name)
    model.to(device)
    q = UnitQuantizer(model, units)
    n = len(units)
    off = [None] * n
    calib_x, _ = load_split(name, "calib")
    q.calibrate(to_input(calib_x[i : i + 256], device) for i in range(0, len(calib_x), 256))
    x, y = load_split(name, "rank")
    q.apply(off, off)
    base = predict(model, x, device)
    acc = lambda pred: float((pred == y).float().mean() * 100)  # noqa: E731
    params = unit_param_counts(units)
    summary = []
    for cond in conditions:
        wb, ab = parse_condition(cond)
        out_dir = out_root / name / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        preds = {}
        q.apply([wb] * n, [ab] * n)
        preds["all"] = predict(model, x, device)
        for i in range(n):
            w, a = [wb] * n, [ab] * n
            w[i], a[i] = None, None
            q.apply(w, a)
            preds[f"restore_{i}"] = predict(model, x, device)
            w, a = list(off), list(off)
            w[i], a[i] = wb, ab
            q.apply(w, a)
            preds[f"only_{i}"] = predict(model, x, device)
            print(name, cond, units[i].name, "restore", round(acc(preds[f'restore_{i}']), 2),
                  "only", round(acc(preds[f'only_{i}']), 2), flush=True)
        # 오차: all 상태 한 번 + only_i 상태 n번
        states = [([wb] * n, [ab] * n)]
        for i in range(n):
            w, a = list(off), list(off)
            w[i], a[i] = wb, ab
            states.append((w, a))
        table = unit_nmse(model, q, states, x[:N_NMSE], device)
        rows = []
        for i, u in enumerate(units):
            rows.append({
                "model": name, "condition": cond, "unit": u.name, "params": params[i],
                "acc_fp32": acc(base), "acc_all": acc(preds["all"]),
                "acc_restore": acc(preds[f"restore_{i}"]), "gain_pp": acc(preds[f"restore_{i}"]) - acc(preds["all"]),
                "acc_only": acc(preds[f"only_{i}"]), "drop_pp": acc(base) - acc(preds[f"only_{i}"]),
                "nmse_all": float(table[0, i]), "nmse_only": float(table[1 + i, i]),
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "restore.csv", index=False)
        # 이미지별 예측은 flip 추적용으로 따로 저장(용량 때문에 git에는 올리지 않음)
        np.savez_compressed(out_dir / "per_image.npz", label=y.numpy(), fp32=base.numpy(),
                            **{k: v.numpy() for k, v in preds.items()})
        summary.append(rank_compare(df, name, cond))
    q.remove()
    return summary


def rank_compare(df: pd.DataFrame, name: str, cond: str) -> dict:
    """오차 순위와 복원 순위의 일치 정도. Spearman ρ와 상위 k개 겹침."""
    row = {"model": name, "condition": cond, "n_units": len(df), "acc_all": df.acc_all.iloc[0]}
    for key in ("nmse_all", "nmse_only", "drop_pp"):
        rho, p = spearmanr(df[key], df.gain_pp)
        row[f"rho_{key}_vs_gain"], row[f"p_{key}_vs_gain"] = rho, p
    top = lambda col, k: set(df.sort_values(col, ascending=False).unit.head(k))  # noqa: E731
    for k in (1, 3):
        row[f"top{k}_overlap_nmse_all"] = len(top("nmse_all", k) & top("gain_pp", k)) / k
    row["top_gain"] = ", ".join(df.sort_values("gain_pp", ascending=False).unit.head(3))
    row["top_nmse_all"] = ", ".join(df.sort_values("nmse_all", ascending=False).unit.head(3))
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=["W4A4"])
    p.add_argument("--out", default="results/imagenet")
    args = p.parse_args()
    device = pick_device()
    out_root = ROOT / args.out
    rows = []
    for name in args.models:
        rows += run(name, args.conditions, out_root, device)
    path = out_root / "summary" / "rank_corr.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():  # 모델·조건을 나눠 실행해도 이전 행이 남도록 합친다
        old = pd.read_csv(path)
        keys = set(zip(new.model, new.condition))
        old = old[[(m, c) not in keys for m, c in zip(old.model, old.condition)]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    # 단위별 표를 하나로 모아 summary에 둔다(개별 restore.csv는 git에 올리지 않으므로)
    parts = [pd.read_csv(f) for f in sorted(out_root.glob("*/*/restore.csv"))]
    pd.concat(parts, ignore_index=True).to_csv(path.parent / "units.csv", index=False)
    print(new.round(3).to_string())


if __name__ == "__main__":
    main()
