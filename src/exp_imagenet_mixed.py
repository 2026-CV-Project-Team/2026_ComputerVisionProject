"""ImageNet 모델의 Mixed-Precision 정책 비교: 상위 k개 단위만 8비트로 올린다.

이하형 문서 11-10 2차 (3): 전수 조사 없이 정책 6개로 상위 1·2·3 단위를 8비트로 올려 비교.
- early:  앞쪽 단위부터 (Early-High)
- late:   뒤쪽 단위부터 (Late-High)
- nmse:   누적 NMSE가 큰 순서
- drop:   단독 양자화 시 정확도 하락이 큰 순서 (단독 민감도)
- gain:   FP32 복원 이득이 큰 순서
- random: 무작위 (seed 여러 개의 평균)

나머지 단위는 기준 비트(예: W4A4)를 유지하고, 고른 단위는 W8A8로 올린다.
순위는 rank 5,000장 결과(summary/units.csv)에서만 정하고, 고른 조합을 rank와 test 모두에서 평가한다.
test 결과를 보고 정책을 고르지 않는다(순환 논증 방지).

잠정 가정
- 승격은 W와 A를 함께 8비트로 올린다(6-Conv의 Mixed와 동일).
- random은 seed 5개.
- 단위 크기가 많이 달라서(ResNet stem 9천 개, layer4.1 470만 개) 같은 k라도 저장 비용이 다르다.
  그래서 평균 가중치 비트(avg_bits = Σ N_i·b_i / Σ N_i)를 같이 기록한다. 동일 예산 비교는 아직 하지 않았다.

실행 예
    python -m src.exp_imagenet_mixed --models resnet18 --bases W4A4 W5A5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from .exp_imagenet import ROOT, parse_condition, pick_device, predict
from .imagenet_data import load_split, to_input
from .imagenet_models import MODELS, load_model, unit_param_counts
from .quant.units import UnitQuantizer

HIGH = 8  # 승격 비트
KS = (1, 2, 3)
RANDOM_SEEDS = range(5)


def choose(policy: str, table: pd.DataFrame, k: int, seed: int = 0) -> list[int]:
    """정책이 고른 단위 번호 k개. table은 단위 순서대로 정렬된 units.csv의 한 조건."""
    n = len(table)
    if policy == "early":
        return list(range(k))
    if policy == "late":
        return list(range(n - k, n))
    if policy == "random":
        return sorted(np.random.default_rng(seed).choice(n, k, replace=False).tolist())
    column = {"nmse": "nmse_all", "drop": "drop_pp", "gain": "gain_pp"}[policy]
    # 값이 같으면 앞쪽 단위를 우선(stable 정렬)해서 결과가 실행마다 달라지지 않게 한다
    order = np.argsort(-table[column].to_numpy(), kind="stable")
    return sorted(order[:k].tolist())


def run(name: str, bases: list[str], units_table: pd.DataFrame, device) -> list[dict]:
    model, units = load_model(name)
    model.to(device)
    q = UnitQuantizer(model, units)
    n = len(units)
    calib_x, _ = load_split(name, "calib")
    q.calibrate(to_input(calib_x[i : i + 256], device) for i in range(0, len(calib_x), 256))
    params = np.array(unit_param_counts(units))
    data = {s: load_split(name, s) for s in ("rank", "test")}
    rows = []
    for base in bases:
        wb, ab = parse_condition(base)
        table = units_table[(units_table.model == name) & (units_table.condition == base)].reset_index(drop=True)
        assert list(table.unit) == [u.name for u in units], f"{name} {base}: units.csv에 단위별 결과가 없음"
        plans = [("base", 0, 0, [])]
        for k in KS:
            for policy in ("early", "late", "nmse", "drop", "gain"):
                plans.append((policy, k, 0, choose(policy, table, k)))
            for seed in RANDOM_SEEDS:
                plans.append(("random", k, seed, choose("random", table, k, seed)))
        cache: dict[tuple, dict[str, float]] = {}  # 같은 조합은 한 번만 평가
        for policy, k, seed, picked in plans:
            key = tuple(picked)
            if key not in cache:
                w, a = [wb] * n, [ab] * n
                for i in picked:
                    w[i], a[i] = HIGH, HIGH
                q.apply(w, a)
                cache[key] = {
                    s: float((predict(model, x, device) == y).float().mean() * 100) for s, (x, y) in data.items()
                }
            bits = np.full(n, wb if wb is not None else 32)
            bits[picked] = HIGH
            rows.append({
                "model": name, "base": base, "policy": policy, "k": k, "seed": seed,
                "units": " ".join(units[i].name for i in picked),
                "avg_bits": float((params * bits).sum() / params.sum()),
                "acc_rank": cache[key]["rank"], "acc_test": cache[key]["test"],
            })
            print(name, base, policy, k, seed, rows[-1]["units"], round(rows[-1]["acc_test"], 2), flush=True)
    q.remove()
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--bases", nargs="+", default=["W4A4"])
    p.add_argument("--out", default="results/imagenet")
    args = p.parse_args()
    summary = ROOT / args.out / "summary"
    units_table = pd.read_csv(summary / "units.csv")
    rows = []
    for name in args.models:
        rows += run(name, args.bases, units_table, pick_device())
    new = pd.DataFrame(rows)
    path = summary / "mixed_topk_all.csv"
    if path.exists():  # 모델·기준을 나눠 실행해도 이전 행이 남도록 합친다
        old = pd.read_csv(path)
        keys = set(zip(new.model, new.base))
        old = old[[(m, b) not in keys for m, b in zip(old.model, old.base)]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    # 정책별 요약: random은 seed 평균과 표준편차
    g = new.groupby(["model", "base", "policy", "k"], sort=False)
    out = g.agg(acc_rank=("acc_rank", "mean"), acc_test=("acc_test", "mean"), sd_test=("acc_test", "std"),
                avg_bits=("avg_bits", "mean"), units=("units", "first"), n=("seed", "count")).reset_index()
    out.loc[out.policy == "random", "units"] = "(seed 평균)"
    out.to_csv(summary / "mixed_topk.csv", index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
