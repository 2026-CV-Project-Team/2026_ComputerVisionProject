"""6-Conv 실험 집계: 조건별 정답률, 층별 효과, 순위 상관, Mixed-Precision 비교.

원칙
- 층 순위는 rank 묶음에서만 산출하고, Mixed-Precision 최종 평가는 test 묶음에서만 한다.
  같은 이미지로 순위도 정하고 평가도 하면 선택 편향으로 결과가 좋게 나온다.
- 모든 표는 seed별로 계산한 뒤 평균과 표준편차로 요약한다. seed 간 편차가 커서
  한 모델의 결과만으로는 결론을 낼 수 없다.
- n = 6(층 수)이라 Spearman ρ는 |ρ| ≥ 0.886이어야 p < 0.05다. 평균 ρ만 보지 말 것.

출력 (<exp>/summary/)
- conditions.csv: 조건별 전 층 양자화 정답률 (test)
- layers.csv:     seed·조건·과제·층별 복원 이득, 단독 하락, margin 변화, NMSE (rank)
- rank_corr.csv:  seed·조건·과제별 Spearman ρ, top-1/top-2 겹침
- mixed.csv:      Mixed-Precision 정책별 test 정답률과 Weight Bit Cost
- mixed_all.csv:  64조합 전체의 test 정답률과 Weight Bit Cost
- pareto.csv:     64조합의 정확도-저장 비용 Pareto frontier
- domain_conditions.csv, domain_layers.csv, domain_rank_corr.csv:
                  도메인 실험(exp_domain.py)을 돌린 경우에만 생성

Mixed-Precision 비교는 목적이 다른 두 방식을 모두 유지한다(이하형 문서 11-10 2차). 서로 다른
실험이므로 결과를 섞어 해석하지 않는다.
- top-k:  순위 상위 k개 층만 8비트. 층 개수를 통제해 "어떤 지표가 중요한 층을 잘 찾는가"를 본다.
          정책마다 저장 비용이 달라지므로 Weight Bit Cost를 함께 적는다.
- budget: 같은 Weight Bit Cost 예산 안에서 고름. "메모리 절약 대비 정확도"를 본다.
          Conv5·6이 가중치의 77%라, 순위대로만 올리면 정책들이 같은 조합을 고르는 문제가 있다.
          그래서 점수를 추가 비트 비용으로 나눈 값(비용 대비)으로 고르는 정책도 함께 계산한다.
- pareto: 64조합을 실제 저장 비용 B = Σ N_i·b_i 로 정렬해 frontier를 구한다.
          frontier는 rank 묶음에서 고르고, test 묶음에서는 고른 조합만 평가한다.
          test를 보고 좋은 조합을 고르면 최종 수치가 낙관적으로 나오기 때문이다.

사용 예:
    python -m src.analyze --exp results/6conv_fold
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .models import CHANNELS, N_LAYERS

LAYERS = range(1, N_LAYERS + 1)
IN_CH = [3] + CHANNELS[:-1]
# 층별 Conv 가중치 수(3x3). Weight Bit Cost 계산에 사용. bias와 Head는 FP32라 제외
PARAMS = np.array([IN_CH[i] * CHANNELS[i] * 9 for i in range(N_LAYERS)])
AVG = "mean3"  # 세 과제 평균을 나타내는 가상의 과제 이름
BUDGET_BITS = (5, 6, 7)  # 평균 가중치 비트 예산(잠정)
TOP_K = (1, 2, 3)


def load(exp: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dirs = sorted(d for d in exp.glob("seed*") if (d / "DONE").exists())
    if not dirs:
        raise SystemExit("완료된 seed 없음")
    # extra_*.csv는 exp_extra.py가 나중에 추가한 조건(예: W5A5). 일부 seed에만 있을 수 있다
    def read(name: str) -> pd.DataFrame:
        files = [d / f for d in dirs for f in (f"{name}.csv", f"extra_{name}.csv") if (d / f).exists()]
        return pd.concat(pd.read_csv(f) for f in files)

    metrics, nmse = read("metrics"), read("nmse")
    keys = ["seed", "set", "condition", "state"]
    mean = metrics.groupby(keys, as_index=False)[["acc", "margin", "margin_drop", "c2w", "w2c"]].mean()
    mean["task"] = AVG
    return pd.concat([metrics, mean], ignore_index=True), nmse


def layer_table(metrics: pd.DataFrame, nmse: pd.DataFrame, set_name: str = "rank") -> pd.DataFrame:
    """seed·조건·과제·층별 효과.

    gain_pp = 복원 이득: (층 i만 FP32 복원한 정답률) − (전 층 양자화 정답률)
    drop_pp = 단독 하락: (FP32 정답률) − (층 i만 양자화한 정답률)
    nmse_all = 누적 양자화 오차, nmse_only = 단독 양자화 오차
    """
    m = metrics[(metrics.set == set_name) & (metrics.condition != "MIX")]
    fp32 = metrics[(metrics.set == set_name) & (metrics.condition == "FP32")].set_index(["seed", "task"])
    all_q = m[m.state == "all"].set_index(["seed", "condition", "task"])[["acc", "margin"]]
    rows = []
    for (seed, cond, task), base in all_q.iterrows():
        if cond == "FP32":
            continue
        fp = fp32.loc[(seed, task)]
        sub = m[(m.seed == seed) & (m.condition == cond) & (m.task == task)].set_index("state")
        for i in LAYERS:
            rows.append(
                {
                    "seed": seed,
                    "condition": cond,
                    "task": task,
                    "layer": i,
                    "acc_all": base.acc,
                    "gain_pp": sub.loc[f"restore_{i}", "acc"] - base.acc,
                    "margin_gain": sub.loc[f"restore_{i}", "margin"] - base.margin,
                    "drop_pp": fp.acc - sub.loc[f"only_{i}", "acc"],
                    "margin_drop_only": fp.margin - sub.loc[f"only_{i}", "margin"],
                }
            )
    table = pd.DataFrame(rows)
    pick = nmse.set_index(["seed", "condition", "state", "layer"]).nmse
    table["nmse_all"] = [pick[(r.seed, r.condition, "all", r.layer)] for r in table.itertuples()]
    table["nmse_only"] = [pick[(r.seed, r.condition, f"only_{r.layer}", r.layer)] for r in table.itertuples()]
    return table


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """두 점수의 상위 k개 층이 겹치는 비율."""
    top_a = set(np.argsort(-a, kind="stable")[:k])
    top_b = set(np.argsort(-b, kind="stable")[:k])
    return len(top_a & top_b) / k


def rank_correlations(table: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("nmse_all", "gain_pp"),
        ("nmse_all", "margin_gain"),
        ("nmse_only", "gain_pp"),
        ("nmse_only", "drop_pp"),
        ("nmse_only", "margin_drop_only"),
        ("gain_pp", "drop_pp"),
        ("margin_gain", "margin_drop_only"),
    ]
    rows = []
    for (seed, cond, task), g in table.groupby(["seed", "condition", "task"]):
        g = g.sort_values("layer")
        for a, b in pairs:
            x, y = g[a].to_numpy(), g[b].to_numpy()
            # 값이 전부 같으면(예: W8A8에서 이득이 모두 0) 순위가 정의되지 않으므로
            # ρ와 겹침을 NaN으로 둔다. 동점을 순서대로 끊어 겹침을 계산하면 인공값이 나온다
            constant = np.ptp(x) == 0 or np.ptp(y) == 0
            rows.append(
                {
                    "seed": seed,
                    "condition": cond,
                    "task": task,
                    "x": a,
                    "y": b,
                    "rho": np.nan if constant else spearmanr(x, y).statistic,
                    "top1": np.nan if constant else overlap(x, y, 1),
                    "top2": np.nan if constant else overlap(x, y, 2),
                }
            )
    return pd.DataFrame(rows)


def mix_name(bits) -> str:
    return "mix_" + "".join(map(str, bits))


def bit_cost(bits) -> float:
    """Weight Bit Cost: 가중치 수로 가중 평균한 비트 수."""
    return float((np.array(bits) * PARAMS).sum() / PARAMS.sum())


def total_bits(bits) -> int:
    """실제 저장 비용 B = Σ N_i · b_i (비트). N_i는 층 i의 가중치 수, b_i는 할당 비트."""
    return int((np.array(bits) * PARAMS).sum())


def greedy(order: np.ndarray, budget: float) -> tuple[int, ...]:
    """순위대로 보면서 예산이 허용하는 층만 8비트로 올림."""
    bits = np.full(N_LAYERS, 4)
    for i in order:
        trial = bits.copy()
        trial[i] = 8
        if bit_cost(trial) <= budget:
            bits = trial
    return tuple(int(b) for b in bits)


def top_k_bits(order: np.ndarray, k: int) -> tuple[int, ...]:
    bits = np.full(N_LAYERS, 4)
    bits[order[:k]] = 8
    return tuple(int(b) for b in bits)


def mixed(metrics: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """층 선택은 rank 묶음의 W4A4 순위로, 평가는 test 묶음의 세 과제 평균 정답률로 한다."""
    mix = metrics[(metrics.condition == "MIX") & (metrics.task == AVG)]
    test = mix[mix.set == "test"].set_index(["seed", "state"]).acc
    rank = mix[mix.set == "rank"].set_index(["seed", "state"]).acc
    base = table[(table.condition == "W4A4") & (table.task == AVG)]
    combos = list(itertools.product((4, 8), repeat=N_LAYERS))
    rows = []

    def add(seed, rule, policy, bits):
        rows.append({"seed": seed, "rule": rule, "policy": policy, "bits": "".join(map(str, bits)),
                     "weight_bit_cost": bit_cost(bits), "test_acc": test[(seed, mix_name(bits))]})

    for seed, g in base.groupby("seed"):
        g = g.sort_values("layer")
        scores = {
            "누적 NMSE 순위": g.nmse_all.to_numpy(),
            "단독 NMSE 순위": g.nmse_only.to_numpy(),
            "복원 이득 순위": g.gain_pp.to_numpy(),
            "복원 margin 순위": g.margin_gain.to_numpy(),
            "단독 하락 순위": g.drop_pp.to_numpy(),
            "앞층부터": -np.arange(N_LAYERS, dtype=float),  # 순위 정보 없이 고르는 기준선
        }
        orders = {name: np.argsort(-score, kind="stable") for name, score in scores.items()}
        add(seed, "기준", "전 층 4비트", (4,) * N_LAYERS)
        add(seed, "기준", "전 층 8비트", (8,) * N_LAYERS)
        # 첫·마지막 층만 8비트로 보호하는 전략(이하형 문서 5절)
        add(seed, "기준", "첫·마지막 층 8비트 보호", (8,) + (4,) * (N_LAYERS - 2) + (8,))
        for budget in BUDGET_BITS:
            rule = f"budget {budget}"
            options = [b for b in combos if bit_cost(b) <= budget]
            for name, order in orders.items():
                add(seed, rule, name, greedy(order, budget))
            # 비용 대비: 점수 / (그 층을 8비트로 올릴 때 늘어나는 비트 수). 층 크기가 달라서
            # 점수 순서대로만 고르면 큰 층 하나가 예산을 다 써버릴 수 있다
            for name, score in scores.items():
                if name != "앞층부터":
                    order = np.argsort(-score / (PARAMS * 4), kind="stable")
                    add(seed, rule, f"{name} (비용 대비)", greedy(order, budget))
            accs = np.array([test[(seed, mix_name(b))] for b in options])
            rows.append({"seed": seed, "rule": rule, "policy": "가능 조합 평균", "bits": f"{len(options)}개",
                         "weight_bit_cost": np.mean([bit_cost(b) for b in options]), "test_acc": accs.mean()})
            add(seed, rule, "rank 전수 조사 최적", max(options, key=lambda b: rank[(seed, mix_name(b))]))
        for k in TOP_K:
            rule = f"top-{k}"
            options = [b for b in combos if b.count(8) == k]
            for name, order in orders.items():
                add(seed, rule, name, top_k_bits(order, k))
            accs = np.array([test[(seed, mix_name(b))] for b in options])
            rows.append({"seed": seed, "rule": rule, "policy": "가능 조합 평균", "bits": f"{len(options)}개",
                         "weight_bit_cost": np.mean([bit_cost(b) for b in options]), "test_acc": accs.mean()})
            add(seed, rule, "rank 전수 조사 최적", max(options, key=lambda b: rank[(seed, mix_name(b))]))
    return pd.DataFrame(rows)


def frontier(cost: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """지배당하지 않는 점의 마스크. 비용이 같거나 작으면서 정답률이 같거나 높은 다른 점이
    있고 둘 중 하나라도 엄격히 더 좋으면 지배당한 것이다."""
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        better = (cost <= cost[i]) & (acc >= acc[i]) & ((cost < cost[i]) | (acc > acc[i]))
        keep[i] = not better.any()
    return keep


def pareto(metrics: pd.DataFrame) -> pd.DataFrame:
    """64조합의 정확도-저장 비용 frontier. 세 과제 평균 정답률 기준.

    scope가 seed 번호면 그 모델 하나의 frontier, 'mean'이면 seed 평균 정답률로 구한 frontier다.
    선택은 rank_acc로만 하고 test_acc는 선택된 조합의 값을 옮겨 적기만 한다.
    """
    mix = metrics[(metrics.condition == "MIX") & (metrics.task == AVG)]
    wide = mix.pivot_table(index=["seed", "state"], columns="set", values="acc").reset_index()
    wide["bits"] = wide.state.str.replace("mix_", "")
    wide["total_bits"] = [total_bits([int(c) for c in b]) for b in wide.bits]
    wide["weight_bit_cost"] = [bit_cost([int(c) for c in b]) for b in wide.bits]
    parts = []
    for seed, g in wide.groupby("seed"):
        g = g[frontier(g.total_bits.to_numpy(), g["rank"].to_numpy())]
        parts.append(g.assign(scope=str(seed), test_sd=np.nan))
    avg = wide.groupby(["bits", "total_bits", "weight_bit_cost"], as_index=False).agg(
        rank=("rank", "mean"), test_sd=("test", "std"), test=("test", "mean"))
    parts.append(avg[frontier(avg.total_bits.to_numpy(), avg["rank"].to_numpy())].assign(scope="mean"))
    out = pd.concat(parts, ignore_index=True).rename(columns={"rank": "rank_acc", "test": "test_acc"})
    out = out.sort_values(["scope", "total_bits"])
    return out[["scope", "bits", "total_bits", "weight_bit_cost", "rank_acc", "test_acc", "test_sd"]]


def domain_tables(exp: Path):
    """도메인별 조건 정답률, W4A4 층별 효과, 순위 상관. 도메인 실험 결과가 없으면 None.

    순위 상관은 두 가지를 본다.
    - nmse_all vs gain_pp: 그 도메인 안에서 누적 NMSE 순위와 복원 이득 순위가 맞는지
    - gain_pp(orig) vs gain_pp(도메인): 입력이 바뀌어도 복원 이득 순위가 유지되는지
    """
    dirs = sorted(d for d in exp.glob("seed*") if (d / "domain_metrics.csv").exists())
    if not dirs:
        return None
    metrics = pd.concat(pd.read_csv(d / "domain_metrics.csv") for d in dirs)
    nmse = pd.concat(pd.read_csv(d / "domain_nmse.csv") for d in dirs)
    keys = ["domain", "seed", "set", "condition", "state"]
    mean = metrics.groupby(keys, as_index=False)[["acc", "margin", "margin_drop", "c2w", "w2c"]].mean()
    metrics = pd.concat([metrics, mean.assign(task=AVG)], ignore_index=True)

    cond = metrics[(metrics.state == "all") & (metrics.set == "test")]
    layers = []
    for domain, m in metrics.groupby("domain", sort=False):
        # layer_table은 조건마다 restore/only 상태가 있어야 하므로 층별 상태를 돌린 W4A4만 넘긴다
        m = m[m.condition.isin(["FP32", "W4A4"])].drop(columns="domain")
        n = nmse[nmse.domain == domain].drop(columns="domain")
        layers.append(layer_table(m, n).assign(domain=domain))
    layers = pd.concat(layers, ignore_index=True)

    rows = []
    avg = layers[layers.task == AVG]
    for (seed, domain), g in avg.groupby(["seed", "domain"], sort=False):
        g = g.sort_values("layer")
        base = avg[(avg.seed == seed) & (avg.domain == "orig")].sort_values("layer")
        rows.append({
            "seed": seed, "domain": domain,
            "rho_nmse_gain": spearmanr(g.nmse_all, g.gain_pp).statistic,
            "top1_nmse_gain": overlap(g.nmse_all.to_numpy(), g.gain_pp.to_numpy(), 1),
            "rho_gain_vs_orig": spearmanr(base.gain_pp, g.gain_pp).statistic,
            "top1_gain_vs_orig": overlap(base.gain_pp.to_numpy(), g.gain_pp.to_numpy(), 1),
        })
    return cond, layers, pd.DataFrame(rows)


def mixed_all(metrics: pd.DataFrame) -> pd.DataFrame:
    mix = metrics[(metrics.condition == "MIX") & (metrics.set == "test")].copy()
    mix["bits"] = mix.state.str.replace("mix_", "")
    mix["weight_bit_cost"] = [bit_cost([int(c) for c in b]) for b in mix.bits]
    mix["total_bits"] = [total_bits([int(c) for c in b]) for b in mix.bits]
    return mix[["seed", "bits", "weight_bit_cost", "total_bits", "task", "acc", "c2w", "w2c"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    args = parser.parse_args()
    out = args.exp / "summary"
    out.mkdir(exist_ok=True)
    metrics, nmse = load(args.exp)
    print("seed:", sorted(int(s) for s in metrics.seed.unique()))
    pd.set_option("display.width", 220, "display.max_rows", 500, "display.max_columns", 30)

    cond = metrics[(metrics.state == "all") & (metrics.condition != "MIX") & (metrics.set == "test")]
    cond.to_csv(out / "conditions.csv", index=False)
    print("\n조건별 test 정답률(%)과 C→W 건수")
    # n은 그 조건을 돌린 seed 수. 추가 조건은 seed 수가 다를 수 있어 함께 적는다
    print(cond.groupby(["condition", "task"]).agg(n=("acc", "size"), acc=("acc", "mean"), sd=("acc", "std"),
                                                 lo=("acc", "min"), hi=("acc", "max"), c2w=("c2w", "mean")).round(2))

    table = layer_table(metrics, nmse)
    table.to_csv(out / "layers.csv", index=False)
    for value in ("gain_pp", "drop_pp", "nmse_all", "nmse_only"):
        print(f"\nW4A4 층별 [{value}] (rank 묶음, seed 평균과 표준편차)")
        sub = table[table.condition == "W4A4"]
        print(sub.pivot_table(index="layer", columns="task", values=value, aggfunc=["mean", "std"]).round(3))

    corr = rank_correlations(table)
    corr.to_csv(out / "rank_corr.csv", index=False)
    print("\n순위 상관(세 과제 평균 기준): seed 평균 ρ, 표준편차, |ρ|≥0.886인 seed 수, top-1·top-2 겹침")
    view = corr[corr.task == AVG].groupby(["condition", "x", "y"]).agg(
        rho=("rho", "mean"), sd=("rho", "std"), sig=("rho", lambda r: int((r.abs() >= 0.886).sum())),
        top1=("top1", "mean"), top2=("top2", "mean"))
    print(view.round(2))

    mix = mixed(metrics, table)
    mix.to_csv(out / "mixed.csv", index=False)
    mixed_all(metrics).to_csv(out / "mixed_all.csv", index=False)
    print("\nMixed-Precision: test 정답률(%, 세 과제 평균)과 Weight Bit Cost(평균 비트)")
    print(mix.groupby(["rule", "policy"], sort=False).agg(
        acc=("test_acc", "mean"), sd=("test_acc", "std"), cost=("weight_bit_cost", "mean")).round(2))

    domain = domain_tables(args.exp)
    if domain is not None:
        d_cond, d_layers, d_corr = domain
        d_cond.to_csv(out / "domain_conditions.csv", index=False)
        d_layers.to_csv(out / "domain_layers.csv", index=False)
        d_corr.to_csv(out / "domain_rank_corr.csv", index=False)
        view = d_cond[d_cond.task == AVG]
        print("\n도메인별 test 정답률(%, 세 과제 평균, seed 평균)")
        print(view.pivot_table(index="domain", columns="condition", values="acc", sort=False).round(2))
        print("\n도메인별 test 정답률 seed 표준편차")
        print(view.pivot_table(index="domain", columns="condition", values="acc", aggfunc="std", sort=False).round(2))
        print("\n도메인별 W4A4 복원 이득(%p, rank 묶음, seed 평균)")
        sub = d_layers[d_layers.task == AVG]
        print(sub.pivot_table(index="domain", columns="layer", values="gain_pp", sort=False).round(2))
        print("\n도메인별 순위 상관(seed 평균)")
        print(d_corr.groupby("domain", sort=False).mean(numeric_only=True).drop(columns="seed").round(2))

    front = pareto(metrics)
    front.to_csv(out / "pareto.csv", index=False)
    print("\nPareto frontier (seed 평균 기준, rank에서 선택하고 test로 평가)")
    print(front[front.scope == "mean"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
