"""최종 출력(logit) 오차 집계. exp_output_error(6-Conv), exp_imagenet_output(ImageNet) 결과를 읽는다.

6-Conv 출력 (results/6conv_fold/summary/)
- output_conditions.csv: 조건별 전 층 양자화 상태의 최종 출력 오차와 정확도 (test, seed별)
- output_layers.csv:     seed·조건·과제·층별 최종 출력 오차 (rank). 기존 layers.csv 열도 함께 붙임
    out_nmse_all     = 전 층 양자화 상태의 최종 출력 오차 (층과 무관하게 같은 값)
    out_nmse_restore = 층 i만 FP32로 되돌렸을 때의 최종 출력 오차
    out_gain         = out_nmse_all − out_nmse_restore  (층 i 복원으로 줄어든 최종 출력 오차)
    out_nmse_only    = 층 i만 양자화했을 때의 최종 출력 오차
    out_kl_*         = 같은 내용을 KL(softmax) 기준으로
- output_rank_corr.csv:  seed·조건·과제별 Spearman ρ (아래 PAIRS)

ImageNet 출력 (results/imagenet/summary/)
- output_conditions.csv, output_units.csv, output_rank_corr.csv: 위와 같은 형식(seed 대신 model)

사용 예:
    python -m src.output_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analyze import AVG, LAYERS, layer_table, load, overlap

# (x, y): x 순위가 y 순위를 얼마나 맞히는지
PAIRS = [
    ("nmse_all", "out_gain"),       # 층 출력 누적 오차가 큰 층 = 복원하면 최종 출력 오차가 많이 줄어드는 층?
    ("out_gain", "gain_pp"),        # 최종 출력 오차 감소 = 정확도 회복?
    ("out_kl_gain", "gain_pp"),
    ("nmse_only", "out_nmse_only"),  # 층 단독 오차가 큰 층 = 최종 출력에도 큰 오차를 주는 층?
    ("out_nmse_only", "drop_pp"),    # 최종 출력 단독 오차 = 단독 정확도 하락?
    ("out_kl_only", "drop_pp"),
]


def corr_rows(table: pd.DataFrame, keys: list[str], item: str) -> pd.DataFrame:
    rows = []
    for key, g in table.groupby(keys):
        g = g.sort_values(item)
        for a, b in PAIRS:
            if a not in g or b not in g:
                continue
            x, y = g[a].to_numpy(), g[b].to_numpy()
            constant = np.ptp(x) == 0 or np.ptp(y) == 0  # 순위가 정의되지 않음 (analyze.py와 같은 처리)
            rows.append({**dict(zip(keys, key)), "x": a, "y": b,
                         "rho": np.nan if constant else spearmanr(x, y).statistic,
                         "top1": np.nan if constant else overlap(x, y, 1),
                         "top3": np.nan if constant else overlap(x, y, 3)})
    return pd.DataFrame(rows)


def six_conv(exp: Path) -> None:
    files = sorted(exp.glob("seed*/output_error.csv"))
    if not files:
        print("6-Conv output_error.csv 없음, 건너뜀")
        return
    out = pd.concat(pd.read_csv(f) for f in files)
    mean = out.groupby(["seed", "set", "condition", "state"], as_index=False)[["out_nmse", "out_kl"]].mean()
    mean["task"] = AVG  # 세 과제 평균
    out = pd.concat([out, mean], ignore_index=True)
    metrics, nmse = load(exp)
    summary = exp / "summary"

    # 조건별 (test)
    acc = metrics[metrics.state == "all"].set_index(["seed", "set", "condition", "task"]).acc
    cond = out[(out.state == "all") & (out.set == "test")].copy()
    cond["acc"] = [acc[(r.seed, r.set, r.condition, r.task)] for r in cond.itertuples()]
    cond.drop(columns="state").to_csv(summary / "output_conditions.csv", index=False)

    # 층별 (rank)
    pick = out[out.set == "rank"].set_index(["seed", "condition", "task", "state"])
    table = layer_table(metrics, nmse)
    table = table[table.seed.isin(out.seed.unique())]
    for metric in ("nmse", "kl"):
        col = f"out_{metric}"
        table[f"{col}_all"] = [pick.loc[(r.seed, r.condition, r.task, "all"), col] for r in table.itertuples()]
        table[f"{col}_restore"] = [pick.loc[(r.seed, r.condition, r.task, f"restore_{r.layer}"), col]
                                   for r in table.itertuples()]
        table[f"{col}_only"] = [pick.loc[(r.seed, r.condition, r.task, f"only_{r.layer}"), col]
                                for r in table.itertuples()]
    table["out_gain"] = table.out_nmse_all - table.out_nmse_restore
    table["out_kl_gain"] = table.out_kl_all - table.out_kl_restore
    table.to_csv(summary / "output_layers.csv", index=False)
    corr = corr_rows(table, ["seed", "condition", "task"], "layer")
    corr.to_csv(summary / "output_rank_corr.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n[6-Conv] 조건별 최종 출력 오차 (test, 세 과제 평균, seed 평균)")
    view = cond[cond.task == AVG].groupby("condition")[["acc", "out_nmse", "out_kl"]].agg(["mean", "std"])
    print(view.round(4))
    print("\n[6-Conv] W4A4 층별 (rank, 세 과제 평균, seed 평균)")
    sub = table[(table.condition == "W4A4") & (table.task == AVG)]
    cols = ["nmse_all", "nmse_only", "gain_pp", "drop_pp", "out_nmse_restore", "out_gain", "out_nmse_only"]
    print(sub.groupby("layer")[cols].mean().round(4))
    print("W4A4 전 층 양자화 최종 출력 NMSE:", round(sub.out_nmse_all.mean(), 4))
    print("\n[6-Conv] 순위 상관 (세 과제 평균 기준, seed 평균)")
    print(corr[corr.task == AVG].groupby(["condition", "x", "y"])[["rho", "top1", "top3"]].mean().round(2))


def imagenet(root: Path) -> None:
    files = sorted(root.glob("*/*/output_error.csv"))
    if not files:
        print("ImageNet output_error.csv 없음, 건너뜀")
        return
    summary = root / "summary"
    units = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cond_files = sorted(root.glob("*/output_conditions.csv"))
    if cond_files:
        pd.concat(pd.read_csv(f) for f in cond_files).to_csv(summary / "output_conditions.csv", index=False)
    # 기존 단위별 표(units.csv)의 복원 이득·단독 하락·층 NMSE를 붙인다
    base = pd.read_csv(summary / "units.csv")
    units = units.merge(base, on=["model", "condition", "unit"], how="left")
    units.to_csv(summary / "output_units.csv", index=False)
    corr = corr_rows(units, ["model", "condition"], "unit_idx")
    corr.to_csv(summary / "output_rank_corr.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n[ImageNet] 단위별 (rank)")
    cols = ["unit", "nmse_all", "gain_pp", "out_nmse_all", "out_nmse_restore", "out_gain", "out_nmse_only", "drop_pp"]
    for (m, c), g in units.groupby(["model", "condition"]):
        print(f"-- {m} {c}")
        print(g.sort_values("unit_idx")[cols].round(4).to_string(index=False))
    print("\n[ImageNet] 순위 상관")
    print(corr.pivot_table(index=["model", "condition"], columns=["x", "y"], values="rho").round(2).to_string())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    p.add_argument("--imagenet", type=Path, default=Path("results/imagenet"))
    args = p.parse_args()
    six_conv(args.exp)
    imagenet(args.imagenet)


if __name__ == "__main__":
    main()
