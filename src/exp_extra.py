"""6-Conv 조건 추가 실험: 이미 학습·보정이 끝난 seed에 정밀도 조건만 더 돌린다.

재학습과 재보정을 하지 않고 저장된 model.pt, ranges.json을 그대로 쓴다. 따라서 본 실험
(exp_6conv)의 조건들과 같은 모델, 같은 보정 범위, 같은 평가 이미지에서 비교된다.

용도 (이하형 문서 11-10 2차)
- W5A5: W6A6은 100%, W4A4는 평균 58%라 그 사이에서 어디부터 무너지는지 확인.
        먼저 1 seed로 보고, 변화가 있으면 5 seed와 층별 분석으로 확장.
- W8A4 등: W와 A 중 어느 쪽이 4비트 하락을 주도하는지 볼 때.

조건 이름은 W{비트}A{비트} 형식이고 32는 FP32 유지를 뜻한다. 예: W5A5, W8A4, W32A5

출력 (본 실험 폴더의 seed{n}/ 아래, 기존 파일은 건드리지 않음)
- extra_metrics.csv: metrics.csv와 같은 형식. 조건마다 all, restore_1~6, only_1~6
- extra_nmse.csv:    nmse.csv와 같은 형식
analyze.py가 두 파일을 자동으로 함께 읽는다.

사용 예:
    python -m src.exp_extra --conditions W5A5 --seeds 0
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .data import check_file, load_split
from .exp_6conv import CONDITIONS, condition_states, eval_set, evaluate, layer_nmse, load_seed
from .quant.quantizer import BITS
from .train import pick_device


def parse_condition(name: str) -> tuple[int | None, int | None]:
    """'W5A5' -> (5, 5), 'W32A4' -> (None, 4)."""
    match = re.fullmatch(r"W(\d+)A(\d+)", name)
    if not match:
        raise SystemExit(f"조건 이름 형식 오류: {name}")
    bits = [None if int(b) == 32 else int(b) for b in match.groups()]
    for b in bits:
        if b is not None and b not in BITS:
            raise SystemExit(f"{name}: {b}비트는 quantizer.BITS {BITS}에 없음")
    return bits[0], bits[1]


def merge_csv(path: Path, new: pd.DataFrame) -> None:
    """같은 조건을 다시 돌리면 예전 행을 새 행으로 바꾼다."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.condition.isin(new.condition.unique())], new], ignore_index=True)
    new.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", nargs="+", default=["W5A5"])
    parser.add_argument("--n-nmse", type=int, default=5000)  # 본 실험과 같은 장수
    args = parser.parse_args()

    conditions = {name: parse_condition(name) for name in args.conditions}
    for name in conditions:
        if name in CONDITIONS:
            raise SystemExit(f"{name}은 본 실험에 이미 있는 조건")
    check_file()
    device = pick_device()
    data, labels, _, _ = load_split(["rank", "test"])
    x_eval, y_eval, masks = eval_set(data, labels)
    x_nmse = data["rank"][: args.n_nmse]

    for seed in args.seeds:
        seed_dir = args.exp / f"seed{seed}"
        if not (seed_dir / "DONE").exists():
            raise SystemExit(f"seed {seed}: 본 실험이 끝나지 않음")
        net, table = load_seed(seed_dir, device)

        # 재현 확인: 불러온 모델로 W4A4를 다시 평가했을 때 본 실험 기록과 같아야 한다.
        # 다르면 모델이나 보정 범위를 잘못 불러온 것이므로 추가 조건 결과도 믿을 수 없다
        check, _ = evaluate(net, table, [condition_states("W4A4", 4, 4)[0]], x_eval, y_eval, masks, seed, device)
        saved = pd.read_csv(seed_dir / "metrics.csv")
        saved = saved[(saved.condition == "W4A4") & (saved.state == "all")].set_index(["set", "task"]).acc
        for row in check:
            assert abs(row["acc"] - saved[(row["set"], row["task"])]) < 1e-6, (seed, row)

        states = [s for name, (w, a) in conditions.items() for s in condition_states(name, w, a)]
        rows, _ = evaluate(net, table, states, x_eval, y_eval, masks, seed, device)
        merge_csv(seed_dir / "extra_metrics.csv", pd.DataFrame(rows))

        nmse_rows = []
        for cond, state, w_bits, a_bits in states:
            if state.startswith("restore"):
                continue  # 본 실험과 같게 all, only 상태만
            for layer, value in enumerate(layer_nmse(net, table, w_bits, a_bits, x_nmse, device), start=1):
                nmse_rows.append({"seed": seed, "condition": cond, "state": state, "layer": layer, "nmse": value})
        merge_csv(seed_dir / "extra_nmse.csv", pd.DataFrame(nmse_rows))

        done = pd.DataFrame(rows)
        view = done[(done.state == "all") & (done.set == "test")].groupby("condition")[["acc", "c2w"]].mean()
        print(f"seed {seed} (W4A4 재현 일치)\n{view.round(3)}", flush=True)


if __name__ == "__main__":
    main()
