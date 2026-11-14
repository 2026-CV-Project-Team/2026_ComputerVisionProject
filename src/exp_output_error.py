"""6-Conv 최종 출력(logit) 양자화 오차 측정.

지금까지 층별로는 출력 오차(NMSE)를 쟀지만, 최종 출력은 정답을 맞혔는지(정확도, margin)만
기록했다. 이 스크립트는 최종 출력층(과제별 Head의 logit)에서도 FP32 대비 오차를 잰다.

지표 (같은 입력의 FP32 logit z, 양자화 모델 logit ẑ, 이미지별로 계산한 뒤 평균)
- out_nmse = ‖z − ẑ‖² / ‖z‖²
    층별 NMSE(exp_6conv.layer_nmse)와 같은 식이라 층별 오차 표에 "최종 출력" 한 줄을 더한 것처럼
    바로 비교할 수 있다.
- out_kl   = KL(softmax(z) ‖ softmax(ẑ))
    확률 분포가 얼마나 바뀌었는지. logit 전체에 같은 값을 더해도 변하지 않으므로, 예측에
    실제로 영향을 주는 변화만 본다. (잠정 보조 지표, 팀 확인 필요)

Head(GAP, Linear)는 양자화하지 않으므로(qmodel.py 규칙), 최종 출력 오차는 전부 앞 층들에서
넘어온 오차다.

상태 (exp_6conv와 같음)
- all:       전 층 양자화                 -> 최종 출력 오차(누적)
- restore_i: 전 층 양자화 후 층 i만 FP32  -> 층 i 복원 시 최종 출력 오차가 얼마나 줄어드는지
- only_i:    층 i만 양자화                -> 층 i 혼자 최종 출력에 주는 오차

재학습·재보정 없이 저장된 model.pt, ranges.json을 그대로 쓴다. 확인용으로 W4A4 all 상태의
예측이 본 실험 per_image.npz 기록과 같은지 검사한다.

출력
- seed{n}/output_error.csv: seed, set, condition, state, task, out_nmse, out_kl
- summary/ 표는 `python -m src.output_summary` 로 만든다.

사용 예:
    python -m src.exp_output_error --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import TASKS, batches, check_file, load_split, to_float
from .exp_6conv import CONDITIONS, condition_states, eval_set, load_seed
from .exp_extra import parse_condition
from .models import N_LAYERS
from .train import pick_device

# 본 실험 조건 + exp_extra로 추가한 W5A5. W5A5는 extra_metrics.csv가 있는 seed에서만 돈다
EXTRA = ["W5A5"]


@torch.no_grad()
def logits_all(net, x: torch.Tensor, device, batch_size: int = 1024) -> dict[str, torch.Tensor]:
    """과제별 logit 전체 (CPU, float32). 6-Conv logit은 작아서(최대 10개) 다 들고 있어도 된다."""
    out = {t: [] for t in TASKS}
    for i, j in batches(len(x), batch_size):
        res = net(to_float(x[i:j]).to(device))
        for t in TASKS:
            out[t].append(res[t].float().cpu())
    return {t: torch.cat(v) for t, v in out.items()}


def output_error(ref: torch.Tensor, out: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """이미지별 (out_nmse, out_kl)."""
    nmse = (ref - out).pow(2).sum(1) / ref.pow(2).sum(1).clamp(min=1e-12)
    log_p, log_q = F.log_softmax(ref, 1), F.log_softmax(out, 1)
    kl = (log_p.exp() * (log_p - log_q)).sum(1)
    return nmse.numpy(), kl.numpy()


def run_seed(seed_dir: Path, seed: int, conditions: dict, x_eval, masks, device) -> pd.DataFrame:
    net, table = load_seed(seed_dir, device)
    off = [None] * N_LAYERS
    table.apply(net, off, off)
    ref = logits_all(net, x_eval, device)

    # 재현 확인: W4A4 all 예측이 본 실험 기록과 같아야 같은 모델·보정 범위를 불러온 것이다
    saved = np.load(seed_dir / "per_image.npz")
    table.apply(net, [4] * N_LAYERS, [4] * N_LAYERS)
    check = logits_all(net, x_eval, device)
    for t in TASKS:
        same = (check[t].argmax(1).numpy() == saved[f"W4A4|all|{t}|pred"]).mean()
        assert same == 1.0, (seed, t, same)

    rows = []
    for cond, (w, a) in conditions.items():
        for _, state, w_bits, a_bits in condition_states(cond, w, a):
            table.apply(net, w_bits, a_bits)
            out = logits_all(net, x_eval, device)
            for t in TASKS:
                nmse, kl = output_error(ref[t], out[t])
                for set_name, mask in masks.items():
                    rows.append({"seed": seed, "set": set_name, "condition": cond, "state": state, "task": t,
                                 "out_nmse": float(nmse[mask].mean()), "out_kl": float(kl[mask].mean())})
    table.apply(net, off, off)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("results/6conv_fold"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    check_file()
    device = pick_device()
    data, labels, _, _ = load_split(["rank", "test"])
    x_eval, _, masks = eval_set(data, labels)
    for seed in args.seeds:
        seed_dir = args.exp / f"seed{seed}"
        if not (seed_dir / "DONE").exists():
            raise SystemExit(f"seed {seed}: 본 실험이 끝나지 않음")
        conditions = dict(CONDITIONS)
        if (seed_dir / "extra_metrics.csv").exists():
            done = set(pd.read_csv(seed_dir / "extra_metrics.csv").condition)
            conditions.update({c: parse_condition(c) for c in EXTRA if c in done})
        df = run_seed(seed_dir, seed, conditions, x_eval, masks, device)
        df.to_csv(seed_dir / "output_error.csv", index=False)
        view = df[(df.state == "all") & (df.set == "test")].groupby("condition")[["out_nmse", "out_kl"]].mean()
        print(f"seed {seed} (W4A4 재현 일치)\n{view.round(4)}", flush=True)


if __name__ == "__main__":
    main()
