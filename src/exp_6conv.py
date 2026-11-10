"""6-Conv 층별 양자화 본 실험 (3D-Shapes, 여러 seed).

한 seed의 순서
  1. FP32 학습 (초기값과 학습 순서만 seed로 바뀜. 데이터 분할은 모든 seed 공통)
  2. Conv-BN folding, folding 전후 FP32 출력 일치 검증
  3. 보정(calibration): 학습과 분리된 2,048장으로 층별 활성값 min/max 수집
  4. 아래 상태(state)를 모두 평가하고 이미지별 예측을 저장
  5. 층별 NMSE 측정

상태
- all:        전 층 양자화
- restore_i:  전 층 양자화 후 층 i만 FP32로 복원  -> 복원 이득
- only_i:     전 층 FP32에서 층 i만 양자화          -> 단독 하락
- mix_XXXXXX: 층별 4/8비트 64조합 전수(W·A 동일 비트). X는 Conv1부터의 비트

출력 (--out 아래)
- eval_index.npz:        평가 이미지의 원본 번호, 묶음(rank/test), 6개 속성 정답
- seed{n}/model.pt:      학습된 FP32 가중치(folding 전)
- seed{n}/fold_check.json: folding 전후 출력 차이
- seed{n}/ranges.json:   층별 활성값 보정 범위
- seed{n}/metrics.csv:   상태·과제·묶음별 집계
- seed{n}/nmse.csv:      층별 NMSE (rank 묶음 앞 n_nmse장)
- seed{n}/per_image.npz: 상태별 이미지별 예측과 logit margin
- seed{n}/DONE:          완료 표시. 있으면 재실행 때 건너뛰므로 중간에 끊겨도 이어서 돌릴 수 있다

사용 예:
    python -m src.exp_6conv --seeds 0                 # 1개 seed 검증
    python -m src.exp_6conv --seeds 0 1 2 3 4         # 완료된 seed는 건너뜀
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import FACTORS, TASKS, batches, check_file, load_split, to_float
from .models import N_LAYERS, SixConv
from .quant.qmodel import FoldedSixConv, QuantTable, calibrate
from .train import pick_device, train_model

# 정밀도 조건: 이름 -> (W 비트, A 비트). None은 FP32 유지
# W4A32 / W32A4는 W와 A의 영향을 분리해 보기 위한 조건이다
CONDITIONS = {
    "W8A8": (8, 8),
    "W6A6": (6, 6),
    "W4A8": (4, 8),
    "W32A4": (None, 4),
    "W4A32": (4, None),
    "W4A4": (4, 4),
}
MIX_BITS = (4, 8)  # Mixed-Precision에서 층마다 고르는 비트


def condition_states(cond: str, w, a) -> list[tuple[str, str, list, list]]:
    """조건 하나의 상태 13개(all, restore_1~6, only_1~6)."""
    states = [(cond, "all", [w] * N_LAYERS, [a] * N_LAYERS)]
    for i in range(N_LAYERS):
        w_r, a_r = [w] * N_LAYERS, [a] * N_LAYERS
        w_r[i], a_r[i] = None, None  # 층 i의 W와 A를 함께 FP32로 복원
        states.append((cond, f"restore_{i + 1}", w_r, a_r))
        w_o, a_o = [None] * N_LAYERS, [None] * N_LAYERS
        w_o[i], a_o[i] = w, a  # 층 i만 양자화
        states.append((cond, f"only_{i + 1}", w_o, a_o))
    return states


def build_states() -> list[tuple[str, str, list, list]]:
    """평가할 모든 상태를 (조건, 상태 이름, 층별 W 비트, 층별 A 비트)로 나열."""
    states = [("FP32", "all", [None] * N_LAYERS, [None] * N_LAYERS)]
    for cond, (w, a) in CONDITIONS.items():
        states += condition_states(cond, w, a)
    for mask in range(2**N_LAYERS):
        bits = [MIX_BITS[mask >> i & 1] for i in range(N_LAYERS)]
        states.append(("MIX", "mix_" + "".join(map(str, bits)), bits, bits))
    return states


@torch.no_grad()
def predict(net, x: torch.Tensor, y: dict, device, batch_size: int = 1024, prep=to_float):
    """과제별 예측 클래스와 logit margin(정답 logit − 나머지 중 최대 logit).

    margin은 정답률이 변하지 않는 조건에서도 예측이 얼마나 흔들렸는지 보여준다.
    prep은 배치를 모델 입력으로 바꾸는 함수다. 기본은 uint8 -> [-0.5, 0.5].
    """
    n = len(x)
    pred = {t: np.empty(n, dtype=np.uint8) for t in TASKS}
    margin = {t: np.empty(n, dtype=np.float32) for t in TASKS}
    for i, j in batches(n, batch_size):
        out = net(prep(x[i:j]).to(device))
        for t in TASKS:
            target = y[t][i:j].to(device)
            logits = out[t]
            true = logits.gather(1, target[:, None]).squeeze(1)
            other = logits.scatter(1, target[:, None], float("-inf")).amax(1)
            pred[t][i:j] = logits.argmax(1).cpu().numpy()
            margin[t][i:j] = (true - other).cpu().numpy()
    return pred, margin


@torch.no_grad()
def layer_nmse(net: FoldedSixConv, table: QuantTable, w_bits, a_bits, x, device) -> list[float]:
    """층별 출력의 ‖y − ŷ‖² / ‖y‖². 이미지별로 계산한 뒤 평균.

    기준 y는 전 층 FP32의 같은 위치 출력이다. 따라서
    - only_i 상태의 층 i 값   = 그 층 혼자 만든 오차(단독 양자화 오차)
    - all 상태의 층 i 값      = 앞 층들의 오차까지 쌓인 오차(누적 양자화 오차)
    """
    totals = torch.zeros(N_LAYERS)
    off = [None] * N_LAYERS
    for i, j in batches(len(x), 512):
        batch = to_float(x[i:j]).to(device)
        table.apply(net, off, off)
        net.capture = []
        net(batch)
        ref = net.capture
        table.apply(net, w_bits, a_bits)
        net.capture = []
        net(batch)
        for k, (a, b) in enumerate(zip(ref, net.capture)):
            a, b = a.flatten(1), b.flatten(1)
            value = (a - b).pow(2).sum(1) / a.pow(2).sum(1).clamp(min=1e-12)
            totals[k] += value.sum().item()
    net.capture = None
    return (totals / len(x)).tolist()


@torch.no_grad()
def check_folding(orig: SixConv, folded: FoldedSixConv, x: torch.Tensor, device) -> dict:
    """folding 전후 FP32 출력 비교. logit 최대 차이와 예측이 달라진 이미지 수."""
    max_diff, changed = 0.0, 0
    for i, j in batches(len(x), 1024):
        batch = to_float(x[i:j]).to(device)
        a, b = orig(batch), folded(batch)
        for t in TASKS:
            max_diff = max(max_diff, (a[t] - b[t]).abs().max().item())
            changed += (a[t].argmax(1) != b[t].argmax(1)).sum().item()
    return {"n": len(x), "max_abs_logit_diff": max_diff, "pred_changed": changed}


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eval_set(data, labels):
    """rank와 test를 이어 붙인 평가 입력, 정답, 묶음별 마스크."""
    x_eval = torch.cat([data["rank"], data["test"]])
    y_eval = {t: torch.cat([labels["rank"][t], labels["test"][t]]) for t in TASKS}
    n_rank = len(data["rank"])
    at = np.arange(len(x_eval))
    return x_eval, y_eval, {"rank": at < n_rank, "test": at >= n_rank}


def evaluate(net, table, states, x_eval, y_eval, masks, seed, device, prep=to_float, log=None):
    """상태 목록을 평가해 (집계 행, 이미지별 예측)을 반환.

    C→W, W→C는 같은 입력에 대한 FP32 예측과 비교한다. 입력이 바뀌는 실험(도메인)에서도
    그 입력의 FP32 결과가 기준이 되므로, 입력 변화의 영향과 양자화의 영향이 섞이지 않는다.
    """
    off = [None] * N_LAYERS
    y_np = {t: y_eval[t].numpy() for t in TASKS}
    table.apply(net, off, off)
    base_pred, base_margin = predict(net, x_eval, y_eval, device, prep=prep)
    if log:
        log("FP32 " + str({t: round(float((base_pred[t] == y_np[t]).mean() * 100), 3) for t in TASKS}))

    per_image: dict[str, np.ndarray] = {}
    rows = []
    for k, (cond, state, w_bits, a_bits) in enumerate(states):
        table.apply(net, w_bits, a_bits)
        pred, margin = predict(net, x_eval, y_eval, device, prep=prep)
        for t in TASKS:
            per_image[f"{cond}|{state}|{t}|pred"] = pred[t]
            per_image[f"{cond}|{state}|{t}|margin"] = margin[t].astype(np.float16)
            ok = pred[t] == y_np[t]
            base_ok = base_pred[t] == y_np[t]
            for set_name, mask in masks.items():
                rows.append(
                    {
                        "seed": seed,
                        "set": set_name,
                        "condition": cond,
                        "state": state,
                        "task": t,
                        "acc": float(ok[mask].mean() * 100),
                        "c2w": int((base_ok & ~ok)[mask].sum()),  # FP32 정답 -> 양자화 후 오답
                        "w2c": int((~base_ok & ok)[mask].sum()),  # FP32 오답 -> 양자화 후 정답
                        "margin": float(margin[t][mask].mean()),
                        "margin_drop": float((base_margin[t] - margin[t])[mask].mean()),
                    }
                )
        if log and k % 20 == 0:
            log(f"상태 {k + 1}/{len(states)}")
    table.apply(net, off, off)
    return rows, per_image


def load_seed(seed_dir: Path, device) -> tuple[FoldedSixConv, QuantTable]:
    """완료된 seed의 학습 가중치와 보정 범위를 불러온다. 재학습·재보정 없이 조건만 추가할 때 사용."""
    orig = SixConv()
    orig.load_state_dict(torch.load(seed_dir / "model.pt", map_location="cpu"))
    net = FoldedSixConv(orig.to(device).eval()).to(device).eval()
    ranges = [torch.tensor(r, device=device) for r in json.loads((seed_dir / "ranges.json").read_text())]
    return net, QuantTable(net, ranges)


def run_seed(seed: int, args, data, labels, device) -> None:
    seed_dir = args.out / f"seed{seed}"
    if (seed_dir / "DONE").exists():
        print(f"seed {seed}: 이미 완료, 건너뜀", flush=True)
        return
    seed_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(message: str) -> None:
        print(f"[seed {seed} {time.time() - started:6.0f}s] {message}", flush=True)

    # 1. 학습
    torch.manual_seed(seed)
    orig = SixConv().to(device)
    train_model(orig, data["train"], labels["train"], args.epochs, device, log)
    torch.save(orig.state_dict(), seed_dir / "model.pt")

    # 2. folding과 검증. 차이가 크면 folding 구현이 틀린 것이므로 중단
    net = FoldedSixConv(orig).to(device).eval()
    fold = check_folding(orig, net, data["val"], device)
    (seed_dir / "fold_check.json").write_text(json.dumps(fold), encoding="utf-8")
    log(f"folding 검증 {fold}")
    assert fold["max_abs_logit_diff"] < 1e-2, fold

    # 3. 보정: folding된 모델의 FP32 순전파에서 범위 수집
    calib = data["calib"]
    ranges = calibrate(net, (to_float(calib[i:j]).to(device) for i, j in batches(len(calib), 256)))
    table = QuantTable(net, ranges)
    (seed_dir / "ranges.json").write_text(json.dumps([r.tolist() for r in ranges]), encoding="utf-8")

    # 4. 상태별 평가. rank와 test를 이어 붙여 한 번에 돌리고 집계만 나눈다
    states = build_states()
    x_eval, y_eval, masks = eval_set(data, labels)
    rows, per_image = evaluate(net, table, states, x_eval, y_eval, masks, seed, device, log=log)

    # 같은 설정에 이르는 두 경로(조건 all, MIX 조합)가 같은 결과인지 확인
    for t in TASKS:
        assert np.array_equal(per_image[f"W4A4|all|{t}|pred"], per_image[f"MIX|mix_444444|{t}|pred"])
        assert np.array_equal(per_image[f"W8A8|all|{t}|pred"], per_image[f"MIX|mix_888888|{t}|pred"])
    write_csv(seed_dir / "metrics.csv", rows)

    # 5. NMSE. restore 상태와 MIX는 순위 산출에 쓰지 않아 제외
    x_nmse = data["rank"][: args.n_nmse]
    nmse_rows = []
    for cond, state, w_bits, a_bits in states:
        if cond in ("FP32", "MIX") or state.startswith("restore"):
            continue
        for layer, value in enumerate(layer_nmse(net, table, w_bits, a_bits, x_nmse, device), start=1):
            nmse_rows.append({"seed": seed, "condition": cond, "state": state, "layer": layer, "nmse": value})
    write_csv(seed_dir / "nmse.csv", nmse_rows)

    np.savez_compressed(seed_dir / "per_image.npz", **per_image)
    (seed_dir / "DONE").write_text(f"{time.time() - started:.0f}s\n", encoding="utf-8")
    log("완료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-nmse", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("results/6conv_fold"))
    args = parser.parse_args()

    check_file()
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1), encoding="utf-8"
    )
    started = time.time()
    data, labels, split, classes = load_split()
    print(f"data {time.time() - started:.0f}s", {k: len(v) for k, v in split.items()}, flush=True)

    eval_idx = np.concatenate([split["rank"], split["test"]])
    np.savez_compressed(
        args.out / "eval_index.npz",
        index=eval_idx,
        set=np.array(["rank"] * len(split["rank"]) + ["test"] * len(split["test"])),
        factors=classes[eval_idx],
        factor_names=np.array(FACTORS),
    )
    for seed in args.seeds:
        run_seed(seed, args, data, labels, device)


if __name__ == "__main__":
    main()
