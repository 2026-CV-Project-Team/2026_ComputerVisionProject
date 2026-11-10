"""6-Conv FP32 모델 학습.

학습은 양자화할 원본 모델을 만드는 준비 단계다. 사전학습 가중치가 없는 자체 구조라 필요하며,
양자화 이후에는 재학습하지 않는다(PTQ).

사용 예:
    python -m src.train --seed 0 --out results/fp32
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import TASKS, batches, check_file, load_split, to_float
from .models import SixConv


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(net: SixConv, x: torch.Tensor, y: dict, epochs: int, device, log=print) -> None:
    """Adam(lr 1e-3) + cosine, batch 256. 세 과제의 cross-entropy를 같은 비중으로 더한다.

    호출 전에 torch.manual_seed(seed)를 걸어 두면 초기값과 학습 순서가 seed로 정해진다.
    학습 설정(epoch 20, lr 등)은 파일럿에서 FP32 정답률이 100%에 도달한 값으로, 잠정값이다.
    """
    n = len(x)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs * (n // 256 + 1))
    for epoch in range(epochs):
        net.train()
        started = time.time()
        order = torch.randperm(n)
        total = 0.0
        for i, j in batches(n, 256):
            idx = order[i:j]
            out = net(to_float(x[idx]).to(device))
            loss = sum(F.cross_entropy(out[t], y[t][idx].to(device)) for t in TASKS)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total += loss.item() * len(idx)
        log(f"  epoch {epoch} loss {total / n:.4f} {time.time() - started:.0f}s")
    net.eval()


@torch.no_grad()
def accuracy(net, x: torch.Tensor, y: dict, device) -> dict[str, float]:
    """과제별 정답률(%)."""
    correct = {t: 0 for t in TASKS}
    for i, j in batches(len(x), 1024):
        out = net(to_float(x[i:j]).to(device))
        for t in TASKS:
            correct[t] += (out[t].argmax(1).cpu() == y[t][i:j]).sum().item()
    return {t: correct[t] / len(x) * 100 for t in TASKS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/fp32"))
    args = parser.parse_args()

    check_file()
    device = pick_device()
    data, labels, _, _ = load_split(["train", "val"])
    torch.manual_seed(args.seed)
    net = SixConv().to(device)
    train_model(net, data["train"], labels["train"], args.epochs, device)
    print("val 정답률(%)", {t: round(v, 2) for t, v in accuracy(net, data["val"], labels["val"], device).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out / f"seed{args.seed}.pt")


if __name__ == "__main__":
    main()
