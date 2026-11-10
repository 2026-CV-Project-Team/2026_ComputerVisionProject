"""6-Conv Multi-Head CNN (학습용 FP32 모델).

구조 (이하형 문서 11-10 확정)
- Conv 6개: 채널 16/16/32/32/64/64, 3x3, padding 1, bias 없음(BN이 bias 역할)
- 각 Conv 뒤 BatchNorm, ReLU
- MaxPool(2)은 Conv2·4·6 뒤에만 적용 -> 64x64 입력이 8x8 특징맵이 됨
- Head: Global Average Pooling 뒤 과제별 Linear 3개 (Shape 4종, Color 10종, Scale 8종)

이 클래스는 학습에만 쓴다. 양자화 실험은 BN을 Conv에 합친(folding) 뒤의
모델(src/quant/qmodel.py)에서 수행한다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .data import TASKS

CHANNELS = [16, 16, 32, 32, 64, 64]
N_LAYERS = len(CHANNELS)
POOL_AFTER = (1, 3, 5)  # 0부터 센 Conv 번호. Conv2·4·6 뒤에 MaxPool


class SixConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = 3
        for ch in CHANNELS:
            self.convs.append(nn.Conv2d(in_ch, ch, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(ch))
            in_ch = ch
        self.heads = nn.ModuleDict({t: nn.Linear(CHANNELS[-1], n) for t, (_, n) in TASKS.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = F.relu(bn(conv(x)))
            if i in POOL_AFTER:
                x = F.max_pool2d(x, 2)
        x = x.mean(dim=(2, 3))  # GAP: (N, 64, 8, 8) -> (N, 64)
        return {t: head(x) for t, head in self.heads.items()}
