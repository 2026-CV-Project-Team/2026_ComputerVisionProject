"""Conv-BN folding: BatchNorm을 앞의 Conv에 합친다.

추론 때 BN은 채널별 고정 선형변환이라 Conv의 가중치와 bias에 흡수할 수 있다.
    BN(conv(x)) = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
              = conv_folded(x) + bias_folded
    W_folded = W * gamma / sqrt(var + eps)        (출력 채널별로 곱)
    b_folded = beta - mean * gamma / sqrt(var + eps)

folding을 먼저 하는 이유(이하형 문서 11-10 확정):
실제 배포되는 정수 추론에서는 BN이 Conv에 합쳐진 가중치를 양자화한다.
합치지 않고 Conv 가중치만 양자화하면 BN의 채널별 배율이 오차를 키우거나 줄여
실제와 다른 결과가 나올 수 있다.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """(folding된 가중치, folding된 bias)를 반환. BN은 eval 통계(running mean/var)를 쓴다."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # 채널별 gamma / sqrt(var + eps)
    weight = conv.weight * scale[:, None, None, None]
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    bias = bn.bias + (conv_bias - bn.running_mean) * scale
    return weight.clone(), bias.clone()
