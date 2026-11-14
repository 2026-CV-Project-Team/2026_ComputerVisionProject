"""모의 양자화(fake quantization) 규칙: 양자화한 값을 곧바로 실수로 되돌린다.

    q  = clamp(round(x / s) + z, q_min, q_max)
    x̂ = s · (q − z)

확정 규칙(이하형 문서 11-10)
- 가중치 W:  대칭(symmetric), 출력 채널별(per-channel), min-max
              q 범위는 ±(2^(b-1) − 1), z = 0. 예: 8비트 ±127, 4비트 ±7
- 활성값 A:  비대칭(asymmetric), 텐서 전체에 하나(per-tensor), 보정 데이터의 min-max
              q 범위는 0 ~ 2^b − 1
- 반올림:    round-to-nearest. 재학습 없음(PTQ)

잠정 가정
- folding으로 생긴 bias는 양자화하지 않고 FP32로 둔다(문서에 규정 없음).
"""
from __future__ import annotations

import torch

# 실험에 쓰는 비트 수. 6비트는 4와 8 사이 민감도 확인용으로 추가됨.
# 5비트는 6비트(100%)와 4비트(급락) 사이에서 어디부터 무너지는지 보려고 추가(이하형 문서 11-10 2차)
BITS = (4, 5, 6, 8)


def weight_qparams(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    """가중치의 (채널별 scale, q_max).

    scale shape은 Conv면 (out_ch, 1, 1, 1), Linear면 (out_features, 1).
    Linear는 출력 뉴런 하나(행 하나)를 채널 하나로 본다(ResNet의 FC, DeiT의 Linear 층용).
    """
    q_max = 2 ** (bits - 1) - 1
    # 채널의 절댓값 최댓값이 정확히 q_max에 대응하도록 scale을 정한다
    dims = tuple(range(1, weight.ndim))  # 출력 채널(0번 축)을 뺀 나머지 축
    scale = weight.detach().abs().amax(dim=dims, keepdim=True) / q_max
    return scale.clamp(min=1e-12), q_max  # 0으로 나누기 방지


def fake_quant_weight(weight: torch.Tensor, scale: torch.Tensor, q_max: int) -> torch.Tensor:
    return scale * torch.round(weight / scale).clamp(-q_max, q_max)


def act_qparams(min_val: torch.Tensor, max_val: torch.Tensor, bits: int):
    """활성값의 (scale, zero-point, q_max).

    범위에 0이 반드시 포함되게 한다. 0이 정확히 표현되지 않으면 ReLU가 만든 0들이
    전부 오차를 갖게 되기 때문이다. ReLU 출력은 min이 0이라 zero-point도 0이 된다.
    """
    q_max = 2**bits - 1
    min_val, max_val = min_val.clamp(max=0.0), max_val.clamp(min=0.0)
    scale = ((max_val - min_val) / q_max).clamp(min=1e-12)
    zero = torch.round(-min_val / scale)
    return scale, zero, q_max


def fake_quant_act(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, q_max: int) -> torch.Tensor:
    # 보정 범위를 벗어난 값은 clamp에서 잘린다(clipping)
    return scale * ((torch.round(x / scale) + zero).clamp(0, q_max) - zero)
