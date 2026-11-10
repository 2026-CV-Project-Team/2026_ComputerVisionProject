"""양자화 실험용 6-Conv: BN을 합친 모델 + 층 단위 양자화 토글.

"층 i"의 정의(이하형 문서 11-10 확정)
- Conv i의 가중치 양자화기 + 그 블록 출력 활성값 양자화기를 한 단위로 묶는다.
- 블록 출력 = ReLU 뒤, MaxPool이 있는 블록은 MaxPool 뒤.
- W와 A를 따로 켜고 끌 수도 있어서 W4A32, W32A4 조건으로 영향을 분리해 볼 수 있다.
- 입력 이미지와 Head(GAP, Linear)는 양자화하지 않는다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..models import N_LAYERS, POOL_AFTER, SixConv
from .fold import fold_conv_bn
from .quantizer import BITS, act_qparams, fake_quant_act, fake_quant_weight, weight_qparams


class FoldedSixConv(nn.Module):
    """학습된 SixConv에서 BN을 Conv에 합쳐 만든 추론 전용 모델."""

    def __init__(self, net: SixConv) -> None:
        super().__init__()
        folded = [fold_conv_bn(conv, bn) for conv, bn in zip(net.convs, net.bns)]
        self.weights = nn.ParameterList([nn.Parameter(w, requires_grad=False) for w, _ in folded])
        self.biases = nn.ParameterList([nn.Parameter(b, requires_grad=False) for _, b in folded])
        self.heads = net.heads
        # 층별 양자화 설정. None이면 그 층의 해당 양자화기를 끈다(FP32)
        self.w_q: list[tuple | None] = [None] * N_LAYERS  # (scale, q_max)
        self.a_q: list[tuple | None] = [None] * N_LAYERS  # (scale, zero, q_max)
        self.observe: list[torch.Tensor] | None = None  # 보정 중일 때 층별 [min, max] 누적
        self.capture: list[torch.Tensor] | None = None  # NMSE 계산용 층별 출력 수집

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for i in range(N_LAYERS):
            w = self.weights[i]
            if self.w_q[i] is not None:
                w = fake_quant_weight(w, *self.w_q[i])
            x = F.relu(F.conv2d(x, w, self.biases[i], padding=1))
            if i in POOL_AFTER:
                x = F.max_pool2d(x, 2)
            if self.observe is not None:
                # 보정은 양자화를 모두 끈 FP32 순전파에서 수행한다
                self.observe[i] = torch.stack(
                    [torch.minimum(self.observe[i][0], x.min()), torch.maximum(self.observe[i][1], x.max())]
                )
            if self.a_q[i] is not None:
                x = fake_quant_act(x, *self.a_q[i])
            if self.capture is not None:
                self.capture.append(x)  # 모든 상태에서 같은 위치(활성값 양자화기 뒤)를 비교
        x = x.mean(dim=(2, 3))
        return {t: head(x) for t, head in self.heads.items()}


@torch.no_grad()
def calibrate(net: FoldedSixConv, batch_iter) -> list[torch.Tensor]:
    """FP32 순전파로 층별 활성값 [min, max]를 수집. batch_iter는 입력 텐서를 내놓는다."""
    device = net.weights[0].device
    net.w_q, net.a_q = [None] * N_LAYERS, [None] * N_LAYERS
    net.observe = [torch.tensor([float("inf"), float("-inf")], device=device) for _ in range(N_LAYERS)]
    for x in batch_iter:
        net(x)
    ranges, net.observe = net.observe, None
    return ranges


class QuantTable:
    """비트 수별 양자화 파라미터를 층마다 미리 계산해 두고, 상태 전환 때 꺼내 쓴다.

    상태 수가 많아서(143개) 매번 다시 계산하지 않도록 한 것이다.
    """

    def __init__(self, net: FoldedSixConv, ranges: list[torch.Tensor], bits=BITS) -> None:
        self.w = {b: [weight_qparams(w, b) for w in net.weights] for b in bits}
        self.a = {b: [act_qparams(r[0], r[1], b) for r in ranges] for b in bits}

    def apply(self, net: FoldedSixConv, w_bits: list, a_bits: list) -> None:
        """층별 비트 수 목록을 모델에 적용. None은 FP32."""
        for i in range(N_LAYERS):
            net.w_q[i] = None if w_bits[i] is None else self.w[w_bits[i]][i]
            net.a_q[i] = None if a_bits[i] is None else self.a[a_bits[i]][i]
