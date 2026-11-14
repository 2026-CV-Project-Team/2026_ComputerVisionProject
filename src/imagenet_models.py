"""ImageNet-1K 사전학습 모델 로드 + 양자화 단위 정의.

모델(이하형 문서 11-10 확정)
- ResNet-18: torchvision, IMAGENET1K_V1 가중치
- DeiT-Tiny: timm `deit_tiny_patch16_224`
- 재학습 없이 사전학습 가중치를 그대로 사용한다.

단위(복원·교체 실험의 단위, 문서 확정)
- ResNet-18: Stem, BasicBlock 8개, FC  -> 10개
- DeiT-Tiny: Patch Embed, Block 12개, Head -> 14개
- 첫·마지막 단위도 다른 단위와 같은 비트 조건을 적용한다(8비트 고정 없음).

잠정 가정(문서에 규정 없음, 팀 확인 필요)
- Stem의 출력은 MaxPool 뒤로 본다(6-Conv의 "MaxPool이 있으면 MaxPool 뒤" 규칙을 따름).
- FC/Head는 가중치만 양자화하고 출력(logit)은 양자화하지 않는다.
  logit을 4비트(16단계)로 자르면 여러 클래스가 같은 값이 되어 순위가 무너지는데,
  이는 층의 민감도라기보다 출력 표현의 문제라서 기본값에서 뺐다. quant_logits=True로 켤 수 있다.
- DeiT의 Patch Embed 출력은 projection 직후(cls token, 위치 임베딩을 더하기 전)로 본다.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .quant.units import Unit

MODELS = ("resnet18", "deit_tiny")

# 두 모델 모두 ImageNet 표준 정규화를 쓴다. 입력 이미지는 양자화 대상이 아니다.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def fold_resnet(model: nn.Module) -> nn.Module:
    """ResNet의 모든 Conv-BN 쌍을 합친다(6-Conv와 같은 순서: folding 후 보정·양자화).

    BN은 추론 때 채널별 곱셈·덧셈이라 바로 앞 Conv의 가중치와 bias에 미리 합칠 수 있다.
    합친 뒤 BN 자리는 Identity로 바꿔 순전파 코드는 그대로 둔다.
    """
    model.eval()
    model.conv1, model.bn1 = fuse_conv_bn_eval(model.conv1, model.bn1), nn.Identity()
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            block.conv1, block.bn1 = fuse_conv_bn_eval(block.conv1, block.bn1), nn.Identity()
            block.conv2, block.bn2 = fuse_conv_bn_eval(block.conv2, block.bn2), nn.Identity()
            if block.downsample is not None:
                # downsample = [1x1 Conv, BN]. 합친 Conv 하나만 남긴다
                block.downsample = nn.Sequential(fuse_conv_bn_eval(block.downsample[0], block.downsample[1]))
    return model


def resnet18_units(model: nn.Module, quant_logits: bool = False) -> list[Unit]:
    units = [Unit("stem", model.maxpool, [model.conv1])]
    for li, layer in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), start=1):
        for bi, block in enumerate(layer):
            convs = [block.conv1, block.conv2]
            if block.downsample is not None:
                convs.append(block.downsample[0])  # shortcut의 1x1 Conv도 같은 블록에 포함
            # 블록 출력 = shortcut을 더한 뒤 ReLU까지 지난 값(BasicBlock의 forward 출력)
            units.append(Unit(f"layer{li}.{bi}", block, convs))
    units.append(Unit("fc", model.fc, [model.fc], quant_act=quant_logits))
    return units


def deit_units(model: nn.Module, quant_logits: bool = False) -> list[Unit]:
    units = [Unit("patch_embed", model.patch_embed, [model.patch_embed.proj])]
    for i, block in enumerate(model.blocks):
        # attention의 qkv·proj, MLP의 fc1·fc2. LayerNorm에는 양자화할 가중치 행렬이 없다
        linears = [block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2]
        units.append(Unit(f"block{i}", block, linears))
    units.append(Unit("head", model.head, [model.head], quant_act=quant_logits))
    return units


def load_model(name: str, pretrained: bool = True, quant_logits: bool = False) -> tuple[nn.Module, list[Unit]]:
    """(추론 전용 모델, 단위 목록). pretrained=False는 테스트용(가중치를 내려받지 않음)."""
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model = fold_resnet(model)
        units = resnet18_units(model, quant_logits)
    elif name == "deit_tiny":
        import timm

        model = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained).eval()
        units = deit_units(model, quant_logits)
    else:
        raise ValueError(f"모르는 모델: {name} (가능: {MODELS})")
    for p in model.parameters():
        p.requires_grad_(False)  # PTQ라 학습하지 않는다
    return model, units


def unit_param_counts(units: list[Unit]) -> list[int]:
    """단위별 가중치 수. Weight Bit Cost(B = Σ N_i · b_i) 계산에 쓴다."""
    return [sum(w.numel() for w in u.fp) if u.fp else sum(m.weight.numel() for m in u.weights) for u in units]
