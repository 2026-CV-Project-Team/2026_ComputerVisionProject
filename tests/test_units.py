import unittest

import torch
from torchvision.models import resnet18

from src.imagenet_models import fold_resnet, load_model, unit_param_counts
from src.quant.quantizer import BITS
from src.quant.units import UnitQuantizer


def randomize_bn(model) -> None:
    """BN 통계가 기본값(평균 0, 분산 1)이면 folding 검증이 약해지므로 임의 값으로 채운다."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.5)


class UnitTest(unittest.TestCase):
    """가중치를 내려받지 않고 무작위 초기값 모델로 토글 동작만 확인한다."""

    def make(self, name):
        torch.manual_seed(0)
        model, units = load_model(name, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        q = UnitQuantizer(model, units)
        with torch.no_grad():
            ref = model(x)
        q.calibrate([x])
        return model, units, q, x, ref

    def run_capture(self, model, q, x, w, a):
        q.apply(w, a)
        q.capture = []
        with torch.no_grad():
            out = model(x)
        captured, q.capture = q.capture, None
        return out, captured

    def test_resnet_folding_matches_original(self):
        torch.manual_seed(0)
        model = resnet18().eval()
        randomize_bn(model)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            expected = model(x)
            out = fold_resnet(model)(x)
        self.assertTrue(torch.allclose(out, expected, atol=1e-3))
        self.assertTrue(torch.equal(out.argmax(1), expected.argmax(1)))

    def test_unit_counts(self):
        for name, n in (("resnet18", 10), ("deit_tiny", 14)):
            _, units = load_model(name, pretrained=False)
            self.assertEqual(len(units), n)
            self.assertFalse(units[-1].quant_act)  # 분류기 logit은 기본적으로 양자화하지 않음

    def test_all_weights_covered(self):
        # 모든 Conv/Linear 가중치가 정확히 한 단위에 속해야 한다(빠지거나 겹치면 안 됨)
        for name in ("resnet18", "deit_tiny"):
            model, units = load_model(name, pretrained=False)
            expected = {id(m) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))}
            covered = [id(m) for u in units for m in u.weights]
            self.assertEqual(len(covered), len(set(covered)))
            self.assertEqual(set(covered), expected)
            self.assertEqual(sum(unit_param_counts(units)), sum(m.weight.numel() for u in units for m in u.weights))

    def test_off_equals_fp32_and_restores(self):
        for name in ("resnet18", "deit_tiny"):
            model, units, q, x, ref = self.make(name)
            n = len(units)
            out4, _ = self.run_capture(model, q, x, [4] * n, [4] * n)
            self.assertFalse(torch.equal(out4, ref))
            out, _ = self.run_capture(model, q, x, [None] * n, [None] * n)
            self.assertTrue(torch.equal(out, ref))  # 양자화 후 FP32로 되돌리면 원래 출력과 완전히 같아야 함

    def test_only_unit_leaves_earlier_units_untouched(self):
        for name, k in (("resnet18", 4), ("deit_tiny", 6)):
            model, units, q, x, _ = self.make(name)
            n = len(units)
            _, ref = self.run_capture(model, q, x, [None] * n, [None] * n)
            w, a = [None] * n, [None] * n
            w[k], a[k] = 4, 4
            _, cap = self.run_capture(model, q, x, w, a)
            for i in range(k):
                self.assertTrue(torch.equal(ref[i], cap[i]), units[i].name)
            self.assertFalse(torch.equal(ref[k], cap[k]))

    def test_levels(self):
        for name in ("resnet18", "deit_tiny"):
            model, units, q, x, _ = self.make(name)
            n = len(units)
            for bits in BITS:
                _, cap = self.run_capture(model, q, x, [bits] * n, [bits] * n)
                for u, value in zip(units, cap):
                    if u.quant_act:
                        self.assertLessEqual(value.unique().numel(), 2**bits, u.name)
                    for module in u.weights:
                        # 채널 하나의 가중치는 최대 2^b - 1개 값만 가진다(대칭이라 -q_max ~ q_max)
                        row = module.weight[0].flatten()
                        self.assertLessEqual(row.unique().numel(), 2**bits - 1, u.name)


if __name__ == "__main__":
    unittest.main()
