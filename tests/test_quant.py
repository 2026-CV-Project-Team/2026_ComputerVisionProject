import unittest

import torch
import torch.nn.functional as F

from src.models import N_LAYERS, SixConv
from src.quant.qmodel import FoldedSixConv, QuantTable, calibrate
from src.quant.quantizer import BITS


def make_nets():
    """BN 통계가 기본값(평균 0, 분산 1)이면 folding 검증이 약해지므로 임의 값으로 채운다."""
    torch.manual_seed(0)
    net = SixConv().eval()
    for bn in net.bns:
        bn.running_mean.normal_(0, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.normal_(0, 0.5)
    return net, FoldedSixConv(net).eval()


class QuantTest(unittest.TestCase):
    def setUp(self):
        self.orig, self.net = make_nets()
        self.x = torch.rand(16, 3, 64, 64) - 0.5
        self.table = QuantTable(self.net, calibrate(self.net, [self.x]))
        self.off = [None] * N_LAYERS

    def outputs(self, w, a):
        self.table.apply(self.net, w, a)
        self.net.capture = []
        with torch.no_grad():
            out = self.net(self.x)
        captured, self.net.capture = self.net.capture, None
        return out, captured

    def test_folding_matches_original(self):
        # folding 전후 FP32 출력이 같아야 한다(부동소수점 오차 범위)
        with torch.no_grad():
            expected = self.orig(self.x)
        out, _ = self.outputs(self.off, self.off)
        for t in out:
            self.assertTrue(torch.allclose(out[t], expected[t], atol=1e-4), t)
            self.assertTrue(torch.equal(out[t].argmax(1), expected[t].argmax(1)))

    def test_weight_levels_and_error(self):
        for bits, q_max in zip(BITS, (7, 15, 31, 127)):
            for i, w in enumerate(self.net.weights):
                s, q = self.table.w[bits][i]
                self.assertEqual(q, q_max)
                self.assertEqual(s.shape, (w.shape[0], 1, 1, 1))  # 출력 채널별 scale
                codes = torch.round(w / s).clamp(-q, q)
                # min-max라 잘리는 값이 없으므로 오차는 반 칸(s/2) 이내
                self.assertLessEqual(codes.abs().max().item(), q_max)
                self.assertTrue(((s * codes - w).abs() <= s / 2 + 1e-6).all())

    def test_activation_levels(self):
        for bits in BITS:
            _, captured = self.outputs(self.off, [bits] * N_LAYERS)
            for i, value in enumerate(captured):
                s, z, q = self.table.a[bits][i]
                self.assertEqual(q, 2**bits - 1)
                self.assertEqual(z.item(), 0)  # ReLU 출력이라 zero-point는 0
                self.assertLessEqual(value.unique().numel(), 2**bits)
                self.assertGreaterEqual(value.min().item(), 0.0)

    def test_only_layer_leaves_earlier_layers_untouched(self):
        _, ref = self.outputs(self.off, self.off)
        w, a = [None] * N_LAYERS, [None] * N_LAYERS
        w[3], a[3] = 4, 4
        _, cap = self.outputs(w, a)
        for i in range(3):
            self.assertTrue(torch.equal(ref[i], cap[i]))
        self.assertFalse(torch.equal(ref[3], cap[3]))

    def test_weight_quant_matches_manual_conv(self):
        w = [None] * N_LAYERS
        w[0] = 4
        _, cap = self.outputs(w, self.off)
        s, q = self.table.w[4][0]
        manual = F.conv2d(self.x, s * torch.round(self.net.weights[0] / s).clamp(-q, q), self.net.biases[0], padding=1)
        self.assertTrue(torch.allclose(cap[0], F.relu(manual)))


if __name__ == "__main__":
    unittest.main()
