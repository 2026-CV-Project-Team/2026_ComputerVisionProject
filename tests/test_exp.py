import unittest

import numpy as np
import torch

from src.analyze import PARAMS, frontier, total_bits
from src.exp_6conv import CONDITIONS, build_states, check_folding, condition_states, layer_nmse
from src.exp_domain import DOMAINS, factor_index, make_domain, swap_background
from src.exp_extra import parse_condition
from src.models import N_LAYERS, SixConv
from src.quant.qmodel import FoldedSixConv, QuantTable, calibrate


class ExpTest(unittest.TestCase):
    def test_states(self):
        states = build_states()
        names = [(c, s) for c, s, _, _ in states]
        self.assertEqual(len(names), len(set(names)))  # 중복 상태 없음
        # FP32 1 + 조건별(all 1 + 복원 6 + 단독 6) + Mixed 64
        self.assertEqual(len(states), 1 + len(CONDITIONS) * (1 + 2 * N_LAYERS) + 64)
        by_name = {(c, s): (w, a) for c, s, w, a in states}
        self.assertEqual(by_name[("W4A4", "restore_2")], ([4, None, 4, 4, 4, 4], [4, None, 4, 4, 4, 4]))
        self.assertEqual(by_name[("W32A4", "only_6")], ([None] * 6, [None] * 5 + [4]))
        self.assertEqual(by_name[("W6A6", "all")], ([6] * 6, [6] * 6))
        self.assertEqual(by_name[("MIX", "mix_844448")][0], [8, 4, 4, 4, 4, 8])

    def test_extra_condition(self):
        self.assertEqual(parse_condition("W5A5"), (5, 5))
        self.assertEqual(parse_condition("W32A4"), (None, 4))
        with self.assertRaises(SystemExit):
            parse_condition("W3A3")  # 준비되지 않은 비트 수
        states = condition_states("W5A5", 5, 5)
        self.assertEqual(len(states), 1 + 2 * N_LAYERS)
        self.assertEqual(states[1], ("W5A5", "restore_1", [None, 5, 5, 5, 5, 5], [None, 5, 5, 5, 5, 5]))

    def test_pareto(self):
        self.assertEqual(PARAMS.tolist(), [432, 2304, 4608, 9216, 18432, 36864])
        self.assertEqual(total_bits([4] * 6), 4 * 71856)
        self.assertEqual(total_bits([8, 4, 4, 4, 4, 4]), 4 * 71856 + 4 * 432)
        # 점 2는 점 1보다 비싸면서 정답률이 낮아 지배당한다. 점 3은 가장 비싸지만 정답률이 가장 높다
        keep = frontier(np.array([10, 20, 30, 40]), np.array([50.0, 80.0, 70.0, 90.0]))
        self.assertEqual(keep.tolist(), [True, True, False, True])

    def test_domain(self):
        # 사전식 번호: 마지막 속성(orientation)이 가장 빨리 바뀐다
        self.assertEqual(factor_index(np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [9, 9, 9, 7, 3, 14]])).tolist(),
                         [1, 15, 479999])
        factors = np.array([[1, 2, 3, 4, 1, 5]] * 50)
        blocked = factor_index(np.array([[0, 0, 3, 4, 1, 5]]))  # 이 이미지는 뽑히면 안 됨
        new = np.array(np.unravel_index(swap_background(factors, blocked), (10, 10, 10, 8, 4, 15))).T
        self.assertTrue((new[:, 2:] == factors[:, 2:]).all())  # 물체 쪽 속성은 그대로
        self.assertTrue((new[:, :2] != factors[:, :2]).all())  # 벽·바닥은 둘 다 바뀜
        self.assertFalse(((new[:, 0] == 0) & (new[:, 1] == 0)).any())
        x = torch.rand(4, 3, 64, 64)
        for name in DOMAINS:
            out = make_domain(name, x, x.flip(0))
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(out.min() >= -0.5 and out.max() <= 0.5)
        self.assertTrue(torch.equal(make_domain("orig", x, x), x - 0.5))
        # 같은 난수에 배율만 다르므로, 잘리지 않은 곳에서는 noise 크기가 σ에 비례
        mid = torch.full((1, 3, 64, 64), 0.5)
        d1, d3 = make_domain("noise_0.01", mid, mid), make_domain("noise_0.03", mid, mid)
        self.assertTrue(torch.allclose(d3, 3 * d1, atol=1e-6))

    def test_nmse_zero_when_off_and_fold_check(self):
        torch.manual_seed(0)
        orig = SixConv().eval()
        net = FoldedSixConv(orig).eval()
        x = (torch.rand(8, 3, 64, 64) * 255).byte()
        device = torch.device("cpu")
        table = QuantTable(net, calibrate(net, [x.float() / 255 - 0.5]))
        off = [None] * N_LAYERS
        self.assertEqual(layer_nmse(net, table, off, off, x, device), [0.0] * N_LAYERS)
        on = layer_nmse(net, table, [4] * N_LAYERS, [4] * N_LAYERS, x, device)
        self.assertTrue(all(v > 0 for v in on))
        table.apply(net, off, off)
        fold = check_folding(orig, net, x, device)
        self.assertLess(fold["max_abs_logit_diff"], 1e-4)
        self.assertEqual(fold["pred_changed"], 0)


if __name__ == "__main__":
    unittest.main()
