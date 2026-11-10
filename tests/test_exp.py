import unittest

import torch

from src.exp_6conv import CONDITIONS, build_states, check_folding, condition_states, layer_nmse
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
