"""최종 출력 오차 지표 확인."""
import unittest

import torch

from src.exp_imagenet_output import output_error as output_error_mean
from src.exp_output_error import output_error


class OutputErrorTest(unittest.TestCase):
    def test_same_logits_zero(self):
        z = torch.randn(8, 10)
        nmse, kl = output_error(z, z.clone())
        self.assertTrue((abs(nmse) < 1e-7).all() and (abs(kl) < 1e-6).all())

    def test_nmse_formula(self):
        z = torch.tensor([[3.0, 4.0]])
        nmse, _ = output_error(z, torch.tensor([[3.0, 0.0]]))
        self.assertAlmostEqual(float(nmse[0]), 16 / 25, places=6)

    def test_kl_shift_invariant(self):
        # logit 전체에 같은 값을 더하면 softmax가 같으므로 KL은 0, NMSE는 0이 아니다
        z = torch.randn(4, 1000)
        nmse, kl = output_error_mean(z, z + 2.0)
        self.assertGreater(nmse, 0)
        self.assertLess(abs(kl), 1e-5)


if __name__ == "__main__":
    unittest.main()
