import unittest

import numpy as np
import torch

from src.data import SPLIT_SIZES, TASKS, fixed_split
from src.models import CHANNELS, SixConv


class ModelTest(unittest.TestCase):
    def test_output_shapes(self):
        net = SixConv().eval()
        with torch.no_grad():
            out = net(torch.rand(2, 3, 64, 64))
        for task, (_, n_class) in TASKS.items():
            self.assertEqual(out[task].shape, (2, n_class))

    def test_three_pools_give_8x8(self):
        # MaxPool은 Conv2·4·6 뒤에만 있으므로 64 -> 32 -> 16 -> 8
        net = SixConv().eval()
        x = torch.rand(1, 3, 64, 64)
        sizes = []
        with torch.no_grad():
            for i, (conv, bn) in enumerate(zip(net.convs, net.bns)):
                x = torch.relu(bn(conv(x)))
                if i in (1, 3, 5):
                    x = torch.max_pool2d(x, 2)
                sizes.append(tuple(x.shape[1:]))
        self.assertEqual([s[0] for s in sizes], CHANNELS)
        self.assertEqual([s[1] for s in sizes], [64, 32, 32, 16, 16, 8])

    def test_split_disjoint_and_fixed(self):
        split = fixed_split()
        joined = np.concatenate(list(split.values()))
        self.assertEqual(len(joined), len(set(joined.tolist())))  # 묶음 간 겹침 없음
        self.assertEqual({k: len(v) for k, v in split.items()}, SPLIT_SIZES)
        again = fixed_split()
        for k in split:  # 다시 불러도 같은 분할
            self.assertTrue(np.array_equal(split[k], again[k]))


if __name__ == "__main__":
    unittest.main()
