import io
import tarfile
import unittest

import numpy as np
from PIL import Image

from src.exp_imagenet import parse_condition
from src.imagenet_data import PREP, preprocess, read_samples


def make_tar(samples) -> io.BytesIO:
    """실제 묶음과 같은 형식(.cls, .jpg, .json 순서)의 작은 tar를 메모리에 만든다."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, label, image in samples:
            for ext, data in (("cls", str(label).encode()), ("jpg", image), ("json", b"{}")):
                info = tarfile.TarInfo(f"{name}.{ext}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class ImageNetDataTest(unittest.TestCase):
    def test_read_samples(self):
        samples = [("a_1", 3, jpeg(40, 30)), ("b_2", 999, jpeg(30, 40))]
        got = list(read_samples(make_tar(samples)))
        self.assertEqual([(n, c) for n, c, _ in got], [("a_1", 3), ("b_2", 999)])
        self.assertEqual(got[0][2], samples[0][2])

    def test_preprocess_shape(self):
        # 가로·세로 비율이 달라도 모델 입력 크기(3, 224, 224)로 맞춰져야 한다
        for model in PREP:
            for w, h in ((500, 375), (300, 800)):
                out = preprocess(jpeg(w, h), model)
                self.assertEqual(out.shape, (3, 224, 224))
                self.assertEqual(out.dtype, np.uint8)

    def test_parse_condition(self):
        self.assertEqual(parse_condition("W4A8"), (4, 8))
        self.assertEqual(parse_condition("W32A4"), (None, 4))


if __name__ == "__main__":
    unittest.main()
