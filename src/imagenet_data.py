"""ImageNet-1K 평가·보정 데이터 준비 (ResNet-18, DeiT-Tiny 공용).

데이터 출처: Hugging Face `timm/imagenet-1k-wds` (약관 동의와 로그인 필요, README 참고)
- 묶음(tar) 안의 샘플 = {이름}.jpg + {이름}.cls(정답 번호 0~999) + {이름}.json

분할(이하형 문서 11-10 확정: 보정 train 2,048 · rank val 5,000 · 최종 평가 val 10,000, 서로 겹치지 않음)
- calib: train에서 2,048장. 보정(활성값 범위 수집)에만 쓴다. val은 보정에 쓰지 않는다.
- rank:  val에서 5,000장. 층 순위·정책 선택용.
- test:  val에서 10,000장. 최종 보고용. rank와 겹치지 않는다.
- 두 모델이 같은 원본 이미지를 쓴다.

잠정 가정(문서에 규정 없음, 팀 확인 필요)
- rank/test는 클래스 균형으로 뽑는다(rank 클래스당 5장, test 클래스당 10장). val은 클래스당 50장이다.
- calib은 train 1,024개 묶음 중 고정 seed로 고른 16개 묶음에서 앞 128장씩 가져온다.
  train 전체(약 154GB)를 받지 않으려고 묶음의 앞부분만 스트리밍으로 읽는다.
- 전처리는 모델별 공식 설정을 따른다(같은 원본 이미지, 다른 크기 조정 방식).
  ResNet-18: 짧은 변 256(bilinear) -> 가운데 224. DeiT-Tiny: 짧은 변 248(bicubic) -> 가운데 224.
  공식 설정을 써야 FP32 정확도가 알려진 수치와 비교 가능하기 때문이다.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
REPO = "timm/imagenet-1k-wds"
RAW = ROOT / "data" / "imagenet" / "raw"  # val 묶음 원본
CALIB_DIR = ROOT / "data" / "imagenet" / "calib"  # 보정용 JPEG
CACHE = ROOT / "data" / "imagenet" / "cache"  # 전처리한 uint8 배열
SPLITS = ROOT / "data" / "splits"  # 분할 목록(커밋 대상)

N_CLASSES = 1000
PER_CLASS = {"rank": 5, "test": 10}
N_CALIB, CALIB_SHARDS, N_TRAIN_SHARDS = 2048, 16, 1024

# 모델별 (크기 조정 후 짧은 변, 보간법). 가운데 224x224를 잘라 쓴다
PREP = {
    "resnet18": (256, InterpolationMode.BILINEAR),
    "deit_tiny": (248, InterpolationMode.BICUBIC),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_samples(fileobj):
    """묶음을 앞에서부터 읽으며 (이름, 정답, JPEG bytes)를 내놓는다. 스트리밍 입력도 가능."""
    name, label, image = None, None, None
    with tarfile.open(fileobj=fileobj, mode="r|") as tar:
        for member in tar:
            stem, ext = member.name.rsplit(".", 1)
            if stem != name:  # 새 샘플 시작
                name, label, image = stem, None, None
            if ext == "cls":
                label = int(tar.extractfile(member).read())
            elif ext == "jpg":
                image = tar.extractfile(member).read()
            if label is not None and image is not None:
                yield name, label, image
                label, image = None, None


def select_val(seed: int = 0) -> dict[str, dict[str, int]]:
    """val 50,000장에서 rank/test를 클래스 균형으로 고른다. 반환: {분할: {이름: 정답}}"""
    by_class: dict[int, list[str]] = {c: [] for c in range(N_CLASSES)}
    for path in sorted(RAW.glob("imagenet1k-validation-*.tar")):
        with tarfile.open(path) as tar:  # 정답 파일(.cls)만 읽는다
            for member in tar:
                if member.name.endswith(".cls"):
                    by_class[int(tar.extractfile(member).read())].append(member.name[:-4])
    assert all(len(v) == 50 for v in by_class.values()), "val 묶음이 다 받아지지 않았음"
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, int]] = {"rank": {}, "test": {}}
    for c in range(N_CLASSES):
        names = sorted(by_class[c])  # 파일 순서에 의존하지 않도록 정렬 후 섞는다
        order = rng.permutation(len(names))
        at = 0
        for split, k in PER_CLASS.items():
            for j in order[at : at + k]:
                out[split][names[j]] = c
            at += k
    return out


def fetch_calib(seed: int = 0) -> dict[str, int]:
    """train 묶음 16개의 앞부분에서 128장씩 스트리밍으로 받아 JPEG로 저장."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(np.random.default_rng(seed).choice(N_TRAIN_SHARDS, CALIB_SHARDS, replace=False).tolist())
    per_shard = N_CALIB // CALIB_SHARDS
    picked: dict[str, int] = {}
    for shard in shards:
        with fs.open(f"datasets/{REPO}/imagenet1k-train-{shard:04d}.tar", "rb") as f:
            for k, (name, label, image) in enumerate(read_samples(f)):
                if k >= per_shard:
                    break  # 필요한 만큼만 읽고 연결을 닫는다
                (CALIB_DIR / f"{name}.jpg").write_bytes(image)
                picked[name] = label
        print(f"calib 묶음 {shard:04d}: 누적 {len(picked)}장", flush=True)
    return picked


def preprocess(image_bytes: bytes, model: str) -> np.ndarray:
    """JPEG bytes -> (3, 224, 224) uint8. 정규화는 배치 단위로 나중에 한다(메모리 절약)."""
    size, interp = PREP[model]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # 흑백·CMYK 이미지도 RGB로 통일
    img = transforms.CenterCrop(224)(transforms.Resize(size, interpolation=interp)(img))
    return np.asarray(img).transpose(2, 0, 1)


def build_cache() -> None:
    """분할 목록에 있는 이미지를 모델별 전처리로 미리 변환해 저장한다.

    상태 수가 많아 매번 JPEG를 다시 풀면 느리므로 uint8 배열로 한 번만 만들어 둔다.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    lists = {s: json.loads((SPLITS / f"imagenet_{s}.json").read_text()) for s in ("calib", "rank", "test")}
    order = {s: {name: i for i, name in enumerate(sorted(v))} for s, v in lists.items()}
    arrays = {
        (m, s): np.lib.format.open_memmap(
            CACHE / f"{m}_{s}_x.npy", mode="w+", dtype=np.uint8, shape=(len(lists[s]), 3, 224, 224)
        )
        for m in PREP
        for s in lists
    }
    for s, v in lists.items():
        np.save(CACHE / f"{s}_y.npy", np.array([v[name] for name in sorted(v)], dtype=np.int64))

    def put(split: str, name: str, image: bytes) -> None:
        for m in PREP:
            arrays[(m, split)][order[split][name]] = preprocess(image, m)

    for name in lists["calib"]:
        put("calib", name, (CALIB_DIR / f"{name}.jpg").read_bytes())
    done = 0
    for path in sorted(RAW.glob("imagenet1k-validation-*.tar")):
        with open(path, "rb") as f:
            for name, _, image in read_samples(f):
                for s in ("rank", "test"):
                    if name in order[s]:
                        put(s, name, image)
                        done += 1
        print(f"{path.name}: 누적 {done}장", flush=True)
    assert done == len(lists["rank"]) + len(lists["test"])
    for a in arrays.values():
        a.flush()


def load_split(model: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(uint8 이미지 텐서, 정답). 이미지는 파일에 매핑된 상태라 메모리를 거의 쓰지 않는다."""
    x = np.load(CACHE / f"{model}_{split}_x.npy", mmap_mode="r")
    y = np.load(CACHE / f"{split}_y.npy")
    return torch.from_numpy(x), torch.from_numpy(y)


def to_input(batch: torch.Tensor, device) -> torch.Tensor:
    """uint8 [0, 255] -> ImageNet 표준 정규화 실수. 입력 이미지는 양자화 대상이 아니다."""
    x = batch.to(device).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calib", action="store_true", help="보정용 train 2,048장 받기")
    p.add_argument("--val", action="store_true", help="받아 둔 val 묶음에서 rank/test 목록 만들기")
    p.add_argument("--cache", action="store_true", help="전처리 배열 만들기")
    args = p.parse_args()
    SPLITS.mkdir(parents=True, exist_ok=True)
    if args.calib:
        (SPLITS / "imagenet_calib.json").write_text(json.dumps(fetch_calib(), indent=0, sort_keys=True))
    if args.val:
        for split, items in select_val().items():
            (SPLITS / f"imagenet_{split}.json").write_text(json.dumps(items, indent=0, sort_keys=True))
            print(split, len(items))
    if args.cache:
        build_cache()


if __name__ == "__main__":
    main()
