# 실험 설명 자료

발표 앞부분에서 양자화와 실험 방법을 쉽게 설명하려고 만든 자료.

| 파일 | 내용 |
|---|---|
| `양자화_설명영상_v2.mp4` | 설명 영상 (2분 46초, 1920x1080). 6-Conv 실험 흐름 + ImageNet 확장 결과 |
| `대본.txt` | 영상 자막 전체와 시각 (render.py가 자동으로 만듦) |
| `render.py` | 영상 렌더러. 장면을 PIL로 그려서 ffmpeg로 묶음 |
| `3DShapes_양자화_쉬운설명_v5.pdf` | 같은 내용을 슬라이드로 정리한 것 |

## 영상 순서

1. 문제 소개: 3D Shapes 그림의 모양, 색, 크기 맞히기
2. 합성곱 필터가 하는 일, 6개 층을 차례로 지나는 과정
3. 양자화: 숫자를 정해진 눈금에 맞춰 짧게 줄이기, 4비트면 크기 1/8
4. 결과: 4비트로 줄이면 정확도 100% → 58%
5. 방법 1(층별 오차 측정): 오차는 6번 층이 가장 큼
6. 방법 2(한 층만 원래 숫자로 되돌리기): 효과는 1번 층이 가장 큼(+19%p), 6번 층은 거의 없음
7. 이유: 앞 층에서 생긴 오차가 뒤로 넘어가며 쌓임
8. 활용: 1번 층만 8비트로 두면 크기 +0.6%, 정확도 58% → 74%
9. ImageNet 확장: ResNet-18 4비트 19% vs 55%, DeiT-Tiny 5비트 25% vs 33%

영상 속 수치는 `results/6conv_fold/summary/`, `results/imagenet/summary/`(PR #10, #11) 값을 반올림한 것.

## 다시 만들기

그림·소리 원본은 재배포 조건 때문에 레포에 넣지 않았다. 다시 렌더링하려면 아래를 `docs/explainer/자료/` 아래에 따로 받아 둬야 한다.

- `자료/cut/`: いらすとや 그림 (배경 투명 처리본)
- `자료/shapes/`, `자료/fm/`: 3D Shapes 예시 이미지와 층별 출력 그림
- 효과음 폴더 (`SFX_DIR` 환경 변수), 배경음 mp3 (`BGM` 환경 변수)
- Pretendard 글꼴 (`~/Library/Fonts`)

```
pip install pillow numpy scipy
FFMPEG=/path/to/ffmpeg SFX_DIR=/path/to/sfx BGM=/path/to/Wallpaper.mp3 python render.py
python render.py --frames 12,40   # 해당 시각 프레임만 PNG로
python render.py --sheet          # 장면별 미리보기 한 장
```

## 출처

- 데이터: 3D Shapes (DeepMind), ImageNet
- 모델: ResNet-18 (torchvision), DeiT-Tiny (timm)
- 그림: いらすとや (irasutoya)
- 효과음: 効果音ラボ (soundeffect-lab.info)
- 배경음: "Wallpaper" Kevin MacLeod (incompetech.com), CC BY 4.0
