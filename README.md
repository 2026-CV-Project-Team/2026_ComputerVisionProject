# 2026 Computer Vision Project

층별 양자화 오차 vs FP32 복원 효과

팀원: 김진서, 이하형, 정채호준

## 주제

재학습 없이 양자화(PTQ)했을 때, 양자화 오차(NMSE)가 큰 층이랑 FP32로 되돌렸을 때 정확도가 많이 회복되는 층이 같은 층인지 본다.
오차가 큰 층이 꼭 예측을 망치는 층은 아닐 수 있다는 게 가설.

실험은 두 단계로 나눠서 진행한다.

1. 6-Conv CNN + 3D-Shapes : 직접 학습한 작은 모델. Shape / Color / Scale 속성별로 어느 층이 영향을 주는지 확인 (진행 중)
2. ResNet-18, DeiT-Tiny + ImageNet-1K : 사전학습 모델에서도 같은 현상이 나오는지 확인 (아직 구현 안 함)

자세한 설정값은 docs/experiment_spec.md 참고. 아직 확정 안 된 값은 잠정이라고 적어놨음.

## 환경

```bash
pip install -r requirements.txt
```

3D-Shapes 데이터(3dshapes.h5, 약 255MB)는 용량 때문에 레포에 안 올렸다. 아래에서 받아서 `data/3dshapes.h5`로 넣으면 됨.

- https://github.com/deepmind/3dshapes-dataset
- https://storage.googleapis.com/3d-shapes/3dshapes.h5

다른 경로에 두고 싶으면 환경변수 `SHAPES3D_H5`에 경로를 넣으면 된다. 실행할 때 파일 해시(SHA-256)를 확인하니까 다른 파일이면 바로 에러 남.

## 실행 방법

```bash
# 테스트
python -m unittest discover -s tests -t .

# FP32 학습만
python -m src.train --seed 0

# 본 실험 (seed 하나에 M4 맥 기준 13분 정도, 끝난 seed는 건너뜀)
python -m src.exp_6conv --seeds 0 1 2 3 4

# 조건 추가 (학습해둔 모델 그대로 쓰고 조건만 더 돌림, seed 하나에 1분 안쪽)
python -m src.exp_extra --conditions W5A5 --seeds 0 1 2 3 4

# 결과 집계 -> results/6conv_fold/summary/
python -m src.analyze --exp results/6conv_fold
```

## 폴더

```
src/
  data.py         데이터 로드, 분할 (모든 seed 동일)
  models.py       6-Conv 모델
  train.py        FP32 학습
  quant/
    fold.py       Conv-BN folding
    quantizer.py  fake quantization
    qmodel.py     folding된 모델 + 층별 on/off, calibration
  exp_6conv.py    본 실험
  exp_extra.py    조건 추가 (W5A5 등)
  analyze.py      집계 (순위 상관, mixed precision)
tests/            단위 테스트
docs/             실험 규격
results/          summary만 올림 (모델, 이미지별 예측은 제외)
```

## 실험 설정 (6-Conv)

- 모델: Conv 6개 (16/16/32/32/64/64), MaxPool은 Conv2, 4, 6 뒤에만, GAP + Linear head 3개
- 양자화: W는 symmetric per-channel, A는 asymmetric per-tensor, 둘 다 min-max. calibration 2,048장
- BN은 folding한 다음에 양자화. Head랑 입력 이미지는 양자화 안 함
- 층 하나 = Conv의 W + 그 블록 출력 A
- 조건: FP32, W8A8, W6A6, W5A5, W4A8, W4A4, W32A4, W4A32 (W5A5는 나중에 추가)
- seed 5개 (0~4). 분할은 고정이고 seed는 초기값이랑 학습 순서만 바꿈
- 층 순위는 rank 1만 장으로 정하고, 최종 평가는 test 1만 장으로 따로 함

측정하는 것

| 이름 | 설명 |
|---|---|
| 단독 하락 | FP32 모델에서 한 층만 양자화했을 때 정확도 하락 |
| 복원 이득 | 전부 양자화한 모델에서 한 층만 FP32로 되돌렸을 때 정확도 회복 |
| NMSE | 층 출력 오차. 그 층만 양자화했을 때(단독)랑 전부 양자화했을 때(누적) 둘 다 |
| C→W, W→C | 양자화 전후로 정답/오답이 바뀐 이미지 수 |
| 순위 상관 | NMSE 순위랑 복원 순위의 Spearman ρ, top-k 겹침 |
| Mixed precision | 층마다 4/8비트 고르는 64가지 조합 전부 |

## 6-Conv 결과 (5 seed)

집계 파일은 results/6conv_fold/summary/에 있음. 정확도는 test 1만 장, 세 과제 평균 기준.

| 조건 | 정확도(%) | seed 범위 |
|---|---|---|
| FP32 | 100 | - |
| W8A8 | 100 | 틀리게 바뀐 이미지 0개 |
| W6A6 | 100 | 틀리게 바뀐 이미지 0개 |
| W5A5 | 98.68 ± 0.75 | 97.86 ~ 99.64 |
| W4A32 | 89.94 ± 4.87 | 82.65 ~ 95.65 |
| W4A8 | 89.89 ± 5.00 | 82.36 ~ 95.59 |
| W32A4 | 85.69 ± 8.52 | 72.35 ~ 93.13 |
| W4A4 | 58.31 ± 13.16 | 43.71 ~ 71.76 |

W4A4에서 층별로 본 것 (rank 1만 장, seed 평균)

| 층 | 복원 이득(%p) | 단독 하락(%p) | 누적 NMSE | 단독 NMSE |
|---|---|---|---|---|
| Conv1 | 19.21 | 5.56 | 0.046 | 0.046 |
| Conv2 | 10.08 | 1.36 | 0.123 | 0.044 |
| Conv3 | 11.73 | 0.63 | 0.224 | 0.056 |
| Conv4 | 12.74 | 2.94 | 0.235 | 0.038 |
| Conv5 | 4.91 | 1.15 | 0.365 | 0.038 |
| Conv6 | -0.45 | 0.00 | 1.302 | 0.082 |

- 8비트, 6비트에서는 정확도가 안 변함. 5비트부터 조금씩 틀리기 시작하고(평균 -1.3%p) 4비트에서 크게 무너짐
  - 5비트에서 제일 많이 떨어지는 과제는 Color (96.95%), Shape은 거의 그대로 (99.91%)
- NMSE 순위랑 복원 이득 순위가 안 맞음. 누적 NMSE 기준 ρ 평균 -0.62, 1등 층은 5개 seed 전부 다름
  - 층이 6개뿐이라 |ρ| 0.886 이상이어야 유의한데, 넘은 건 seed 1개
  - 누적 NMSE는 뒤 층일수록 앞 층 오차가 쌓여서 커지는 것도 감안해야 함
- Conv6은 단독 NMSE가 5개 중 4개 seed에서 제일 큰데, 혼자 양자화했을 때 정확도 하락은 전부 0
- Conv1 ~ 5만 8비트로 두고 Conv6만 4비트로 해도 (평균 5.95비트) 5개 seed 전부 100%
- seed마다 편차가 커서 층 사이의 작은 차이는 해석하기 어려움
- BN folding은 결과에 거의 영향이 없었음. 11/9 예비 실험(folding 없음)이랑 수치가 거의 같음

Mixed precision 비교 기준(Weight Bit Cost)은 아직 못 정했다. 평균 비트 예산으로 자르면 Conv5, 6이 너무 커서 정책끼리 차이가 안 남. mixed.csv에 예산 방식이랑 top-k 방식 둘 다 넣어놨음.

예비 실험 집계는 results/prelim_1109_nofold/에 따로 있음.

## 브랜치

- main : 검증된 코드만
- feature/6conv, feature/imagenet, feature/quantization, feature/experiments, feature/analysis

main에 바로 push하지 말고 기능별 브랜치에서 작업한 다음 PR로 합친다. 설정을 바꾸면 docs/experiment_spec.md도 같이 고치기.

## 참고

- Burgess, Kim. 3D Shapes Dataset, 2018
- Yuan et al. PTQ4ViT, ECCV 2022
- Lin et al. FQ-ViT, IJCAI 2022
- Dong et al. HAWQ, ICCV 2019
