# 실험 규격

모델과 양자화 설정을 한곳에 기록해 같은 조건으로 재현할 수 있게 한다.
근거 문서: 이하형 「11_10 양자화모델보안 cnnViT 연구진행방향」. 표의 "상태"가 잠정인 항목은 팀 확정 전이다.

## 1. 6-Conv + 3D-Shapes

### 데이터

| 항목 | 값 | 상태 |
|---|---|---|
| 데이터 | DeepMind 3D Shapes, 48만 장, 64x64 | 확정 |
| 예측 과제 | Shape 4종, Object Color 10종, Scale 8종 | 확정 |
| 분할 | seed 0으로 한 번 섞어 모든 모델 seed가 공유, 묶음 간 겹침 없음 | 확정 |
| 학습 | 60,000장 | 잠정 |
| 보정(calibration) | 2,048장, 학습과 분리 | 확정 |
| val | 5,000장 (folding 검증용) | 잠정 |
| rank | 10,000장 (층 순위 산출 전용) | 잠정 |
| test | 10,000장 (최종 평가 전용) | 잠정 |
| 입력 | [0, 255] -> [-0.5, 0.5], 양자화하지 않음 | 확정 |

### 모델과 학습

| 항목 | 값 | 상태 |
|---|---|---|
| Conv | 6개, 채널 16/16/32/32/64/64, 3x3, padding 1 | 확정 |
| MaxPool | Conv2·4·6 뒤 3회, 최종 특징맵 8x8 | 확정 |
| Head | GAP + Linear 3개, FP32 유지 | 확정 |
| BN | 학습 후 Conv-BN folding, folding 전후 FP32 출력 일치 검증, folding된 모델에서 보정 | 확정 |
| 학습 | Adam lr 1e-3, cosine, batch 256, 20 epoch, 세 과제 cross-entropy 합 | 잠정 |
| seed | 5개 (0~4). 초기값과 학습 순서만 바뀜 | 확정 |

### 양자화

| 항목 | 값 | 상태 |
|---|---|---|
| 방식 | PTQ, fake quantization, 재학습 없음, round-to-nearest | 확정 |
| 가중치 W | 대칭, per-channel(출력 채널), min-max, 범위 ±(2^(b-1) − 1) | 확정 |
| 활성값 A | 비대칭, per-tensor, 보정 min-max, 범위 0 ~ 2^b − 1 | 확정 |
| A 양자화 지점 | 각 Conv의 ReLU 뒤, MaxPool이 있으면 MaxPool 뒤 | 확정 |
| 층 단위 | Conv i의 W + 그 블록 출력 A를 한 단위로 취급 | 확정 |
| 첫·마지막 층 | 단독·복원 실험에서는 고정 없이 같은 조건. Mixed-Precision에서 8비트 보호 전략을 따로 비교 | 확정 |
| 정밀도 조건 | FP32, W8A8, W6A6, W5A5, W4A8, W4A4, W32A4, W4A32. W5A5는 11-10 2차 회신으로 추가(`src/exp_extra.py`, 기존 모델·보정 범위 재사용) | 확정 |
| folding된 bias | FP32 유지 | 잠정 (문서에 규정 없음) |

### 실험

| 실험 | 내용 |
|---|---|
| 단독 양자화 | 전 층 FP32에서 층 i만 양자화 -> 정답률 하락 |
| 개별 복원 | 전 층 양자화 후 층 i만 FP32 복원 -> 정답률 회복 |
| NMSE | 같은 특징맵 위치에서 ‖y − ŷ‖² / ‖y‖². 단독 양자화 오차와 누적 양자화 오차를 구분 |
| Mixed-Precision | 층별 4/8비트 64조합 전수, W·A 동일 비트 |

저장 항목: 이미지별 정답, FP32·양자화 예측, C→W, W→C, logit margin, NMSE, 복원 결과.

### 미정 사항

- Mixed-Precision의 Weight Bit Cost 비교 기준. Conv5·6이 가중치의 77%라 평균 비트 예산 5·6에서는 모든 정책이 같은 조합을 고른다. `src/analyze.py`는 예산 방식과 top-k 방식을 모두 계산한다.
- 도메인 실험 2차(벽·바닥 색 조합 일부를 학습에서 제외, (Wall + 3·Floor) mod 10 = 0 인 10조합)의 실행 여부. 재학습이 필요함.

## 2. ImageNet 모델 (구현 전)

| 항목 | 값 |
|---|---|
| 데이터 | ImageNet-1K, 사전학습 가중치 그대로 사용 |
| 분할 | 보정 train 2,048장 / rank val 5,000장 / 최종 평가는 rank와 겹치지 않는 val 10,000장 |
| CNN | ResNet-18 (torchvision). 단위: Stem Conv, BasicBlock 8개, FC |
| ViT | DeiT-Tiny (timm) 우선, 필요하면 DeiT-Small. 단위: Patch Embedding, Block 12개, Head |
| ResNet 규칙 | BN folding, ReLU와 Residual Add는 FP32, Downsample Conv는 블록에 포함 |
| DeiT 규칙 | QKV·Proj·FC1·FC2의 W와 지정 A 양자화. LayerNorm·GELU·Softmax·Residual Add·행렬곱은 FP32 |

## 3. 예비 실험 (11-09)

BN folding 없이 수행한 6-Conv 5 seed 결과는 본 결과와 섞지 않고 `results/prelim_1109_nofold/`에 따로 둔다.

본 실험과 수치가 거의 같다(seed 0 기준 공통 780개 수치 중 588개 동일, 최대 차이 0.10%p). 가중치를 출력 채널별 대칭 min-max로 양자화하면 BN의 채널별 배율이 scale에 흡수되기 때문으로 해석한다.

## 도메인 실험 1차 (11-10 2차 회신으로 확정, `src/exp_domain.py`)

| 항목 | 설정 | 상태 |
|---|---|---|
| 모델 | 본 실험 5 seed 모델 그대로. 재학습 없음 | 확정 |
| 배경 | Shape·Object Color·Scale·Orientation 유지, 벽·바닥 색 변경. 학습·보정 이미지는 제외 | 확정 |
| Noise | Gaussian, σ = 0.01, 0.03 (이미지 값 0~1 기준) | 확정 |
| Blur | Gaussian, 5x5 커널, σ = 0.5, 1.0 | 확정 |
| 배경 선택 방법 | 벽·바닥 둘 다 원래와 다른 색으로 무작위(seed 0) | 잠정 |
| Noise 후 처리 | 0~1 밖은 잘라냄. 두 σ는 같은 난수에 배율만 다름 | 잠정 |
| 보정 범위 | 원본 이미지로 구한 범위 그대로(재보정 없음) | 잠정 |
| 측정 상태 | FP32, 조건별 전 층 양자화, W4A4의 복원·단독·NMSE | 잠정 |

배경 변경은 모델이 학습에서 이미 본 색이라 분포 변화가 아니다. 같은 물체에 배경만 다를 때 결과가 달라지는지 보는 대조 실험으로 해석한다.
