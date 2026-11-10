# 층별 양자화 오차와 FP32 복원 효과 비교

2026 컴퓨터비전 팀 프로젝트 (김진서, 이하형, 정채호준)

재학습 없는 PTQ에서 **수치 양자화 오차가 큰 층**과 **FP32로 되돌렸을 때 정확도가 크게 회복되는 층**이 같은지 확인한다.

- 통제 실험: 직접 학습한 6-Conv Multi-Head CNN + 3D-Shapes (속성별 영향 확인)
- 실제 모델 검증: ResNet-18, DeiT-Tiny + ImageNet-1K (예정)

## 준비

```bash
pip install -r requirements.txt
```

데이터는 DeepMind 3D Shapes의 `3dshapes.h5`(약 255MB)를 받아 `data/3dshapes.h5`에 둔다.

- 저장소: https://github.com/deepmind/3dshapes-dataset
- 파일: https://storage.googleapis.com/3d-shapes/3dshapes.h5
- SHA-256: `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` (실행 시 자동 확인)

## 코드 구조

| 파일 | 내용 |
|---|---|
| `src/data.py` | 3D-Shapes 읽기, 모든 seed 공통 분할 |
| `src/models.py` | 6-Conv Multi-Head CNN |
| `src/train.py` | FP32 학습 |

## 실행

```bash
python -m src.train --seed 0
python -m unittest discover -s tests -t .
```
