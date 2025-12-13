# 자율주행 시스템 MVP 프로젝트

KITTI 데이터셋을 활용한 비전 기반 및 라이다 기반 자율주행 시스템 MVP 구현

## 프로젝트 구조

```
automotive/
├── MVP-V/                    # 비전 기반 프로젝트 (테슬라 방식)
│   └── notebooks/
│       ├── 01_data_loading.ipynb          # KITTI 이미지/레이블 로더
│       ├── 02_yolo_detection.ipynb        # YOLO 객체 탐지
│       ├── 03_depth_estimation.ipynb      # 스테레오 깊이 추정
│       ├── 04_tracking.ipynb              # ByteTrack 추적
│       ├── 05_pipeline_integration.ipynb  # 전체 파이프라인 통합
│       └── 06_test_evaluation.ipynb       # 테스트 및 평가
│
├── MVP-L/                    # 라이다 기반 프로젝트
│   └── notebooks/
│       ├── 01_data_loading.ipynb          # 포인트 클라우드/칼리브레이션 로더
│       ├── 02_pointpillars_training.ipynb # PointPillars 모델 학습
│       ├── 03_bev_occupancy.ipynb         # BEV 주행가능영역 생성
│       ├── 04_tracking_3d.ipynb           # 3D 추적 (Kalman + AB3DMOT)
│       ├── 05_pipeline_integration.ipynb  # 전체 파이프라인 통합
│       └── 06_test_evaluation.ipynb       # 테스트 및 평가
│
├── dataset/                   # KITTI 데이터셋 (공통)
├── requirements.txt           # 전체 라이브러리 의존성
└── README.md                  # 프로젝트 설명서
```

## 설치 방법

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터셋 준비

KITTI 데이터셋을 `dataset/` 폴더에 배치하세요:
- `dataset/kitti/` - 비전 데이터 (이미지, 레이블)
- `dataset/training/` - 라이다 데이터 (velodyne, calib, label_2)

## 사용 방법

### MVP-V (비전 기반)

1. `MVP-V/notebooks/01_data_loading.ipynb`부터 순서대로 실행
2. 각 노트북은 이전 노트북의 결과를 활용

### MVP-L (라이다 기반)

1. `MVP-L/notebooks/01_data_loading.ipynb`부터 순서대로 실행
2. 각 노트북은 이전 노트북의 결과를 활용

## 주요 기능

### MVP-V (비전 기반)
- YOLO 기반 2D 객체 탐지
- 스테레오 깊이 추정
- ByteTrack 객체 추적
- 규칙 기반 경로 계획

### MVP-L (라이다 기반)
- PointPillars 3D 객체 탐지
- BEV occupancy grid 생성
- 3D Kalman Filter 추적
- 규칙 기반 경로 계획

## 참고 문헌

### MVP-V
- **YOLO**: Redmon et al. "You Only Look Once" (CVPR 2016)
- **SAM**: Kirillov et al. "Segment Anything" (ICCV 2023)
- **ByteTrack**: Zhang et al. "ByteTrack: Multi-Object Tracking" (ECCV 2022)

### MVP-L
- **PointPillars**: Lang et al. "PointPillars" (CVPR 2019)
- **AB3DMOT**: Weng et al. "AB3DMOT: A Baseline for 3D Multi-Object Tracking" (ICRA 2020)

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.
