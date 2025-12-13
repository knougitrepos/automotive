# 자율주행 AI 파이프라인 (비전 라인)

CARLA 시뮬레이터에서 정책(action) 데이터를 수집하고, KITTI로 self-supervised 사전학습을 수행하여, 지침을 따르는 자율주행 모델을 학습 및 평가하는 프로젝트입니다.

## 환경

- **GPU**: GTX 1080Ti (11GB VRAM)
- **파이프라인**: 비전 라인 (Tesla-like)
- **구현**: Jupyter Notebook 중심

## 프로젝트 구조

```
automotive/
├── notebook/                  # 핵심 구현 노트북
│   ├── 01_carla_setup.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_kitti_exploration.ipynb
│   ├── 04_ssl_pretraining.ipynb
│   ├── 05_bc_training.ipynb
│   ├── 06_safety_shield.ipynb
│   └── 07_evaluation.ipynb
├── config/                    # 설정 파일
├── dataset/                   # 데이터
├── utils/                     # 유틸리티
├── plan.md                    # 구현 계획서
└── requirements.txt
```

## 시작하기

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# CARLA 설치 (0.9.15)
# https://github.com/carla-simulator/carla/releases/tag/0.9.15
pip install carla==0.9.15
```

### 2. CARLA 서버 실행

```powershell
cd C:\CARLA_0.9.15
.\CarlaUE4.exe -quality-level=Low
```

### 3. 노트북 순차 실행

1. `01_carla_setup.ipynb` - CARLA 연결 테스트
2. `02_data_collection.ipynb` - 데이터 수집
3. ... (순차적으로)

## 참고 논문

- SimCLR (Chen et al., ICML 2020) - Self-supervised 학습
- CARLA (Dosovitskiy et al., CoRL 2017) - 시뮬레이터
- PilotNet (Bojarski et al., 2016) - End-to-end 주행

자세한 내용은 [plan.md](plan.md) 참조
