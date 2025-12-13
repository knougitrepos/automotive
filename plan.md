# 자율주행 AI 파이프라인 구현 계획서 (비전 라인)

> **목표**: CARLA 시뮬레이터에서 정책(action) 데이터를 수집하고, KITTI로 self-supervised 사전학습을 수행하여, 신호등 준수/속도 제한/보행자 회피 등 지침을 따르는 자율주행 모델을 학습 및 평가

---

## 1. 환경 정보

| 항목 | 값 |
|------|------|
| **CPU** | Intel i5-10600K |
| **RAM** | 64GB |
| **GPU** | GTX 1080Ti (11GB VRAM) |
| **CARLA** | 미설치 → 설치 필요 |
| **파이프라인 방향** | 비전 라인 (Tesla-like) |
| **구현 형태** | **Jupyter Notebook 중심** |
| **목표** | 완전한 기능 구현 |

### GPU 메모리 최적화 전략 (11GB)

- Batch size: 8~16 권장
- Mixed Precision (FP16) 사용
- Gradient Checkpointing 적용
- ResNet-50 기반 (ViT는 메모리 부담)

---

## 2. CARLA 설치 가이드

> **CARLA 0.9.15 설치가 필요합니다**
> 
> GTX 1080Ti에서 원활히 동작하며, Windows 지원이 안정적입니다.

### 2.1 설치 방법 (Windows)

```powershell
# 1. CARLA 다운로드 (약 15GB)
# https://github.com/carla-simulator/carla/releases/tag/0.9.15
# CARLA_0.9.15.zip 다운로드

# 2. 압축 해제: F:\CARLA\WindowsNoEditor

# 3. CARLA 서버 실행
cd F:\CARLA\WindowsNoEditor
.\CarlaUE4.exe -quality-level=Low -RenderOffScreen
# 또는 GUI 모드: .\CarlaUE4.exe

# 4. Python 클라이언트 설치
pip install carla==0.9.15
```

### 2.2 설치 확인

```python
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print(f"CARLA 연결 성공: {world.get_map().name}")
```

---

## 3. 프로젝트 구조 (노트북 중심)

```
c:\git\automotive\
├── notebook/                          # 핵심 구현 (노트북)
│   ├── 01_carla_setup.ipynb          # CARLA 설치 확인 및 기본 테스트
│   ├── 02_data_collection.ipynb      # CARLA 데이터 수집
│   ├── 03_kitti_exploration.ipynb    # KITTI 데이터 탐색
│   ├── 04_ssl_pretraining.ipynb      # Self-supervised 사전학습
│   ├── 05_bc_training.ipynb          # 행동 클로닝 학습
│   ├── 06_safety_shield.ipynb        # Safety Shield 구현
│   └── 07_evaluation.ipynb           # CARLA 평가 및 시각화
│
├── config/                            # 설정 파일
│   ├── carla_config.yaml
│   ├── training_config.yaml
│   └── model_config.yaml
│
├── dataset/                           # 데이터
│   ├── kitti/                        # 기존 KITTI 데이터
│   ├── kitti_tracking/
│   └── carla_collected/              # CARLA 수집 데이터 (생성됨)
│
├── utils/                             # 공용 유틸리티 (최소한의 .py)
│   ├── __init__.py
│   ├── carla_utils.py                # CARLA 헬퍼 함수
│   ├── data_utils.py                 # 데이터 처리 유틸
│   └── viz_utils.py                  # 시각화 유틸
│
├── plan.md                            # 이 문서
├── requirements.txt
└── README.md
```

> **노트북 중심 구조의 장점**
> - 각 단계별 결과를 시각적으로 확인 가능
> - 실험 과정이 문서화됨
> - 디버깅과 수정이 용이
> - 셀 단위 실행으로 메모리 관리 용이 (11GB GPU에 적합)

---

## 4. 노트북별 상세 구성

### 4.1 `01_carla_setup.ipynb` - CARLA 설치 확인

```
Cell 1: 라이브러리 임포트 및 CARLA 연결
Cell 2: 월드/맵 정보 확인
Cell 3: 차량 스폰 테스트
Cell 4: 센서 부착 테스트 (카메라)
Cell 5: autopilot 주행 테스트
Cell 6: 스크린샷 캡처 및 저장
```

---

### 4.2 `02_data_collection.ipynb` - CARLA 데이터 수집

```
Cell 1: 설정 및 라이브러리
Cell 2: CARLA 클라이언트 연결
Cell 3: 센서 설정 (RGB 카메라)
Cell 4: Expert Agent (autopilot) 설정
Cell 5: 데이터 수집 루프 정의
Cell 6: 에피소드 수집 실행
Cell 7: 수집 데이터 검증 및 시각화
Cell 8: 데이터 저장 (Parquet + 이미지)
```

#### 수집 데이터 스키마

| 필드 | 타입 | 설명 |
|------|------|------|
| `frame_id` | int | 프레임 번호 |
| `image_path` | str | 이미지 파일 경로 |
| `ego_speed` | float | 자차 속도 (m/s) |
| `ego_location` | [x, y, z] | 위치 |
| `ego_rotation` | [pitch, yaw, roll] | 방향 |
| `traffic_light` | str | red/yellow/green/none |
| `speed_limit` | float | 속도 제한 (km/h) |
| `steer` | float | 핸들 [-1, 1] |
| `throttle` | float | 가속 [0, 1] |
| `brake` | float | 제동 [0, 1] |
| `collision` | bool | 충돌 여부 |
| `lane_invasion` | bool | 차선 침범 |

---

### 4.3 `03_kitti_exploration.ipynb` - KITTI 데이터 탐색

```
Cell 1: KITTI 디렉토리 구조 확인
Cell 2: 이미지 시퀀스 로드 및 시각화
Cell 3: 연속 프레임 분석 (optical flow 패턴)
Cell 4: 데이터 통계 (이미지 크기, 프레임 수)
Cell 5: Self-supervised 학습용 데이터셋 구성
Cell 6: Train/Val 분할
```

---

### 4.4 `04_ssl_pretraining.ipynb` - Self-supervised 사전학습

```
Cell 1: 라이브러리 및 설정
Cell 2: KITTI 데이터셋 클래스 정의
Cell 3: 데이터 증강 파이프라인 (SimCLR 스타일)
Cell 4: Vision Encoder 모델 정의 (ResNet-50)
Cell 5: Contrastive Loss 정의
Cell 6: 학습 루프 구현
Cell 7: 학습 실행 및 모니터링
Cell 8: 학습 곡선 시각화
Cell 9: 체크포인트 저장
Cell 10: 학습된 표현 품질 평가 (t-SNE)
```

#### Self-supervised 학습 방법론

| 방법 | 구현 복잡도 | 성능 | 메모리 사용 |
|------|------------|------|------------|
| **SimCLR** | 낮음 | 좋음 | 중간 |
| DINO | 중간 | 매우 좋음 | 높음 |
| MoCo v3 | 중간 | 매우 좋음 | 중간 |

> **권장**: SimCLR (11GB GPU에 적합, 구현 단순)

---

### 4.5 `05_bc_training.ipynb` - 행동 클로닝 학습

```
Cell 1: 라이브러리 및 설정
Cell 2: CARLA 수집 데이터 로드
Cell 3: 데이터셋 클래스 정의
Cell 4: 전체 모델 정의
        - Vision Encoder (사전학습 로드)
        - Policy Head (MLP)
Cell 5: 손실 함수 정의
Cell 6: 학습 루프 구현
Cell 7: 학습 실행
Cell 8: 학습 곡선 시각화
Cell 9: 검증 세트 평가
Cell 10: 모델 저장
```

#### 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        전체 모델 구조                            │
├─────────────────────────────────────────────────────────────────┤
│  Input: RGB Image (800 × 600 × 3)                               │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Vision Encoder (ResNet-50, KITTI 사전학습)              │    │
│  │  - 출력: 2048-dim feature vector                        │    │
│  │  - 옵션: freeze / fine-tune                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Policy Head (MLP)                                       │    │
│  │  - Linear(2048, 512) → ReLU → Dropout(0.3)              │    │
│  │  - Linear(512, 256) → ReLU → Dropout(0.3)               │    │
│  │  - Linear(256, 3) → Tanh (steer) / Sigmoid (throttle/brake)│ │
│  └─────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  Output: [steer, throttle, brake]                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4.6 `06_safety_shield.ipynb` - Safety Shield 구현

```
Cell 1: Shield 개념 설명 및 규칙 정의
Cell 2: SafetyShield 클래스 구현
Cell 3: 규칙별 테스트
        - 빨간불 → brake 강제
        - 속도 초과 → throttle clamp
        - 장애물 근접 → brake override
Cell 4: Policy + Shield 통합
Cell 5: 시뮬레이션 테스트 (오프라인)
Cell 6: 규칙 파라미터 튜닝
```

#### Shield 규칙 정의

```python
class SafetyShield:
    """
    규칙 기반 안전 장치
    Policy 출력을 받아 안전하지 않으면 override
    """
    def __init__(self):
        self.rules = {
            'red_light_stop': True,      # 빨간불 정지
            'speed_limit': True,          # 속도 제한
            'collision_avoid': True,      # 충돌 회피
            'pedestrian_priority': True,  # 보행자 우선
        }
    
    def apply(self, action, state):
        """
        action: [steer, throttle, brake]
        state: {traffic_light, speed, speed_limit, obstacle_dist, ...}
        """
        safe_action = action.copy()
        
        # 규칙 1: 빨간불 → 정지
        if state['traffic_light'] == 'red':
            safe_action['throttle'] = 0.0
            safe_action['brake'] = 1.0
        
        # 규칙 2: 속도 초과 → throttle 제한
        if state['speed'] > state['speed_limit']:
            safe_action['throttle'] = 0.0
        
        # 규칙 3: 장애물 근접 → 긴급 제동
        if state['obstacle_dist'] < 5.0:  # 5m 이내
            safe_action['brake'] = min(1.0, 5.0 / state['obstacle_dist'])
        
        return safe_action
```

---

### 4.7 `07_evaluation.ipynb` - CARLA 평가 및 시각화

```
Cell 1: 평가 설정 및 시나리오 정의
Cell 2: 모델 로드 (Policy + Shield)
Cell 3: CARLA 환경 연결
Cell 4: 평가 루프 구현
Cell 5: 시나리오별 평가 실행
        - 직선 주행
        - 신호등 정지
        - 속도 제한 구역
        - 보행자 횡단
Cell 6: 주행 영상 녹화
Cell 7: 메트릭 계산 및 시각화
Cell 8: 결과 리포트 생성
```

#### 평가 메트릭

| 카테고리 | 메트릭 | 설명 |
|----------|--------|------|
| **주행 성능** | Success Rate | 목표 도달 비율 |
| | Collision Rate | 충돌 횟수/에피소드 |
| | Lane Invasion Rate | 차선 침범 빈도 |
| **지침 준수** | Traffic Light Compliance | 신호 준수율 |
| | Speed Limit Compliance | 속도 제한 준수율 |
| | Pedestrian Safety | 보행자 안전 정지율 |
| **효율성** | Avg Speed | 평균 주행 속도 |
| | Route Completion Time | 경로 완주 시간 |

---

## 5. 의존성 패키지

```
# requirements.txt

# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0

# CARLA (별도 설치 후)
carla==0.9.15

# Self-supervised / Models
timm>=0.9.0

# Jupyter
jupyter>=1.0.0
ipywidgets>=8.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Data
pyyaml>=6.0
pyarrow>=14.0.0
tqdm>=4.65.0

# Logging
wandb>=0.15.0

# 선택적
scikit-learn>=1.3.0  # t-SNE, 평가
```

---

## 6. 구현 로드맵

### Phase 1: 환경 설정 (1-2일)
- CARLA 설치 및 테스트
- 프로젝트 구조 생성

### Phase 2: 데이터 (3-5일)
- KITTI 탐색 및 준비
- CARLA 데이터 수집

### Phase 3: 사전학습 (3-5일)
- SimCLR 구현
- KITTI 사전학습 실행

### Phase 4: 정책 학습 (3-5일)
- 행동 클로닝 구현
- 학습 및 튜닝

### Phase 5: 안전 & 평가 (3-5일)
- Safety Shield 구현
- CARLA 평가
- 결과 시각화

---

## 7. 학습 설정 (GTX 1080Ti 최적화)

```yaml
# training_config.yaml

# Self-supervised 사전학습
ssl_training:
  batch_size: 16          # 11GB에서 안정적
  learning_rate: 0.001
  epochs: 100
  mixed_precision: true   # FP16 사용
  
# 행동 클로닝
bc_training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  encoder_freeze: true    # 초기에는 인코더 고정
  
# 모델 설정
model:
  encoder: resnet50       # ViT 대신 ResNet 사용 (메모리 절약)
  feature_dim: 2048
  policy_hidden: [512, 256]
  dropout: 0.3
```

---

## 8. 참고 논문 및 출처

| 주제 | 논문 | 핵심 아이디어 |
|------|------|---------------|
| **Self-supervised** | [SimCLR (Chen et al., ICML 2020)](https://arxiv.org/abs/2002.05709) | Contrastive learning, NT-Xent loss로 유사한 이미지는 가깝게, 다른 이미지는 멀게 학습 |
| **Self-supervised** | [DINO (Caron et al., ICCV 2021)](https://arxiv.org/abs/2104.14294) | Self-distillation, teacher-student 네트워크로 라벨 없이 의미있는 표현 학습 |
| **행동 클로닝** | [ALVINN (Pomerleau, 1988)](https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) | 최초의 End-to-end 자율주행, 뉴럴넷으로 직접 조향 예측 |
| **End-to-end 주행** | [End to End Learning for Self-Driving Cars (Bojarski et al., 2016)](https://arxiv.org/abs/1604.07316) | NVIDIA의 PilotNet, CNN으로 이미지→조향 직접 학습 |
| **모방 학습** | [DAgger (Ross et al., AISTATS 2011)](https://arxiv.org/abs/1011.0686) | Distribution shift 문제 해결, 온라인 데이터 수집과 학습 반복 |
| **Safe RL** | [Safe RL Survey (García & Fernández, 2015)](https://jmlr.org/papers/v16/garcia15a.html) | 안전 강화학습 방법론 종합, constraint 기반 접근 |
| **CARLA** | [CARLA (Dosovitskiy et al., CoRL 2017)](https://arxiv.org/abs/1711.03938) | 자율주행 연구용 오픈소스 시뮬레이터, 다양한 센서와 날씨 조건 지원 |
| **World Model** | [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) | 잠재 공간에서 미래 상태 예측, model-based RL |

---

## 9. 다음 단계

1. **CARLA 설치** - 설치 가이드에 따라 CARLA 0.9.15 설치
2. **프로젝트 구조 생성** - 디렉토리 및 requirements.txt 생성
3. **01_carla_setup.ipynb** - CARLA 연결 테스트 노트북 작성
4. **순차적으로 나머지 노트북 구현**
