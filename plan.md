# Donkeycar MVP 구현 계획서

## 개요

현재 프로젝트를 Donkeycar 프레임워크 전용으로 리팩토링합니다.
- CARLA 관련 코드 제거
- Donkeycar 시뮬레이터(gym-donkeycar) 사용
- Parts + Vehicle Loop 아키텍처 적용

## 프로젝트 구조

```
automotive/
├── manage.py              # 진입점 (drive/collect/train)
├── myconfig.py            # 사용자 설정
├── vehicle.py             # Vehicle Loop
├── parts/                 # Donkeycar Parts
│   ├── __init__.py
│   ├── camera.py          # SimCameraPart
│   ├── actuator.py        # ActuatorPart
│   ├── pilot.py           # KerasPilot
│   └── datastore.py       # TubWriter
├── models/                # 학습된 모델
├── data/                  # Tub 데이터
└── docs/
    └── mvp_flow.md        # MVP 흐름 문서
```

## 핵심 Parts

### 1. SimCameraPart (camera.py)
시뮬레이터에서 카메라 이미지 획득

### 2. ActuatorPart (actuator.py)
조향/스로틀 제어 명령 전달

### 3. KerasPilot (pilot.py)
CNN 모델 기반 자율주행 (image → steering, throttle)

### 4. TubWriter (datastore.py)
Donkeycar Tub 형식 데이터 저장

## MVP 흐름

```
Camera → Pilot Model → Actuator → Simulator
         ↓
     Tub Writer (데이터 저장)
```

## 명령어

```bash
# 시뮬레이터 연결 + 수동운전
python manage.py drive

# 자율주행
python manage.py drive --model models/mypilot.h5

# 모델 학습
donkey train --tub ./data --model models/mypilot.h5
```

## 참고 자료

| 항목 | 참고 |
|------|------|
| Donkeycar 문서 | https://docs.donkeycar.com |
| gym-donkeycar | https://github.com/tawnkramer/gym-donkeycar |
| Behavioral Cloning | Bojarski et al., 2016 |
