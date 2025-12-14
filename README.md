# Donkeycar 자율주행 MVP

Donkeycar 프레임워크 기반 자율주행 파이프라인입니다.

## 구조

```
automotive/
├── manage.py              # 진입점
├── myconfig.py            # 설정
├── vehicle.py             # Vehicle Loop
├── parts/                 # Donkeycar Parts
│   ├── camera.py
│   ├── actuator.py
│   ├── pilot.py
│   └── datastore.py
├── models/                # 학습된 모델
├── data/                  # Tub 데이터
└── docs/
    └── mvp_flow.md
```

## 시작하기

### 1. 환경 설정

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 시뮬레이터 설치

1. [Donkey Simulator](https://github.com/tawnkramer/gym-donkeycar/releases) 다운로드
2. `myconfig.py`에서 `DONKEY_SIM_PATH` 수정

### 3. 실행

```bash
# 드라이브 모드 (데이터 수집)
python manage.py drive

# 자율주행 모드
python manage.py drive --model models/mypilot.h5

# 학습
donkey train --tub ./data --model models/mypilot.h5
```

## MVP 흐름

```
Camera → Pilot → Actuator → Simulator
           ↓
       Tub Writer
```

## 참고

- [Donkeycar 공식 문서](https://docs.donkeycar.com)
- [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)
- NVIDIA PilotNet (Bojarski et al., 2016)
