# -*- coding: utf-8 -*-
"""
Donkeycar 설정 파일
==================

사용자 설정을 정의합니다.
이 파일을 수정하여 환경에 맞게 설정하세요.
"""

# ===== Donkeycar 시뮬레이터 설정 =====

# Donkeycar Gym 사용 여부
DONKEY_GYM = True

# 시뮬레이터 경로 (OS에 맞게 수정)
# Windows: DonkeySimWin/donkey_sim.exe
# Mac: DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim
# Linux: DonkeySimLinux/donkey_sim.x86_64
DONKEY_SIM_PATH = "path/to/donkey_sim.exe"

# Gym 환경 이름
# - donkey-generated-track-v0
# - donkey-warehouse-v0
# - donkey-avc-sparkfun-v0
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0"


# ===== 카메라 설정 =====

CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
CAMERA_DEPTH = 3  # RGB


# ===== 모델 설정 =====

# 학습된 모델 경로 (자율주행 모드에서 사용)
MODEL_PATH = 'models/mypilot.h5'

# 모델 유형: 'keras' 또는 'pytorch'
MODEL_TYPE = 'keras'


# ===== 드라이빙 설정 =====

# Vehicle Loop 주파수 (Hz)
DRIVE_LOOP_HZ = 20

# 최대 스로틀 (0.0 ~ 1.0)
MAX_THROTTLE = 0.5

# 조이스틱 사용 여부
USE_JOYSTICK = False


# ===== 데이터 수집 설정 =====

# 데이터 저장 경로
DATA_PATH = './data'

# 녹화 여부 (drive 모드에서)
RECORDING = True

# 최대 레코드 수
MAX_RECORDS = 10000


# ===== 학습 설정 =====

# 배치 사이즈
BATCH_SIZE = 128

# 학습률
LEARNING_RATE = 0.001

# 에포크
EPOCHS = 100


# ===== 로깅 =====

LOG_LEVEL = 'INFO'


print("✅ myconfig.py 로드 완료")
