#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donkeycar 진입점
================

Usage:
    python manage.py drive                    # 수동 운전 + 데이터 수집
    python manage.py drive --model MODEL_PATH # 자율주행 모드
    python manage.py train                    # 모델 학습

참고:
- Donkeycar 공식 문서: https://docs.donkeycar.com
"""

import os
import sys
import argparse
import logging

# 프로젝트 루트 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """myconfig.py 로드"""
    try:
        import myconfig as cfg
        return cfg
    except ImportError:
        logger.warning("myconfig.py를 찾을 수 없습니다. 기본 설정 사용.")
        return None


def drive(cfg, model_path=None, use_joystick=False):
    """
    드라이브 모드
    
    Args:
        cfg: 설정 객체
        model_path: 모델 경로 (자율주행 시)
        use_joystick: 조이스틱 사용 여부
    """
    from vehicle import Vehicle
    from parts.camera import SimCameraPart, MockCameraPart
    from parts.actuator import ActuatorPart, MockActuatorPart
    from parts.pilot import KerasPilot, MockPilot
    from parts.datastore import TubWriter
    
    logger.info("="*50)
    logger.info("Donkeycar Drive Mode")
    logger.info("="*50)
    
    # Vehicle 생성
    v = Vehicle()
    
    # Donkeycar Gym 환경 설정
    env = None
    if cfg and getattr(cfg, 'DONKEY_GYM', False):
        try:
            import gym
            import gym_donkeycar
            
            env = gym.make(
                cfg.DONKEY_GYM_ENV_NAME,
                exe_path=cfg.DONKEY_SIM_PATH,
                port=9091
            )
            logger.info(f"Donkey Gym 환경 생성: {cfg.DONKEY_GYM_ENV_NAME}")
        except ImportError:
            logger.warning("gym-donkeycar 설치 필요: pip install gym-donkeycar")
            logger.info("Mock 환경 사용")
        except Exception as e:
            logger.error(f"Gym 환경 생성 실패: {e}")
            logger.info("Mock 환경 사용")
    
    # 카메라 Part
    if env:
        camera = SimCameraPart(
            env=env,
            width=getattr(cfg, 'CAMERA_WIDTH', 160),
            height=getattr(cfg, 'CAMERA_HEIGHT', 120)
        )
    else:
        camera = MockCameraPart()
    v.add(camera, outputs=['camera/image'])
    
    # 파일럿 Part
    if model_path:
        pilot = KerasPilot(model_path=model_path)
        logger.info(f"자율주행 모드: {model_path}")
    else:
        pilot = MockPilot()
        logger.info("수동 운전 모드 (MockPilot)")
    v.add(pilot, inputs=['camera/image'], outputs=['pilot/steering', 'pilot/throttle'])
    
    # Actuator Part
    if env:
        actuator = ActuatorPart(env=env)
    else:
        actuator = MockActuatorPart()
    v.add(actuator, inputs=['pilot/steering', 'pilot/throttle'])
    
    # 데이터 수집
    if getattr(cfg, 'RECORDING', True):
        tub_writer = TubWriter(base_path=getattr(cfg, 'DATA_PATH', './data'))
        tub_writer.start_recording()
        v.add(tub_writer, inputs=['camera/image', 'pilot/steering', 'pilot/throttle'])
        logger.info("데이터 수집 ON")
    
    # 루프 시작
    rate_hz = getattr(cfg, 'DRIVE_LOOP_HZ', 20)
    logger.info(f"Vehicle Loop 시작: {rate_hz} Hz")
    logger.info("종료: Ctrl+C")
    
    try:
        v.start(rate_hz=rate_hz)
    finally:
        if env:
            env.close()


def train(cfg, tub_path=None, model_path=None):
    """
    모델 학습
    
    Args:
        cfg: 설정 객체
        tub_path: Tub 데이터 경로
        model_path: 저장할 모델 경로
    """
    logger.info("="*50)
    logger.info("Donkeycar Train Mode")
    logger.info("="*50)
    
    tub_path = tub_path or getattr(cfg, 'DATA_PATH', './data')
    model_path = model_path or getattr(cfg, 'MODEL_PATH', 'models/mypilot.h5')
    
    try:
        from parts.datastore import TubReader
        from train import train_model
        
        logger.info(f"Tub 경로: {tub_path}")
        logger.info(f"모델 저장: {model_path}")
        
        # 학습 실행
        train_model(
            tub_path=tub_path,
            model_path=model_path,
            batch_size=getattr(cfg, 'BATCH_SIZE', 128),
            epochs=getattr(cfg, 'EPOCHS', 100),
            learning_rate=getattr(cfg, 'LEARNING_RATE', 0.001)
        )
        
    except ImportError:
        logger.error("train.py 모듈이 필요합니다.")
        logger.info("donkey train --tub ./data --model ./models/mypilot.h5 명령어 사용")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Donkeycar 관리 스크립트')
    
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # drive 명령어
    drive_parser = subparsers.add_parser('drive', help='드라이브 모드')
    drive_parser.add_argument('--model', '-m', type=str, help='모델 경로')
    drive_parser.add_argument('--js', action='store_true', help='조이스틱 사용')
    
    # train 명령어
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--tub', '-t', type=str, help='Tub 데이터 경로')
    train_parser.add_argument('--model', '-m', type=str, help='모델 저장 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    cfg = load_config()
    
    if args.command == 'drive':
        drive(cfg, model_path=args.model, use_joystick=args.js)
    elif args.command == 'train':
        train(cfg, tub_path=args.tub, model_path=args.model)
    else:
        parser.print_help()
        print("\n예시:")
        print("  python manage.py drive")
        print("  python manage.py drive --model models/mypilot.h5")
        print("  python manage.py train --tub ./data --model models/mypilot.h5")


if __name__ == "__main__":
    main()
