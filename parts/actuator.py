# -*- coding: utf-8 -*-
"""
Actuator Part
=============

차량 제어(조향, 스로틀, 브레이크) Part.
시뮬레이터에 제어 명령을 전달합니다.

Donkeycar 시뮬레이터 API:
- steering: -1.0 ~ 1.0 (좌/우)
- throttle: -1.0 ~ 1.0 (후진/전진)
- brake: 0.0 ~ 1.0
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class ActuatorPart:
    """
    차량 제어 Actuator Part
    
    gym-donkeycar 환경에 조향/스로틀 명령을 전달합니다.
    """
    
    def __init__(self, env=None):
        """
        Args:
            env: gym-donkeycar 환경 객체
        """
        self.env = env
        self.steering = 0.0
        self.throttle = 0.0
        self.on = True
        
        logger.info("ActuatorPart 초기화")
    
    def run(self, steering: float, throttle: float) -> Tuple[float, float]:
        """
        제어 명령 실행
        
        Args:
            steering: 조향값 (-1.0 ~ 1.0)
            throttle: 스로틀값 (-1.0 ~ 1.0)
            
        Returns:
            (steering, throttle) 튜플
        """
        # 값 클리핑
        self.steering = max(-1.0, min(1.0, steering))
        self.throttle = max(-1.0, min(1.0, throttle))
        
        # 시뮬레이터에 명령 전달
        if self.env is not None:
            action = [self.steering, self.throttle]
            self.env.step(action)
        
        return self.steering, self.throttle
    
    def run_threaded(self, steering: float, throttle: float):
        """
        쓰레드 모드에서 제어값 저장
        """
        self.steering = max(-1.0, min(1.0, steering))
        self.throttle = max(-1.0, min(1.0, throttle))
    
    def update(self):
        """
        저장된 제어값을 환경에 전달 (별도 쓰레드에서 호출)
        """
        while self.on:
            if self.env is not None:
                action = [self.steering, self.throttle]
                self.env.step(action)
    
    def shutdown(self):
        """
        Part 종료 (안전 정지)
        """
        self.on = False
        self.steering = 0.0
        self.throttle = 0.0
        
        if self.env is not None:
            self.env.step([0.0, 0.0])
        
        logger.info("ActuatorPart 종료 (안전 정지)")


class MockActuatorPart:
    """테스트용 더미 Actuator"""
    
    def run(self, steering: float, throttle: float) -> Tuple[float, float]:
        return steering, throttle
    
    def shutdown(self):
        pass
