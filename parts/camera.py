# -*- coding: utf-8 -*-
"""
Camera Part
===========

Donkeycar 시뮬레이터 카메라 Part.
시뮬레이터에서 RGB 이미지를 획득합니다.

참고: gym-donkeycar API
- img_w: 160 (기본값)
- img_h: 120 (기본값)
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SimCameraPart:
    """
    Donkeycar 시뮬레이터 카메라 Part
    
    gym-donkeycar 환경에서 카메라 이미지를 획득합니다.
    
    Attributes:
        env: gym-donkeycar 환경
        image: 현재 카메라 이미지
    """
    
    def __init__(self, env=None, width: int = 160, height: int = 120):
        """
        Args:
            env: gym-donkeycar 환경 객체
            width: 이미지 너비
            height: 이미지 높이
        """
        self.env = env
        self.width = width
        self.height = height
        self.image: Optional[np.ndarray] = None
        self.on = True
        
        logger.info(f"SimCameraPart 초기화: {width}x{height}")
    
    def run(self) -> Optional[np.ndarray]:
        """
        현재 프레임 이미지 반환
        
        Returns:
            RGB 이미지 (H, W, 3) 또는 None
        """
        return self.image
    
    def run_threaded(self) -> Optional[np.ndarray]:
        """
        쓰레드 모드에서 이미지 반환
        """
        return self.image
    
    def update(self):
        """
        환경에서 새 이미지 획득 (별도 쓰레드에서 호출)
        """
        while self.on:
            if self.env is not None:
                # gym-donkeycar에서 observation은 이미지
                obs, _, _, _ = self.env.step([0, 0])  # 정지 상태
                self.image = obs
    
    def set_image(self, image: np.ndarray):
        """
        외부에서 이미지 설정 (Vehicle Loop에서 사용)
        
        Args:
            image: RGB 이미지
        """
        self.image = image
    
    def shutdown(self):
        """
        Part 종료
        """
        self.on = False
        logger.info("SimCameraPart 종료")


# 테스트용 더미 카메라
class MockCameraPart:
    """테스트용 더미 카메라"""
    
    def __init__(self, width: int = 160, height: int = 120):
        self.width = width
        self.height = height
        
    def run(self) -> np.ndarray:
        """랜덤 이미지 반환"""
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
    
    def shutdown(self):
        pass
