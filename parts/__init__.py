# -*- coding: utf-8 -*-
"""
Donkeycar Parts 패키지
=====================

Donkeycar 스타일 Parts를 정의합니다.
각 Part는 run() 메서드를 통해 Vehicle Loop에서 주기적으로 실행됩니다.

참고:
- Donkeycar 공식 문서: https://docs.donkeycar.com
- gym-donkeycar: https://github.com/tawnkramer/gym-donkeycar
"""

from .camera import SimCameraPart
from .actuator import ActuatorPart
from .pilot import KerasPilot
from .datastore import TubWriter

__all__ = [
    'SimCameraPart',
    'ActuatorPart', 
    'KerasPilot',
    'TubWriter',
]

print("✅ parts 패키지 로드 완료")
