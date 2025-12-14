# -*- coding: utf-8 -*-
"""
Vehicle Loop
============

Donkeycar 스타일 Vehicle Loop.
Parts를 조합하여 주기적으로 실행합니다.

참고: Donkeycar 공식 문서
- https://docs.donkeycar.com/parts/about/
"""

import time
import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Vehicle 공유 메모리
    
    Parts 간 데이터 교환을 위한 키-값 저장소입니다.
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
    
    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)
    
    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def keys(self) -> List[str]:
        return list(self._data.keys())
    
    def values(self) -> Dict[str, Any]:
        return dict(self._data)


class PartEntry:
    """Part 등록 정보"""
    
    def __init__(self, part, inputs: List[str], outputs: List[str], 
                 run_threaded: bool = False):
        self.part = part
        self.inputs = inputs
        self.outputs = outputs
        self.run_threaded = run_threaded


class Vehicle:
    """
    Donkeycar 스타일 Vehicle Loop
    
    Parts를 추가하고 주기적으로 실행합니다.
    
    Example:
        v = Vehicle()
        v.add(camera, outputs=['image'])
        v.add(pilot, inputs=['image'], outputs=['steering', 'throttle'])
        v.add(actuator, inputs=['steering', 'throttle'])
        v.start(rate_hz=20)
    """
    
    def __init__(self):
        self.parts: List[PartEntry] = []
        self.memory = MemoryStore()
        self.running = False
        self.loop_count = 0
        
        logger.info("Vehicle 초기화")
    
    def add(self, part, inputs: List[str] = None, outputs: List[str] = None,
            run_threaded: bool = False):
        """
        Part 추가
        
        Args:
            part: Part 객체 (run() 메서드 필요)
            inputs: 메모리에서 읽을 키 목록
            outputs: 메모리에 쓸 키 목록
            run_threaded: 쓰레드 모드 사용 여부
        """
        inputs = inputs or []
        outputs = outputs or []
        
        entry = PartEntry(part, inputs, outputs, run_threaded)
        self.parts.append(entry)
        
        part_name = type(part).__name__
        logger.info(f"Part 추가: {part_name} (in={inputs}, out={outputs})")
    
    def start(self, rate_hz: float = 20, max_loops: Optional[int] = None):
        """
        Vehicle Loop 시작
        
        Args:
            rate_hz: 루프 주파수 (Hz)
            max_loops: 최대 루프 수 (None이면 무한)
        """
        self.running = True
        self.loop_count = 0
        loop_time = 1.0 / rate_hz
        
        logger.info(f"Vehicle Loop 시작: {rate_hz} Hz")
        
        try:
            while self.running:
                start_time = time.time()
                
                # 모든 Part 실행
                self._run_parts()
                
                self.loop_count += 1
                
                # 최대 루프 확인
                if max_loops is not None and self.loop_count >= max_loops:
                    logger.info(f"최대 루프 도달: {max_loops}")
                    break
                
                # 타이밍 조절
                elapsed = time.time() - start_time
                sleep_time = loop_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > loop_time * 1.5:
                    logger.warning(f"루프 지연: {elapsed:.3f}s > {loop_time:.3f}s")
                    
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트 감지")
        finally:
            self.shutdown()
    
    def _run_parts(self):
        """모든 Part 실행"""
        for entry in self.parts:
            # 입력 수집
            inputs = [self.memory[key] for key in entry.inputs]
            
            # Part 실행
            try:
                if entry.run_threaded:
                    outputs = entry.part.run_threaded(*inputs)
                else:
                    outputs = entry.part.run(*inputs)
            except Exception as e:
                part_name = type(entry.part).__name__
                logger.error(f"Part 실행 오류 ({part_name}): {e}")
                outputs = None
            
            # 출력 저장
            if outputs is not None:
                if len(entry.outputs) == 1:
                    self.memory[entry.outputs[0]] = outputs
                else:
                    # 튜플/리스트 언패킹
                    if isinstance(outputs, (tuple, list)):
                        for key, value in zip(entry.outputs, outputs):
                            self.memory[key] = value
                    else:
                        self.memory[entry.outputs[0]] = outputs
    
    def shutdown(self):
        """모든 Part 종료"""
        self.running = False
        
        for entry in self.parts:
            if hasattr(entry.part, 'shutdown'):
                try:
                    entry.part.shutdown()
                except Exception as e:
                    part_name = type(entry.part).__name__
                    logger.error(f"Part 종료 오류 ({part_name}): {e}")
        
        logger.info(f"Vehicle 종료 (총 {self.loop_count} 루프)")


def create_vehicle_from_config(config: Dict) -> Vehicle:
    """
    설정에서 Vehicle 생성
    
    Args:
        config: myconfig.py에서 로드한 설정
        
    Returns:
        구성된 Vehicle 객체
    """
    from parts import SimCameraPart, ActuatorPart, KerasPilot, TubWriter
    
    v = Vehicle()
    
    # 카메라
    camera = SimCameraPart(
        width=config.get('CAMERA_WIDTH', 160),
        height=config.get('CAMERA_HEIGHT', 120)
    )
    v.add(camera, outputs=['camera/image'])
    
    # 파일럿 (모델 있을 경우)
    model_path = config.get('MODEL_PATH')
    if model_path:
        pilot = KerasPilot(model_path=model_path)
        v.add(pilot, inputs=['camera/image'], outputs=['pilot/steering', 'pilot/throttle'])
    
    # Actuator
    actuator = ActuatorPart()
    v.add(actuator, inputs=['pilot/steering', 'pilot/throttle'])
    
    # 데이터 저장
    if config.get('RECORDING', False):
        tub_writer = TubWriter(base_path=config.get('DATA_PATH', './data'))
        v.add(tub_writer, inputs=['camera/image', 'pilot/steering', 'pilot/throttle'])
    
    return v


if __name__ == "__main__":
    # 간단한 테스트
    from parts.camera import MockCameraPart
    from parts.pilot import MockPilot
    from parts.actuator import MockActuatorPart
    
    v = Vehicle()
    v.add(MockCameraPart(), outputs=['image'])
    v.add(MockPilot(), inputs=['image'], outputs=['steering', 'throttle'])
    v.add(MockActuatorPart(), inputs=['steering', 'throttle'])
    
    v.start(rate_hz=10, max_loops=50)
    print("테스트 완료!")
