"""
CARLA 유틸리티 함수
- 클라이언트 연결
- 센서 설정
- 데이터 변환
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

# 로거 설정
logger = logging.getLogger(__name__)


def connect_carla(host: str = 'localhost', port: int = 2000, timeout: float = 10.0):
    """
    CARLA 서버에 연결
    
    Args:
        host: CARLA 서버 호스트
        port: CARLA 서버 포트
        timeout: 연결 타임아웃 (초)
    
    Returns:
        tuple: (client, world) 또는 연결 실패 시 (None, None)
    """
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        logger.info(f"CARLA 연결 성공: {world.get_map().name}")
        return client, world
    except ImportError:
        logger.error("CARLA 패키지가 설치되지 않았습니다. 'pip install carla==0.9.15' 실행 필요")
        return None, None
    except Exception as e:
        logger.error(f"CARLA 연결 실패: {e}")
        return None, None


def setup_camera(world, vehicle, config: Dict[str, Any]):
    """
    차량에 RGB 카메라 부착
    
    Args:
        world: CARLA world 객체
        vehicle: 차량 액터
        config: 카메라 설정 딕셔너리
    
    Returns:
        카메라 센서 액터
    """
    try:
        import carla
        
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # 카메라 속성 설정
        camera_bp.set_attribute('image_size_x', str(config.get('width', 800)))
        camera_bp.set_attribute('image_size_y', str(config.get('height', 600)))
        camera_bp.set_attribute('fov', str(config.get('fov', 100)))
        
        # 부착 위치
        pos = config.get('position', [1.5, 0.0, 2.4])
        rot = config.get('rotation', [0, 0, 0])
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
        )
        
        camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        logger.info(f"카메라 부착 완료: {config.get('width')}x{config.get('height')}")
        return camera
        
    except Exception as e:
        logger.error(f"카메라 설정 실패: {e}")
        return None


def carla_image_to_numpy(carla_image) -> np.ndarray:
    """
    CARLA 이미지를 numpy 배열로 변환
    
    Args:
        carla_image: CARLA 센서 이미지
    
    Returns:
        numpy 배열 (H, W, 3) RGB 형식
    """
    # BGRA → RGB 변환
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]  # BGRA → BGR
    array = array[:, :, ::-1]  # BGR → RGB
    return array


def get_vehicle_speed(vehicle) -> float:
    """
    차량 속도 계산 (m/s)
    
    Args:
        vehicle: CARLA 차량 액터
    
    Returns:
        속도 (m/s)
    """
    velocity = vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed


def get_traffic_light_state(vehicle) -> str:
    """
    차량 앞 신호등 상태 확인
    
    Args:
        vehicle: CARLA 차량 액터
    
    Returns:
        'red', 'yellow', 'green', 또는 'none'
    """
    try:
        import carla
        
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            state = traffic_light.get_state()
            
            if state == carla.TrafficLightState.Red:
                return 'red'
            elif state == carla.TrafficLightState.Yellow:
                return 'yellow'
            elif state == carla.TrafficLightState.Green:
                return 'green'
        
        return 'none'
    except Exception:
        return 'none'


def spawn_vehicle(world, blueprint_name: str = 'vehicle.tesla.model3', 
                  spawn_point: Optional[Any] = None):
    """
    차량 스폰
    
    Args:
        world: CARLA world 객체
        blueprint_name: 차량 블루프린트 이름
        spawn_point: 스폰 위치 (None이면 랜덤)
    
    Returns:
        차량 액터
    """
    try:
        import carla
        
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find(blueprint_name)
        
        if spawn_point is None:
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"차량 스폰 완료: {blueprint_name}")
        return vehicle
        
    except Exception as e:
        logger.error(f"차량 스폰 실패: {e}")
        return None


def cleanup_actors(actors: list):
    """
    액터들 정리 (삭제)
    
    Args:
        actors: 삭제할 액터 리스트
    """
    for actor in actors:
        if actor is not None:
            try:
                actor.destroy()
            except Exception as e:
                logger.warning(f"액터 삭제 실패: {e}")
    
    logger.info(f"{len(actors)}개 액터 정리 완료")
