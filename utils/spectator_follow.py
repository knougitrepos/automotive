# -*- coding: utf-8 -*-
"""
CARLA Spectator Follow Utility
=============================

서버 창(언리얼 GUI)에서 ego 차량을 자동 추적하는 유틸리티입니다.

Features:
---------
1. 3인칭 자동 추적 (ego 차량 뒤에 카메라 고정)
2. 실시간 Spectator Transform 업데이트

Usage (단독 실행):
-----------------
python spectator_follow.py

Usage (다른 스크립트에서 사용):
-----------------------------
from spectator_follow import SpectatorFollower
follower = SpectatorFollower(world)
follower.follow(ego_vehicle)  # 단일 호출
# 또는
follower.start_follow_loop(ego_vehicle)  # 무한 루프

참고 논문/문서:
- CARLA Simulator: https://carla.readthedocs.io/
- Spectator 설정: https://carla.readthedocs.io/en/latest/core_world/
"""

import carla
import math
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectatorFollower:
    """
    Spectator (서버 창 카메라)를 ego 차량 뒤에서 3인칭으로 추적하는 클래스
    
    Parameters:
    -----------
    world : carla.World
        CARLA 월드 객체
    distance : float
        차량 뒤로 떨어진 거리 (기본값: 6.0m)
    height : float
        차량 위로 올라간 높이 (기본값: 2.5m)
    pitch : float
        카메라 아래로 기울어진 각도 (기본값: -15.0도)
    """
    
    def __init__(self, world, distance=6.0, height=2.5, pitch=-15.0):
        self.world = world
        self.spectator = world.get_spectator()
        
        # 카메라 설정
        self.distance = distance
        self.height = height
        self.pitch = pitch
        
        logger.info(f"SpectatorFollower 초기화 완료 (distance={distance}, height={height}, pitch={pitch})")
    
    def follow(self, vehicle):
        """
        단일 프레임에서 spectator를 차량 뒤로 이동
        
        Parameters:
        -----------
        vehicle : carla.Vehicle
            추적할 ego 차량
        """
        if vehicle is None:
            return
            
        tr = vehicle.get_transform()
        yaw_rad = math.radians(tr.rotation.yaw)
        
        # 차량 뒤쪽으로 distance만큼, 위로 height만큼 오프셋
        offset = carla.Location(
            x=-self.distance * math.cos(yaw_rad),
            y=-self.distance * math.sin(yaw_rad),
            z=self.height
        )
        
        cam_location = tr.location + offset
        cam_rotation = carla.Rotation(
            pitch=self.pitch,
            yaw=tr.rotation.yaw,
            roll=0.0
        )
        
        self.spectator.set_transform(carla.Transform(cam_location, cam_rotation))
    
    def start_follow_loop(self, vehicle, fps=30):
        """
        무한 루프로 차량을 추적 (별도 스레드에서 실행 권장)
        
        Parameters:
        -----------
        vehicle : carla.Vehicle
            추적할 ego 차량
        fps : int
            업데이트 주기 (초당 프레임)
        """
        interval = 1.0 / fps
        logger.info(f"Spectator follow 루프 시작 (FPS: {fps})")
        
        try:
            while True:
                self.follow(vehicle)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Spectator follow 루프 종료")


def find_ego_vehicle(world, role_name='hero'):
    """
    월드에서 ego 차량 찾기
    
    Parameters:
    -----------
    world : carla.World
        CARLA 월드 객체
    role_name : str
        차량의 role_name 속성 (기본값: 'hero')
    
    Returns:
    --------
    carla.Vehicle or None
    """
    # role_name='hero'로 스폰된 차량 찾기
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            logger.info(f"Ego 차량 발견: {actor.type_id} (role_name={role_name})")
            return actor
    
    # 없으면 첫 번째 차량 반환
    vehicles = world.get_actors().filter("vehicle.*")
    if len(vehicles) > 0:
        ego = vehicles[0]
        logger.info(f"Ego 차량 (기본): {ego.type_id}")
        return ego
    
    logger.warning("차량을 찾을 수 없습니다")
    return None


def main():
    """단독 실행 시 Spectator Follow 데모"""
    print("=" * 50)
    print("CARLA Spectator Follow Utility")
    print("=" * 50)
    print("\n3인칭 자동 추적을 시작합니다...")
    print("종료하려면 Ctrl+C를 누르세요.\n")
    
    # CARLA 연결
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✅ CARLA 연결 성공: {world.get_map().name}")
    except Exception as e:
        print(f"❌ CARLA 연결 실패: {e}")
        print("\nCarlaUE4.exe가 실행 중인지 확인하세요.")
        return
    
    # Ego 차량 찾기
    ego = find_ego_vehicle(world)
    if ego is None:
        print("❌ 차량을 찾을 수 없습니다. 먼저 차량을 스폰하세요.")
        return
    
    print(f"✅ 추적 대상: {ego.type_id}")
    print("\n서버 창(언리얼 GUI)에서 자동 추적이 시작됩니다...")
    
    # Spectator Follow 시작
    follower = SpectatorFollower(world)
    follower.start_follow_loop(ego)


if __name__ == "__main__":
    main()
