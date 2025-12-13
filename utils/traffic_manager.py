# -*- coding: utf-8 -*-
"""
CARLA Traffic Manager Utility
=============================

NPC 차량, 보행자, 신호등을 관리하는 유틸리티입니다.

Features:
---------
1. NPC 차량 스폰 및 자동 주행
2. 보행자 스폰 및 랜덤 이동
3. Traffic Manager 설정

Usage:
------
from traffic_manager import TrafficSpawner
spawner = TrafficSpawner(client, world, config)
spawner.spawn_all()
# ... 데이터 수집 ...
spawner.cleanup()

참고 논문/문서:
- CARLA Traffic Manager: https://carla.readthedocs.io/en/latest/adv_traffic_manager/
- CARLA Actors: https://carla.readthedocs.io/en/latest/core_actors/
"""

import carla
import random
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficSpawner:
    """
    CARLA Traffic Manager를 활용한 NPC 차량 및 보행자 스폰 관리 클래스
    
    Parameters:
    -----------
    client : carla.Client
        CARLA 클라이언트 객체
    world : carla.World
        CARLA 월드 객체
    config : dict
        설정 딕셔너리 (carla_config.yaml에서 로드)
    """
    
    def __init__(self, client, world, config):
        self.client = client
        self.world = world
        self.config = config
        
        self.blueprint_library = world.get_blueprint_library()
        self.spawn_points = world.get_map().get_spawn_points()
        
        # 스폰된 액터 추적
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
        
        # Traffic Manager
        self.traffic_manager = None
        
        # 설정 로드
        self.traffic_config = config.get('traffic', {})
        
        logger.info("TrafficSpawner 초기화 완료")
    
    def setup_traffic_manager(self):
        """Traffic Manager 초기화 및 설정"""
        tm_config = self.traffic_config.get('traffic_manager', {})
        tm_port = tm_config.get('port', 8000)
        
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        
        # Global 설정
        if tm_config.get('hybrid_physics_mode', False):
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(
                tm_config.get('hybrid_physics_radius', 70.0)
            )
        
        if tm_config.get('respawn_dormant_vehicles', False):
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            tm_config.get('global_distance_to_leading', 2.5)
        )
        
        logger.info(f"Traffic Manager 초기화 완료 (port: {tm_port})")
        return self.traffic_manager
    
    def spawn_vehicles(self, num_vehicles=None):
        """
        NPC 차량 스폰
        
        Parameters:
        -----------
        num_vehicles : int, optional
            스폰할 차량 수 (None이면 설정에서 읽음)
        
        Returns:
        --------
        list : 스폰된 차량 리스트
        """
        vehicle_config = self.traffic_config.get('vehicles', {})
        
        if not vehicle_config.get('enabled', True):
            logger.info("NPC 차량 스폰 비활성화됨")
            return []
        
        if num_vehicles is None:
            num_vehicles = vehicle_config.get('num_vehicles', 30)
        
        # 차량 블루프린트 가져오기
        vehicle_filter = vehicle_config.get('filter', 'vehicle.*')
        blueprints = self.blueprint_library.filter(vehicle_filter)
        
        # 안전 스폰 (ego 차량과 겹치지 않도록)
        available_spawn_points = self.spawn_points.copy()
        random.shuffle(available_spawn_points)
        
        # 필요한 만큼만 사용
        num_vehicles = min(num_vehicles, len(available_spawn_points))
        
        # 배치 스폰 준비
        batch = []
        for i, spawn_point in enumerate(available_spawn_points[:num_vehicles]):
            blueprint = random.choice(blueprints)
            
            # 색상 랜덤화
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            # 드라이버 ID 설정 (선택적)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            batch.append(
                carla.command.SpawnActor(blueprint, spawn_point)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.traffic_manager.get_port()))
            )
        
        # 배치 실행
        results = self.client.apply_batch_sync(batch, True)
        
        for result in results:
            if not result.error:
                self.vehicles.append(result.actor_id)
        
        # 차량 행동 설정
        self._configure_vehicle_behavior()
        
        logger.info(f"NPC 차량 {len(self.vehicles)}대 스폰 완료")
        return self.vehicles
    
    def _configure_vehicle_behavior(self):
        """스폰된 차량들의 행동 설정"""
        behavior_config = self.traffic_config.get('vehicle_behavior', {})
        
        for vehicle_id in self.vehicles:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle is None:
                continue
            
            # 신호 무시 확률
            ignore_lights = behavior_config.get('ignore_lights_percentage', 0.0)
            self.traffic_manager.ignore_lights_percentage(vehicle, ignore_lights)
            
            # 표지판 무시 확률
            ignore_signs = behavior_config.get('ignore_signs_percentage', 0.0)
            self.traffic_manager.ignore_signs_percentage(vehicle, ignore_signs)
            
            # 차선 변경 확률
            left_change = behavior_config.get('random_left_lane_change', 0.1)
            right_change = behavior_config.get('random_right_lane_change', 0.1)
            self.traffic_manager.random_left_lanechange_percentage(vehicle, left_change * 100)
            self.traffic_manager.random_right_lanechange_percentage(vehicle, right_change * 100)
            
            # 속도 편차
            speed_diff = behavior_config.get('speed_difference_percentage', 30)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_diff)
    
    def spawn_pedestrians(self, num_pedestrians=None):
        """
        보행자 스폰
        
        Parameters:
        -----------
        num_pedestrians : int, optional
            스폰할 보행자 수 (None이면 설정에서 읽음)
        
        Returns:
        --------
        list : 스폰된 보행자 리스트
        """
        ped_config = self.traffic_config.get('pedestrians', {})
        
        if not ped_config.get('enabled', True):
            logger.info("보행자 스폰 비활성화됨")
            return []
        
        if num_pedestrians is None:
            num_pedestrians = ped_config.get('num_pedestrians', 20)
        
        # 보행자 블루프린트
        ped_filter = ped_config.get('filter', 'walker.pedestrian.*')
        walker_blueprints = self.blueprint_library.filter(ped_filter)
        
        # 보행자 컨트롤러 블루프린트
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        
        # 보행자 스폰 위치 (네비게이션 메시 위)
        spawn_locations = []
        for _ in range(num_pedestrians):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_locations.append(spawn_point)
        
        if len(spawn_locations) == 0:
            logger.warning("보행자 스폰 위치를 찾을 수 없습니다")
            return []
        
        # 배치 스폰 - 보행자
        batch = []
        walker_speeds = []
        run_probability = ped_config.get('run_probability', 0.2)
        
        for spawn_point in spawn_locations:
            walker_bp = random.choice(walker_blueprints)
            
            # 무적 모드 해제 (충돌 감지용)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            # 걷기/뛰기 속도 결정
            if walker_bp.has_attribute('speed'):
                if random.random() < run_probability:
                    # 뛰기
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    # 걷기
                    walker_speeds.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speeds.append(0)
            
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        # 보행자 스폰 실행
        results = self.client.apply_batch_sync(batch, True)
        
        walkers_list = []
        for i, result in enumerate(results):
            if not result.error:
                walkers_list.append({
                    'id': result.actor_id,
                    'speed': walker_speeds[i]
                })
        
        # 배치 스폰 - 컨트롤러
        batch = []
        for walker_info in walkers_list:
            batch.append(
                carla.command.SpawnActor(
                    walker_controller_bp, 
                    carla.Transform(), 
                    walker_info['id']
                )
            )
        
        # 컨트롤러 스폰 실행
        results = self.client.apply_batch_sync(batch, True)
        
        for i, result in enumerate(results):
            if not result.error:
                walkers_list[i]['controller'] = result.actor_id
        
        # 월드 틱 (컨트롤러 초기화)
        self.world.tick()
        
        # 컨트롤러 시작 및 목적지 설정
        crossing_factor = ped_config.get('crossing_factor', 0.3)
        self.world.set_pedestrians_cross_factor(crossing_factor)
        
        for walker_info in walkers_list:
            if 'controller' not in walker_info:
                continue
                
            controller = self.world.get_actor(walker_info['controller'])
            if controller is None:
                continue
            
            # 컨트롤러 시작
            controller.start()
            
            # 랜덤 목적지 설정
            destination = self.world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            
            # 속도 설정
            controller.set_max_speed(float(walker_info['speed']))
            
            self.walkers.append(walker_info['id'])
            self.walker_controllers.append(walker_info['controller'])
        
        logger.info(f"보행자 {len(self.walkers)}명 스폰 완료")
        return self.walkers
    
    def spawn_all(self):
        """모든 교통 참여자 스폰 (차량 + 보행자)"""
        # Traffic Manager 설정
        self.setup_traffic_manager()
        
        # 차량 스폰
        self.spawn_vehicles()
        
        # 보행자 스폰
        self.spawn_pedestrians()
        
        # 안정화 대기
        time.sleep(1.0)
        for _ in range(10):
            self.world.tick()
        
        logger.info(f"교통 환경 설정 완료: 차량 {len(self.vehicles)}대, 보행자 {len(self.walkers)}명")
        return {
            'vehicles': len(self.vehicles),
            'pedestrians': len(self.walkers)
        }
    
    def cleanup(self):
        """모든 스폰된 액터 제거"""
        # 컨트롤러 정지
        for controller_id in self.walker_controllers:
            controller = self.world.get_actor(controller_id)
            if controller is not None:
                controller.stop()
        
        # 배치 삭제
        actor_ids = self.vehicles + self.walkers + self.walker_controllers
        
        if actor_ids:
            batch = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
            self.client.apply_batch_sync(batch, True)
        
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
        
        logger.info("모든 NPC 액터 정리 완료")
    
    def get_nearby_actors(self, ego_vehicle, radius=50.0):
        """
        ego 차량 주변의 액터 정보 수집
        
        Parameters:
        -----------
        ego_vehicle : carla.Vehicle
            기준 차량
        radius : float
            검색 반경 (m)
        
        Returns:
        --------
        dict : {'vehicles': [...], 'pedestrians': [...]}
        """
        ego_location = ego_vehicle.get_transform().location
        
        nearby_vehicles = []
        nearby_pedestrians = []
        
        # 주변 차량
        for vehicle_id in self.vehicles:
            vehicle = self.world.get_actor(vehicle_id)
            if vehicle is None:
                continue
            
            vehicle_location = vehicle.get_transform().location
            distance = ego_location.distance(vehicle_location)
            
            if distance <= radius:
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
                
                nearby_vehicles.append({
                    'id': vehicle_id,
                    'type': vehicle.type_id,
                    'distance': distance,
                    'speed': speed,
                    'location': [vehicle_location.x, vehicle_location.y, vehicle_location.z]
                })
        
        # 주변 보행자
        for walker_id in self.walkers:
            walker = self.world.get_actor(walker_id)
            if walker is None:
                continue
            
            walker_location = walker.get_transform().location
            distance = ego_location.distance(walker_location)
            
            if distance <= radius:
                nearby_pedestrians.append({
                    'id': walker_id,
                    'type': walker.type_id,
                    'distance': distance,
                    'location': [walker_location.x, walker_location.y, walker_location.z]
                })
        
        return {
            'vehicles': sorted(nearby_vehicles, key=lambda x: x['distance']),
            'pedestrians': sorted(nearby_pedestrians, key=lambda x: x['distance'])
        }


def main():
    """단독 실행 테스트"""
    import yaml
    from pathlib import Path
    
    print("=" * 50)
    print("CARLA Traffic Manager Test")
    print("=" * 50)
    
    # 설정 로드
    config_path = Path(__file__).parent.parent / 'config' / 'carla_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # CARLA 연결
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"✅ CARLA 연결 성공: {world.get_map().name}")
    except Exception as e:
        print(f"❌ CARLA 연결 실패: {e}")
        return
    
    # Traffic Spawner 생성
    spawner = TrafficSpawner(client, world, config)
    
    try:
        # 동기 모드 설정
        settings = world.get_settings()
        original_settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # 교통 환경 스폰
        result = spawner.spawn_all()
        print(f"\n✅ 스폰 완료: 차량 {result['vehicles']}대, 보행자 {result['pedestrians']}명")
        print("\n30초 동안 시뮬레이션 실행 중... (Ctrl+C로 종료)")
        
        # 시뮬레이션 실행
        for i in range(300):  # 30초
            world.tick()
            time.sleep(0.1)
            if i % 50 == 0:
                print(f"  {i//10}초 경과...")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    
    finally:
        # 정리
        spawner.cleanup()
        world.apply_settings(original_settings)
        print("✅ 정리 완료")


if __name__ == "__main__":
    main()
