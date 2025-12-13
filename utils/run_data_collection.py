# -*- coding: utf-8 -*-
"""
CARLA Data Collection with Traffic Environment
===============================================

다양한 교통 환경(NPC 차량, 보행자, 신호등)에서 자율주행 데이터를 수집하는 스크립트입니다.

Features:
---------
1. Ego 차량 + NPC 차량 30대 + 보행자 20명
2. Spectator 3인칭 자동 추적
3. 신호등 및 Traffic Manager 활성화
4. 에피소드별 데이터 저장

Usage:
------
python run_data_collection.py

참고 논문/문서:
- CARLA Simulator (Dosovitskiy et al., CoRL 2017)
- End-to-End Learning (Bojarski et al., 2016)
- DAgger (Ross et al., 2011)
- CARLA Traffic Manager: https://carla.readthedocs.io/en/latest/adv_traffic_manager/
"""

import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import cv2
import yaml
import math
import threading
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CARLA 임포트
try:
    import carla
    logger.info("CARLA 패키지 로드 성공")
except ImportError:
    logger.error("CARLA 패키지를 설치해주세요: pip install carla==0.9.15")
    sys.exit(1)

# 유틸리티 임포트
from utils.traffic_manager import TrafficSpawner
from utils.spectator_follow import SpectatorFollower, find_ego_vehicle


class SensorManager:
    """센서 관리 클래스"""
    
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = {}
        self.data = {}
        self.blueprint_library = world.get_blueprint_library()
    
    def setup_camera(self, sensor_config):
        """RGB 카메라 설정"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(sensor_config['width']))
        camera_bp.set_attribute('image_size_y', str(sensor_config['height']))
        camera_bp.set_attribute('fov', str(sensor_config['fov']))
        
        pos = sensor_config['position']
        rot = sensor_config.get('rotation', [0, 0, 0])
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
        )
        
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._process_camera(image))
        
        self.sensors['rgb_front'] = camera
        logger.info(f"카메라 설정 완료: {sensor_config['width']}x{sensor_config['height']}")
    
    def setup_collision_sensor(self):
        """충돌 센서 설정"""
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        collision_sensor.listen(lambda event: self._process_collision(event))
        self.sensors['collision'] = collision_sensor
        self.data['collision_history'] = []
        logger.info("충돌 센서 설정 완료")
    
    def setup_lane_invasion_sensor(self):
        """차선 침범 센서 설정"""
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        lane_sensor.listen(lambda event: self._process_lane_invasion(event))
        self.sensors['lane_invasion'] = lane_sensor
        self.data['lane_invasion_history'] = []
        logger.info("차선 침범 센서 설정 완료")
    
    def _process_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        self.data['current_image'] = array
        self.data['image_frame'] = image.frame
    
    def _process_collision(self, event):
        self.data['collision_history'].append({
            'frame': event.frame,
            'actor': event.other_actor.type_id if event.other_actor else 'unknown',
            'intensity': event.normal_impulse.length()
        })
    
    def _process_lane_invasion(self, event):
        self.data['lane_invasion_history'].append({
            'frame': event.frame,
            'lane_types': [str(x) for x in event.crossed_lane_markings]
        })
    
    def get_current_image(self):
        return self.data.get('current_image')
    
    def check_collision(self, frame):
        for col in self.data.get('collision_history', []):
            if col['frame'] == frame:
                return True
        return False
    
    def check_lane_invasion(self, frame):
        for lane in self.data.get('lane_invasion_history', []):
            if lane['frame'] == frame:
                return True
        return False
    
    def reset_events(self):
        self.data['collision_history'] = []
        self.data['lane_invasion_history'] = []
    
    def destroy(self):
        for name, sensor in self.sensors.items():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.sensors = {}
        logger.info("센서 정리 완료")


class DataCollectorWithTraffic:
    """교통 환경이 포함된 CARLA 데이터 수집기"""
    
    def __init__(self, client, world, config, save_dir):
        self.client = client
        self.world = world
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.vehicle = None
        self.sensor_manager = None
        self.traffic_spawner = None
        self.spectator_follower = None
        
        self.current_episode = 0
        self.frames = []
        self.images = []
        
        self.blueprint_library = world.get_blueprint_library()
        self.spawn_points = world.get_map().get_spawn_points()
        
        logger.info(f"DataCollector 초기화 완료 (저장 위치: {save_dir})")
    
    def spawn_ego_vehicle(self, blueprint_name=None):
        """Ego 차량 스폰"""
        if blueprint_name is None:
            blueprint_name = self.config.get('vehicle', {}).get('blueprint', 'vehicle.tesla.model3')
        
        vehicle_bp = self.blueprint_library.find(blueprint_name)
        
        # role_name 설정 (hero로 설정하여 ego 차량 식별)
        vehicle_bp.set_attribute('role_name', 'hero')
        
        spawn_point = np.random.choice(self.spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"Ego 차량 스폰: {blueprint_name}")
        return self.vehicle
    
    def setup_sensors(self):
        """센서 설정"""
        sensor_config = self.config['sensors']['rgb_front']
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        self.sensor_manager.setup_camera(sensor_config)
        self.sensor_manager.setup_collision_sensor()
        self.sensor_manager.setup_lane_invasion_sensor()
    
    def setup_traffic(self):
        """교통 환경 설정 (NPC 차량 + 보행자)"""
        self.traffic_spawner = TrafficSpawner(self.client, self.world, self.config)
        result = self.traffic_spawner.spawn_all()
        logger.info(f"교통 환경: 차량 {result['vehicles']}대, 보행자 {result['pedestrians']}명")
        return result
    
    def setup_spectator_follow(self):
        """Spectator 3인칭 추적 설정"""
        self.spectator_follower = SpectatorFollower(self.world)
        logger.info("Spectator Follow 설정 완료")
    
    def get_vehicle_state(self):
        """차량 상태 정보 수집"""
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        control = self.vehicle.get_control()
        
        traffic_light = 'none'
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            state = tl.get_state()
            if state == carla.TrafficLightState.Red:
                traffic_light = 'red'
            elif state == carla.TrafficLightState.Yellow:
                traffic_light = 'yellow'
            elif state == carla.TrafficLightState.Green:
                traffic_light = 'green'
        
        speed_limit = 50.0
        
        # 주변 객체 정보
        nearby = {'vehicles': 0, 'pedestrians': 0}
        if self.traffic_spawner:
            nearby_actors = self.traffic_spawner.get_nearby_actors(self.vehicle, radius=30.0)
            nearby['vehicles'] = len(nearby_actors['vehicles'])
            nearby['pedestrians'] = len(nearby_actors['pedestrians'])
        
        return {
            'ego_speed': speed,
            'ego_location': [location.x, location.y, location.z],
            'ego_rotation': [rotation.pitch, rotation.yaw, rotation.roll],
            'traffic_light': traffic_light,
            'speed_limit': speed_limit,
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake,
            'nearby_vehicles': nearby['vehicles'],
            'nearby_pedestrians': nearby['pedestrians']
        }
    
    def collect_frame(self, frame_id):
        """단일 프레임 수집"""
        world_frame = self.world.tick()
        
        # Spectator 업데이트
        if self.spectator_follower and self.vehicle:
            self.spectator_follower.follow(self.vehicle)
        
        time.sleep(0.05)
        
        image = self.sensor_manager.get_current_image()
        if image is None:
            return None
        
        state = self.get_vehicle_state()
        
        collision = self.sensor_manager.check_collision(world_frame)
        lane_invasion = self.sensor_manager.check_lane_invasion(world_frame)
        
        frame_data = {
            'frame_id': frame_id,
            'world_frame': world_frame,
            'collision': collision,
            'lane_invasion': lane_invasion,
            **state
        }
        
        self.frames.append(frame_data)
        self.images.append(image.copy())
        
        return frame_data
    
    def run_episode(self, episode_length, enable_autopilot=True):
        """에피소드 실행"""
        self.frames = []
        self.images = []
        self.sensor_manager.reset_events()
        
        if enable_autopilot:
            self.vehicle.set_autopilot(True)
        
        # 안정화 대기
        for _ in range(10):
            self.world.tick()
            if self.spectator_follower and self.vehicle:
                self.spectator_follower.follow(self.vehicle)
        time.sleep(0.5)
        
        # 데이터 수집
        collision_count = 0
        for frame_id in tqdm(range(episode_length), desc=f"Episode {self.current_episode}"):
            frame_data = self.collect_frame(frame_id)
            
            if frame_data is None:
                continue
            
            if frame_data['collision']:
                collision_count += 1
                if collision_count > 5:
                    logger.warning("다중 충돌로 에피소드 조기 종료")
                    break
        
        self.vehicle.set_autopilot(False)
        
        return len(self.frames)
    
    def save_episode(self):
        """에피소드 저장"""
        episode_dir = self.save_dir / f"episode_{self.current_episode:03d}"
        episode_dir.mkdir(exist_ok=True)
        
        image_dir = episode_dir / 'images'
        image_dir.mkdir(exist_ok=True)
        
        for i, (frame, image) in enumerate(zip(self.frames, self.images)):
            image_path = image_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frame['image_path'] = str(image_path)
        
        df = pd.DataFrame(self.frames)
        df.to_parquet(episode_dir / 'frames.parquet', index=False)
        
        episode_info = {
            'episode_id': self.current_episode,
            'num_frames': len(self.frames),
            'total_collisions': sum(1 for f in self.frames if f['collision']),
            'total_lane_invasions': sum(1 for f in self.frames if f['lane_invasion']),
            'avg_nearby_vehicles': np.mean([f['nearby_vehicles'] for f in self.frames]),
            'avg_nearby_pedestrians': np.mean([f['nearby_pedestrians'] for f in self.frames]),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(episode_dir / 'metadata.json', 'w') as f:
            json.dump(episode_info, f, indent=2)
        
        logger.info(f"에피소드 {self.current_episode} 저장 완료: {len(self.frames)} 프레임")
        self.current_episode += 1
        
        return episode_info
    
    def reset(self):
        """에피소드 리셋"""
        if self.sensor_manager:
            self.sensor_manager.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        
        time.sleep(0.5)
    
    def cleanup(self):
        """완전 정리"""
        self.reset()
        if self.traffic_spawner:
            self.traffic_spawner.cleanup()
        logger.info("데이터 수집기 정리 완료")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("CARLA Data Collection with Traffic Environment")
    print("=" * 60)
    
    # 설정 로드
    config_path = PROJECT_ROOT / 'config' / 'carla_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n설정 파일: {config_path}")
    print(f"에피소드 수: {config['collection']['num_episodes']}")
    print(f"에피소드당 프레임: {config['collection']['episode_length']}")
    print(f"NPC 차량: {config['traffic']['vehicles']['num_vehicles']}대")
    print(f"보행자: {config['traffic']['pedestrians']['num_pedestrians']}명")
    
    # CARLA 연결
    try:
        client = carla.Client(config['server']['host'], config['server']['port'])
        client.set_timeout(config['server']['timeout'])
        world = client.get_world()
        print(f"\n✅ CARLA 연결 성공: {world.get_map().name}")
    except Exception as e:
        print(f"\n❌ CARLA 연결 실패: {e}")
        print("CarlaUE4.exe가 실행 중인지 확인하세요.")
        return
    
    # 동기 모드 설정
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / config['collection']['fps']
    world.apply_settings(settings)
    print(f"✅ 동기 모드 활성화 (FPS: {config['collection']['fps']})")
    
    # 저장 디렉토리
    save_dir = PROJECT_ROOT / 'dataset' / 'carla_traffic'
    
    # 데이터 수집기 생성
    collector = DataCollectorWithTraffic(client, world, config, save_dir)
    
    # 수집 통계
    all_episodes_info = []
    
    try:
        # 에피소드 수집
        num_episodes = config['collection']['num_episodes']
        episode_length = config['collection']['episode_length']
        
        for ep in range(num_episodes):
            print(f"\n{'=' * 50}")
            print(f"에피소드 {ep + 1}/{num_episodes} 시작")
            print(f"{'=' * 50}")
            
            # Ego 차량 스폰
            collector.spawn_ego_vehicle()
            
            # 센서 설정
            collector.setup_sensors()
            
            # 교통 환경 설정 (첫 에피소드에서만)
            if ep == 0:
                collector.setup_traffic()
                collector.setup_spectator_follow()
            
            # 에피소드 실행
            num_frames = collector.run_episode(episode_length)
            
            # 저장
            episode_info = collector.save_episode()
            all_episodes_info.append(episode_info)
            
            # 리셋 (교통 환경은 유지)
            collector.reset()
            
            print(f"\n✅ 에피소드 {ep + 1} 완료:")
            print(f"   - 수집 프레임: {num_frames}")
            print(f"   - 충돌: {episode_info['total_collisions']}회")
            print(f"   - 차선 침범: {episode_info['total_lane_invasions']}회")
            print(f"   - 평균 주변 차량: {episode_info['avg_nearby_vehicles']:.1f}대")
            print(f"   - 평균 주변 보행자: {episode_info['avg_nearby_pedestrians']:.1f}명")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단됨")
    
    finally:
        collector.cleanup()
        world.apply_settings(original_settings)
    
    # 요약
    print(f"\n\n{'=' * 60}")
    print("데이터 수집 완료!")
    print(f"{'=' * 60}")
    print(f"저장 위치: {save_dir}")
    print(f"수집 에피소드: {len(all_episodes_info)}")
    
    if all_episodes_info:
        total_frames = sum(ep['num_frames'] for ep in all_episodes_info)
        print(f"총 프레임: {total_frames:,}")
    
    # 데이터셋 정보 저장
    dataset_info = {
        'num_episodes': len(all_episodes_info),
        'episodes': all_episodes_info,
        'config': {
            'num_vehicles': config['traffic']['vehicles']['num_vehicles'],
            'num_pedestrians': config['traffic']['pedestrians']['num_pedestrians'],
            'episode_length': config['collection']['episode_length'],
            'fps': config['collection']['fps']
        },
        'collection_date': datetime.now().isoformat()
    }
    
    with open(save_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n다음 단계: dataset/carla_traffic 폴더의 데이터를 확인하세요.")


if __name__ == "__main__":
    main()
