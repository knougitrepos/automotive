import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import cv2
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import carla

# 프로젝트 루트 설정 (노트북과 동일한 구조 가정)
PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_carla(host='localhost', port=2000, timeout=10.0):
    """CARLA 서버에 연결하고 world 객체 반환"""
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        logger.info(f"CARLA 연결 성공: {world.get_map().name}")
        return client, world
    except Exception as e:
        logger.error(f"CARLA 연결 실패: {e}")
        raise ConnectionError("CARLA 서버에 연결할 수 없습니다. CarlaUE4.exe가 실행 중인지 확인해주세요.")

class SensorManager:
    """센서 관리 클래스 (RGB 카메라, 충돌, 차선 침범)"""
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = {}
        self.data = {}
        self.blueprint_library = world.get_blueprint_library()

    def setup_camera(self, sensor_config):
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
        array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
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

class DataCollector:
    """CARLA 데이터 수집기"""
    def __init__(self, world, save_dir):
        self.world = world
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vehicle = None
        self.sensor_manager = None
        self.current_episode = 0
        self.frames = []
        self.images = []
        self.blueprint_library = world.get_blueprint_library()
        self.spawn_points = world.get_map().get_spawn_points()

    def spawn_vehicle(self, blueprint_name='vehicle.tesla.model3'):
        vehicle_bp = self.blueprint_library.find(blueprint_name)
        spawn_point = np.random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"차량 스폰: {blueprint_name}")
        return self.vehicle

    def setup_sensors(self, sensor_config):
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        self.sensor_manager.setup_camera(sensor_config)
        self.sensor_manager.setup_collision_sensor()
        self.sensor_manager.setup_lane_invasion_sensor()

    def get_vehicle_state(self):
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
        return {
            'ego_speed': speed,
            'ego_location': [location.x, location.y, location.z],
            'ego_rotation': [rotation.pitch, rotation.yaw, rotation.roll],
            'traffic_light': traffic_light,
            'speed_limit': speed_limit,
            'steer': control.steer,
            'throttle': control.throttle,
            'brake': control.brake
        }

    def collect_frame(self, frame_id):
        world_frame = self.world.tick()
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
        self.frames = []
        self.images = []
        self.sensor_manager.reset_events()
        if enable_autopilot:
            self.vehicle.set_autopilot(True)
        for _ in range(10):
            self.world.tick()
        time.sleep(0.5)
        collision_count = 0
        for frame_id in tqdm(range(episode_length), desc=f"Episode {self.current_episode}"):
            frame_data = self.collect_frame(frame_id)
            if frame_data is None:
                continue
            if frame_data['collision']:
                collision_count += 1
                if collision_count > 3:
                    logger.warning("다중 충돌로 에피소드 조기 종료")
                    break
        self.vehicle.set_autopilot(False)
        return len(self.frames)

    def save_episode(self):
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
            'timestamp': datetime.now().isoformat()
        }
        with open(episode_dir / 'metadata.json', 'w') as f:
            json.dump(episode_info, f, indent=2)
        logger.info(f"에피소드 {self.current_episode} 저장 완료: {len(self.frames)} 프레임")
        self.current_episode += 1
        return episode_info

    def reset(self):
        if self.sensor_manager:
            self.sensor_manager.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        time.sleep(0.5)

    def cleanup(self):
        self.reset()
        logger.info("데이터 수집기 정리 완료")

print("✅ utils.carla_helper 모듈 로드 완료")
