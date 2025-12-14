
import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import cv2
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

PROJECT_ROOT = Path().absolute()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

@dataclass
class VehicleState:
    speed: float = 0.0
    speed_limit: float = 50.0
    traffic_light: str = 'none'
    distance_to_traffic_light: float = float('inf')
    obstacle_distance: float = float('inf')
    obstacle_type: str = 'none'
    lane_invasion: bool = False
    
    def speed_kmh(self) -> float:
        return self.speed * 3.6
    
    def is_over_speed_limit(self, margin: float = 0.0) -> bool:
        return self.speed_kmh() > (self.speed_limit + margin)

@dataclass
class Action:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

class SafetyShield:
    def __init__(self, config):
        self.enabled = config.get('enabled', True)
        self.rules = config.get('rules', {})
        self.params = config.get('params', {})
        self.intervention_count = 0
    
    def apply(self, action: Action, state: VehicleState) -> tuple:
        if not self.enabled:
            return action, []
        
        safe_action = Action(
            steer=action.steer,
            throttle=action.throttle,
            brake=action.brake
        )
        applied_rules = []
        
        # 빨간불 정지
        if self.rules.get('red_light_stop', True) and state.traffic_light == 'red':
            safe_action.throttle = 0.0
            safe_action.brake = 1.0
            applied_rules.append('red_light_stop')
        
        # 속도 제한
        if self.rules.get('speed_limit', True):
            margin = self.params.get('speed_limit_margin', 5.0)
            if state.is_over_speed_limit(margin):
                safe_action.throttle = 0.0
                applied_rules.append('speed_limit')
        
        # 충돌 회피 (Simple placeholder logic as standard evaluation might not have obstacles populated)
        if self.rules.get('collision_avoid', True):
            emergency_dist = self.params.get('emergency_brake_distance', 5.0)
            if state.obstacle_distance < emergency_dist:
                brake_force = min(1.0, emergency_dist / max(state.obstacle_distance, 0.1))
                safe_action.brake = max(safe_action.brake, brake_force)
                safe_action.throttle = 0.0
                applied_rules.append('collision_avoid')

        if applied_rules:
            self.intervention_count += 1
        
        return safe_action, applied_rules

class DrivingPolicyModel(nn.Module):
    def __init__(self, encoder_type='resnet50'):
        super().__init__()
        HIDDEN_DIMS = [512, 256]
        DROPOUT = 0.3
        
        if encoder_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=False)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = models.resnet34(pretrained=False)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_dim, HIDDEN_DIMS[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[1], 3)
        )
        
        self.steer_activation = nn.Tanh()
        self.throttle_brake_activation = nn.Sigmoid()
    
    def forward(self, x):
        features = self.encoder(x)
        raw_output = self.policy_head(features)
        steer = self.steer_activation(raw_output[:, 0:1])
        throttle = self.throttle_brake_activation(raw_output[:, 1:2])
        brake = self.throttle_brake_activation(raw_output[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)

class CARLAEvaluator:
    def __init__(self, model, shield, transform, device):
        self.model = model
        self.shield = shield
        self.transform = transform
        self.device = device
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.current_image = None
    
    def connect(self, host='localhost', port=2000):
        if not CARLA_AVAILABLE: return False
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            return True
        except Exception as e:
            return False
    
    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        spawn_point.location.z += 2.0
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        return self.vehicle
    
    def setup_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self._process_image)
    
    def _process_image(self, image):
        array = np.array(image.raw_data, dtype=np.uint8).copy()
        array = array.reshape((image.height, image.width, 4))
        self.current_image = array[:, :, :3][:, :, ::-1] # RGB
    
    def get_vehicle_state(self):
        vel = self.vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        # Note: Traffic light state and obstacles are mocked here as they require more complex sensors
        return VehicleState(speed=speed)

    def predict_action(self, image: np.ndarray):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_tensor = self.model(image_tensor)
            action_np = action_tensor.cpu().numpy()[0]
        return action_np 

    def run_episode(self, max_frames=200):
        print("Starting Fixed Episode (Model + Shield)...")
        # Warmup
        for _ in range(20):
            self.world.tick()
            
        for frame in range(max_frames):
            self.world.tick()
            if self.current_image is None: continue
            
            raw_action = self.predict_action(self.current_image)
            
            # --- FIX APPLIED HERE ---
            # 1. Throttle Boost
            boosted_throttle = min(float(raw_action[1]) * 2.0, 1.0)
            # 2. Brake Thresholding
            clean_brake = float(raw_action[2])
            if clean_brake < 0.1: clean_brake = 0.0
            
            # Create Action object
            action = Action(
                steer=float(raw_action[0]),
                throttle=boosted_throttle,
                brake=clean_brake
            )
            
            # Apply Safety Shield
            state = self.get_vehicle_state()
            safe_action, applied_rules = self.shield.apply(action, state)
            
            if frame % 10 == 0:
                print(f"[Frame {frame}] Speed: {state.speed_kmh():.2f} km/h | Rules: {applied_rules}")
                print(f"  Raw     -> T: {raw_action[1]:.3f}, B: {raw_action[2]:.3f}")
                print(f"  Boosted -> T: {action.throttle:.3f}, B: {action.brake:.3f}")
                print(f"  Safe    -> T: {safe_action.throttle:.3f}, B: {safe_action.brake:.3f}")
            
            control = carla.VehicleControl(
                steer=safe_action.steer,
                throttle=safe_action.throttle,
                brake=safe_action.brake
            )
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)
    
    def cleanup(self):
        if self.camera: self.camera.destroy()
        if self.vehicle: self.vehicle.destroy()
        if self.world:
             settings = self.world.get_settings()
             settings.synchronous_mode = False
             self.world.apply_settings(settings)

def main():
    if not CARLA_AVAILABLE:
        print("CARLA module not found.")
        return

    # Load Config for Shield (Mocking config loading or use default)
    shield_config = {'enabled': True, 'rules': {'red_light_stop': True, 'speed_limit': True, 'collision_avoid': True, 'pedestrian_priority': True}}
    shield = SafetyShield(shield_config)

    # Load Model
    model = DrivingPolicyModel()
    ckpt_path = PROJECT_ROOT / 'checkpoints' / 'bc' / 'best_bc_model.pth'
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded: {ckpt_path}")
    model = model.to(device).eval()
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    evaluator = CARLAEvaluator(model, shield, transform, device)
    if evaluator.connect():
        try:
            evaluator.spawn_vehicle()
            evaluator.setup_camera()
            time.sleep(1) 
            evaluator.run_episode(max_frames=200)
        finally:
            evaluator.cleanup()

if __name__ == "__main__":
    main()
