
import sys
import os
import time
import logging
import numpy as np
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path().absolute()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    print("⚠️ CARLA Import Failed")

@dataclass
class VehicleState:
    speed: float = 0.0
    def speed_kmh(self) -> float:
        return self.speed * 3.6

class CARLAEvaluator:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
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
            print(f"Connection error: {e}")
            return False

    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        spawn_point.location.z += 2.0  # Spawn slightly higher to avoid ground collision
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle

    def get_vehicle_state(self):
        vel = self.vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return VehicleState(speed=speed)

    def run_debug_episode(self, duration_frames=100):
        print("\n--- Starting FORCE THROTTLE Test ---")
        self.vehicle.set_light_state(carla.VehicleLightState.HighBeam)
        
        # 워밍업: 물리 엔진 안정화
        for _ in range(20):
            self.world.tick()
            
        for frame in range(duration_frames):
            self.world.tick()
            
            # 강제 직진 명령
            control = carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0)
            control.manual_gear_shift = False
            self.vehicle.apply_control(control)
            
            state = self.get_vehicle_state()
            
            if frame % 10 == 0:
                print(f"Frame {frame}: Speed {state.speed_kmh():.2f} km/h, Throttle: 1.0")
                
    def cleanup(self):
        if self.vehicle: self.vehicle.destroy()
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

if __name__ == "__main__":
    evaluator = CARLAEvaluator()
    if evaluator.connect():
        try:
            evaluator.spawn_vehicle()
            evaluator.run_debug_episode()
        finally:
            evaluator.cleanup()
