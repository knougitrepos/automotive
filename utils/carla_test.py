import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.carla_helper import connect_carla, DataCollector

# CARLA 연결
client, world = connect_carla(host='localhost', port=2000, timeout=10.0)

# 데이터 수집기 초기화
SAVE_DIR = PROJECT_ROOT / 'data' / 'carla_episodes'
collector = DataCollector(world, save_dir=str(SAVE_DIR))
collector.spawn_vehicle()
collector.setup_sensors({'width': 800, 'height': 600, 'fov': 90, 'position': [2.0, 0.0, 1.4]})
collector.run_episode(episode_length=500)
collector.save_episode()
collector.cleanup()