# Utils 패키지
"""
자율주행 AI 파이프라인 유틸리티 모듈

CARLA 관련 모듈:
- spectator_follow: Spectator 3인칭 추적
- traffic_manager: NPC 차량 및 보행자 관리
- run_data_collection: 교통 환경 데이터 수집 스크립트
"""

# CARLA 유틸리티
try:
    from .spectator_follow import SpectatorFollower, find_ego_vehicle
    from .traffic_manager import TrafficSpawner
except ImportError:
    # CARLA가 설치되지 않은 경우 무시
    pass
