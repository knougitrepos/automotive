"""
데이터 처리 유틸리티
- 데이터 로드/저장
- 변환
- 증강
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml

# 로거 설정
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        설정 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 로드 완료: {config_path}")
        return config
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return {}


def save_episode_data(episode_dir: str, frames: List[Dict], images: List[np.ndarray]):
    """
    에피소드 데이터 저장
    
    Args:
        episode_dir: 에피소드 저장 디렉토리
        frames: 프레임 메타데이터 리스트
        images: 이미지 배열 리스트
    """
    import cv2
    
    episode_path = Path(episode_dir)
    episode_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 디렉토리
    image_dir = episode_path / 'images'
    image_dir.mkdir(exist_ok=True)
    
    # 이미지 저장
    for i, (frame, image) in enumerate(zip(frames, images)):
        image_filename = f"{i:06d}.jpg"
        image_path = image_dir / image_filename
        
        # RGB → BGR (OpenCV 저장용)
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame['image_path'] = str(image_path)
    
    # 메타데이터 저장 (Parquet)
    df = pd.DataFrame(frames)
    parquet_path = episode_path / 'frames.parquet'
    df.to_parquet(parquet_path, index=False)
    
    logger.info(f"에피소드 저장 완료: {episode_dir} ({len(frames)} 프레임)")


def load_episode_data(episode_dir: str) -> Tuple[pd.DataFrame, Path]:
    """
    에피소드 데이터 로드
    
    Args:
        episode_dir: 에피소드 디렉토리
    
    Returns:
        (프레임 DataFrame, 이미지 디렉토리 Path)
    """
    episode_path = Path(episode_dir)
    
    parquet_path = episode_path / 'frames.parquet'
    df = pd.read_parquet(parquet_path)
    
    image_dir = episode_path / 'images'
    
    logger.info(f"에피소드 로드 완료: {episode_dir} ({len(df)} 프레임)")
    return df, image_dir


def get_kitti_sequences(kitti_dir: str) -> List[str]:
    """
    KITTI 시퀀스 목록 가져오기
    
    Args:
        kitti_dir: KITTI 데이터 디렉토리
    
    Returns:
        시퀀스 디렉토리 경로 리스트
    """
    kitti_path = Path(kitti_dir)
    sequences = []
    
    # training/image_02 구조 확인
    image_dirs = list(kitti_path.rglob('image_02'))
    
    for image_dir in image_dirs:
        # 시퀀스 디렉토리들 찾기
        for seq_dir in image_dir.iterdir():
            if seq_dir.is_dir():
                sequences.append(str(seq_dir))
    
    logger.info(f"KITTI 시퀀스 {len(sequences)}개 발견")
    return sequences


def load_kitti_sequence(sequence_dir: str) -> List[str]:
    """
    KITTI 시퀀스 이미지 경로 로드
    
    Args:
        sequence_dir: 시퀀스 디렉토리
    
    Returns:
        이미지 경로 리스트 (정렬됨)
    """
    seq_path = Path(sequence_dir)
    image_paths = sorted(seq_path.glob('*.png'))
    
    if not image_paths:
        image_paths = sorted(seq_path.glob('*.jpg'))
    
    return [str(p) for p in image_paths]


def normalize_action(steer: float, throttle: float, brake: float) -> Dict[str, float]:
    """
    액션 값 정규화
    
    Args:
        steer: 핸들 각도 [-1, 1]
        throttle: 가속 [0, 1]
        brake: 제동 [0, 1]
    
    Returns:
        정규화된 액션 딕셔너리
    """
    return {
        'steer': np.clip(steer, -1.0, 1.0),
        'throttle': np.clip(throttle, 0.0, 1.0),
        'brake': np.clip(brake, 0.0, 1.0)
    }


def train_val_split(data: List[Any], val_ratio: float = 0.2, 
                    seed: int = 42) -> Tuple[List[Any], List[Any]]:
    """
    학습/검증 데이터 분할
    
    Args:
        data: 데이터 리스트
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
    
    Returns:
        (학습 데이터, 검증 데이터) 튜플
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    
    val_size = int(len(data) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    logger.info(f"데이터 분할: 학습 {len(train_data)}, 검증 {len(val_data)}")
    return train_data, val_data
