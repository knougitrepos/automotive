# -*- coding: utf-8 -*-
"""
Datastore Part
==============

Donkeycar Tub 형식으로 데이터를 저장하는 Part.

Tub 형식:
- data/ 디렉토리에 이미지 + JSON 메타데이터 저장
- 각 레코드: {image, steering, throttle, timestamp}
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class TubWriter:
    """
    Donkeycar Tub 형식 데이터 저장 Part
    
    주행 중 카메라 이미지와 제어 입력을 저장합니다.
    
    저장 구조:
    data/
    ├── tub_YYYYMMDD_HHMMSS/
    │   ├── images/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   ├── records.json
    │   └── manifest.json
    """
    
    def __init__(self, base_path: str = './data', max_records: int = 10000):
        """
        Args:
            base_path: 데이터 저장 기본 경로
            max_records: 최대 레코드 수
        """
        self.base_path = Path(base_path)
        self.max_records = max_records
        self.tub_path: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.records = []
        self.record_count = 0
        self.recording = False
        
        # 기본 디렉토리 생성
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TubWriter 초기화: {base_path}")
    
    def start_recording(self, name: Optional[str] = None):
        """
        녹화 시작 - 새 Tub 생성
        
        Args:
            name: Tub 이름 (없으면 타임스탬프 사용)
        """
        if name is None:
            name = datetime.now().strftime("tub_%Y%m%d_%H%M%S")
        
        self.tub_path = self.base_path / name
        self.image_dir = self.tub_path / 'images'
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        self.records = []
        self.record_count = 0
        self.recording = True
        
        # Manifest 생성
        manifest = {
            'name': name,
            'created': datetime.now().isoformat(),
            'inputs': ['camera/image_array'],
            'outputs': ['pilot/steering', 'pilot/throttle']
        }
        with open(self.tub_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"녹화 시작: {self.tub_path}")
    
    def run(self, image: np.ndarray, steering: float, throttle: float):
        """
        프레임 저장
        
        Args:
            image: RGB 이미지 (H, W, 3)
            steering: 조향값
            throttle: 스로틀값
        """
        if not self.recording or image is None:
            return
        
        if self.record_count >= self.max_records:
            logger.warning(f"최대 레코드 수 도달: {self.max_records}")
            self.stop_recording()
            return
        
        try:
            # 이미지 저장
            image_name = f"{self.record_count:06d}.jpg"
            image_path = self.image_dir / image_name
            
            import cv2
            # RGB → BGR (OpenCV 저장용)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 레코드 생성
            record = {
                'index': self.record_count,
                'image': image_name,
                'steering': float(steering),
                'throttle': float(throttle),
                'timestamp': time.time()
            }
            self.records.append(record)
            self.record_count += 1
            
        except Exception as e:
            logger.error(f"레코드 저장 오류: {e}")
    
    def stop_recording(self):
        """
        녹화 중지 - 레코드 저장
        """
        if not self.recording:
            return
        
        self.recording = False
        
        # 레코드 저장
        if self.tub_path and self.records:
            records_path = self.tub_path / 'records.json'
            with open(records_path, 'w') as f:
                json.dump(self.records, f, indent=2)
            
            logger.info(f"녹화 완료: {self.record_count} 레코드 저장됨")
        
        self.records = []
    
    def shutdown(self):
        """Part 종료"""
        self.stop_recording()
        logger.info("TubWriter 종료")


class TubReader:
    """
    Donkeycar Tub 데이터 읽기
    """
    
    def __init__(self, tub_path: str):
        """
        Args:
            tub_path: Tub 디렉토리 경로
        """
        self.tub_path = Path(tub_path)
        self.records = []
        self.current_index = 0
        
        self._load()
    
    def _load(self):
        """레코드 로드"""
        records_path = self.tub_path / 'records.json'
        if records_path.exists():
            with open(records_path, 'r') as f:
                self.records = json.load(f)
            logger.info(f"Tub 로드: {len(self.records)} 레코드")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx: int):
        """레코드 및 이미지 반환"""
        import cv2
        
        record = self.records[idx]
        image_path = self.tub_path / 'images' / record['image']
        
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return {
            'image': image,
            'steering': record['steering'],
            'throttle': record['throttle']
        }
    
    def get_generator(self):
        """학습용 제너레이터"""
        import cv2
        
        for record in self.records:
            image_path = self.tub_path / 'images' / record['image']
            image = cv2.imread(str(image_path))
            
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield image, record['steering'], record['throttle']
