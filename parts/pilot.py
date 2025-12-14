# -*- coding: utf-8 -*-
"""
Pilot Part
==========

딥러닝 모델 기반 자율주행 Pilot Part.
카메라 이미지를 입력받아 조향/스로틀을 예측합니다.

참고 논문:
- NVIDIA PilotNet (Bojarski et al., 2016)
- End-to-End Learning for Self-Driving Cars
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class KerasPilot:
    """
    Keras/TensorFlow 기반 자율주행 파일럿
    
    이미지를 입력받아 (steering, throttle)을 예측합니다.
    
    모델 구조 (권장):
    - Input: (None, 120, 160, 3) RGB 이미지
    - Output: (steering, throttle) 2개 값
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 학습된 모델 파일 경로 (.h5)
        """
        self.model = None
        self.model_path = model_path
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            logger.warning(f"모델 파일 없음: {model_path}. 기본 제어 사용.")
    
    def load(self, model_path: str):
        """
        모델 로드
        
        Args:
            model_path: .h5 모델 파일 경로
        """
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            logger.info(f"모델 로드 완료: {model_path}")
        except ImportError:
            logger.error("TensorFlow/Keras가 설치되지 않았습니다.")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    def run(self, image: np.ndarray) -> Tuple[float, float]:
        """
        이미지 → (steering, throttle) 예측
        
        Args:
            image: RGB 이미지 (H, W, 3)
            
        Returns:
            (steering, throttle) 튜플
        """
        if self.model is None or image is None:
            return 0.0, 0.0
        
        try:
            # 전처리
            img = self._preprocess(image)
            
            # 추론
            output = self.model.predict(img, verbose=0)
            
            # 출력 파싱
            if len(output.shape) == 2 and output.shape[1] == 2:
                steering = float(output[0, 0])
                throttle = float(output[0, 1])
            else:
                steering = float(output[0])
                throttle = 0.3  # 기본 스로틀
            
            return steering, throttle
            
        except Exception as e:
            logger.error(f"추론 오류: {e}")
            return 0.0, 0.0
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리
        
        Args:
            image: RGB 이미지
            
        Returns:
            배치 텐서 (1, H, W, 3), 정규화됨
        """
        img = image.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def shutdown(self):
        """Part 종료"""
        logger.info("KerasPilot 종료")


class TorchPilot:
    """
    PyTorch 기반 자율주행 파일럿
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Args:
            model_path: 학습된 모델 파일 경로 (.pth)
            device: 'cuda' 또는 'cpu'
        """
        self.model = None
        self.device = device
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """모델 로드"""
        try:
            import torch
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"PyTorch 모델 로드: {model_path}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    def run(self, image: np.ndarray) -> Tuple[float, float]:
        """이미지 → (steering, throttle) 예측"""
        if self.model is None or image is None:
            return 0.0, 0.0
        
        try:
            import torch
            import torchvision.transforms as T
            
            # 전처리
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((120, 160)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
            
            steering = float(output[0, 0].cpu())
            throttle = float(output[0, 1].cpu()) if output.shape[1] > 1 else 0.3
            
            return steering, throttle
            
        except Exception as e:
            logger.error(f"추론 오류: {e}")
            return 0.0, 0.0
    
    def shutdown(self):
        logger.info("TorchPilot 종료")


class MockPilot:
    """테스트용 더미 파일럿"""
    
    def run(self, image: np.ndarray) -> Tuple[float, float]:
        """간단한 시뮬레이션 (항상 직진)"""
        return 0.0, 0.3  # 직진, 저속
    
    def shutdown(self):
        pass
