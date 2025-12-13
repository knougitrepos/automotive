"""
시각화 유틸리티
- 학습 곡선
- 이미지/비디오
- 메트릭
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

# 로거 설정
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (선택적)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         title: str = "Training Curves"):
    """
    학습 곡선 시각화
    
    Args:
        history: {'train_loss': [...], 'val_loss': [...], ...}
        save_path: 저장 경로 (None이면 표시만)
        title: 그래프 제목
    """
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    
    if len(history) == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values, label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"학습 곡선 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_action_distribution(actions: Dict[str, np.ndarray],
                             save_path: Optional[str] = None):
    """
    액션 분포 시각화
    
    Args:
        actions: {'steer': [...], 'throttle': [...], 'brake': [...]}
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, (name, values) in zip(axes, actions.items()):
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(name)
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Distribution')
        ax.axvline(np.mean(values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(values):.3f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"액션 분포 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_evaluation_metrics(metrics: Dict[str, float],
                            save_path: Optional[str] = None,
                            title: str = "Evaluation Metrics"):
    """
    평가 메트릭 막대 그래프
    
    Args:
        metrics: {'metric_name': value, ...}
        save_path: 저장 경로
        title: 제목
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    
    bars = ax.barh(names, values, color=colors)
    
    # 값 표시
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center')
    
    ax.set_xlabel('Value')
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"메트릭 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def show_image_grid(images: List[np.ndarray], 
                    titles: Optional[List[str]] = None,
                    ncols: int = 4,
                    figsize: tuple = (16, 8),
                    save_path: Optional[str] = None):
    """
    이미지 그리드 표시
    
    Args:
        images: 이미지 리스트
        titles: 제목 리스트
        ncols: 열 개수
        figsize: 그림 크기
        save_path: 저장 경로
    """
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, image in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        
        axes[row][col].imshow(image)
        axes[row][col].axis('off')
        
        if titles and idx < len(titles):
            axes[row][col].set_title(titles[idx])
    
    # 빈 셀 숨기기
    for idx in range(len(images), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"이미지 그리드 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tsne(features: np.ndarray, 
              labels: Optional[np.ndarray] = None,
              save_path: Optional[str] = None,
              title: str = "t-SNE Visualization"):
    """
    t-SNE 시각화 (표현 학습 품질 평가용)
    
    Args:
        features: 특징 벡터 (N, D)
        labels: 라벨 (선택적)
        save_path: 저장 경로
        title: 제목
    """
    from sklearn.manifold import TSNE
    
    logger.info("t-SNE 계산 중...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                             c=labels, cmap='tab10', alpha=0.7, s=10)
        plt.colorbar(scatter)
    else:
        ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.7, s=10)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"t-SNE 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()
