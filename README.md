# 석사 논문용 term project 1번
# 자율주행 AI 파이프라인 (비전 라인)

CARLA 시뮬레이터에서 정책(action) 데이터를 수집하고, KITTI로 self-supervised 사전학습을 수행하여, 지침을 따르는 자율주행 모델을 학습 및 평가하는 프로젝트입니다.

## 환경

- **GPU**: GTX 1080Ti (11GB VRAM)
- **OS**: Windows 10/11
- **Python**: 3.12 (CARLA 0.9.16 호환)
- **Simulator**: CARLA 0.9.16
- **파이프라인**: 비전 라인 (Tesla-like)
- **구현**: Jupyter Notebook 중심

## 프로젝트 구조

```
automotive/
├── notebook/                  # 핵심 구현 노트북
│   ├── 01_carla_setup.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_kitti_exploration.ipynb
│   ├── 04_ssl_pretraining.ipynb
│   ├── 05_bc_training.ipynb
│   ├── 06_safety_shield.ipynb
│   └── 07_evaluation.ipynb
├── config/                    # 설정 파일
├── dataset/                   # 데이터
├── utils/                     # 유틸리티
├── plan.md                    # 구현 계획서 (기본)
├── expansion_plan.md          # 확장 계획서 (Advanced)
└── requirements.txt
```

## 시작하기

### 1. 환경 설정

```powershell
# 1. Python 3.12 가상환경 생성
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. CARLA 0.9.16 Python API 설치
# (CARLA 설치 경로의 wheel 파일 사용)
pip install "F:\CARLA\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
```

### 2. CARLA 서버 실행

```powershell
# CARLA 설치 경로로 이동하여 실행
cd F:\CARLA\CARLA_0.9.16\WindowsNoEditor
.\CarlaUE4.exe -quality-level=Low
```

### 3. 노트북 실행

1. VS Code에서 노트북 파일 열기 (`notebook/01_carla_setup.ipynb`)
2. 커널 선택: `Python 3.12 (CARLA)` (또는 `.venv312`)
3. 셀 순차 실행

---

## Literature Research (참고 논문)

### 1. Survey & Optimization (종합 연구) [Q1]
[1] L. Chen et al., "End-to-end Autonomous Driving: Challenges and Frontiers," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2024. [Available: https://arxiv.org/abs/2306.16927]
[2] Y. Huang et al., "A Survey on Trajectory-Prediction Techniques for Autonomous Driving," *IEEE Transactions on Intelligent Transportation Systems (T-ITS)*, 2022.
[3] S. Mozaffari et al., "Deep Learning for Multimodal Representation Learning: A Review," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

### 2. Core Foundation (기반 기술)
[4] T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," in *Proc. ICML*, 2020. (SimCLR)
[5] A. Dosovitskiy et al., "CARLA: An Open Urban Driving Simulator," in *Proc. CoRL*, 2017.
[6] M. Bojarski et al., "End to End Learning for Self-Driving Cars," *arXiv preprint*, 2016. (PilotNet)

### 3. Self-Supervised & Representation Learning
[7] M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv preprint arXiv:2304.07193*, 2023.
[8] K. He et al., "Masked Autoencoders Are Scalable Vision Learners," in *Proc. CVPR*, 2022.
[9] M. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," in *Proc. CVPR*, 2023. (I-JEPA)

### 4. End-to-End Autonomous Driving (SOTA)
[10] Y. Hu et al., "Planning-oriented Autonomous Driving," in *Proc. CVPR*, 2023. (UniAD - Best Paper)
[11] K. Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving," *IEEE TPAMI*, 2022.
[12] H. Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," in *Proc. ICCV*, 2023.
[13] P. Wu et al., "Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline," in *Proc. NeurIPS*, 2022. (TCP)

### 5. World Models & Generative AI
[14] Z. Hu et al., "MILE: Model-Based Imitation Learning for Urban Driving," in *Proc. NeurIPS*, 2022.
[15] D. Hafner et al., "Mastering Diverse Domains through World Models," *arXiv preprint*, 2023. (DreamerV3)
[16] T. Wang et al., "DriveWM: The First World Model for End-to-end Autonomous Driving," *arXiv preprint arXiv:2311.17918*, 2023.
[17] A. Van den Oord et al., "Neural Discrete Representation Learning," in *Proc. NeurIPS*, 2017. (VQ-VAE)
[18] A. Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2," in *Proc. NeurIPS*, 2019.
[19] P. Esser et al., "Taming Transformers for High-Resolution Image Synthesis," in *Proc. CVPR*, 2021. (VQ-GAN)
[20] W. Zheng et al., "OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving," *arXiv preprint*, 2024.

### 6. Vision-Language & Multimodal
[21] A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. ICML*, 2021. (CLIP)
[22] H. Liu et al., "Visual Instruction Tuning," in *Proc. NeurIPS*, 2023. (LLaVA)
[23] L. Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data," in *Proc. CVPR*, 2024.
[24] A. Kirillov et al., "Segment Anything," in *Proc. ICCV*, 2023. (SAM)

### 7. 3D Perception & Occupancy (Advanced Vision)
[25] Z. Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers," in *Proc. ECCV*, 2022.
[26] Y. Wei et al., "SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving," in *Proc. ICCV*, 2023.

### 8. Imitation Learning & Safety
[27] C. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," in *Proc. RSS*, 2023.
[28] T. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Arms," in *Proc. CoRL*, 2023. (ACT)
[29] M. Alshiekh et al., "Safe Reinforcement Learning via Shielding," in *Proc. AAAI*, 2018.

### 9. Video Understanding
[30] Z. Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training," in *Proc. NeurIPS*, 2022.
[31] G. Bertasius et al., "Is Space-Time Attention All You Need for Video Understanding?," in *Proc. ICML*, 2021. (TimeSformer)

---

자세한 내용은 [plan.md](plan.md) 및 [expansion_plan.md](expansion_plan.md) 참조
