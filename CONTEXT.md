# Medical AI Diagnostic Platform - Context

## 1. Project Objective
To build an AI-based medical diagnostic platform (xAI) focusing on explainability. The system supports multimodality, including chest X-rays and 3D brain MRI, providing clinical-grade confidence and visualization.

## 2. System Architecture
- **Backend:** Django Framework, REST API.
- **Task Queue:** Celery + Redis (Asynchronous processing of heavy AI tasks).
- **AI Engine:** PyTorch, MONAI (SwinUNETR for 3D Segmentation).
- **Pre-processing Engine:** HD-BET (High-Definition Brain Extraction Tool) for automated skull-stripping.
- **Frontend:** Tailwind CSS (Dark/Glassmorphism aesthetic), NiiVue v0.71.0 (Optimized 3D Medical Visualizer).

## 3. Development History & Current Status
- **Phase 1 (X-ray):** Completed classification of 14 lung diseases (NIH dataset). Multi-layer Grad-CAM integration allows physicians to view specific lesion areas for each disease.
- **Phase 2 (MRI):** - Automated DICOM to NIfTI converter.
    - Integrated HD-BET to clean out-of-domain noise (skull, eyes, neck).
    - Built an automatic router: detects BraTS standard ZIPs (T1, T2, T1c, FLAIR) for direct fusion, or routes single-modality files through HD-BET.
    - Implemented strict clinical post-processing filters (Volume thresholding > 10,000 voxels, probability calibration).
- **Status:** The entire data flow from upload -> HD-BET preprocessing -> SwinUNETR inference -> Post-processing -> WebGL 3D rendering is 100% smooth, robust against edge cases, and visually stable.

## 4. Training Configuration (Hardware)
- CPU: Ryzen 5 5600X
- GPU: NVIDIA RTX 5060 8GB VRAM (Patch-based training optimized 96x96x96).