# Medical AI Diagnostic Platform - Context

## 1. Project Objective
To build an AI-based medical diagnostic platform (xAI) focusing on explainability. The system supports multimodality, including chest X-rays and 3D brain MRI, providing clinical-grade confidence and visualization.

## 2. System Architecture
- **Backend:** Django Framework, REST API.
- **Task Queue:** Celery + Redis.
- **AI Engine:** PyTorch, MONAI (SwinUNETR), HD-BET.
- **Frontend:** Tailwind CSS, NiiVue (Custom Brats Colormap).

## 3. Development History & Current Status
- **Phase 1 (X-ray):** Completed classification of 14 lung diseases. Implemented Optimal Threshold Calibration and OpenCV-based Tight xAI Masking for precise visualization.
- **Phase 2 (MRI):** - Automated DICOM/ZIP to NIfTI converter.
    - Integrated HD-BET to clean out-of-domain noise.
    - Implemented strict clinical post-processing filters (Volume thresholding > 10,000 voxels).
- **Status:** The entire AI logic (X-ray and MRI) is clinically robust and visually accurate. We are now moving towards UI cleanup and Physician Validation workflows.

## 4. Training Configuration (Hardware)
- CPU: Ryzen 5 5600X
- GPU: NVIDIA RTX 5060 8GB VRAM (CUDA:0).