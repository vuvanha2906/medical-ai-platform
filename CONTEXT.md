# Medical AI Diagnostic Platform - Context

## 1. Project Objective
To build an AI-based medical diagnostic platform (xAI) focusing on explainability. The system supports multimodality, including chest X-rays and brain MRI.

## 2. System Architecture
- **Backend:** Django Framework, REST API.

- **Task Queue:** Celery + Redis (Asynchronous processing of heavy AI tasks).

- **AI Engine:** PyTorch, MONAI (For 3D medical data).

- **Frontend:** Tailwind CSS (Glassmorphism aesthetic), NiiVue (3D Medical Visualizer).

## 3. Development History & Current Status
- **Phase 1 (X-ray):** Completed classification of 14 lung diseases (NIH dataset). Multi-layer Grad-CAM integration allows physicians to view specific lesion areas for each disease.

- **Phase 2 (MRI):** Complete the 3D data processing pipeline. Build an automated DICOM to NIfTI converter.

- **Status:** The entire data flow from upload -> preprocessing -> AI simulation -> 3D rendering is now smooth. The system is ready for loading actual weights from the SwinUNETR model.

## 4. Training Configuration (Hardware)
- CPU: Ryzen 5 5600X
- GPU: NVIDIA RTX 5060 8GB VRAM (Patch-based training optimized 96x96x96).