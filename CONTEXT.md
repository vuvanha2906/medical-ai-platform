# Project Context: Medical AI Platform
This is a research-oriented Medical AI Platform built on Windows. 
The goal is to provide web-based medical image diagnosis using Deep Learning, specifically focusing on handling class imbalance and Explainable AI (Grad-CAM).

## Tech Stack
* **Backend:** Django 4.2, Django REST Framework (DRF).
* **Task Queue:** Celery (currently running synchronously via `CELERY_TASK_ALWAYS_EAGER = True` for local dev) + SQLite.
* **AI Engine:** PyTorch (CPU for dev), torchvision, pytorch-grad-cam.
* **Frontend:** Django Templates + Tailwind CSS (via CDN) - No React. Modern, academic, clinical UI.

## Current Progress
* [x] Project structure initialized using `uv`.
* [x] Django settings configured for a decoupled `frontend/` directory.
* [x] Database models created: `Study` (Patient info, modality, status) and `Prediction` (JSON results, heatmap path).
* [x] Basic Frontend UI (Dashboard) is working.
* [x] Mock Celery task (`time.sleep(3)`) is currently handling the processing flow.

## Next Immediate Goal (Month 1 MVP)
Implement the end-to-end X-ray pipeline replacing the mock task with a real PyTorch `DenseNet121` pretrained model, and generate basic Grad-CAM heatmaps.