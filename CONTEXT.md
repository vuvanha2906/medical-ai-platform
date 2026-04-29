# Project Context: Medical AI Platform

## Tech Stack & Environment
* **Backend:** Django 4.2, Django REST Framework (DRF).
* **Task Queue:** Celery (Synchronous mode for local dev) + SQLite.
* **AI Engine:** PyTorch (DenseNet121) + pytorch-grad-cam.
* **Frontend:** Django Templates + Tailwind CSS (No React). Vanilla JS Fetch API.

## Current Implementation Overview

### 1. Active Django Apps & Models
* **`studies` App:**
  * `Study`: UUID, `patient_name`, `image` (X-ray), `created_at`, `status` (Processing/Completed).
  * `Prediction`: FK to `Study`, `prediction_label`, `probability`, `heatmap_url`.
* **`users` App:** Custom `CustomUser` model (AbstractBaseUser).

### 2. End-to-End Architecture
1. **Image Upload:** User uploads via UI -> `StudyUploadView` creates `Study` record & anonymous name -> Triggers `process_xray_study` Celery task -> Returns Processing status.
2. **AI Inference (`studies/tasks.py`):** Loads image -> Preprocesses -> Runs PyTorch `XrayPredictor` -> Generates Grad-CAM -> Saves to `Prediction` model.
3. **UI/UX Flow:**
  * `Dashboard`: Lists recent studies with dynamic status.
  * `Report Detail`: Shows patient info, AI prediction stats, and Original vs Heatmap visual comparison.
  * `Analytics`: Charts and summaries fetched from `api/analytics/`.

## Immediate Next Steps
* Implement PDF generation for clinical reports.
* Fine-tune the PyTorch model on specific medical datasets.