# Project Context: Medical AI Platform

## Tech Stack & Environment
* **Backend:** Django 4.2, Django REST Framework (DRF).
* **Task Queue:** Celery (Synchronous mode for local dev) + SQLite.
* **AI Engine:** PyTorch (DenseNet121) + pytorch-grad-cam.
* **PDF Generation:** `xhtml2pdf` for clinical report exports.
* **Frontend:** Django Templates + Tailwind CSS (No React). Vanilla JS Fetch API. UI features a Dark Theme with Glassmorphism aesthetic. Chart.js for Analytics.

## Current Implementation Overview

### 1. Active Django Apps & Models
* **`studies` App:**
  * `Study`: Integer ID (Auto-increment), `patient_name`, `image` (X-ray/MRI/CT/US), `created_at`, `status` (Processing/Completed/Pending/Failed), `modality`.
  * `Prediction`: FK to `Study`, `prediction_label`, `probability`, `heatmap_url`.
* **`users` App:** Custom `CustomUser` model (AbstractBaseUser).

### 2. End-to-End Architecture
1. **Image Upload:** User uploads via UI -> `StudyUploadView` creates `Study` record & anonymous name -> Triggers `process_xray_study` Celery task -> Returns JSON status.
2. **AI Inference (`studies/tasks.py`):** Loads image -> Preprocesses -> Runs PyTorch `XrayPredictor` -> Generates Grad-CAM -> Saves to `Prediction` model.
3. **UI/UX Flow:**
  * `Dashboard`: Landing page showing recent uploads and upload form.
  * `Studies`: Main data table displaying the history of all studies. Serves as the gateway to detailed reports.
  * `Report Detail`: Shows patient info, AI prediction stats, Original vs Heatmap visual comparison (Side-by-side/Overlay), and provides PDF Export functionality.
  * `Analytics`: Charts and summaries fetched from `api/analytics/` (`AnalyticsDataView`).

## Immediate Next Steps
* Fine-tune the PyTorch model on specific medical datasets to improve accuracy across different modalities.
* Implement user authentication (Login/Logout) and role-based access control (Radiologist vs. Admin).