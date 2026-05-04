# Project Context: Medical AI Platform

## Tech Stack & Environment
* **Backend:** Django 4.2, Django REST Framework (DRF).
* **Task Queue:** Celery (Synchronous mode for local dev) + SQLite.
* **AI Engine:** PyTorch (DenseNet121) + pytorch-grad-cam.
* **PDF Generation:** `xhtml2pdf` for clinical report exports.
* **Frontend:** Django Templates + Tailwind CSS (No React). Vanilla JS Fetch API. UI features a Dark Theme with Glassmorphism aesthetic. Chart.js for Analytics.

## Current Implementation Overview

### 1. Active Django Apps & Models
* **`studies` App:** `Study` (Integer ID, patient info, modality, status) and `Prediction` (FK to Study, label, probability, heatmap_url).
* **`users` App:** Custom `CustomUser` model handling system authentication.

### 2. End-to-End Architecture & Security
1. **Authentication:** Integrated Django Auth. Custom Glassmorphism login page (`users/login.html`). All main views and APIs are secured using `LoginRequiredMixin` and `@login_required` / `IsAuthenticated`.
2. **Image Upload & Processing:** Authenticated user uploads via UI -> `StudyUploadView` creates record -> Celery task `process_xray_study` runs PyTorch inference -> Generates Grad-CAM -> Saves to DB.
3. **UI/UX Flow:**
  * `Dashboard` & `Studies`: Secured history and upload gateways.
  * `Report Detail`: Visual xAI comparison and PDF Export functionality.
  * `Analytics`: Chart.js dashboards rendering metrics from secured APIs.

## Immediate Next Steps
* Integrate the custom Chest X-ray training dataset.
* Write the PyTorch training loop (`train.py`) to fine-tune the model locally on Windows.