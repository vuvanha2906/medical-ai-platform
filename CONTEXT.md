# Project Context: Medical AI Platform

## Tech Stack & Environment
* **Backend:** Django 4.2, Django REST Framework (DRF).
* **Task Queue:** Celery (Synchronous mode for local dev) + SQLite.
* **AI Engine:** PyTorch (DenseNet121) + pytorch-grad-cam.
* **PDF Generation:** `xhtml2pdf` for clinical report exports.
* **Frontend:** Django Templates + Tailwind CSS (No React). Vanilla JS Fetch API. UI features a Dark Theme with Glassmorphism aesthetic. Chart.js for Analytics.

## Current Implementation Overview

### 1. Active Django Apps & Models
* **`studies` App:** `Study` (Integer ID, patient info, modality, status) and `Prediction` (FK to Study, label, probability, heatmap_url, and a new `heatmaps` JSONField to store multiple multi-disease attention maps).
* **`users` App:** Custom `CustomUser` model handling system authentication.

### 2. End-to-End Architecture & Security
1. **Authentication:** Integrated Django Auth. Custom Glassmorphism login page (`users/login.html`). All main views and APIs are secured using `LoginRequiredMixin` and `@login_required` / `IsAuthenticated`.
2. **Image Upload & Processing:** Authenticated user uploads via UI -> `StudyUploadView` creates record -> Celery task `process_xray_study` runs multi-label PyTorch inference -> Thresholds > 0.5 trigger multi-class Grad-CAM heatmaps -> Saves dynamic dictionary to DB.
3. **UI/UX Flow:**
  * `Dashboard` & `Studies`: Secured history and upload gateways.
  * `Report Detail`: Visual xAI comparison with interactive JS toggles for multi-disease heatmaps, plus PDF Export functionality.
  * `Analytics`: Chart.js dashboards rendering metrics from secured APIs.

## Immediate Next Steps
* Integrate the VinBigData Chest X-ray dataset to upgrade the system from Classification to an Object Detection / Hybrid xAI paradigm.
* Optimize local inference performance (batching and mixed precision) on the RTX 5060 GPU environment.