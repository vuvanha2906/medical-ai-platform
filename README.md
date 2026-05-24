# Medical AI Platform

Web-based Medical Image Diagnosis using Deep Learning
**X-ray Lung Pathology** • **Brain Tumor 3D MRI Segmentation** • **Explainable AI (xAI)**

[![View Web Portfolio](https://img.shields.io/badge/View-Web_Portfolio-10b981?style=for-the-badge)](https://vuvanha-portfolio.vercel.app/)

---

## 📸 System Showcase

**<img width="773" height="433" alt="demo_xray" src="https://github.com/user-attachments/assets/60acdf99-3427-4e8e-b1cf-ebb009c0a204" />**
> *Chest X-ray multi-label classification with OpenCV-blended tight Grad-CAM masking and optimal threshold calibration.*

**<img width="774" height="399" alt="demo_mri" src="https://github.com/user-attachments/assets/857c24f3-c671-4657-8061-364d7ddba2ab" />**
> *3D Brain MRI (BraTS) semantic segmentation visualized natively in browser via NiiVue WebGL, featuring HD-BET skull-stripping and clinical-grade volume filtering.*
> 
**<img width="765" height="397" alt="demo_chatbot" src="https://github.com/user-attachments/assets/59b396c4-1024-4f27-9435-981ff535b58c" />**
> *Context-aware clinical RAG assistant powered by DeepSeek LLM and ChromaDB vector retrieval, featuring strict anti-hallucination guardrails for evidence-based medical reasoning.*

## Tech Stack

| Layer           | Library                                      |
| --------------- | -------------------------------------------- |
| Web             | Django 4.2, Django REST Framework, SimpleJWT |
| Task Queue      | Celery 5 + Redis                             |
| Deep Learning   | PyTorch, torchvision, MONAI                  |
| Explainability  | pytorch-grad-cam                             |
| LLM & RAG       | LangChain, DeepSeek API, ChromaDB, HuggingFace|
| Medical Imaging | nibabel, HD-BET, pydicom, OpenCV             |
| Frontend        | Tailwind CSS, NiiVue (WebGL 3D Viewer)       |

---

# Quickstart (uv)

This project uses **uv** for fast Python environment and dependency management.

## 1. Install `uv`

### Ubuntu (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update PATH (or restart terminal)
source $HOME/.cargo/env
```

### Windows

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Initialize Python & Virtual Environment

`uv` will automatically download the appropriate Python version
(e.g. Python 3.12) and create a virtual environment.

### Windows (PowerShell)

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

`uv` can synchronize dependencies directly from `pyproject.toml` or `uv.lock`.

```bash
uv sync
```

---

## 4. Install PyTorch (CPU / GPU)

Since PyTorch manages wheels through dedicated package indexes, the recommended way is to install using `uv pip` with `--index-url`.

### CPU Environment (Development / Non-GPU Machines)

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GPU Environment (Training / Production Inference)

Check your CUDA version using:

```bash
nvidia-smi
```

or

```bash
nvcc --version
```

Then install the matching PyTorch build.

Example for **CUDA 13.2**:

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

For the latest compatibility matrix, refer to:
[https://pytorch.org/](https://pytorch.org/)

---

## 5. Install HD-BET (Skull Stripping)

```bash
pip install hd-bet
```

---

## 6. Run the System

Apply database migrations and start the backend server:

```bash
python backend/manage.py migrate
python backend/manage.py createsuperuser
python backend/manage.py runserver
```

Run the Celery worker (open a separate terminal):

```bash
celery -A config worker -l info --pool=solo
```

---

# Project Structure

```bash
medical_ai_platform/
├── pyproject.toml          ← uv dependencies
├── requirements.txt        ← Plain-text dependency reference
├── .env.example
├── docker-compose.yml
│
├── notebooks/              ← Jupyter notebooks (Training, EDA, Experiments)
│   ├── 01_nih_chest_xray_training.ipynb
│   └── 02_alzheimer_mri_training.ipynb
│
├── backend/
│   ├── manage.py
│   ├── config/
│   │   ├── settings/
│   │   │   ├── base.py
│   │   │   ├── local.py
│   │   │   └── production.py
│   │   ├── urls.py
│   │   ├── celery.py
│   │   └── wsgi.py
│   │
│   └── apps/
│       ├── users/          ← Authentication, JWT
│       └── studies/        ← Study + Prediction Models
│
├── ai_engine/
├── common/             
│   ├── xray/               ← DenseNet121 + Grad-CAM
│   ├── mri_tumor/          ← Multi-view CNN + NIfTI pipeline
│   ├── knowledge_base/     ← Medical PDFs for RAG
│   ├── chroma_db/          ← Vector database storage
│   ├── ingest.py           ← Document embedding script
│   └── rag_queery.py          ← Medical RAG Assistant logic
│
├── frontend/               ← Django templates + static assets
└── tests/
```

---

# Model Training & Research

All exploratory analysis (EDA), training, fine-tuning, and experimentation for PyTorch models are performed independently.
The source notebooks are stored under the `notebooks/` directory.

You can also view or run the training pipelines directly on Kaggle, where datasets and GPU environments are pre-configured.

### Kaggle Notebooks

* 🔗 **NIH Chest X-ray 14 – DenseNet121 Training**
  [https://www.kaggle.com/code/vuvanha2906/nih-chest-x-ray-14-densenet121-pytorch](https://www.kaggle.com/code/vuvanha2906/nih-chest-x-ray-14-densenet121-pytorch)

* 🔗 **3D MRI Brain Tumor Segmentation – SwinUNETR**
  [https://www.kaggle.com/code/vuvanha2906/3d-mri-brain-tumor-segmentation-swinunetr](https://www.kaggle.com/code/vuvanha2906/3d-mri-brain-tumor-segmentation-swinunetr)

* Additional training notebooks will be added later.

---

# Datasets

| Module | Dataset          | Link                                                                                                                                                       |
| ------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| X-ray  | NIH ChestX-ray14 | [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)                                                                 |
| MRI    | BraTS2020        | [https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) |
| ...    | ...              | ...                                                                                                                                                        |

---

# Development Roadmap

```bash
[x] Phase 1: Django Core Architecture + Authentication + Upload API

[x] Phase 2: X-ray Diagnosis
    - DenseNet121
    - Grad-CAM
    - Threshold Calibration

[x] Phase 3: MRI Tumor Segmentation
    - SwinUNETR
    - HD-BET Skull Stripping
    - Clinical Volume Filtering

[x] Phase 4: LLM-Assisted Clinical RAG
    - DeepSeek API Integration
    - ChromaDB Vector Storage & HuggingFace Embeddings
    - Anti-hallucination Prompt Engineering

[ ] Phase 5: Physician Validation Workflow
    - Longitudinal Patient Tracking
    - Multi-study Comparison
    - Clinical Review Pipeline
```

---

# Key Features

* Chest X-ray pathology detection using Deep Learning
* 3D Brain Tumor MRI segmentation
* Explainable AI visualization with Grad-CAM
* JWT-based authentication
* Async background inference with Celery + Redis
* DICOM / NIfTI medical imaging support
* WebGL-based MRI visualization using NiiVue
* Modular AI engine for future clinical models

---

# Future Improvements

* Multi-class disease severity scoring
* Radiology report generation (LLM-assisted)
* PACS integration
* Multi-modal diagnosis (X-ray + MRI + CT)
* Physician feedback loop for continuous learning
* Model monitoring & drift detection

---
