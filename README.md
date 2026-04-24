# Medical AI Platform

Web-based Medical Image Diagnosis using Deep Learning  
**X-ray Lung Pathology** · **Alzheimer MRI Classification** · **Explainable AI (Grad-CAM)**

---

## Tech stack

| Layer | Library |
|---|---|
| Web | Django 4.2, DRF, SimpleJWT |
| Task queue | Celery 5 + Redis |
| Deep learning | PyTorch, torchvision |
| Explainability | pytorch-grad-cam |
| Medical imaging | nibabel, nilearn, SimpleITK |
| Experiment tracking | MLflow / Weights & Biases |

---

## Quickstart (uv)

### 1. Install uv (nếu chưa có)

```

### 4. Cài PyTorch (CPU — dev)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **GPU (training thật):** thay `cpu` bằng `cu121` hoặc `cu118` tuỳ CUDA version:
> ```bash
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Kiểm tra CUDA version: `nvcc --version` hoặc `nvidia-smi`

### 5. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env: DATABASE_URL, REDIS_URL, SECRET_KEY...
```

### 6. Chạy services (Docker)

```bash
docker compose up -d postgres redis
```

### 7. Migrate & chạy server

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

### 8. Chạy Celery worker (terminal riêng)

```bash
celery -A config worker -l info
celery -A config beat -l info    # scheduler (optional)
```

---

## Cấu trúc thư mục

```
medical_ai_platform/
├── pyproject.toml          ← uv / dependencies
├── requirements.txt        ← plain-text reference
├── .env.example
├── docker-compose.yml
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
│       ├── users/          ← auth, JWT
│       ├── studies/        ← Study + Prediction models
│       └── ai_engine/      ← Celery tasks, inference wrappers
│
├── ai_engine/
│   ├── common/             ← base model, utils
│   ├── xray/               ← DenseNet121, Grad-CAM
│   └── alzheimer/          ← Multi-view CNN, NIfTI pipeline
│
├── frontend/               ← Django templates + static
└── tests/
```

---

## Datasets

| Module | Dataset | Link |
|---|---|---|
| X-ray | NIH ChestX-ray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| X-ray | CheXpert (Stanford) | https://stanfordmlgroup.github.io/competitions/chexpert/ |
| Alzheimer | ADNI | https://adni.loni.usc.edu (đăng ký ~1–2 tuần) |
| Alzheimer | OASIS-3 | https://www.oasis-brains.org |
| Alzheimer | Kaggle 4-class | https://www.kaggle.com/datasets/tourist55/alzheimers-dataset |

---

## Thứ tự phát triển (research roadmap)

```
Phase 1  Django skeleton + auth + upload API
Phase 2  X-ray: DenseNet121 + Grad-CAM + API endpoint
Phase 3  Alzheimer: NIfTI pipeline + Multi-view CNN + heatmap
Phase 4  Frontend viewer + PDF report + Docker + evaluation
```