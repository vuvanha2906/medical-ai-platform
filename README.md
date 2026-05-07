# Medical AI Platform

Web-based Medical Image Diagnosis using Deep Learning  
**X-ray Lung Pathology** • **Alzheimer MRI Classification** • **Explainable AI (Grad-CAM)**

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

Dự án sử dụng `uv` để quản lý Python và các thư viện một cách nhanh nhất.

### 1. Cài đặt `uv`

*   **Trên Ubuntu (Linux):**
    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    # Sau đó chạy lệnh dưới để cập nhật PATH (hoặc khởi động lại terminal)
    source $HOME/.cargo/env
    ```
*   **Trên Windows:**
    Mở PowerShell và chạy:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    ```

### 2. Khởi tạo Python và Môi trường ảo

`uv` sẽ tự động tải phiên bản Python phù hợp (ví dụ Python 3.12) và tạo môi trường ảo `.venv`.

*   **Trên Ubuntu:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```
*   **Trên Windows (PowerShell):**
    ```powershell
    uv venv
    .venv\Scripts\Activate.ps1
    ```

### 3. Cài đặt Dependencies

`uv` hỗ trợ đồng bộ hóa cực nhanh tất cả thư viện từ tệp `pyproject.toml` hoặc `uv.lock`.
```bash
uv sync
```

### 4. Cài đặt PyTorch (Tùy chọn CPU / GPU)

Vì PyTorch quản lý wheel ở các index riêng, cách tốt nhất với `uv` là cài đặt kèm cờ `--index-url`.

*   **Môi trường CPU (Phục vụ dev hoặc máy không có GPU):**
    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

*   **Môi trường GPU (Training thật):**
    Kiểm tra phiên bản CUDA bằng lệnh `nvidia-smi` hoặc `nvcc --version` và chọn index phù hợp (Ví dụ GTX 5060 8GB CUDA version 13.2 sử dụng `/nightly/cu132`):
    ```bash
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
    ```
    Tham khảo https://pytorch.org/ để kiểm tra phiên bản CUDA

### 5. Cấu hình môi trường

Sao chép file cấu hình mẫu và điền các thông số cần thiết (Database URL, Redis URL...):
*   **Trên Ubuntu:** `cp .env.example .env`
*   **Trên Windows:** `copy .env.example .env`

### 6. Chạy các dịch vụ (Docker)

Đảm bảo bạn đã cài đặt Docker và Docker Compose. Khởi chạy cơ sở dữ liệu và Redis:
```bash
docker compose up -d postgres redis
```

### 7. Migrate & Chạy Server

Cập nhật cấu trúc database và khởi tạo tài khoản quản trị:
```bash
python backend/manage.py migrate
python backend/manage.py createsuperuser
python backend/manage.py runserver
```

### 8. Chạy Celery worker (Mở terminal riêng)

Chuyển hướng vào thư mục chứa file `manage.py` và chạy lệnh sau để xử lý hàng đợi tác vụ AI:

*   **Trên Ubuntu:**
    ```bash
    celery -A config worker -l info
    ```
*   **Trên Windows:**
    *(Celery trên Windows cần thêm cờ `--pool=solo` hoặc `--pool=gevent` để hoạt động ổn định)*
    ```bash
    celery -A config worker -l info --pool=solo
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
│       ├── users/          ← auth, JWT
│       └── studies/        ← Study + Prediction models
│           
│
├── ai_engine/
│   ├── common/             ← base model, utils
│   ├── xray/               ← DenseNet121, Grad-CAM
│   │      ├── weights/
│   │      │   └── best_nih_densenet121.pth  <-- Trọng số nằm ở đây
│   │      ├── model.py
│   │      ├── grad_cam.py
│   │      └── inference.py
│   └── alzheimer/          ← Multi-view CNN, NIfTI pipeline
│
├── frontend/               ← Django templates + static
└── tests/
```
---

## Model Training & Research
Quá trình nghiên cứu (EDA) và huấn luyện (Training & Fine-tuning) các mô hình PyTorch được thực hiện độc lập và mã nguồn được lưu trữ tại thư mục notebooks/.

Bạn có thể xem hoặc chạy trực tiếp mã nguồn Training trên Kaggle với dữ liệu và GPU đã được thiết lập sẵn:

- 🔗 Kaggle Notebook: NIH Chest X-ray 14 - DenseNet121 Training https://www.kaggle.com/code/vuvanha2906/nih-chest-x-ray-14-densenet121-pytorch

- (Các link training khác sẽ được cập nhật...)


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
