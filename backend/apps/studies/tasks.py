import os
import zipfile
import shutil
import nibabel as nib
import time
from pathlib import Path
from celery import shared_task
from PIL import Image
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from django.conf import settings
from ai_engine.xray.inference import XrayPredictor
from .models import Study, Prediction

PROJECT_ROOT = Path(settings.BASE_DIR).parent
# Khởi tạo predictor 1 lần duy nhất khi worker chạy (để không tốn thời gian load weights mỗi lần dự đoán)
# Đảm bảo file .pth đã nằm đúng chỗ
WEIGHTS_PATH = PROJECT_ROOT / 'ai_engine' / 'xray' / 'weights' / 'best_nih_densenet121.pth'
print(f"Loading weights from: {WEIGHTS_PATH}")
predictor = XrayPredictor(weights_path=WEIGHTS_PATH)

def convert_dicom_to_png(study):
    """Đọc DICOM, chuyển thành PNG và cập nhật lại đường dẫn cho mô hình và Web UI."""
    file_path = study.image.path
    if file_path.lower().endswith(('.dcm', '.dicom')):
        # 1. Đọc file DICOM
        dicom_data = pydicom.dcmread(file_path)
        pixel_array = dicom_data.pixel_array.astype(float)

        # 2. Chuẩn hóa mức xám (0 - 255)
        pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        pixel_array = np.uint8(pixel_array)

        # # 3. Tạo file PNG mới
        # img = Image.fromarray(pixel_array)
        # new_filename = study.image.name.replace('.dcm' '.dicom', '.png')
        # new_filepath = file_path.replace('.dcm' '.dicom', '.png')
        base_name_obj, _ = os.path.splitext(study.image.name)
        new_filename = f"{base_name_obj}.png"

        base_path_disk, _ = os.path.splitext(file_path)
        new_filepath = f"{base_path_disk}.png"

        # 2. Tạo và lưu file PNG
        img = Image.fromarray(pixel_array)

        # Lưu file PNG đè lên
        img.save(new_filepath)

        # 4. Cập nhật Database để UI nhận diện file PNG
        study.image.name = new_filename
        study.save()
        return new_filepath
    return file_path


@shared_task
def process_xray_study(study_id, image_path):
    try:
        study = Study.objects.get(id=study_id)
        study.status = 'Processing'
        study.save()

        # ==========================================
        # 1. TIỀN XỬ LÝ: CHUYỂN ĐỔI DICOM -> PNG
        # ==========================================
        processed_image_path = convert_dicom_to_png(study)

        # Đường dẫn gốc (Absolute Path để lưu file xuống ổ cứng)
        heatmap_base_filename = f"heatmap_study_{study_id}"
        heatmap_base_rel_path = os.path.join('heatmaps', heatmap_base_filename)
        heatmap_base_full_path = os.path.join(settings.MEDIA_ROOT, heatmap_base_rel_path)

        os.makedirs(os.path.dirname(heatmap_base_full_path), exist_ok=True)

        # ==========================================
        # 2. CHẠY SUY LUẬN AI
        # LƯU Ý: Truyền processed_image_path thay vì image_path gốc!
        # ==========================================
        result = predictor.predict_and_explain(
            image_path=processed_image_path,
            output_heatmap_base_path=heatmap_base_full_path
        )

        # ==========================================
        # 3. CHUYỂN ĐỔI ĐƯỜNG DẪN Ổ CỨNG -> URL WEB
        # ==========================================
        # result['heatmaps'] đang chứa đường dẫn vật lý, vd: /home/.../media/heatmaps/study_1_Pneumonia.png
        # Ta cần đổi thành dạng web URL: /media/heatmaps/study_1_Pneumonia.png
        heatmap_web_urls = {}
        for disease, abs_path in result['heatmaps'].items():
            filename = os.path.basename(abs_path)
            heatmap_web_urls[disease] = f"{settings.MEDIA_URL}heatmaps/{filename}"

        # ==========================================
        # 4. LƯU KẾT QUẢ XUỐNG DATABASE
        # ==========================================
        Prediction.objects.update_or_create(
            study=study,
            defaults={
                'prediction_label': result['predicted_class'],
                'probability': result['probability'],
                'results': result['all_probabilities'],  # Lưu bảng % xác suất
                'heatmaps': heatmap_web_urls,  # Lưu bộ từ điển đường dẫn ảnh mới tạo

                # Cột này giữ lại cho UI hiển thị ảnh mặc định đầu tiên
                'heatmap_url': heatmap_web_urls.get(result['predicted_class']) or heatmap_web_urls.get('Original')
            }
        )

        study.status = 'Completed'
        study.save()
        return {'study_id': study_id, 'status': 'Success'}

    except Exception as e:
        if 'study' in locals():
            study.status = 'Failed'
            study.save()
        print(f"Error processing study {study_id}: {str(e)}")
        return {'study_id': study_id, 'status': 'Failed', 'error': str(e)}


# =================================================================
# HELPER: CHUYỂN ĐỔI BẤT KỲ ĐỊNH DẠNG NÀO SANG NIFTI (.nii.gz)
# =================================================================
def ensure_nifti_format(file_path, output_dir):
    """
    Nhận vào .dcm, .nii.gz hoặc .zip.
    Đảm bảo trả về đúng 1 file .nii.gz hợp lệ để làm nền hiển thị.
    """
    ext = file_path.lower()

    # TRƯỜNG HỢP 1: Đã là NIfTI
    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        return file_path

    # TRƯỜNG HỢP 2: Là DICOM đơn lẻ
    elif ext.endswith('.dcm') or ext.endswith('.dicom'):
        dicom_data = pydicom.dcmread(file_path)
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        # Thêm chiều Z để biến ảnh 2D thành khối 3D (H, W, 1)
        if len(pixel_array.shape) == 2:
            pixel_array = np.expand_dims(pixel_array, axis=-1)

        # Lưu thành file NIfTI
        nii_img = nib.Nifti1Image(pixel_array, affine=np.eye(4))
        nii_path = os.path.join(output_dir, 'converted_dicom.nii.gz')
        nib.save(nii_img, nii_path)
        return nii_path

    # TRƯỜNG HỢP 3: Là file ZIP
    elif ext.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Tìm file .nii.gz hoặc .dcm đầu tiên thấy được trong thư mục giải nén
        for root, _, files in os.walk(output_dir):
            for file in files:
                f_path = os.path.join(root, file)
                if file.lower().endswith(('.nii', '.nii.gz')):
                    return f_path
                elif file.lower().endswith('.dcm'):
                    return ensure_nifti_format(f_path, output_dir)  # Đệ quy để convert DICOM

        raise ValueError("Không tìm thấy bất kỳ file ảnh y tế nào (.nii, .dcm) trong file ZIP.")

    else:
        raise ValueError("Định dạng file không được hỗ trợ. Vui lòng tải lên .nii.gz, .dcm hoặc .zip")

# backend/apps/ai_engine/tasks.py (Phác thảo logic)

@shared_task
def process_mri_study(study_id, uploaded_file_path):
    try:
        study = Study.objects.get(id=study_id)
        study.status = 'Processing'
        study.save()

        # Thư mục tạm thời
        work_dir = os.path.join(settings.MEDIA_ROOT, 'temp', f'study_{study_id}')
        os.makedirs(work_dir, exist_ok=True)

        # 1. Đảm bảo có file NIfTI gốc để hiển thị
        print(f"Đang xử lý file MRI: {uploaded_file_path}")
        nifti_file = ensure_nifti_format(uploaded_file_path, work_dir)

        # 2. CHẠY AI SUY LUẬN (Tích hợp mô hình SwinUNETR vào đây)
        # TẠM THỜI GIẢ LẬP ĐỂ TEST LUỒNG
        time.sleep(2)

        # Lấy kích thước thật của ảnh vừa nạp
        img_nii = nib.load(nifti_file)
        img_data = img_nii.get_fdata()
        affine = img_nii.affine

        # Tạo khối u giả lập ở chính giữa bức ảnh
        fake_mask = np.zeros(img_data.shape, dtype=np.uint8)
        cx, cy, cz = img_data.shape[0] // 2, img_data.shape[1] // 2, img_data.shape[2] // 2
        fake_mask[cx - 20:cx + 20, cy - 20:cy + 20, cz - 5:cz + 5] = 2  # Tạo cục u nhỏ ở giữa

        mask_nii = nib.Nifti1Image(fake_mask, affine)

        # 3. Chuẩn bị đường dẫn lưu file vào thư mục public media
        bg_filename = f"mri_bg_study_{study_id}.nii.gz"
        mask_filename = f"mri_mask_study_{study_id}.nii.gz"

        bg_rel_path = os.path.join('heatmaps', bg_filename)
        mask_rel_path = os.path.join('heatmaps', mask_filename)

        bg_full_path = os.path.join(settings.MEDIA_ROOT, bg_rel_path)
        mask_full_path = os.path.join(settings.MEDIA_ROOT, mask_rel_path)

        os.makedirs(os.path.dirname(bg_full_path), exist_ok=True)

        # Copy file gốc và lưu mask
        shutil.copy(nifti_file, bg_full_path)
        nib.save(mask_nii, mask_full_path)

        # 4. Cập nhật Database
        Prediction.objects.update_or_create(
            study=study,
            defaults={
                'prediction_label': 'High-grade Glioma',
                'probability': '94.2%',
                'heatmaps': {
                    # SỬA LỖI Ở ĐÂY: Dùng f-string nối chuỗi trực tiếp với dấu "/", không dùng os.path.join
                    'background_url': f"{settings.MEDIA_URL}heatmaps/{bg_filename}",
                    'mask_url': f"{settings.MEDIA_URL}heatmaps/{mask_filename}"
                }
            }
        )

        study.status = 'Completed'
        study.save()

        # Dọn dẹp thư mục tạm
        shutil.rmtree(work_dir, ignore_errors=True)
        return {'study_id': study_id, 'status': 'Success'}

    except Exception as e:
        if 'study' in locals():
            study.status = 'Failed'
            study.save()
        print(f"Lỗi hệ thống MRI: {str(e)}")
        return {'study_id': study_id, 'status': 'Failed', 'error': str(e)}