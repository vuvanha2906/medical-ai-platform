import os
import traceback
import zipfile
import shutil
import dicom2nifti
import nibabel as nib
from pathlib import Path
from celery import shared_task
from PIL import Image
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from django.conf import settings
from ai_engine.xray.inference import XrayPredictor
from ai_engine.mri_tumor.inference import MRITumorPredictor
from .models import Study, Prediction

PROJECT_ROOT = Path(settings.BASE_DIR).parent
WEIGHTS_PATH = PROJECT_ROOT / 'ai_engine' / 'xray' / 'weights' / 'best_nih_densenet121.pth'
MRI_WEIGHTS_PATH = PROJECT_ROOT / 'ai_engine' / 'mri_tumor' / 'weights' / 'best_swinunetr_brats.pth'

_xray_predictor = None
_mri_predictor = None


def get_xray_predictor():
    global _xray_predictor
    if _xray_predictor is None:
        print(f"Loading XRAY weights from: {WEIGHTS_PATH}")
        _xray_predictor = XrayPredictor(weights_path=WEIGHTS_PATH)
    return _xray_predictor


def get_mri_predictor():
    global _mri_predictor
    if _mri_predictor is None:
        print(f"Loading MRI weights from: {MRI_WEIGHTS_PATH}")
        _mri_predictor = MRITumorPredictor(weights_path=MRI_WEIGHTS_PATH)
    return _mri_predictor


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
        img = Image.fromarray(pixel_array).convert('RGB')

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
        predictor = get_xray_predictor()
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
# HELPER MỚI: PHÁT HIỆN MẶT PHẲNG CHỤP TỪ METADATA DICOM
# =================================================================
def get_dicom_plane(image_orientation):
    """
    Sử dụng vector chỉ phương của ảnh DICOM để xác định mặt phẳng:
    Sagittal, Coronal, hoặc Axial.
    """
    row_cosine = np.array(image_orientation[:3])
    col_cosine = np.array(image_orientation[3:])

    # Tính tích có hướng (Cross product) để tìm Vector pháp tuyến
    normal_vector = np.cross(row_cosine, col_cosine)
    abs_normal = np.abs(normal_vector)

    # Trục nào có giá trị lớn nhất thì mặt phẳng vuông góc với trục đó
    dominant_axis = np.argmax(abs_normal)

    if dominant_axis == 0:
        return 'Sagittal'
    elif dominant_axis == 1:
        return 'Coronal'
    else:
        return 'Axial'


# =================================================================
# HELPER: CHUYỂN ĐỔI BẤT KỲ ĐỊNH DẠNG NÀO SANG NIFTI (.nii.gz)
# =================================================================
def ensure_nifti_format(file_path, output_dir):
    """
    Trả về: (nifti_file_path, has_3_planes_flag)
    """
    ext = file_path.lower()

    # TRƯỜNG HỢP 1: Đã là NIfTI (BraTS) -> Mặc định là 3D chuẩn
    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        return file_path, True

    # TRƯỜNG HỢP 2: Là 1 file DICOM đơn lẻ hoặc Multi-frame DICOM
    elif ext.endswith('.dcm') or ext.endswith('.dicom'):
        dicom_data = pydicom.dcmread(file_path)
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        # Nếu là Multi-frame DICOM (Ví dụ file của bạn có 21 ảnh -> shape: 21, 512, 512)
        if len(pixel_array.shape) == 3:
            # SỬA LỖI ĐẢO TRỤC: (Z, Y, X) -> (X, Y, Z)
            pixel_array = np.transpose(pixel_array, (2, 1, 0))

        # Nếu là 1 ảnh DICOM 2D duy nhất (shape: 512, 512)
        elif len(pixel_array.shape) == 2:
            pixel_array = np.transpose(pixel_array)  # (Y, X) -> (X, Y)
            pixel_array = np.expand_dims(pixel_array, axis=-1)

        nii_img = nib.Nifti1Image(pixel_array, affine=np.eye(4))
        nii_path = os.path.join(output_dir, 'converted_single_dicom.nii.gz')
        nib.save(nii_img, nii_path)
        return nii_path, False

    elif ext.endswith('.zip'):
        extract_dir = os.path.join(output_dir, 'extracted_dicom')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 👉 SỬA LỖI TẠI ĐÂY: Quét tìm NIfTI nhưng trả về THƯ MỤC thay vì 1 file
        has_nifti = False
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.nii', '.nii.gz')):
                    has_nifti = True
                    break
            if has_nifti: break

        # Trả về toàn bộ thư mục giải nén để inference.py tự quét sâu vào trong
        if has_nifti:
            return extract_dir, True


# backend/apps/ai_engine/tasks.py (Phác thảo logic)

@shared_task
def process_mri_study(study_id, uploaded_file_path):
    try:
        study = Study.objects.get(id=study_id)
        study.status = 'Processing'
        study.save()

        work_dir = os.path.join(settings.MEDIA_ROOT, 'temp', f'study_{study_id}')
        os.makedirs(work_dir, exist_ok=True)

        print(f"Đang xử lý MRI: {uploaded_file_path}")
        nifti_file, has_3_planes = ensure_nifti_format(uploaded_file_path, work_dir)

        print("Đang chạy mô hình AI SwinUNETR...")

        bg_filename = f"mri_bg_study_{study_id}.nii.gz"
        mask_filename = f"mri_mask_study_{study_id}.nii.gz"

        bg_rel_path = os.path.join('heatmaps', bg_filename)
        mask_rel_path = os.path.join('heatmaps', mask_filename)

        bg_full_path = os.path.join(settings.MEDIA_ROOT, bg_rel_path)
        mask_full_path = os.path.join(settings.MEDIA_ROOT, mask_rel_path)

        os.makedirs(os.path.dirname(bg_full_path), exist_ok=True)

        # ==========================================
        # 👉 FIX LỖI PERMISSION: CHỌN ĐÚNG FILE ĐỂ COPY LÀM ẢNH NỀN WEB
        # ==========================================
        if os.path.isdir(nifti_file):
            # Nếu là thư mục (đã quét ZIP 4 file), lục tìm file T1 làm đại diện
            nii_files = [os.path.join(root, f) for root, _, files in os.walk(nifti_file) for f in files if
                         f.lower().endswith(('.nii', '.nii.gz'))]
            bg_source = nii_files[0]  # Mặc định lấy file đầu tiên
            for f in nii_files:
                # Ưu tiên lấy file T1 (t1.nii) để hiển thị cấu trúc não rõ nhất
                if 't1' in f.lower() and 'ce' not in f.lower() and 'c' not in f.lower():
                    bg_source = f
                    break
            shutil.copy(bg_source, bg_full_path)
        else:
            # Nếu chỉ là 1 file đơn lẻ thì copy bình thường
            shutil.copy(nifti_file, bg_full_path)

        # Gọi mô hình dự đoán
        predictor = get_mri_predictor()
        inference_result = predictor.predict_and_save_mask(nifti_file, mask_full_path)

        # Cập nhật kết quả vào CSDL
        heatmaps_dict = {
            'background_url': f"{settings.MEDIA_URL}heatmaps/{bg_filename}",
            'force_2d': not has_3_planes
        }

        if inference_result['has_tumor']:
            heatmaps_dict['mask_url'] = f"{settings.MEDIA_URL}heatmaps/{mask_filename}"

        Prediction.objects.update_or_create(
            study=study,
            defaults={
                'prediction_label': inference_result['prediction_label'],
                'probability': inference_result['probability'],
                'heatmaps': heatmaps_dict
            }
        )

        study.status = 'Completed'
        study.save()
        shutil.rmtree(work_dir, ignore_errors=True)
        return {'study_id': study_id, 'status': 'Success'}

    except Exception as e:
        if 'study' in locals():
            study.status = 'Failed'
            study.save()
        print(f"\n❌ LỖI CRASH HỆ THỐNG MRI TẠI STUDY {study_id}:")
        traceback.print_exc()
        print("-" * 50)
        return {'study_id': study_id, 'status': 'Failed', 'error': str(e)}
