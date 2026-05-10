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

    # TRƯỜNG HỢP 3: Là file ZIP chứa chuỗi DICOM
    elif ext.endswith('.zip'):
        extract_dir = os.path.join(output_dir, 'extracted_dicom')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # A. Quét Metadata DICOM để tìm các mặt phẳng
        detected_planes = set()
        has_nifti = False
        nifti_path = None

        for root, _, files in os.walk(extract_dir):
            for file in files:
                f_path = os.path.join(root, file)
                if file.lower().endswith(('.nii', '.nii.gz')):
                    has_nifti = True
                    nifti_path = f_path
                else:
                    # Thử đọc Header DICOM (Rất nhanh vì không đọc pixel)
                    try:
                        dcm = pydicom.dcmread(f_path, stop_before_pixels=True)
                        if 'ImageOrientationPatient' in dcm:
                            plane = get_dicom_plane(dcm.ImageOrientationPatient)
                            detected_planes.add(plane)
                    except:
                        continue

        # Nếu trong ZIP chứa NIfTI (như BraTS), bỏ qua DICOM, xem như 3D
        if has_nifti:
            return nifti_path, True

        # KIỂM TRA ĐIỀU KIỆN 3 MẶT PHẲNG CỦA BẠN:
        has_3_planes = (len(detected_planes) == 3)
        print(f"Các mặt phẳng quét được trong DICOM: {detected_planes} -> Đủ 3 mặt: {has_3_planes}")

        # B. Dùng dicom2nifti chuyển chuỗi DICOM thành 1 khối NIfTI (.nii.gz)
        nifti_out_dir = os.path.join(output_dir, 'nifti_converted')
        os.makedirs(nifti_out_dir, exist_ok=True)

        try:
            dicom2nifti.convert_directory(extract_dir, nifti_out_dir, compression=True, reorient=True)
            generated_files = [f for f in os.listdir(nifti_out_dir) if f.endswith('.nii.gz')]
            if generated_files:
                return os.path.join(nifti_out_dir, generated_files[0]), has_3_planes
        except Exception as e:
            print(f"Cảnh báo gom DICOM: {e}")
            # Fallback
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.dcm', '.dicom')):
                        fallback_path, _ = ensure_nifti_format(os.path.join(root, file), output_dir)
                        return fallback_path, False

        raise ValueError("Không tìm thấy dữ liệu hợp lệ trong file ZIP.")
    else:
        raise ValueError("Định dạng file không được hỗ trợ.")


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
        # NHẬN BIẾN has_3_planes TỪ HELPER
        nifti_file, has_3_planes = ensure_nifti_format(uploaded_file_path, work_dir)

        print("Đang chạy mô hình AI SwinUNETR...")

        bg_filename = f"mri_bg_study_{study_id}.nii.gz"
        mask_filename = f"mri_mask_study_{study_id}.nii.gz"

        bg_rel_path = os.path.join('heatmaps', bg_filename)
        mask_rel_path = os.path.join('heatmaps', mask_filename)

        bg_full_path = os.path.join(settings.MEDIA_ROOT, bg_rel_path)
        mask_full_path = os.path.join(settings.MEDIA_ROOT, mask_rel_path)

        os.makedirs(os.path.dirname(bg_full_path), exist_ok=True)
        shutil.copy(nifti_file, bg_full_path)

        # ==========================================
        # GỌI SUY LUẬN THẬT VÀ NHẬN KẾT QUẢ
        # ==========================================
        predictor = get_mri_predictor()
        inference_result = predictor.predict_and_save_mask(nifti_file, mask_full_path)
        # ==========================================
        # CẬP NHẬT DATABASE VỚI KẾT QUẢ ĐỘNG
        # ==========================================
        heatmaps_dict = {
            'background_url': f"{settings.MEDIA_URL}heatmaps/{bg_filename}",
            'force_2d': not has_3_planes
        }

        # Chỉ truyền mask_url cho giao diện nếu AI thực sự tìm thấy khối u
        if inference_result['has_tumor']:
            heatmaps_dict['mask_url'] = f"{settings.MEDIA_URL}heatmaps/{mask_filename}"

        Prediction.objects.update_or_create(
            study=study,
            defaults={
                'prediction_label': inference_result['prediction_label'],  # Nhãn lấy từ mô hình
                'probability': inference_result['probability'],  # Xác suất lấy từ mô hình
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

        traceback.print_exc()  # Bắt buộc phải có dòng này để in lỗi đỏ

        print("-" * 50)

        return {'study_id': study_id, 'status': 'Failed', 'error': str(e)}
