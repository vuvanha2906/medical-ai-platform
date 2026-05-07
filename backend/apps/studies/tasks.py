import os
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