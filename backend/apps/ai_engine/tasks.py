import os
import sys
from celery import shared_task
from django.apps import apps
from django.conf import settings
from ai_engine.xray.inference import XrayPredictor
from backend.apps.studies.models import Study, Prediction

PROJECT_ROOT = os.path.dirname(settings.BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# Khởi tạo predictor 1 lần duy nhất khi worker chạy (để không tốn thời gian load weights mỗi lần dự đoán)
# Đảm bảo file .pth đã nằm đúng chỗ
WEIGHTS_PATH = os.path.join(settings.BASE_DIR, 'apps', 'ai_engine', 'xray', 'weights', 'nih_densenet_best.pth')
print(f"Loading weights from: {WEIGHTS_PATH}")
predictor = XrayPredictor(weights_path=WEIGHTS_PATH)

@shared_task
def process_xray_study(study_id, image_path):
    try:
        study = Study.objects.get(id=study_id)
        study.status = 'Processing'
        study.save()

        # Đường dẫn lưu file Heatmap
        heatmap_filename = f"heatmap_study_{study_id}.png"
        heatmap_rel_path = os.path.join('heatmaps', heatmap_filename)
        heatmap_full_path = os.path.join(settings.MEDIA_ROOT, heatmap_rel_path)

        # Đảm bảo thư mục media/heatmaps tồn tại
        os.makedirs(os.path.dirname(heatmap_full_path), exist_ok=True)

        # Chạy AI Suy Luận + GradCAM
        result = predictor.predict_and_explain(
            image_path=image_path,
            output_heatmap_path=heatmap_full_path
        )

        # Lưu kết quả vào Database
        Prediction.objects.update_or_create(
            study=study,
            defaults={
                'prediction_label': result['predicted_class'],
                'probability': result['probability'],
                'results': result['all_probabilities'],
                'heatmap_url': f"{settings.MEDIA_URL}{heatmap_rel_path}"
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