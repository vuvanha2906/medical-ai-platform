import time
import random


def run_inference_service(study_instance):
    """
    Hàm này đóng vai trò Facade (mặt tiền) để gọi các model AI thực tế.
    Input: Instance của Study model
    Output: Cập nhật instance đó với kết quả
    """
    try:
        print(f"--> AI Engine started for: {study_instance.id} ({study_instance.study_type})")

        # Giả lập thời gian xử lý AI
        # Trong thực tế, đoạn này sẽ load model PyTorch và predict
        time.sleep(2)

        results = {}

        if study_instance.study_type == 'CXR':
            # Mock logic X-ray
            results = {
                "pneumonia": round(random.uniform(0.1, 0.9), 2),
                "edema": round(random.uniform(0.0, 0.3), 2),
                "normal": round(random.uniform(0.1, 0.5), 2)
            }
        elif study_instance.study_type == 'MRI':
            # Mock logic MRI
            labels = ["CN", "MCI", "AD"]
            diagnosis = random.choice(labels)
            results = {
                "diagnosis": diagnosis,
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "hippocampus_volume": "reduced" if diagnosis == "AD" else "normal"
            }

        # Cập nhật kết quả vào DB
        study_instance.ai_results = results
        study_instance.status = 'SUCCESS'

        # (Optional) study_instance.heatmap_image = ... (Path to generated heatmap)

        study_instance.save()
        print(f"--> AI Engine finished: {study_instance.status}")

    except Exception as e:
        study_instance.status = 'FAILED'
        study_instance.error_log = str(e)
        study_instance.save()