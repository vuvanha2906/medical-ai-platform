import time
from celery import shared_task
from django.utils import timezone
from apps.studies.models import Study, Prediction

@shared_task
def process_medical_image(study_id):
    try:
        study = Study.objects.get(id=study_id)
        # Simulate AI processing time
        time.sleep(3)
        # Update the study status to 'Completed'
        study.status = 'Completed'
        study.save()
        # Create a Prediction record
        prediction = Prediction.objects.create(
            study=study,
            inference_results={"Pneumonia": 0.85},
            heatmap_image=None,
            execution_time=timezone.now()
        )
        print(f"Processed study {study_id} and created prediction {prediction.id}")
    except Study.DoesNotExist:
        print(f"Study with id {study_id} does not exist")