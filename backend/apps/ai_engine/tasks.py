import os
from celery import shared_task
from django.apps import apps
from ai_engine.xray.inference import XrayPredictor
@shared_task
def process_medical_image(study_id: int):
    # Import the Study and Prediction models
    Study = apps.get_model('studies', 'Study')
    Prediction = apps.get_model('studies', 'Prediction')

    # Get the study instance
    study = Study.objects.get(id=study_id)

    # Check if the study has an uploaded image
    if not study.image:
        raise ValueError("No image uploaded for this study.")

    # Get the image path
    image_path = study.image.path

    # Instantiate the XrayPredictor
    predictor = XrayPredictor()

    # Predict the top 3 classes
    predictions = predictor.predict(image_path)

    # Save the predictions to the Prediction model
    prediction = Prediction(
            study=study,
        results=predictions,
        heatmap_path=None  # Placeholder for future Grad-CAM heatmap
        )
    prediction.save()

    # Mark the study as completed
    study.status = 'Completed'
    study.save()
