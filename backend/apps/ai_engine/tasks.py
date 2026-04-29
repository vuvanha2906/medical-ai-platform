import os
from celery import shared_task
from django.apps import apps
from django.conf import settings
from ai_engine.xray.inference import XrayPredictor

@shared_task
def process_xray_study(study_id: int, image_path: str) -> dict:
    """Process the X-ray study and return the prediction results and heatmap path."""
    # Import the Study and Prediction models
    Study = apps.get_model('studies', 'Study')
    Prediction = apps.get_model('studies', 'Prediction')

    # Get the study instance
    study = Study.objects.get(id=study_id)

    # Check if the study has an uploaded image
    if not study.image:
        raise ValueError("No image uploaded for this study.")

    ai_provider = XrayPredictor()

    # Predict the probabilities
    prediction_result = ai_provider.predict(image_path)

    # Generate the Grad-CAM heatmap
    heatmap_filename = f'heatmap_{study_id}.png'
    heatmap_path = os.path.join(settings.MEDIA_ROOT, heatmap_filename)
    ai_provider.generate_heatmap(image_path, heatmap_path)
    # Save the predictions to the Prediction model
    prediction = Prediction(
            study=study,
        results=prediction_result['probabilities'],
        heatmap_path=heatmap_filename
        )
    prediction.save()

    # Mark the study as completed
    study.status = 'Completed'
    study.save()

    # Return the results
    return {
        'study_id': study_id,
        'prediction': prediction_result['probabilities'],
        'heatmap_path': heatmap_filename
    }
