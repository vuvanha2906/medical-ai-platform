from django.db import models
import uuid
from django.utils import timezone

# Define choices for modality and status
MODALITY_CHOICES = [
    ('X-ray', 'X-ray'),
    ('MRI', 'MRI')
]

STATUS_CHOICES = [
    ('Pending', 'Pending'),
    ('Processing', 'Processing'),
    ('Completed', 'Completed'),
    ('Failed', 'Failed')
]


class Study(models.Model):
    patient_name = models.CharField(max_length=100)
    image = models.FileField(upload_to='studies/')

    modality = models.CharField(max_length=50, blank=True, null=True)
    status = models.CharField(max_length=50, default='Pending')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.patient_name} - {self.modality}"


class Prediction(models.Model):
    study = models.OneToOneField(Study, on_delete=models.CASCADE, related_name='prediction')
    prediction_label = models.CharField(max_length=255, null=True, blank=True)
    probability = models.CharField(max_length=50, null=True, blank=True)
    results = models.JSONField(default=dict, null=True, blank=True)
    heatmap_url = models.URLField(blank=True, null=True)
    heatmaps = models.JSONField(default=dict, null=True, blank=True)

    def __str__(self):
        return f"Prediction for {self.study}"
