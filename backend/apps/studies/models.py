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
    modality = models.CharField(max_length=50)
    status = models.CharField(max_length=50, default='Pending')
    image = models.ImageField(upload_to='studies/')

    def __str__(self):
        return f"{self.patient_name} - {self.modality}"

class Prediction(models.Model):
    study = models.OneToOneField(Study, on_delete=models.CASCADE, related_name='prediction')
    results = models.JSONField()
    heatmap_url = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"Prediction for {self.study}"