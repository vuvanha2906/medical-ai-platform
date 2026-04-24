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
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    anonymous_patient_info = models.CharField(max_length=255)
    modality = models.CharField(max_length=10, choices=MODALITY_CHOICES)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='Pending')

    def __str__(self):
        return f"Study {self.id} - {self.anonymous_patient_info} ({self.modality})"

class Prediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    study = models.ForeignKey(Study, related_name='predictions', on_delete=models.CASCADE)
    inference_results = models.JSONField()
    heatmap_image = models.ImageField(upload_to='heatmaps/')
    execution_time = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Prediction {self.id} for Study {self.study.id}"