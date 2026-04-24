from rest_framework import serializers
from .models import Study, Prediction

class StudySerializer(serializers.ModelSerializer):
    class Meta:
        model = Study
        fields = ['id', 'anonymous_patient_info', 'modality', 'status']

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'study', 'inference_results', 'heatmap_image', 'execution_time']