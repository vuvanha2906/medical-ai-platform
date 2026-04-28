from rest_framework import serializers
from .models import Study, Prediction

class StudySerializer(serializers.ModelSerializer):
    class Meta:
        model = Study
        fields = ['id', 'patient_name', 'modality', 'status', 'image']

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'study', 'results', 'heatmap_url']