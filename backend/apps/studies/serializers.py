from rest_framework import serializers
from .models import Study


class StudyUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Study
        fields = ['image', 'study_type']

    def validate_image(self, value):
        # Validate file size hoặc extension ở đây nếu cần (Production check)
        if value.size > 50 * 1024 * 1024:  # Giới hạn 50MB
            raise serializers.ValidationError("File size too large!")
        return value


class StudyDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Study
        fields = [
            'id', 'created_at', 'study_type', 'status',
            'image', 'ai_results', 'heatmap_image'
        ]
        read_only_fields = fields  # Output chỉ đọc