import uuid
from django.db import models


class Study(models.Model):
    # Định nghĩa các loại chẩn đoán
    STUDY_TYPES = (
        ('CXR', 'Chest X-Ray'),
        ('MRI', 'Brain MRI'),
    )

    # Trạng thái xử lý (Workflow status)
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('SUCCESS', 'Success'),
        ('FAILED', 'Failed'),
    )

    # Sử dụng UUID để bảo mật hơn trong môi trường production
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Metadata cơ bản
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Input: File ảnh hoặc file MRI (.dcm, .nii.gz)
    image = models.FileField(upload_to='uploads/%Y/%m/%d/')
    study_type = models.CharField(max_length=10, choices=STUDY_TYPES)

    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')

    # Output: Kết quả AI (Lưu JSON để linh hoạt cấu trúc)
    # Ví dụ Xray: {"pneumonia": 0.9, "normal": 0.1}
    # Ví dụ MRI: {"diagnosis": "AD", "confidence": 0.85}
    ai_results = models.JSONField(null=True, blank=True)

    # Explainable AI: Đường dẫn tới ảnh Heatmap/Grad-CAM
    heatmap_image = models.ImageField(upload_to='heatmaps/%Y/%m/%d/', null=True, blank=True)

    # Ghi log lỗi nếu có
    error_log = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.study_type} - {self.id} - {self.status}"