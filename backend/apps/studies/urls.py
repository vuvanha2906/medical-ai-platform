# backend/apps/studies/urls.py
from django.urls import path
from .views import (
    StudyUploadFormView, StudyRetrieveView, # API Views cũ
    home_view, upload_study_view, result_view # Web Views mới
)

urlpatterns = [
    # --- API ENDPOINTS (Dành cho Mobile App sau này) ---
    path('api/upload/', StudyUploadFormView.as_view(), name='api-study-upload'),
    path('api/<uuid:pk>/', StudyRetrieveView.as_view(), name='api-study-detail'),

    # --- WEB INTERFACE (Dành cho trình duyệt) ---
    path('', home_view, name='home-view'),
    path('upload/', upload_study_view, name='study-upload'), # Action của Form
    path('result/<uuid:pk>/', result_view, name='study-result'),
]