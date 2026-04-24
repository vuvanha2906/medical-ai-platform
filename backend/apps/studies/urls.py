from django.urls import path
from .views import StudyUploadView, DashboardView, XrayAnalysisView, MriAlzheimerView

urlpatterns = [
    path('upload/', StudyUploadView.as_view(), name='study-upload'),
    path('', DashboardView, name='dashboard'),
    path('xray-analysis/', XrayAnalysisView, name='xray_analysis'),
    path('mri-alzheimer/', MriAlzheimerView, name='mri_alzheimer'),
]