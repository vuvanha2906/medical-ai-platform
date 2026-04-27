from django.urls import path
from .views import StudyUploadView, DashboardView, XrayAnalysisView, MriAlzheimerView

app_name = 'studies'

urlpatterns = [
    path('upload/', StudyUploadView.as_view(), name='upload'),
    path('', DashboardView, name='dashboard'),
    path('xray-analysis/', XrayAnalysisView, name='xray_analysis'),
    path('mri-alzheimer/', MriAlzheimerView, name='mri_alzheimer'),
]