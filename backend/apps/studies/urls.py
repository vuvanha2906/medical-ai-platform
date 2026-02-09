from django.urls import path
from .views import StudyUploadView, StudyRetrieveView

urlpatterns = [
    path('upload/', StudyUploadView.as_view(), name='study-upload'),
    path('<uuid:pk>/', StudyRetrieveView.as_view(), name='study-detail'),
]