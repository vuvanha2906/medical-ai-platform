# backend/apps/studies/urls.py
from django.urls import path
from . import views

app_name = "studies"

urlpatterns = [
    path('', views.DashboardView, name='dashboard'),
    path('upload/', views.StudyUploadView.as_view(), name='upload'),
    path('studies/', views.StudyListView.as_view(), name='study_list'),

    # Chỉ giữ lại link chi tiết và link tải PDF
    path('reports/<int:pk>/', views.ReportDetailView.as_view(), name='report_detail'),
    path('reports/<int:pk>/pdf/', views.DownloadReportPDFView.as_view(), name='download_pdf'),

    path('analytics/', views.AnalyticsView.as_view(), name='analytics'),
    path('api/analytics/', views.AnalyticsDataView.as_view(), name='analytics_data'),
    path('api/chat/', views.chat_with_ai, name='chat_with_ai'),
]