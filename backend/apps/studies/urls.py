# backend/apps/studies/urls.py
from django.urls import path
from . import views

app_name = "studies"

urlpatterns = [
    path('', views.DashboardView, name='dashboard'),
    path('upload/', views.StudyUploadView.as_view(), name='upload'),
    path('studies/', views.StudyListView.as_view(), name='study_list'),
    path('reports/', views.ReportListView.as_view(), name='report_detail'),
    path('reports/<int:pk>/', views.ReportListView.as_view(), name='report_detail'),
    path('analytics/', views.AnalyticsView.as_view(), name='analytics'),
    path('api/analytics/', views.AnalyticsDataView.as_view(), name='analytics_data'),
]