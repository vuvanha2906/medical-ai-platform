from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from rest_framework.response import Response
from rest_framework import status, generics
from .models import Study, Prediction
from .serializers import StudySerializer
from apps.ai_engine.tasks import process_xray_study
import uuid
from django.views.generic import TemplateView, ListView, DetailView
from rest_framework.views import APIView
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count

class StudyUploadView(generics.CreateAPIView):
    """
    Endpoint to upload a new study (image + metadata).
    Handles missing `patient_name` by generating an anonymous identifier.
    """
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def post(self, request, *args, **kwargs):
        # DRF’s request.data can be immutable (e.g., QueryDict), so copy it.
        data = request.data.copy()

        # If the frontend does not supply a patient_name, generate an anonymous one.
        if not data.get('patient_name'):
            anon_name = f"Anonymous_Patient_{uuid.uuid4().hex[:6]}"
            data['patient_name'] = anon_name

        # Use the serializer with the (potentially) augmented data.
        serializer = self.get_serializer(data=data)

        if serializer.is_valid():
            study = serializer.save()
            # Trigger Celery task for AI processing.
            process_xray_study.delay(study.id, study.image.path)
            return Response({'study_id': study.id, 'status': 'Processing'}, status=status.HTTP_201_CREATED)

        # Print serializer errors to the terminal for debugging (as requested earlier).
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def DashboardView(request):
    studies_list = Study.objects.all().order_by('-id')
    return render(request, 'studies/dashboard.html', {'studies': studies_list})

def XrayAnalysisView(request):
    return HttpResponse("X-ray Analysis Module coming soon")

def MriAlzheimerView(request):
    return HttpResponse("MRI Alzheimer Module coming soon")

class StudyListCreateView(generics.ListCreateAPIView):
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        study = serializer.save()

        # Start the Celery task
        task = process_xray_study.delay(study.id, study.image.path)

        # Return the task ID immediately
        return JsonResponse({'task_id': task.id})

class StudyDetailView(generics.RetrieveAPIView):
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def get(self, request, *args, **kwargs):
        study = self.get_object()
        prediction = getattr(study, 'prediction', None)
        if prediction:
            return JsonResponse({
                'study_id': study.id,
                'patient_name': study.patient_name,
                'modality': study.modality,
                'status': study.status,
                'results': prediction.results,
                'heatmap_url': prediction.heatmap_url
            })
        else:
            return JsonResponse({
                'study_id': study.id,
                'patient_name': study.patient_name,
                'modality': study.modality,
                'status': study.status,
                'results': {},
                'heatmap_url': None
            })

class TaskStatusView(generics.GenericAPIView):
    def get(self, request, task_id):
        task_result = AsyncResult(task_id)
        if task_result.ready():
            result = task_result.get()
            study = Study.objects.get(id=result['study_id'])
            prediction, created = Prediction.objects.get_or_create(study=study)
            prediction.results = result['prediction']
            prediction.heatmap_url = result['heatmap_path']
            prediction.save()
            return JsonResponse({
                'status': 'Completed',
                'results': result['prediction'],
                'heatmap_url': result['heatmap_path']
            })
        else:
            return JsonResponse({'status': 'Processing'})

class StudyListView(ListView):
    """
    Displays a list of all studies, ordered by newest first.
    """
    model = Study
    template_name = "studies/study_list.html"
    context_object_name = "studies"
    paginate_by = 25  # Optional pagination; adjust as needed

    def get_queryset(self):
        # Order by creation date descending (newest first).
        # If `created_at` does not exist, fallback to ordering by primary key.
        if hasattr(Study, "created_at"):
            return Study.objects.all().order_by("-created_at")
        return Study.objects.all().order_by("-id")

class ReportListView(ListView):
    """
    Trang Reports: Chỉ hiển thị danh sách các ca đã phân tích xong (Completed).
    Tái sử dụng lại giao diện bảng của trang study_list.
    """
    model = Study
    template_name = "studies/study_list.html" # Dùng lại UI cực đẹp của bạn
    context_object_name = "studies"
    paginate_by = 25

    def get_queryset(self):
        # Lọc ra những ca có status là 'Completed' và sắp xếp mới nhất lên đầu
        if hasattr(Study, "created_at"):
            return Study.objects.filter(status='Completed').order_by('-created_at')
        return Study.objects.filter(status='Completed').order_by('-id')

class AnalyticsView(TemplateView):
    """
    Serves the analytics dashboard page (`analytics.html`).
    The page will later fetch data from `AnalyticsDataView` via AJAX.
    """
    template_name = "studies/analytics.html"

class AnalyticsDataView(APIView):
    """
    API endpoint trả về dữ liệu tổng hợp (aggregated data) cho biểu đồ trên trang Analytics.
    """
    def get(self, request, *args, **kwargs):
        # 1. Tổng số ca phân tích (Total Studies)
        total_studies = Study.objects.count()

        # 2. Phân bố theo loại ảnh chụp (Modality Distribution)
        # Kết quả: [{'modality': 'X-ray', 'count': 50}, {'modality': 'MRI', 'count': 20}]
        modality_counts = Study.objects.values('modality').annotate(count=Count('id')).order_by('-count')

        # 3. Phân bố theo trạng thái (Status Distribution)
        # Kết quả: [{'status': 'Completed', 'count': 80}, {'status': 'Processing', 'count': 5}]
        status_counts = Study.objects.values('status').annotate(count=Count('id')).order_by('-count')

        # 4. Xu hướng số ca phân tích trong 7 ngày qua (7-day Trend)
        # Nhóm dữ liệu theo ngày để vẽ biểu đồ đường (Line chart)
        seven_days_ago = timezone.now() - timedelta(days=7)
        trend_data = (
            Study.objects.filter(created_at__gte=seven_days_ago)
            .annotate(date=TruncDate('created_at'))  # Ép kiểu datetime về date
            .values('date')
            .annotate(count=Count('id'))
            .order_by('date')
        )

        # Đóng gói dữ liệu trả về cho frontend
        data = {
            'total_studies': total_studies,
            'modality_distribution': list(modality_counts),
            'status_distribution': list(status_counts),
            'trend_data': list(trend_data)
        }

        return Response(data)