from django.http import JsonResponse
from django.shortcuts import render, HttpResponse, get_object_or_404
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
from .utils import render_to_pdf  # Kéo tính năng PDF vào
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from rest_framework.permissions import IsAuthenticated
from celery.result import AsyncResult


class StudyUploadView(generics.CreateAPIView):
    """
    Endpoint to upload a new study (image + metadata).
    Handles missing `patient_name` by generating an anonymous identifier.
    """
    permission_classes = [IsAuthenticated]
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def post(self, request, *args, **kwargs):
        # DRF's request.data can be immutable (e.g., QueryDict), so copy it.
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


@login_required
def DashboardView(request):
    studies_list = Study.objects.all().order_by('-id')
    return render(request, 'studies/dashboard.html', {'studies': studies_list})


@login_required
def XrayAnalysisView(request):
    return HttpResponse("X-ray Analysis Module coming soon")


@login_required
def MriAlzheimerView(request):
    return HttpResponse("MRI Alzheimer Module coming soon")


class StudyListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
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
    permission_classes = [IsAuthenticated]
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
    permission_classes = [IsAuthenticated]
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


class StudyListView(LoginRequiredMixin, ListView):
    """
    Displays a list of all studies, ordered by newest first.
    """
    model = Study
    template_name = "studies/study_list.html"
    context_object_name = "studies"
    paginate_by = 25

    def get_queryset(self):
        if hasattr(Study, "created_at"):
            return Study.objects.all().order_by("-created_at")
        return Study.objects.all().order_by("-id")

class ReportDetailView(LoginRequiredMixin, DetailView):
    """
    Shows the AI diagnostic report for a specific study.
    """
    model = Study
    template_name = "studies/report_detail.html"
    context_object_name = "study"
    pk_url_kwarg = "pk"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            prediction = Prediction.objects.get(study=self.object)
            context["prediction"] = prediction
        except Prediction.DoesNotExist:
            context["prediction"] = None
        return context


# ---> TÍNH NĂNG MỚI ĐƯỢC CHÈN VÀO ĐÂY <---
class DownloadReportPDFView(LoginRequiredMixin, DetailView):
    """
    Generates and downloads a PDF version of the AI diagnostic report.
    """
    model = Study
    template_name = "studies/report_pdf.html"
    pk_url_kwarg = "pk"

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        try:
            prediction = Prediction.objects.get(study=self.object)
        except Prediction.DoesNotExist:
            prediction = None

        context = {
            "study": self.object,
            "prediction": prediction,
        }

        pdf_response = render_to_pdf(self.template_name, context)
        return pdf_response


# ----------------------------------------

class AnalyticsView(LoginRequiredMixin, TemplateView):
    """
    Serves the analytics dashboard page (`analytics.html`).
    """
    template_name = "studies/analytics.html"


class AnalyticsDataView(APIView):
    """
    API endpoint trả về dữ liệu tổng hợp.
    """
    permission_classes = [IsAuthenticated]
    def get(self, request, *args, **kwargs):
        total_studies = Study.objects.count()
        modality_counts = Study.objects.values('modality').annotate(count=Count('id')).order_by('-count')
        status_counts = Study.objects.values('status').annotate(count=Count('id')).order_by('-count')

        seven_days_ago = timezone.now() - timedelta(days=7)
        trend_data = (
            Study.objects.filter(created_at__gte=seven_days_ago)
            .annotate(date=TruncDate('created_at'))
            .values('date')
            .annotate(count=Count('id'))
            .order_by('date')
        )

        data = {
            'total_studies': total_studies,
            'modality_distribution': list(modality_counts),
            'status_distribution': list(status_counts),
            'trend_data': list(trend_data)
        }

        return Response(data)