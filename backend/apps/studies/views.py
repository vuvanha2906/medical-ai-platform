from django.shortcuts import render, HttpResponse
from rest_framework.response import Response
from rest_framework import status, generics
from .models import Study, Prediction
from .serializers import StudySerializer
from .tasks import process_xray_study, process_mri_study
import uuid
from django.views.generic import TemplateView, ListView, DetailView
from rest_framework.views import APIView
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta
from django.db.models import Count
from .utils import render_to_pdf
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from rest_framework.permissions import IsAuthenticated


class StudyUploadView(generics.CreateAPIView):
    """Endpoint to upload a new study (image + metadata)."""
    queryset = Study.objects.all()
    serializer_class = StudySerializer
    permission_classes = [IsAuthenticated]  # Thêm bảo mật

    def post(self, request, *args, **kwargs):
        if hasattr(request.data, '_mutable'):
            request.data._mutable = True

        if not request.data.get('patient_name'):
            request.data['patient_name'] = f"Anonymous_Patient_{uuid.uuid4().hex[:6]}"

        if hasattr(request.data, '_mutable'):
            request.data._mutable = False

        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            study = serializer.save()

            # ĐIỀU HƯỚNG TASK DỰA VÀO MODALITY
            modality = study.modality.lower() if study.modality else ''
            if 'mri' in modality:
                process_mri_study.delay(study.id, study.image.path)
            else:
                process_xray_study.delay(study.id, study.image.path)
            return Response({'study_id': study.id, 'status': 'Processing'}, status=status.HTTP_201_CREATED)

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

class StudyListView(LoginRequiredMixin, ListView):
    model = Study
    template_name = "studies/study_list.html"
    context_object_name = "studies"
    paginate_by = 25

    def get_queryset(self):
        if hasattr(Study, "created_at"):
            return Study.objects.all().order_by("-created_at")
        return Study.objects.all().order_by("-id")

class ReportDetailView(LoginRequiredMixin, DetailView):
    model = Study
    template_name = "studies/report_detail.html"
    context_object_name = "study"
    pk_url_kwarg = "pk"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            context["prediction"] = Prediction.objects.get(study=self.object)
        except Prediction.DoesNotExist:
            context["prediction"] = None
        return context


class DownloadReportPDFView(LoginRequiredMixin, DetailView):
    model = Study
    template_name = "studies/report_pdf.html"
    pk_url_kwarg = "pk"

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        try:
            prediction = Prediction.objects.get(study=self.object)
        except Prediction.DoesNotExist:
            prediction = None

        pdf_response = render_to_pdf(self.template_name, {"study": self.object, "prediction": prediction})
        return pdf_response


class AnalyticsView(LoginRequiredMixin, TemplateView):
    template_name = "studies/analytics.html"

class AnalyticsDataView(APIView):
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

        return Response({
            'total_studies': total_studies,
            'modality_distribution': list(modality_counts),
            'status_distribution': list(status_counts),
            'trend_data': list(trend_data)
        })