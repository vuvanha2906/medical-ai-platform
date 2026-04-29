from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from .models import Study, Prediction
from .serializers import StudySerializer, PredictionSerializer
from apps.ai_engine.tasks import process_xray_study
import uuid

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
